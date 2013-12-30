// Copyright 2009 Carsten Eie Frigaard.
//
// License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>.
// This is free software: you are free to change and redistribute it.
// There is NO WARRANTY, to the extent permitted by law.
//
// This software is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// any later version.
//
// This software is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this software.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __CUDAPRINTF_H__
#define __CUDAPRINTF_H__

#if CUDA_DEBUG_GX > 1
#define THREAD_MESSAGE_SIZE 4096

struct DevMsg
{
	char* m;
	size_t i;
	size_t end;

	size_t tid;
	size_t sz;
};

struct Char64
{
    char c[64];
};

__device__
struct Char64 CopyString(const char* s)
{
	//FUN_MESSAGE(6,"CopyString(%s)",s);

	struct Char64 b;
	int i;
	for(i=0;i<64;++i) {
		b.c[i]=s[i];
		if (s[i]==0) break;
	}
	if (i<63) b.c[i+1]=0;
	b.c[63]=0;

	return b;
}

__device__
struct Char64 itoa_device(const int val)
{
	//FUN_MESSAGE(6,"itoa_device(%d)",val);

	struct Char64 b;
	int x;
	for(x=0;x<64;++x) b.c[x]=0;

	x=0;
	if (val<0) b.c[x++]='-';
	if (val==0) b.c[x++]='0';

	int v=val >= 0 ? val : -val;
	int m=1000000000;
	int prezero=1;

	if (v>m) for(int j=0;j<6;++j) b.c[x++]='?'; else
	while(m>0){
		const int f=v/m;
		if (f>=0) {
			if (f!=0 && prezero) prezero=0;
			if (prezero==0) b.c[x++]='0' + f;
			v %= m;
		}
		m/=10;
		if (x>=64) return CopyString("_ITOA_OVERFLOW"); // avoid any overflow
	}

	return b;
}

/*
typedef union {
	long L;
	float F;
	} LF_t;

__device__
struct Char64 ftoa_device_old(const float val)
{
	//FUN_MESSAGE(6,"itof_device(%g)",val);

	struct Char64 b;
	if (val==0.0)
	{
		b.c[0] = '0';
		b.c[1] = '.';
		b.c[2] = '0';
		b.c[3] = 0;
		return b;
	}

	LF_t y;
	y.F=val;

	const short exp2 = (unsigned char)(y.L >> 23) - 127;
	const long mantissa = (y.L & 0xFFFFFF) | 0x800000;
	long frac_part = 0;
	long int_part = 0;

	if (exp2 >= 31)      return CopyString("_FTOA_TOO_LARGE");
	else if (exp2 < -23) return CopyString("_FTOA_TOO_SMALL");
	else if (exp2 >= 23) int_part = mantissa << (exp2 - 23);
	else if (exp2 >= 0) {
		int_part  = mantissa >> (23 - exp2);
		frac_part = (mantissa << (exp2 + 1)) & 0xFFFFFF;
	}
	else frac_part = (mantissa & 0xFFFFFF) >> -(exp2 + 1);  // if (exp2 < 0)
	b=itoa_device(int_part);
	const struct Char64 fp=itoa_device(frac_part);

	int x=0,j=0;
	while(x<64 && b.c[x]!=0) ++x;
	if(x<63) b.c[x++]='.';
	for(;x<64;++x) b.c[x]=fp.c[j++];
	b.c[63]=0;

	return b;
}
*/

__device__
struct Char64 ftoa_device(const float val,const int prec=3,int fstyle=0) // fstyle=1 -> %f, others %g
{
	//FUN_MESSAGE(6,"itof_device(%g)",x);

	struct Char64 b;
	int bi;
	for(bi=0;bi<64;++bi) b.c[bi]=0;
	bi=0;

	float x=val;
	if (x==0.0)
	{
		b.c[0] = '0';
		b.c[1] = '.';
		b.c[2] = '0';
		b.c[3] = 0;
		return b;
	}

	// converts a floating point number to an ascii string
	// x is stored into str, which should be at least 30 chars long
	int ie=0, i, k, ndig;
	double y;
	//if (nargs() != 7) IEHzap("ftoa  ");

	ndig = ( prec<=0) ? 7 : (prec > 22 ? 23 : prec+1);

	// print in e format unless last arg is 'f'
	// if x negative, write minus and reverse
	if ( x < 0)
	{
		b.c[bi++] =  '-';
		if (bi>63) return CopyString("_FTOA_BAD_1");
		x = -x;
	}

	// put x in range 1 <= x < 10
	i=0;
	if (x > 0.0) while (x < 1.0)
	{
		x *= 10.0;
		ie--;
		if (++i>20) return CopyString("_FTOA_BAD_2");
	}

	i=0;
	while (x >= 10.0)
	{
		x /= 10.0;
		ie++;
		if (++i>20) return CopyString("_FTOA_BAD_4");
	}

	if (ie<3 && ie>-3) fstyle=1; // return ftoa_device_old(val);

	// in f format, number of digits is related to size
	if (fstyle) ndig += ie;

	if (ndig>24) return CopyString("_FTOA_BAD_5"); // XXX mod, caf

	// round. x is between 1 and 10 and ndig will be printed to
	// right of decimal point so rounding is ...
	for (y = i = 1; i < ndig; i++) y = y/10.;

	x += y/2.;
	if (x >= 10.0) {x = 1.0; ie++;} // repair rounding disasters

	// now loop.  put out a digit (obtain by multiplying by
	// 10, truncating, subtracting) until enough digits out
	// if fstyle, and leading zeros, they go out special
	if (fstyle && ie<0)
	{
		b.c[bi++] = '0';
		if (bi>63) return CopyString("_FTOA_BAD_6");

		b.c[bi++] = '.';
		if (bi>63) return CopyString("_FTOA_BAD_7");

		if (ndig < 0) ie = ie-ndig; // limit zeros if underflow
			for (i = -1; i > ie; i--) {
				b.c[bi++] = k + '0';
				if (bi>63) return CopyString("_FTOA_BAD_8");
			}
	}

	for (i=0; i < ndig; i++)
	{
		k = x;

		b.c[bi++] = k + '0';
		if (bi>63) return CopyString("_FTOA_BAD_9");

		if (i == (fstyle ? ie : 0))  {
			b.c[bi++] =  '.'; // where is decimal point
			if (bi>63) return CopyString("_FTOA_BAD_10");
		}

		x -= (y=k);
		x *= 10.0;
	}

	// now, in estyle,  put out exponent if not zero
	if (!fstyle && ie!= 0)
	{
		b.c[bi++] = 'E';
		if (bi>63) return CopyString("_FTOA_BAD_11");

		if (ie < 0)
		{
			ie = -ie;

			b.c[bi++] = '-';
			if (bi>63) return CopyString("_FTOA_BAD_12");
		}
		for (k=100; k > ie; k /= 10) ;

		for (; k > 0; k /= 10)
		{
			b.c[bi++] = ie/k + '0';
			if (bi>63) return CopyString("_FTOA_BAD_13");

			ie = ie%k;
		}
	}
	return b;
}

__device__
void Set_device_thread_state(struct DevMsg* msg,const char state)
{
	//FUN_MESSAGE(6,"Set_device_thread_state(p,%c)",state);

	char* p=&msg->m[msg->tid*THREAD_MESSAGE_SIZE];
	if      (*p==0   && state=='S') *p=state;
	else if (*p=='S' && state=='D') *p=state;
	else if (*p=='S' && state=='F') *p=state;
	else if (*p=='F' && state=='F') *p=state;
	else  *p='?';
}

__device__
void PrintDevMsg(struct DevMsg*const msg,const char*const s,const int append)
{
	//FUN_MESSAGE(6,"PrintDevMsg(p,%s,%d) [msg.tid=%d, msg.sz=%d]",s,append,msg->tid,msg->sz);

	if (msg==0 || s==0) return;

	int sok=1;
	size_t i=0;
	while(sok && msg->i<msg->end) {
		const char c=s[i];

		if(c==0) sok=0;
		msg->m[msg->i]=c;
		++(msg->i);
		++i;
	}

	if (append && msg->i+1<msg->end && i>0) {
		msg->m[msg->i+1]=0;
		msg->m[msg->i-1]=',';
		if (append==2) --(msg->i);
	}
}

__device__
struct DevMsg ResetMsg(char*const base,const size_t tid,const size_t sz)
{
	//FUN_MESSAGE(6,"ResetMsg(p,%d)\n",tid);

	struct DevMsg msg;
	msg.m=base;
	msg.tid=tid;
	msg.sz=sz;
	msg.i=tid*THREAD_MESSAGE_SIZE;
	msg.end=(tid+1)*THREAD_MESSAGE_SIZE;

	if (msg.i+THREAD_MESSAGE_SIZE>sz){
		#ifdef __DEVICE_EMULATION__
			printf("** ResetMsg() index out of bounds");
		#endif
		msg.m=0;
		msg.tid=msg.sz=msg.i=msg.end=0;
		return msg; // invalid msg
	}

	//size_t i;
	//for(i=0;i<THREAD_MESSAGE_SIZE;++i) msg.m[msg.i+i]=0;

	Set_device_thread_state(&msg,'S');

	++msg.i;
	msg.m[msg.i]=0;
	msg.m[msg.end-1]=0;

	return msg;
}

	#define SET_DEVICE_THREAD_STATE(state) Set_device_thread_state(&devmsg,state);

	#define PRINT_DEV_MSG_S(value) {PrintDevMsg(&devmsg,value,1);}
	#define PRINT_DEV_MSG_I(value) {PrintDevMsg(&devmsg,__STRING(value),2); PrintDevMsg(&devmsg,"=",2); PrintDevMsg(&devmsg,itoa_device(value).c,1);}
	#define PRINT_DEV_MSG_D(value) {PrintDevMsg(&devmsg,__STRING(value),2); PrintDevMsg(&devmsg,"=",2); PrintDevMsg(&devmsg,ftoa_device(value).c,1);}
#else // CUDA_DEBUG_GX <= 0
	#define SET_DEVICE_THREAD_STATE(state)
	#define PRINT_DEV_MSG_S(value)
	#define PRINT_DEV_MSG_I(value)
	#define PRINT_DEV_MSG_D(value)
#endif

#endif // __CUDAPRINTF_H__
