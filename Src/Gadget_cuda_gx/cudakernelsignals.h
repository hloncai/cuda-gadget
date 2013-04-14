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

#ifndef __CUDA_KERNELSIGNALS_GX_H__
#define __CUDA_KERNELSIGNALS_GX_H__

#ifdef __CUDACC__

#include "cudautils.h"
#include "stdio.h"

struct KernelSignals
{
	#define SIG_MESSAGE_SIZE (1024-12*4) // allocate 1024 bytes
	char sig_msg[SIG_MESSAGE_SIZE]; // NOTE: points directly on __FILE__ chars

	int sig_exit;
	int sig_abort;
	int sig_line;

	unsigned int sig_tid;
	unsigned int sig_flag;

	// extra debug info types
	int m,o,p,q,r;
	float s,t;
};

struct KernelSignals* Initialize_KernelSignals();
void  Finalize_KernelSignals();
void  Free_KernelSignals();

/*
__device__
int MutexLock(const unsigned int* lock)
{
	ASSERT_DEVICE_GX(lock!=NULL);
	unsigned int f=1;

	int k=0;
	while(f){
		f=atomicCAS(lock,0,1);
		if (f==0) return 1;
		else {
			// sleep some
			for(int i=0;i<100;++i)
				for(int j=0;j<100;++j) k=i*j;
		}
	}
	k=0;
	return k; // avoid compiler warning
}

__device__
void MutexUnlock(const unsigned int* lock)
{
	ASSERT_DEVICE_GX(lock!=NULL);
	const unsigned int f=atomicDec(lock,0);
	ASSERT_DEVICE_GX( f==1 );
}
*/

// __device__ struct KernelSignals* g_kernelsignals_shared; // NOTE: causes warning in CUDA2.X (just ignore it): " Advisory: Cannot tell what pointer points to, assuming global memory space"

/* This function causes out-of-bounds memory errors. kernalsignals is allocated in a bizarre way, so need to check and fix that... */
/*
__forceinline__ __device__
int SetKernelSignal(const int code,const int exitorabort,const char* file,const int line,const char* msg,const int m,const int o,const int p,const int q,const int r,const float s,const float t)
{
	const unsigned int f=atomicAdd(&g_kernelsignals_shared->sig_flag,1);
	if (f>0) return code; // avoid race condition

	if (g_kernelsignals_shared->sig_exit!=0 || g_kernelsignals_shared->sig_abort!=0) return code; // a signal already there

	int setmsg=0;
	if (exitorabort) {
		if (g_kernelsignals_shared->sig_exit==0)  {g_kernelsignals_shared->sig_exit=code; setmsg=1;}
	} else {
		if (g_kernelsignals_shared->sig_abort==0) {g_kernelsignals_shared->sig_abort=code; setmsg=1;}
	}
	if (setmsg){
		// 1D tid calculation
		const char* badindex="bad thread index in SetKernelSignal, assumed 1D thread indexing";
		if (!(blockDim.y==1 && blockDim.z==1 && gridDim.y==1 && gridDim.z==1 )) file=badindex;

		const int tx = threadIdx.x;
		const int bx = blockIdx.x;
		const size_t tid =tx*gridDim.x + bx; // assumes 1D thread ids

		g_kernelsignals_shared->sig_tid=tid;
		g_kernelsignals_shared->sig_line=line;
		g_kernelsignals_shared->m=m;
		g_kernelsignals_shared->o=o;
		g_kernelsignals_shared->p=p;
		g_kernelsignals_shared->q=q;
		g_kernelsignals_shared->r=r;
		// g_kernelsignals_shared->s=s;	// this variable is not allocated properly
		// g_kernelsignals_shared->t=t;	// this variable is not allocated properly

		int i=0;
		if (msg!=NULL)  for(i=0;i<SIG_MESSAGE_SIZE-1 &&  msg[i]!=NULL;++i) g_kernelsignals_shared->sig_msg[i]=msg[i];
		#ifdef __DEVICE_EMULATION__
			// NOTE: macro __FILE__ produces garbage output under nvcc in GPU mode
			int j;
			if (file!=NULL) for(j=0;i<SIG_MESSAGE_SIZE-1 && file[j]!=NULL;++i,++j) g_kernelsignals_shared->sig_msg[i]=file[j];
		#else
			if (file); // avoid compiler warning
		#endif
	}







	return code;
}
*/

__forceinline__ __device__ int  exit_device_fun (const int n,const char* file,const int line,const char* msg,const int m=0,const int o=0,const int p=0,const int q=0,const int r=0,const float s=0,const float t=0)
{ 
	//SetKernelSignal( n,1,file,line,msg,m,o,p,q,r,s,t);
	/* Use CUDA native printf (CC>=2.0) */
	printf("cuprintf exit_device_fun %s %s: %s\n",file,line,msg);
	return n;
}

__forceinline__ __device__ void abort_device_fun(            const char* file,const int line,const char* msg,const int m=0,const int o=0,const int p=0,const int q=0,const int r=0,const float s=0,const float t=0)
{
	// SetKernelSignal(-1,0,file,line,msg,m,o,p,q,r,s,t);
	/* Use CUDA native printf (CC>=2.0) */
	printf("cuprintf abort_device_fun %s %s: %s\n",file,line,msg);
}

#define exit_device(n)            exit_device_fun(n,__FILE__,__LINE__,NULL,0,0,0)
#define exit_device_msg(n,msg)    exit_device_fun(n,__FILE__,__LINE__,msg,0,0,0)
#define exit_device_info(n,m,o,p,q,r,s,t) exit_device_fun(n,__FILE__,__LINE__,NULL,m,o,p,q,r,s,t)
#define abort_device()            abort_device_fun(__FILE__,__LINE__,NULL,0,0,0)
#define abort_device_msg(msg)     abort_device_fun(__FILE__,__LINE__,NULL,0,0,0)

#endif // __CUDACC__

#endif // CUDA_KERNELSIGNALS
