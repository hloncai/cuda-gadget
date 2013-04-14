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

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#include <mpi.h>

#include "allvars.h"

#define USE_MUTEX_STDOUT_LOCK_GX // try to print to stdout using a lock, preventing overlaid printoutput from the mpi-system

#ifdef USE_MUTEX_STDOUT_LOCK_GX
	// NOTE: if you get an compile error in this error then disable the mutex
	//       locking by elliminating the USE_MUTEX_STDOUT_LOCK_GX predefine above

	//#define FILE_LOCK_GX // use slow, global file lock as mutex instead of pthread mutex'es

	#ifdef FILE_LOCK_GX
		#include <sys/types.h>
		#include <sys/stat.h>
		#include <fcntl.h>
		#include <unistd.h>
	#else
		#include <pthread.h>
		static pthread_mutex_t mutex_mpd_print_lock = PTHREAD_MUTEX_INITIALIZER;
	#endif

	void Mutexlock(void)
	{
		#ifdef FILE_LOCK_GX
			int r=open("mutex.lock.txt",O_CREAT|O_EXCL);
			while(r<=0) {
				usleep(10);
				r=open("mutex.lock.txt",O_CREAT|O_EXCL);
			}
		#else
			pthread_mutex_lock( &mutex_mpd_print_lock);
		#endif
	}

	void Mutexunlock(void)
	{
		#ifdef FILE_LOCK_GX
			close(r);
			unlink("mutex.lock.txt");
		#else
			pthread_mutex_unlock( &mutex_mpd_print_lock );
		#endif
	}
#else
	// do nothing...
	void Mutexlock  () {};
	void Mutexunlock() {};
#endif

void PrintFun(FILE* fp,const char* msg)
{
	if (fp!=NULL && msg!=NULL) {
		Mutexlock();
		fprintf(fp,"%s",msg);
		Mutexunlock();
	}
}

void PrintVariableArgs(char* b,const int n,const char* format,va_list args)
{
	b[0]=0;
	vsnprintf(b,n,format,args);
}

#define ERROR_OR_WARNING_GX(msg)\
	va_list args;\
	va_start(args,format);\
	va_end(args);\
	\
	char b0[1024];\
	b0[0]=0;\
	PrintVariableArgs(b0,1024,format,args);\
	char b1[2*1024];\
	b1[0]=0;\
	snprintf(b1,2*1024,"\nNode: %d, %s:%d: %s %s\n",ThisTask,file,line,msg,b0);\
	\
	PrintFun(stderr,b1)

void Error_gx(const char* file,const int line,const char* format,...)
{
	ERROR_OR_WARNING_GX("ERROR");
	MPI_Abort(MPI_COMM_WORLD, 99);
	// abort();
}

void Warning_gx(const char* file,const int line,const char* format,...)
{
	ERROR_OR_WARNING_GX("WARNING");
}

// for printing purpose only
static int g_ThisTask=-1;

void SetThisTask(const int thistask)
{
	if (g_ThisTask!=-1) Error_gx(__FILE__,__LINE__,"g_ThisTask was already setted");
	else{
		if (thistask<0) Error_gx(__FILE__,__LINE__,"bad thistask value, must be >=0");
		g_ThisTask=thistask;
	}
}

void Message_gx(const char* file,const int line,const char* format,...)
{
	static unsigned int c=0;
	va_list args;
	va_start(args,format);
	va_end(args);

	char b0[1024];
	b0[0]=0;
	PrintVariableArgs(b0,1024,format,args);
	char b1[1024];
	b1[0]=0;

	#ifdef __CUDACC__
		const char prefix='#';
	#else
		const char prefix='*';
	#endif
	if (g_ThisTask<0) Warning_gx(__FILE__,__LINE__,"bad thistask value, must be >=0, did you remember to call SetThisTask()?");
	snprintf (b1,1024,"%c%c {%5u,%2d} %s\n",prefix,prefix,c,g_ThisTask,b0);

	++c;
	PrintFun(stdout,b1);
	if (file || line) {}; // avoid compiler warning
}

#ifdef __CUDACC__
__host__
#endif
void Assert_fail_gx(const char* expr,const char* file,const int line,const char* msg)
{
	fprintf(stderr,"\n%s:%d: ASSERT_GX, expression '%s' failed\n",file,line,expr);
	if (msg!=0) fprintf(stderr,"\tmessage : %s\n",msg);
	fprintf(stderr,"\taborting now...\n\n");
	fflush(stdout);
	fflush(stderr);
	exit(-2);
}

