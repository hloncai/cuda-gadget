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

#ifndef __CUDAMALLOC_GX_H__
#define __CUDAMALLOC_GX_H__

#include "Gadget2/tools_gx.h"

extern "C"
void* Malloc_cuda_gx(const size_t t,const char* msg,const char* file,const int line);

extern "C"
void Free_cuda_gx(void** p);

extern "C"
void* Malloc_host_cuda_gx(const size_t t,const char* msg,const char* file,const int line);

extern "C"
void Free_host_cuda_gx(void** p);

extern "C"
void Memcpy_cuda_gx(void* dst,const void*const src,const size_t t,const int direction);

#ifdef __CUDACC__
#if CUDA_DEBUG_GX > 1

// NOTE: printkernelstate defined here due to include problems with malloc/assert
struct printkernelstate
{
	const unsigned int maxthreads;
	const size_t sz_msg;
	char*const msg;
	char*const msg_cpy;

	printkernelstate(const dim3 grid,const dim3 threads)
	: maxthreads(threads.x*threads.y*grid.x*grid.y)
	, sz_msg(sizeof(char)*maxthreads*THREAD_MESSAGE_SIZE)
	, msg((char*)Malloc_cuda_gx(sz_msg,"thread messages",__FILE__,__LINE__))
	, msg_cpy((char*)malloc(sz_msg+1))
	{
		memset(msg_cpy,0,sz_msg);
		Memcpy_cuda_gx(msg,msg_cpy,sz_msg,0);
		ASSERT_GX( maxthreads>=1 && sz_msg==maxthreads*THREAD_MESSAGE_SIZE && msg && msg_cpy);
	}

	~printkernelstate()
	{
		ASSERT_GX( maxthreads>=1 && sz_msg==maxthreads*THREAD_MESSAGE_SIZE && msg && msg_cpy);
		Free_cuda_gx((void**)&msg);
		free((void*)msg_cpy);
		memset(this,sizeof(printkernelstate),0);
	}

	void ShowThreadInfo(const int onlyfailed=1)
	{
		ASSERT_GX( maxthreads>=1 && sz_msg==maxthreads*THREAD_MESSAGE_SIZE && msg && msg_cpy);
		Memcpy_cuda_gx(msg_cpy,msg,sz_msg,1);
		size_t i;
		for(i=0;i<maxthreads;++i){
			const size_t m=i*THREAD_MESSAGE_SIZE;
			ASSERT_GX(m+THREAD_MESSAGE_SIZE-1<sz_msg+1);
			msg_cpy[m+THREAD_MESSAGE_SIZE-1]=0; // truncate
			if (msg_cpy[m]!=0) {
				//int n=0;
				//while(newlines && msg_cpy[m+n]!=0) {if (msg_cpy[m+n]==',') msg_cpy[m+n]='\t'; ++n;} // cannot handle newlines yet, replaced by tabs
				const char s=msg_cpy[m];
				if (!(s=='D' && msg_cpy[m+1]==0)){
					char st[16];
					if      (s=='D') strcpy(st,"DONE");
					else if (s=='F') strcpy(st,"FAILED");
					else if (s=='S') strcpy(st,"STARTED");
					else if (s=='?') strcpy(st,"UNKNOWN");
					else             strcpy(st,"N/A    ");
					if   (!onlyfailed || s!='D') MESSAGE("thread(%d): state=%s info='%s'",i,st,&(msg_cpy[m+1]));
				}
			}
		}
	}
};
#endif
#endif

#endif // __CUDAMALLOC_GX_H__
