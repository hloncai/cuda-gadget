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

#include "cudautils.h"

void* Malloc_cuda_gx(const size_t t,const char* msg,const char* file,const int line)
{
	FUN_MESSAGE(5,"Malloc_cuda_gx(%t)",t);

	const size_t residual=0; // t%16; no need to do residual

	#if CUDA_DEBUG_GX>0
		if (t/1024/1024>0) MESSAGE("Malloc_cuda_gx(%s:%d): %d bytes = %0.1f Mb %s",file,line,t,1.0*t/1024/1024,msg);
		else               MESSAGE("Malloc_cuda_gx(%s:%d): %d bytes = %0.1f Kb %s",file,line,t,1.0*t/1024     ,msg);
	#endif

	if (t==0) return NULL;
	void* p=NULL;
	const cudaError_t r=cudaMalloc(&p,t+residual);
	if (r!=cudaSuccess) {
		// DebugMem_Dump(1);
		ERROR("%s, cudaMalloc failed to allocate %d bytes = %g Mb",cudaGetErrorString(r),t,t/1024./1024.);
	}
	ASSERT_GX( p!=NULL );

	#if CUDA_DEBUG_GX > 0
		const cudaError_t r2=cudaMemset(p,0,t+residual);
		if (r2!=cudaSuccess) ERROR("%s, cudaMemset failed",cudaGetErrorString(r2));
	#endif

	DebugMem_Insert(p,t+residual,msg,file,line);
	if (msg || file || line); // avoid compiler warning
	return p;
}

void Free_cuda_gx(void** p)
{
	FUN_MESSAGE(5,"Free_cuda_gx(%p)",*p);

	ASSERT_GX( p!=NULL );
	if(*p==NULL) return;
	ASSERT_GX(*p!=NULL);

	DebugMem_Remove(*p);
	const cudaError_t r=cudaFree(*p);
	if (r!=cudaSuccess) ERROR("cudaFree failed to free pointer %xd",*p);
	*p=NULL;
}

void* Malloc_host_cuda_gx(const size_t t,const char* msg,const char* file,const int line)
{
	FUN_MESSAGE(5,"Malloc_host_cuda_gx(%t)",t);

	if (t==0) return NULL;
	void* p=NULL;

	const cudaError_t r=cudaHostAlloc((void **)&p,t,cudaHostAllocMapped);

	if (r!=cudaSuccess) {
		// DebugMem_Dump(1);
		ERROR("%s, cudaHostAlloc failed to allocate %d bytes = %g Mb",cudaGetErrorString(r),t,t/1024./1024.);
	}
	ASSERT_GX( p!=NULL );

	DebugMem_Insert(p,t,msg,file,line);
	if (msg || file || line); // avoid compiler warning
	return p;
}

void Free_host_cuda_gx(void** p)
{
	FUN_MESSAGE(5,"Free_host_cuda_gx(%p)",*p);

	ASSERT_GX( p!=NULL );
	if(*p==NULL) return;
	ASSERT_GX(*p!=NULL);

	DebugMem_Remove(*p);
	const cudaError_t r=cudaFreeHost(*p);
	if (r!=cudaSuccess) ERROR("cudaFree failed to free pointer %xd",*p);
	*p=NULL;
}

void Memcpy_cuda_gx(void* dst,const void*const src,const size_t t,const int direction)
{
	FUN_MESSAGE(5,"Memcpy_cuda_gx(%p,%p,%d,%d)",dst,src,t,direction);
	ASSERT_GX(direction==0 || direction==1);

	if (t==0) {
		ASSERT_GX( src==NULL );
		return;
	}
	ASSERT_GX( dst!=NULL && src!=NULL && t>0);
	const cudaError_t r=direction==0 ? cudaMemcpy(dst,src,t,cudaMemcpyHostToDevice) : cudaMemcpy(dst,src,t,cudaMemcpyDeviceToHost);
	if (r!=cudaSuccess) ERROR("cudaMemcpy failed to copy %d bytes = %g Mb",t,t/1024./1024.);
}

