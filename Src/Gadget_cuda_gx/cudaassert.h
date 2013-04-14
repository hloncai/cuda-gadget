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

#ifndef __ASSERT_GX_H__
#define __ASSERT_GX_H__

#define CUDA_PRINTFUN_GX 0 // levels for function call print: -1=none, 0=some, 1=more, etc.

#ifndef __CUDACC__
	#include "../../Gadget_cuda_gx/cudautils.h"
#endif

#if CUDA_DEBUG_GX > 0
	#define ASSERT_GX(expr)                  ((void)((expr) ? 0 : (Assert_fail_gx       (__STRING(expr), __FILE__, __LINE__,0),0)))
	#if CUDA_DEBUG_GX > 1
		#define ASSERT_DEVICE_GX(expr)     ((void)((expr) ? 0 : (Assert_fail_device_gx(__STRING(expr), __FILE__, __LINE__,0,&devmsg),0)))
	#else
		#define ASSERT_DEVICE_GX(expr)     ((void)((expr) ? 0 : (Assert_fail_device_gx(__STRING(expr), __FILE__, __LINE__,0,0),0)))
	#endif
#else
	#define ASSERT_GX(expr)
	#define ASSERT_DEVICE_GX(expr)
#endif

#if (defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L)
	#define   ERROR(format, ...)   Error_gx(__FILE__,__LINE__,format, ## __VA_ARGS__)
	#define WARNING(format, ...) Warning_gx(__FILE__,__LINE__,format, ## __VA_ARGS__)
	#define MESSAGE(format, ...) Message_gx(__FILE__,__LINE__,format, ## __VA_ARGS__)
	#if CUDA_DEBUG_GX > 0
		#define FUN_MESSAGE(level,format,...) if (level<=CUDA_PRINTFUN_GX) Message_gx(__FILE__,__LINE__,format, ## __VA_ARGS__)
	#else
		#define FUN_MESSAGE(level,format,...)
	#endif
#else
	#define   ERROR(format...)     Error_gx(__FILE__,__LINE__,format)
	#define WARNING(format...)   Warning_gx(__FILE__,__LINE__,format)
	#define MESSAGE(format...)   Message_gx(__FILE__,__LINE__,format)
	#if CUDA_DEBUG_GX > 0
		#define FUN_MESSAGE(level,format...) if (level<=CUDA_PRINTFUN_GX) Message_gx(__FILE__,__LINE__,format)
	#else
		#define FUN_MESSAGE(level,format...)
	#endif
#endif

// error handling funs
void Error_gx  (const char* file,const int line,const char* format,...);
void Warning_gx(const char* file,const int line,const char* format,...);
void Message_gx(const char* file,const int line,const char* format,...);
void Assert_fail_gx(const char* expr,const char* file,const int line,const char* msg);
void SetThisTask(const int thistask);

#ifdef __CUDACC__
	__device__ void Assert_fail_device_gx(const char* expr,const char* file,const int line,const char* msg,struct DevMsg* pdevmsg);
#endif

#endif // __ASSERT_GX_H__

