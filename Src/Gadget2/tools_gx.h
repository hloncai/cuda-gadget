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

#if CUDA_DEBUG_GX > 1
	#define ENABLE_TOOLS_FUN_GX

	#include "defines_gx.h"

	EXTERN_C_GX  void   DebugMem_Insert(const void* p,const size_t sz,const char*const msg,const char* file,const int line);
	EXTERN_C_GX  void   DebugMem_Remove(const void* p);
	EXTERN_C_GX  size_t DebugMem_Size  ();
	EXTERN_C_GX  void   DebugMem_Dump  (const int v);

	int  SortParticles_Init(const size_t N,const struct particle_data* particles,const int Ti_Current,const int sphmode);
	void SortParticles_Sort(const int Np,int*const sorted);
#else
	#define DebugMem_Insert(p,sz,msg,file,line)
	#define DebugMem_Remove(p)
	#define DebugMem_Dump(v)

	#define SortParticles_Init(N,particles,Ti_Current,sphmode) N
	#define SortParticles_Sort(Np,result)
#endif
