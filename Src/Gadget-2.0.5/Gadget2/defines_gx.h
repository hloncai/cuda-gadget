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

// NOTE: G2X setup defines

// N.B. For large problem sizes (~512^3 or more, textures must be disabled.  CUDA does not support large 1D textures and will crash inexplicably if you attempt this.
//#define CUDA_GX_USE_TEXTURES                 // use texture cache for gadget structure lookups
//#define CUDA_GX_NO_SPH_SUPPORT             // enable/diable GPU hydra calc
//#define CUDA_GX_DEBUG_MEMORY_ERROR         // inject extra commands to check for memory errors! Will cause use of lmem
#define CUDA_GX_SHARED_NGBLIST               // put NGB list in shared mem  (1 elem list)

#define MAX_NGB_GX                     2000  // defines maximum length of neighbour list, GPU def, can be diff from GADGET2 def
#define MIN_FORCE_PARTICLES_FOR_GPU_GX 1  // minimum number of particles participating in a GPU force calc
#define MIN_SPH_PARTICLES_FOR_GPU_GX   2000  // -                                              GPU sph calc

// NOTE: demo defines only, not tested properly

//#define CUDA_GX_BITWISE_EXPORTFLAGS        // exportflags are in bit-arrays instead of char-arrays
//#define CUDA_GX_CONSTANT_VARS_IN_SHARED_MEM  // put constant data in shared mem, won't add any speed
//#define CUDA_GX_SEGMENTED 1                  // chunckup kernels into smaller seqments, only really needed for chunckmanager
//#define CUDA_GX_CHUNCK_MANAGER               // reassign chuncks to avalible nodes
//#define CUDA_GX_CHUNCK_MANAGER_SPH           // reassign chuncks to avalible nodes

// NOTE: G2X system defines

#ifdef __CUDACC__
	#define EXTERN_C_GX extern "C"
#else
	#define EXTERN_C_GX
#endif

// #ifdef FLOAT
// 	#undef FLOAT
// #endif
// #ifdef DOUBLE
// 	#undef DOUBLE
// #endif

// #define FLOAT   "ERROR: do not use FLOAT directly, use FLOAT_INTERNAL_GX"
// #define float   "ERROR: do not use float directly, use FLOAT_INTERNAL_GX"
// #define DOUBLE  "ERROR: do not use DOUBLE directly, use FLOAT_INTERNAL_GX"
// #define double  "ERROR: do not use double directly, use FLOAT_INTERNAL_GX"

// Float range, somewhat arbitrary range
// NOTE: INF and SUB normale not defined in CUDA2.2
#define CUDA_GX_INFINITY  1E300
#define CUDA_GX_SUBNORMAL 1E-300
