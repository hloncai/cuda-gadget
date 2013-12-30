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

#ifndef __GADGET_CUDA_GX_H__
#define __GADGET_CUDA_GX_H__

EXTERN_C_GX int  isEmulationMode                   (void);
EXTERN_C_GX int Initialize_cuda_gx(const int argc,char*const*const argv,const int thistask,const int localrank);
EXTERN_C_GX void InitializeResults_cuda_gx         (const struct state_gx s_gx);
EXTERN_C_GX void InitializeResultsHydro_cuda_gx    (const struct state_gx s_gx);
EXTERN_C_GX void InitializeScratch_cuda_gx         (const struct state_gx s_gx);
EXTERN_C_GX void InitializeCalculation_cuda_gx     (const struct state_gx s,const int sphmode);
EXTERN_C_GX void InitializeHydraCalculation_cuda_gx(const struct state_gx s);
EXTERN_C_GX void FinalizeCalculation_cuda_gx       (void);
EXTERN_C_GX void CopyAuxData_cuda_gx               (const struct state_gx s);
EXTERN_C_GX char GetExportflag_gx                  (const struct state_gx* s,const unsigned int n,const unsigned int bitelemsize,const unsigned int index);
EXTERN_C_GX int  GetExportflag_size_gx             (const unsigned int N,const unsigned int bitelemsize);
EXTERN_C_GX void Trunckernelsize_gx                (const int N,const int NTask,struct state_gx*const s);
EXTERN_C_GX void WalkNodes                         (const int MaxPart,const int MaxNodes,const struct NODE_gx* Nodes,int* walknomin,int* walknomax);
EXTERN_C_GX void PrintConfig                       (void);
#ifdef CUDA_GX_CHUNCK_MANAGER
	EXTERN_C_GX void ReLaunchChunkManager        ();
	EXTERN_C_GX void ManageChuncks               (const int sphmode);
#else
	#define ReLaunchChunkManager()
	#define ManageChuncks(sphmode)
#endif

// force and hydro calculations
EXTERN_C_GX void force_treeevaluate_shortrange_range_cuda_gx(const int mode,const unsigned int NumPart,const struct state_gx s,const struct parameters_gx p);
EXTERN_C_GX void hydro_evaluate_range_cuda_gx               (const int mode,const unsigned int N_gas  ,const struct state_gx s,const struct parameters_gx p,const struct parameters_hydro_gx h);

// for use in Gadget src dir only
EXTERN_C_GX void* Malloc_cuda_gx      (const size_t t,const char* msg,const char* file,const int line);
EXTERN_C_GX void  Free_cuda_gx        (void** p);
EXTERN_C_GX void* Malloc_host_cuda_gx (const size_t t,const char* msg,const char* file,const int line);
EXTERN_C_GX void  Free_host_cuda_gx   (void** p);
EXTERN_C_GX void  Memcpy_cuda_gx (void* dst,const void*const src,const size_t t,const int direction);

#endif // __GADGET_CUDA_GX_H__
