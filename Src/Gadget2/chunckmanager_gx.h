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

#ifdef CUDA_GX_CHUNCK_MANAGER
	#ifndef CUDA_GX_SEGMENTED
		ERROR: cannot define CUDA_GX_CHUNCK_MANAGER without CUDA_GX_SEGMENTED
	#endif

	#define CHUNK_MANAGER_SIZE_GX         1024
	#define CHUNK_MANAGER_ALIGN_GX        16
	#define CHUNK_MANAGER_MALLOC_EXTRA_GX 1024*1024*16
	#define CHUNK_MANAGER_SLEEP_GX        100000 // in microsec for polling sleeps

	#define CHUNK_MANAGER_MIN_SEG_LEFT_FORCE  4 // minimum segments left for sending a seg to anohter node, for force calculation
	#define CHUNK_MANAGER_MIN_SEG_LEFT_SPH    2 // - , for SPH calculation
	#define CHUNK_MANAGER_THREAD_DIV_SPH      1
	#define CHUNK_MANAGER_ALLOCATE_NODES      1
	#define CHUNK_MANAGER_MAX_ALLOC_NODES     12

	EXTERN_C_GX void  Lock                    ();
	EXTERN_C_GX void  Unlock                  ();
	EXTERN_C_GX int   PendingResults          ();
	EXTERN_C_GX int   PendingMessages         ();

	EXTERN_C_GX void  SendChunckDone          (double time_kernel);
	EXTERN_C_GX int   WaitAllChunkNodesDone   ();
	EXTERN_C_GX int   RecvChunckAvailNodes    (const int nodeneeded,int*const avaliblenodes);
	EXTERN_C_GX char* AdjustKernelData        (const char*const v,char* c,struct state_gx* s,struct parameters_gx* p,struct parameters_hydro_gx* h,const unsigned int sz);

	EXTERN_C_GX void  SendKernelDataToNode    (const int node,const struct state_gx*const s,const struct parameters_gx*const p,const struct parameters_hydro_gx* h,const int repack);
	EXTERN_C_GX char* RecvKernelDataFromNode  (const int node,struct state_gx** s,struct parameters_gx** p,struct parameters_hydro_gx** h,unsigned int*const sz);

	EXTERN_C_GX int   SendKernelResultToNode  (const int node,const struct state_gx*const s,unsigned int sz,unsigned int offset,const int sphmode);
	EXTERN_C_GX int   RecvKernelResultFromNode(const struct state_gx*const s,const int sphmode,int relaunch);

	EXTERN_C_GX void  ReLaunchChunkManager    ();
#else
	#define ReLaunchChunkManager()
#endif
