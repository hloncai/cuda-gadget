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

#if CUDA_DEBUG_GX > 0
	int isFloatOK                            (const FLOAT_GX x,const char*const file,const int line);
	EXTERN_C_GX int isParticleDataOK         (const struct particle_data_gx p,const char*const file,const int line);
	EXTERN_C_GX int isSphParticleDataOK      (const struct sph_particle_data_gx p,const char*const file,const int line);
	EXTERN_C_GX int isEtcDataOK              (const struct etc_gx r,const char*const file,const int line);
	EXTERN_C_GX int isResultDataOK           (const struct result_gx        r,const char*const file,const int line);
	EXTERN_C_GX int isResultHydraDataOK      (const struct result_hydro_gx  r,const char*const file,const int line);
	EXTERN_C_GX int  EqualState              (struct state_gx s1,struct state_gx s2,const int cmppointers,const int cmpall,const char*const file,const int line);
	EXTERN_C_GX void ValidateParameters      (const struct parameters_gx p      ,const char*const file,const int line);
	EXTERN_C_GX void ValidateParameters_hydra(const struct parameters_hydro_gx p,const char*const file,const int line);
	EXTERN_C_GX void ValidateState           (const struct state_gx s,const int MaxNodes,const int checkparticledata,const char*const file,const int line);

	EXTERN_C_GX void PrintResult             (const struct result_gx r,const char*const msg);
	EXTERN_C_GX void PrintNode               (const struct NODE_gx r,const char*const msg);
	EXTERN_C_GX void PrintParameters         (const struct parameters_gx*const       s,const char*const msg);
	EXTERN_C_GX void PrintParametersHydro    (const struct parameters_hydro_gx*const h,const char*const msg);
	EXTERN_C_GX void PrintState              (const struct state_gx*const            p,const char*const msg);

	double DistRMS                           (const size_t N,const struct particle_data *const s,const int Ti_current);
	double DistRMSGravdata                   (const size_t N,const struct gravdata_in   *const s);

	int TestGetAuxData                       (const int N);
#else
	#define ValidateParameters(p,file,line)
	#define ValidateParameters_hydra(p,file,line)
	#define ValidateState(s,MaxNodes,checkparticledata,file,line)
	#define EqualState(s1,s2,cmppointers,cmpall,file,line)
#endif

#if CUDA_DEBUG_GX > 1
	void AssertsOnhasGadgetDataBeenModified_gx(const int no_particle_change,const int print,const int ignore_ti_current);
#else
	#define AssertsOnhasGadgetDataBeenModified_gx(no_particle_change,print,ignore_ti_current)
#endif

#ifndef __CUDACC__
	// NOTE: debug/info funs avalible for any config
	int  PrintInfoInitialize (const int N,const int mode,const int sphmode);
	void PrintInfoFinalize   (const struct state_gx s,const int ndone,const int Np,const double starttime,const double cpytime,const double subtime,const int printmode,const int iter,const int lev,const int not_timestepped_gx,const int count_exported_gx,const int nexport,const int nexportsum,const int exporthash_gx,const double costtotal);
#endif
