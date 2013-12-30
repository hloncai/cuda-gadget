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

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h> // for Error and Warning calls
#include <string.h> // for memcpy
#include <math.h>
#include <signal.h>
#include <mpi.h>

#include "allvars.h"
#include "interface_gx.h"
#include "handshake_gx.h"
#include "tools_gx.h"
#include "timer_gx.h"
#include "debug_gx.h"
#include "../gadget_cuda_gx.h"

struct parameters_gx       p_gx;
#ifdef SPH
struct parameters_hydro_gx h_gx;
#endif
struct state_gx            s_gx;
int                        g_init=-1; // global initialization var, setted in GX part
const int                  g_alloc_extra=16*1024; // alloc this extra elements (not bytes)

#ifdef __ICC
        void  __gxx_personality_v0() {} // avoid compiler warning, link btw gcc and icc gives warn about this missing fun
#endif

size_t Total_allocated_gx(const struct state_gx s)
{
	const size_t allocmem=
		sizeof(float)                       * s.sz_shortrange_table +
		sizeof(struct NODE_gx)              * s.sz_Nodes_base +
		sizeof(int)                         * s.sz_Nextnode +
		sizeof(int)                         * s.sz_DomainTask +
		sizeof(int)                         * s.sz_Exportflag +
		sizeof(struct result_gx)            * s.sz_result +
		sizeof(struct particle_data_gx)     * s.sz_P +
		#ifdef SPH
		sizeof(struct sph_particle_data_gx) * s.sz_SphP +
		#endif
		sizeof(struct etc_gx)               * s.sz_etc +
		sizeof(struct extNODE_gx)           * s.sz_extNodes_base +
		sizeof(int)                         * s.sz_Ngblist +
		#ifdef SPH
		sizeof(struct result_hydro_gx)      * s.sz_result_hydro +
		#endif
		sizeof(int)                         * s.sz_Psorted +
		sizeof(float)                       * s.sz_scratch;

	return allocmem;
}

void CheckEnoughMemory_realusage(const size_t sz_memory_limit,const int cudamode,const struct state_gx s)
{
	const size_t allocmem=Total_allocated_gx(s);

	MESSAGE("Allocation sizes=[shortrange=%d Nodes=%d Nextnode=%d DomainTaks=%d Exportflag=%d result=%d P=%d SphP=%d etc=%d extNodes=%d Ngblist=%d result_hydro=%d Psorted=%d scratch=%d] bytes",
			sizeof(float)                       * s.sz_shortrange_table,
			sizeof(struct NODE_gx)              * s.sz_Nodes_base,
			sizeof(int)                         * s.sz_Nextnode,
			sizeof(int)                         * s.sz_DomainTask,
			sizeof(char)                        * s.sz_Exportflag,
			sizeof(struct result_gx)            * s.sz_result,
			sizeof(struct particle_data_gx)     * s.sz_P,
			#ifdef SPH
			sizeof(struct sph_particle_data_gx) * s.sz_SphP,
			#endif
			sizeof(struct etc_gx)               * s.sz_etc,
			sizeof(struct extNODE_gx)           * s.sz_extNodes_base,
			sizeof(int)                         * s.sz_Ngblist,
			#ifdef SPH
			sizeof(struct result_hydro_gx)      * s.sz_result_hydro,
			#endif
			sizeof(int)                         * s.sz_Psorted,
			sizeof(float)                       * s.sz_scratch
	);
	MESSAGE("                =[shortrange=%2.1f Nodes=%2.1f Nextnode=%2.1f DomainTask=%2.1f Exportflag=%2.1f result=%2.1f P=%2.1f Sph=%2.1f etc=%2.1f extNodes=%2.1f Ngblist=%2.1f result_hydro=%2.1f Psorted=%2.1f scratch=%2.1f] %c",

			100.0 * sizeof(float)                       * s.sz_shortrange_table / allocmem,
			100.0 * sizeof(struct NODE_gx)              * s.sz_Nodes_base       / allocmem,
			100.0 * sizeof(int)                         * s.sz_Nextnode         / allocmem,
			100.0 * sizeof(int)                         * s.sz_DomainTask       / allocmem,
			100.0 * sizeof(char)                        * s.sz_Exportflag       / allocmem,
			100.0 * sizeof(struct result_gx)            * s.sz_result           / allocmem,
			100.0 * sizeof(struct particle_data_gx)     * s.sz_P                / allocmem,
			#ifdef SPH
			100.0 * sizeof(struct sph_particle_data_gx) * s.sz_SphP             / allocmem,
			#endif
			100.0 * sizeof(struct etc_gx)               * s.sz_etc              / allocmem,
			100.0 * sizeof(struct extNODE_gx)           * s.sz_extNodes_base    / allocmem,
			100.0 * sizeof(int)                         * s.sz_Ngblist          / allocmem,
			#ifdef SPH
			100.0 * sizeof(struct result_hydro_gx)      * s.sz_result_hydro     / allocmem,
			#endif
			100.0 * sizeof(int)                         * s.sz_Psorted          / allocmem,
			100.0 * sizeof(float)                       * s.sz_scratch          / allocmem,

			 '%'
		);

	if      (allocmem>s.sz_memory_limit)     ERROR("Needs more memory that is on the cuda device, needed mem=%d kB, avail mem=%d kB",allocmem/1024,s.sz_memory_limit/1024);
	else if (1.3*allocmem>s.sz_memory_limit) WARNING("Needed cuda memory close to maximum, needed mem=%d kB, avail mem=%d kB",allocmem/1024,s.sz_memory_limit/1024);
	else MESSAGE("Total needed cuda memory=%d Mb",allocmem/1024/1024);
}

void CheckEnoughMemory(const size_t sz_memory_limit,const int cudamode)
{
	FUN_MESSAGE(4,"CheckEnoughMemory(%u,%d)",sz_memory_limit,cudamode);

	struct state_gx s;
	s.cudamode=cudamode;
	s.sz_shortrange_table=1000; // XXX assumption
	s.sz_Nodes_base=MaxNodes+1; // XXX not correct when using optimal n
	s.sz_Nextnode=All.MaxPart + MAXTOPNODES;
	s.sz_DomainTask=NTask>1 ? MAXTOPNODES : 0;
	s.sz_Exportflag=NTask>1 ? GetExportflag_size_gx(NumPart,NTask) : 0;
	s.sz_result=NumPart;
	s.sz_P=NumPart;
	s.sz_SphP=N_gas;
	s.sz_etc=s.sz_P;
	s.sz_extNodes_base=s.sz_Nodes_base;
	s.sz_Ngblist=
	#ifdef SPH
	s.sz_result_hydro=s.sz_SphP;
	#endif
	s.sz_memory_limit=sz_memory_limit;

	CheckEnoughMemory_realusage(sz_memory_limit,cudamode,s);
}

// structures
struct Char99
{
	char s[99];
};

struct Char99 HumanByteSize(const size_t sz)
{
	struct Char99 c;
	c.s[0]='\0';

	if      (sz<1024) return c;
	else if (sz/1024/1024/1024>0) snprintf(c.s,99,"%0.1f Gb",1.0*sz/1024/1024/1024);
	else if (sz/1024/1024>0)      snprintf(c.s,99,"%0.1f Mb",1.0*sz/1024/1024);
	else if (sz/1024>0)           snprintf(c.s,99,"%0.1f Kb",1.0*sz/1024);
	return c;
}

void Memcpy_gx(void*const d,const void* const s,const size_t bytes,const int cudamode)
{
	FUN_MESSAGE(3,"Memcpy_gx(%p,%p,%d,%d)",d,s,bytes,cudamode);
	if (cudamode<2) memcpy(d,s,bytes);
	else            Memcpy_cuda_gx(d,s,bytes,0);
}

void* Malloc_gx(const size_t sz,const char* msg,const char* file,const int line)
{
	FUN_MESSAGE(3,"Malloc_gx(%d,'%s',%s,%d)",sz,msg,file,line);

	if (g_init<0)  ERROR("GX not initialized, remember a call to Initialize_gx()");
	#if CUDA_DEBUG_GX>0
	if (msg!=NULL) {
		const struct Char99 c=HumanByteSize(sz);
		// N.B. Should print size_t with %zu, not %d
		MESSAGE("Malloc_gx(%s:%d): %zu bytes = %s %s",file,line,sz,c.s,msg);
	}
	#endif

	// leave 2 Mb of overhead data
	// N.B. Should print size_t with %zu, not %d
	#warning Compiling Malloc_gx that may not work correctly!
	if (Total_allocated_gx(s_gx)+1024*1024*2>s_gx.sz_memory_limit) WARNING("Memory allocation exceeding graphical card limit, current allocated = %zu bytes, limit = %d bytes, requested = %zu bytes",Total_allocated_gx(s_gx),s_gx.sz_memory_limit,sz);

	if (sz==0) ERROR("cannot malloc zero bytes, %s:%d",file,line);
	ASSERT_GX( sz!=0 );

	void* d=malloc(sz);
	if (d==NULL) ERROR("Could not allocate %d bytes = %d Mb for data structure, %s:%d",sz,sz/1024/1024,file,line);

	return d;
}

void Free_gx(void** p)
{
	FUN_MESSAGE(3,"Free_gx(%p)",*p);

	if (g_init<0) ERROR("GX not initialized, remember a call to Initialize_gx()");
	if (*p==NULL) ERROR("cannot free zero pointer");

	ASSERT_GX( p!=NULL && *p!=NULL );
	free(*p);
	*p=NULL;
}

void AllocateData_gx(const int N,const int extra,void** p_dat,unsigned int*const p_sz,unsigned int*const p_sz_max,const int sizeofdata,const int iscudamem,const int ishostmem,const char*const msg,const char*const file,const int line)
{
	FUN_MESSAGE(3,"AllocateData_gx(N=%d,extra=%d)",N,extra);
	ASSERT_GX( N>=0 && extra>=0 && sizeofdata>0 && p_dat!=NULL && p_sz!=NULL);
	ASSERT_GX( iscudamem==0 || iscudamem==1 );
	ASSERT_GX( ishostmem==0 || ishostmem==1 );

	if (*p_dat==NULL || N>*p_sz) {
		if (p_sz_max==NULL || N>*p_sz_max){
			if (*p_dat!=NULL) {
				if (iscudamem) {
					if (ishostmem) Free_host_cuda_gx(p_dat);
					else           Free_cuda_gx(p_dat);
				}
				else           Free_gx(p_dat);
			}

			const size_t bz=sizeofdata * (N+extra);
			if (iscudamem) {
				if (ishostmem) *p_dat=Malloc_host_cuda_gx(bz,msg,file,line);
				else           *p_dat=Malloc_cuda_gx(bz,msg,file,line);
			}
			else           *p_dat=Malloc_gx     (bz,msg,file,line);

			if (p_sz_max!=NULL) *p_sz_max=N+extra;
		}
	}

	ASSERT_GX( N==0 || *p_dat!=NULL );
	*p_sz=N;
}

void AllocateData_once_gx(const int N,const int extra,void** p_dat,unsigned int*const p_sz,unsigned int*const p_sz_max,const int sizeofdata,const int iscudamem,const char*const msg,const char*const file,const int line)
{
	ASSERT_GX( N>=0 && extra==0 && sizeofdata>0 && p_dat!=NULL && p_sz!=NULL && p_sz_max==NULL);

	if (*p_dat==NULL)  AllocateData_gx(N,0,p_dat,p_sz,p_sz_max,sizeofdata,iscudamem,0,msg,file,line);
	else if (N!=*p_sz) ERROR("table size changed unexpected, %s",msg);
}

#define ALLOCATOR_ONCE_GX(N,name,sizeofdata,iscudamem,msg)                          AllocateData_once_gx(N,0    ,(void**)&s_gx.name,&s_gx.sz_##name,0                  ,sizeofdata,iscudamem,msg,__FILE__,__LINE__)
#define ALLOCATOR_WITH_HEADROOM_GX(N,extra,name,sizeofdata,iscudamem,ishostmem,msg) AllocateData_gx     (N,extra,(void**)&s_gx.name,&s_gx.sz_##name,&s_gx.sz_max_##name,sizeofdata,iscudamem,0,msg,__FILE__,__LINE__)

int FindTypeSizes_Sub_gx(const unsigned short ptype,const int ptype_last,struct parameters_gx*const p)
{
	FUN_MESSAGE(5,"FindTypeSizes_Sub_gx()");

	p->nonsortedtypes=1; // YYY  NOTE: disable sort/nonsorting for now!
	ASSERT_GX(p!=NULL && ptype<6);
	if (ptype!=ptype_last) {
		if (ptype_last!=-1 && ptype<ptype_last) p->nonsortedtypes=1;
	}
	p->typesizes[ptype] += 1;
	return ptype;
}

int CountParticlesInTimeStep_gx(const size_t N,const struct particle_data*const p,const int Ti_Current,const int sphmode)
{
	FUN_MESSAGE(3,"CountParticlesInTimeStep_gx(%d)",N);
	int i,Np=0;
	for(i=0;i<N;++i){
		if (p[i].Ti_endstep == Ti_Current) {
			ASSERT_GX( !sphmode || p[i].Type==0 );
			++Np;
		}
	}
	return Np;
}

struct parameters_gx FillParameters_gx()
{
	FUN_MESSAGE(3,"FillParameters_gx()");
	struct parameters_gx p;
	memset(&p,0,sizeof(struct parameters_gx));

	p.MaxPart   =All.MaxPart;
	p.MaxNodes  =MaxNodes;

	p.Ti_Current=All.Ti_Current;

#ifdef PMGRID
	p.Asmth[0]  =All.Asmth[0];
	p.Asmth[1]  =All.Asmth[1];
	p.Rcut[0]   =All.Rcut[0];
	p.Rcut[1]   =All.Rcut[1];
#endif
	p.BoxSize   =All.BoxSize;
	p.ErrTolTheta   =All.ErrTolTheta;
	p.ErrTolForceAcc=All.ErrTolForceAcc;
	#ifdef ADAPTIVE_GRAVSOFT_FORGAS
		p.soft=All.soft;
	#endif
	int i;
	for(i=0;i<6;++i) {
		p.ForceSoftening[i]=All.ForceSoftening[i];
		ASSERT_GX(All.MassTable[i]>=0);
		p.Masses[i]=All.MassTable[i];
		p.typesizes[i]=0;
	}

	p.ComovingIntegrationOn=All.ComovingIntegrationOn;
	p.Timebase_interval=All.Timebase_interval;
	p.export_P_type=-1;

	ValidateParameters(p,__FILE__,__LINE__);

	return p;
}

struct result_gx GetTarget(const int target,int n)
{
	ASSERT_GX( s_gx.mode==0 );

	ASSERT_GX( s_gx.sz_result_buffer==s_gx.sz_Psorted);
	ASSERT_GX( s_gx.sz_max_result_buffer==s_gx.sz_max_Psorted);
	ASSERT_GX( target>=0 && target<s_gx.sz_result_buffer);
	ASSERT_GX( target<s_gx.sz_Psorted );
	ASSERT_GX( n==s_gx.Psorted[target] );

	const struct result_gx r=s_gx.result_buffer[target];

	#ifdef CUDA_GX_DEBUG_MEMORY_ERROR
		ASSERT_GX( r.realtarget==n );
		ASSERT_GX( r.temp2==(s_gx.mode==0 ? target+100000 : target-100000) );
		ASSERT_GX( r.type>=0 &&  r.type<6);
	#endif

	return r;
}

int GetType(const int tag)
{
	const int type=tag & 7;
	ASSERT_GX( type>=0 && type<8 ); // or type<6
	return type;
}

int GetTypeAndRealtarget(const int type,const int realtarget)
{
	if ((realtarget<<3)>>3!=realtarget) ERROR("cannot splice type and target together when taget>2^(32-3)");

	ASSERT_GX( type>=0 && type<8 ); // or type<6
	ASSERT_GX( realtarget>=0 && (realtarget<<3)>>3==realtarget);

	const int x=(realtarget<<3) | type;

	ASSERT_GX( (x >> 3)==realtarget );
	ASSERT_GX( GetType(x)==type );

	return x;
}

void WriteResultData_gx(const size_t N,const struct particle_data*const p)
{
	FUN_MESSAGE(3,"WriteResultData_gx(N=%d)",N);

	ASSERT_GX( s_gx.mode==0 );
	ASSERT_GX( s_gx.sz_result<=N && s_gx.Np==s_gx.sz_result );

	int i,Np=0;
	for(i=0;i<N;++i){
		if (p[i].Ti_endstep==p_gx.Ti_Current) {
			ASSERT_GX( Np<s_gx.sz_result );

			// write auxilary data to result temporary structure
			s_gx.result[Np].acc_x=p[i].OldAcc;
			s_gx.result[Np].acc_y=0;
			s_gx.result[Np].acc_z=0;

			s_gx.result[Np].ninteractions=GetTypeAndRealtarget(p[i].Type,i);

			#ifdef CUDA_GX_DEBUG_MEMORY_ERROR
				s_gx.result[Np].oldacc=p[i].OldAcc;
				s_gx.result[Np].temp2=Np;     // write some dummy data
				s_gx.result[Np].realtarget=i;
				s_gx.result[Np].type=p[i].Type;
			#endif

			ASSERT_GX( isResultDataOK(s_gx.result[Np],__FILE__,__LINE__));
			ASSERT_GX( s_gx.result[Np].acc_y>=0 );
			ASSERT_GX( s_gx.result[Np].acc_z==0 );

			++Np;
		}
	}

	ASSERT_GX( Np<=N );
	ASSERT_GX( Np==s_gx.Np && Np==s_gx.sz_result );
}

void WriteResultDataSorted_gx(const struct particle_data*const p)
{
	ASSERT_GX( s_gx.mode==0 );
	ASSERT_GX( s_gx.sz_result==s_gx.sz_Psorted );

	int i;
	for(i=0;i<s_gx.sz_result;++i){
		const int target=s_gx.Psorted[i];
		ASSERT_GX( target<NumPart );
		const int type=p[target].Type;
		s_gx.result[i].ninteractions=GetTypeAndRealtarget(type,target);
	}
}

void WriteResultDataExport_gx(const size_t N)
{
	FUN_MESSAGE(3,"WriteResultDataExport_gx(%d)",N);

	ASSERT_GX( s_gx.mode==1 );
	ASSERT_GX( N==s_gx.sz_result && N==s_gx.Np );

	int i,ptype_last=-1;
	for(i=0;i<6;++i) p_gx.typesizes[i]=0; // reset type sizes

	for(i=0;i<N;++i) {
		ASSERT_GX( i<s_gx.sz_scratch );
		s_gx.scratch[i]=GravDataGet[i].w.OldAcc;

		s_gx.result[i].acc_x=GravDataGet[i].u.Pos[0];
		s_gx.result[i].acc_y=GravDataGet[i].u.Pos[1];
		s_gx.result[i].acc_z=GravDataGet[i].u.Pos[2];
		#ifdef UNEQUALSOFTENINGS
			const int ptype=GravDataGet[i].Type;
		#else
			const int ptype=p_gx.export_P_type;
		#endif

		#ifdef CUDA_GX_DEBUG_MEMORY_ERROR
			s_gx.result[i].oldacc=GravDataGet[i].w.OldAcc;
			s_gx.result[i].temp2=-i;
			s_gx.result[i].realtarget=i;
			s_gx.result[i].type=ptype;
			ASSERT_GX(isFloatOK(s_gx.result[i].oldacc,__FILE__,__LINE__));
		#endif

		s_gx.result[i].ninteractions=GetTypeAndRealtarget(ptype,i);
		ptype_last=FindTypeSizes_Sub_gx(ptype,ptype_last,&p_gx);

		ASSERT_GX(ptype>=0 && ptype<6);
		ASSERT_GX( isResultDataOK(s_gx.result[i],__FILE__,__LINE__) );
	}
}

void WritePsortedData_gx(void)
{
	FUN_MESSAGE(3,"WritePsortData_gx()");

	ASSERT_GX( s_gx.mode==0 );
	ASSERT_GX( s_gx.sz_result==s_gx.sz_Psorted );
	ASSERT_GX( s_gx.Np==s_gx.sz_Psorted && s_gx.Np==s_gx.sz_result );

	int i;
	for(i=0;i<s_gx.sz_result;++i){
		const int realtarget=s_gx.result[i].ninteractions>>3;
		ASSERT_GX( realtarget>=0 && realtarget<s_gx.sz_P );
		s_gx.Psorted[i]=realtarget;
	}
}

void WriteEtcData_gx(const int N,const struct particle_data*const p
#ifdef SPH
,const struct sph_particle_data*const sph
#endif
)

{
	FUN_MESSAGE(3,"WriteEtcData_gx(%d)",N);
	ASSERT_GX(N==s_gx.sz_etc);

	int i;
	for(i=0;i<N;++i){
		// write etc data
		s_gx.etc[i].Type=p[i].Type;
		s_gx.etc[i].Ti_begstep=p[i].Ti_begstep;
		s_gx.etc[i].Ti_endstep=p[i].Ti_endstep;

		// NOTE: Hsml part unimplemeted/tested
		#ifdef SPH
		if (sph!=NULL && p[i].Type==0) {
			ASSERT_GX( i<s_gx.sz_SphP );
			s_gx.etc[i].Hsml=sph[i].Hsml;
		} else s_gx.etc[i].Hsml=0;
		#endif
	}
}

void WriteParticleData_gx(const size_t N,const struct particle_data*const p)
{
	FUN_MESSAGE(3,"WriteParticleData_gx(%d)",N);
	ASSERT_GX(N==s_gx.sz_P);

	int i,Np=0;
	int ptype_last=-1;
	int sortmessage=0;

	for(i=0;i<N;++i){
		// copy particle data
		s_gx.P[i].Pos[0]=p[i].Pos[0];
		s_gx.P[i].Pos[1]=p[i].Pos[1];
		s_gx.P[i].Pos[2]=p[i].Pos[2];
		s_gx.P[i].Mass=p[i].Mass;

		ASSERT_GX(isParticleDataOK(s_gx.P[i],__FILE__,__LINE__));

		// find and write ptype sizes to typesizes array
		const int ptype=p[i].Type;
		ptype_last=FindTypeSizes_Sub_gx(ptype,ptype_last,&p_gx);

		if(i<N_gas && ptype!=0 && !sortmessage) {
			sortmessage=1;
			MESSAGE("i=%d, s_gx.N_gas=%d, ptype=%d",i,N_gas,ptype);
			WARNING("gas particles not sorted");
		}

		//if (!(ptype!=0 || i<NtypeLocal[0] )) MESSAGE("ptype=%d, i=%d, N=%d, NtypeLocal[0]=%d, Ntype[0]=%d",ptype,i,N,NtypeLocal[0],Ntype[0]);
		//YYY ASSERT_GX( ptype!=0 || i<NtypeLocal[0] );
		//YYY s_gx.etc[i].Hsml=ptype==0 ? sph[i].Hsml : -1;

		if (p[i].Ti_endstep==p_gx.Ti_Current) ++Np;
	}

	s_gx.Np=Np;
	ValidateParameters(p_gx,__FILE__,__LINE__);
}

void UpdateParticleData_gx(const size_t N,const struct particle_data*const p,const int extra)
{
	FUN_MESSAGE(3,"UpdateParticleData_gx(%d)",N);

	ASSERT_GX( s_gx.NumPart==s_gx.sz_P );
	ASSERT_GX( N==s_gx.sz_etc && (s_gx.sz_Exportflag>0 || s_gx.NTask==1) );

	ALLOCATOR_WITH_HEADROOM_GX(N,extra,P,sizeof(struct particle_data_gx),0,0,"particle mirror data");

	s_gx.NumPart=N;

	WriteParticleData_gx(N,p);
}

void UpdateShortrangeTableData_gx(const int N,const float *const s)
{
	FUN_MESSAGE(3,"UpdateShortrangeTableData_gx(%d)",N);
	if (s_gx.cudamode==0) return;

	ALLOCATOR_ONCE_GX(N,shortrange_table,sizeof(float),0,"shortrange_table mirror data");

	size_t i;
	for(i=0;i<N;++i) s_gx.shortrange_table[i]=s[i];
	//Memcpy_gx(s_gx.shortrange_table,s,t,0);
}

void UpdateNodeData_gx(const int M,const int N,const struct NODE* const s,const int All_MaxPart_gx,int extra)
{
	// NOTE: M unused, M=larget set of nodes, N=optimal number nodes
	FUN_MESSAGE(3,"UpdateNodeData_gx(%d,%d)",M,N);

	ASSERT_GX( s_gx.MaxPart==All_MaxPart_gx );
	if (sizeof(struct NODE)!=sizeof(struct NODE_gx)) ERROR("Node size mismatch");
	
	MESSAGE("N: %d, extra: %d, Nodes_base: %d, MaxNodes: %d", N,extra,Nodes_base,MaxNodes);
	// 'Nodes_base' below is not the global variable! careful!
	ALLOCATOR_WITH_HEADROOM_GX(N,extra,Nodes_base,sizeof(struct NODE_gx),s_gx.cudamode>=2,0,"node mirror data");

	s_gx.sz_Nodes_base=N;
	s_gx.Nodes=s_gx.Nodes_base-All_MaxPart_gx;

	const size_t bz=sizeof(struct NODE_gx) * N;
	if (s_gx.cudamode<2) memcpy(s_gx.Nodes_base,s,bz);
	else                 Memcpy_cuda_gx(s_gx.Nodes_base,s,bz,0);
}

void UpdateNextNodeData_gx(const int N,const int *const s)
{
	FUN_MESSAGE(3,"UpdateNextNodeData_gx(%d)",N);

	ALLOCATOR_ONCE_GX(N,Nextnode,sizeof(int),s_gx.cudamode>=2,"nextnode mirror data");

	const size_t bz=sizeof(int)*s_gx.sz_Nextnode;
	if (s_gx.cudamode<2) memcpy(s_gx.Nextnode,s,bz);
	else                 Memcpy_cuda_gx(s_gx.Nextnode,s,bz,0);
}

void UpdateDomainTaskData_gx(const int N,const int*const s)
{
	FUN_MESSAGE(3,"UpdateDomainTaskData_gx(%d)",N);
	ASSERT_GX((NTask>1 && s!=NULL) || (N==0 && s==0));

	if (N==0){
		ASSERT_GX( s_gx.NTask==1 && s_gx.DomainTask==0 && s_gx.sz_DomainTask==0 );
		return;
	}

	ALLOCATOR_ONCE_GX(N,DomainTask,sizeof(int),s_gx.cudamode>=2,"domaintask mirror data");
	ASSERT_GX( (size_t)s_gx.ThisTask<s_gx.sz_DomainTask );

	const size_t bz=sizeof(int) * s_gx.sz_DomainTask;
	if (s_gx.cudamode<2) memcpy(s_gx.DomainTask,s,bz);
	else                 Memcpy_cuda_gx(s_gx.DomainTask,s,bz,0);
}

void UpdateExportflagData_gx(const int M,const int extra)
{
	FUN_MESSAGE(3,"UpdateExportflagData_gx(%d)",M);

	const int N=GetExportflag_size_gx(M,NTask); // NOTE: should really just be Np
	ASSERT_GX(NTask>1 || N==0);

	if (N==0){
		ASSERT_GX( s_gx.NTask==1 && s_gx.Exportflag==NULL && s_gx.sz_Exportflag==0);
		return;
	}

    	ALLOCATOR_WITH_HEADROOM_GX(N,extra,Exportflag,sizeof(char),0,0,"Exportflag data");

	ASSERT_GX( s_gx.NTask==1 || ((s_gx.Exportflag!=NULL && s_gx.sz_Exportflag>0) || M==0) );
	memset(s_gx.Exportflag,0,s_gx.sz_Exportflag);
}

void UpdateEtcData_gx(const int N,const struct particle_data*const p,
#ifdef SPH
const struct sph_particle_data*const sph,
#endif
const int extra)
{
	FUN_MESSAGE(3,"UpdateEtcData_gx(%d)",N);
	ALLOCATOR_WITH_HEADROOM_GX(N,extra,etc,sizeof(struct etc_gx),0,0,"etc data");
	WriteEtcData_gx(N,p
		#ifdef SPH
		,sph
		#endif
	);
}

void UpdateResultData_gx(const int N,const struct particle_data*const p,const int extra,const int exportmode)
{
	FUN_MESSAGE(3,"UpdateResultData_gx(%d)",N);

//const int b=s_gx.sz_max_result;
	ALLOCATOR_WITH_HEADROOM_GX(s_gx.Np,extra,result,sizeof(struct result_gx),0,0,"result data");
//if (s_gx.sz_max_result>b) MESSAGE("result=%d, max=%d, head=%d, extra=%d",s_gx.sz_result,s_gx.sz_max_result,s_gx.sz_max_result-s_gx.sz_result,s_gx.sz_max_result-b);

	#ifdef CUDA_GX_DEBUG_MEMORY_ERROR
		#if CUDA_DEBUG_GX>0
			int i;
			for(i=0;i<s_gx.sz_max_result;++i) s_gx.result[i].temp2=0;
		#endif
	#endif

	if (!exportmode) {
		ASSERT_GX( p!=NULL );
		WriteResultData_gx(N,p);
	} else{
		ASSERT_GX( p==NULL );
		WriteResultDataExport_gx(N);
	}

	ASSERT_GX( s_gx.sz_result==s_gx.Np );
}

void UpdateResultBufferData_gx(const int extra)
{
	FUN_MESSAGE(3,"UpdateResultBufferData_gx()");
	ASSERT_GX( s_gx.mode==0 );

	ALLOCATOR_WITH_HEADROOM_GX(s_gx.sz_result,extra,result_buffer,sizeof(struct result_gx),0,0,"result bufferdata");

	ASSERT_GX( s_gx.sz_result_buffer==s_gx.sz_result );
}

void WriteResultBufferData_gx()
{
	FUN_MESSAGE(3,"WriteResultBufferData_gx()");

	ASSERT_GX( s_gx.sz_result_buffer==s_gx.sz_result );
	memcpy(s_gx.result_buffer,s_gx.result,sizeof(struct result_gx)*s_gx.sz_result );
}

void UpdatePsortedData_gx(const int extra)
{
	FUN_MESSAGE(3,"UpdatePsortData_gx()");
	ALLOCATOR_WITH_HEADROOM_GX(s_gx.sz_result,extra,Psorted,sizeof(int),0,0,"Psorted data");

	WritePsortedData_gx();
}

/*
int WritePsortedData_gx(const int N)
{
	FUN_MESSAGE(3,"WritePsortDataHydro_gx()");

	ASSERT_GX( s_gx.mode==0 );
	ASSERT_GX( s_gx.Np==s_gx.sz_Psorted

	int i,Np=0;
	for(i=0;i<N;++i){
		if (p[i].Ti_endstep==p_gx.Ti_Current) {

			ASSERT_GX( Np<s_gx.sz_Psorted );
			ASSERT_GX( p[i].Type==0 );

			s_gx.Psorted[Np]=i;
			++Np;
		}
	}

	ASSERT_GX( Np<=N );
	ASSERT_GX( Np==s_gx.Np && Np==s_gx.sz_Psorted );
	return Np;
}

void UpdatePsortedDataHydro_gx(const int N,const int Np,const int extra)
{
	FUN_MESSAGE(3,"UpdatePsortDataHydro_gx()");
	ASSERT_GX( s_gx.Np==Np && Np<=N );
	ALLOCATOR_WITH_HEADROOM_GX(Np,extra,Psorted,sizeof(int),0,0,"Psorted data");

	const int Np2=WritePsortedDataHydro_gx(N);
	ASSERT_GX( Np2==Np );
}
*/

void UpdateScratchData_gx(const int N,const int extra)
{
	FUN_MESSAGE(3,"UpdateScratchData_gx(%d)",N);
	ALLOCATOR_WITH_HEADROOM_GX(N,extra,scratch,sizeof(FLOAT_GX),0,0,"scratch data");
}

void Initialize_gx(const int argc,char*const*const argv,const int thistask,const int ntask,const int localrank)
{
	ASSERT_GX(thistask==ThisTask);

	memset(&p_gx,0,sizeof(struct parameters_gx));
	memset(&s_gx,0,sizeof(struct state_gx));

	s_gx.ThisTask=thistask;
	s_gx.NTask=ntask;
	SetThisTask(thistask);
	s_gx.cudamode=2;  // default cudamode: use GPU

	FUN_MESSAGE(3,"Initialize_gx(thistask=%d,NTask=%d)",thistask,NTask);

	int i;
	for(i=1;i<argc;++i) {
		if(strcmp(argv[i],"-cuda")==0) {
			++i;
			if (i>=argc) ERROR("expected program argument -cuda <mode>");
			s_gx.cudamode=atoi(argv[i]);
			if (s_gx.cudamode<0 || s_gx.cudamode>3) ERROR("expected program argument cuda mode in range [0;2]");
		}
		if(strcmp(argv[i],"-cudadebug")==0) {
			++i;
			if (i<argc) s_gx.debugval=atoi(argv[i]);
		}
	}
	if (s_gx.cudamode==0) return;

	if (g_init!=-1) ERROR("GX not re-entrant,  call Initialize_gx() only once");
	ASSERT_GX( ThisTask<ntask );

	g_init=1;

	const int h=TestHandshakedata_gx();
	if (h<=0) ERROR("Handshake of data structures failed, mismatch in sizeof, structure or alignment data between this compiler and CUDA nvcc [h=%d]",h);

	s_gx.sz_memory_limit=Initialize_cuda_gx(argc,argv,thistask,localrank);
	ASSERT_GX( s_gx.sz_memory_limit>=1024*1024*16 ); // at least 16 Mb of mem present

	if (thistask==0) {
		PrintConfig();

		char defconfig[16*1024];
		#ifdef __GNUC__
			sprintf(defconfig,"GNUC=%d.%d.%d",__GNUC__,__GNUC_MINOR__,__GNUC_PATCHLEVEL__);

		#endif
		#ifdef __ICC // or __INTEL_COMPILER"
			sprintf(defconfig,"ICC=%d",__ICC);
		#endif

		MESSAGE("MPICC Config    =[%s]",defconfig);
	}
}

int InitializeCalculation_gx(const int N,const struct particle_data*const p,const int sphmode)
{
	FUN_MESSAGE(3,"InitializeCalculation_gx(%d)",N);

	p_gx=FillParameters_gx();
	s_gx.MaxPart=All.MaxPart;
	s_gx.iteration++;
	s_gx.sphmode=sphmode;

	ASSERT_GX(NumPart==N);
	ASSERT_GX(s_gx.mode==0);

	UpdateScratchData_gx     (0,g_alloc_extra);
	UpdateEtcData_gx         (N,p,
		#ifdef SPH
		NULL,
		#endif
		g_alloc_extra);
	UpdateExportflagData_gx  (NTask>1 ? N : 0,g_alloc_extra);
	UpdateParticleData_gx    (N,p,g_alloc_extra);

	UpdateResultData_gx      (N,p,g_alloc_extra,s_gx.mode);
	UpdateResultBufferData_gx(g_alloc_extra);
	UpdatePsortedData_gx     (g_alloc_extra);

	Trunckernelsize_gx(s_gx.Np,s_gx.NTask,&s_gx);

	ASSERT_GX( TestGetAuxData(N) );

	#if CUDA_DEBUG_GX>2
		const int M=SortParticles_Init(N,p,All.Ti_Current,0);
		//SortParticles_Sort(Np,s_gx.Psorted);
		//WriteResultDataSorted_gx(p);
		ASSERT_GX(M<=N);
		ASSERT_GX(s_gx.Np==M || M==N);
	#endif

	ASSERT_GX(s_gx.P!=NULL && s_gx.sz_P==NumPart && s_gx.result!=NULL && (s_gx.sz_result<=s_gx.sz_P || s_gx.mode==1));
	AssertsOnhasGadgetDataBeenModified_gx(1,1,0);

	ValidateParameters(p_gx,__FILE__,__LINE__);
	ValidateState     (s_gx,p_gx.MaxNodes,1,__FILE__,__LINE__);

	if (s_gx.cudamode>=2) InitializeCalculation_cuda_gx(s_gx,sphmode);

	static int firstcall=1;
	if (firstcall==1){
		firstcall=0;
		CheckEnoughMemory_realusage(s_gx.sz_memory_limit,s_gx.cudamode,s_gx);
	}

	return s_gx.Np;
}

int InitializeProlog_gx(const int N)
{
	FUN_MESSAGE(3,"InitializeProlog_gx(%d)",N);

	if (s_gx.MaxPart==0) s_gx.MaxPart=All.MaxPart;

	ASSERT_GX(s_gx.mode==0);
	ASSERT_GX(s_gx.MaxPart==All.MaxPart);
	ASSERT_GX(MaxNodes+1==(int)(All.TreeAllocFactor * All.MaxPart + 1));
	ASSERT_GX(NumPart<=All.MaxPart);
	ASSERT_GX(N==NumPart || N==N_gas);

	UpdateNextNodeData_gx  (All.MaxPart + MAXTOPNODES,Nextnode);
	UpdateNodeData_gx      (MaxNodes+1,Numnodestree,Nodes_base,All.MaxPart,10*g_alloc_extra); // NOTE: allocate some more extra for nodes
	UpdateDomainTaskData_gx(NTask>1 ? MAXTOPNODES : 0,NTask>1 ? DomainTask : 0);

	int Np=0;
	if (s_gx.cudamode>1) {
		Np=CountParticlesInTimeStep_gx(N,P,All.Ti_Current,0);
		CopyAuxData_cuda_gx(s_gx);
	}

	return Np;
}

void InitializeExportCalculation_gx(const int N,const int typeP0)
{
	FUN_MESSAGE(3,"InitializeExportCalculation_gx(N=%d,typeP0=%d)",N,typeP0);

	AssertsOnhasGadgetDataBeenModified_gx(1,1,0);
	ASSERT_GX( s_gx.mode==0 );

	s_gx.mode=1;
	s_gx.Np=N;

	ASSERT_GX( s_gx.cudamode>0 || N<=s_gx.sz_result ); // NOTE: dunno if this hold in cudamode=0?
	ASSERT_GX( p_gx.export_P_type==-1 );
	#ifndef UNEQUALSOFTENINGS
		p_gx.export_P_type=typeP0;
		ASSERT_GX( p_gx.export_P_type>=0 &&  p_gx.export_P_type<6);
	#endif

	UpdateScratchData_gx(N,g_alloc_extra);
	UpdateResultData_gx (N,0,g_alloc_extra,1);

	ASSERT_GX( N==s_gx.sz_result );
	ValidateParameters(p_gx,__FILE__,__LINE__);

	if (s_gx.cudamode>=2) {
		InitializeResults_cuda_gx(s_gx);
		InitializeScratch_cuda_gx(s_gx);
	}
}

void Finalize_gx()
{
	FUN_MESSAGE(3,"Finalize_gx()");
	ASSERT_GX(s_gx.mode==0);

	if (g_init!=-1) FinalizeCalculation_cuda_gx();
}

double FinalizeExportCalculation_gx(const int N)
{
	FUN_MESSAGE(3,"FinalizeExportCalculation_gx(%d)",N);

	ASSERT_GX(s_gx.mode==1);
	ASSERT_GX(s_gx.cudamode==2);
	ASSERT_GX(N==s_gx.sz_result);

	s_gx.mode=0;
	#ifdef UNEQUALSOFTENINGS
		ASSERT_GX( p_gx.export_P_type==-1 );
	#else
		ASSERT_GX( p_gx.export_P_type>=0 &&  p_gx.export_P_type<6);
	#endif

	int i;
	double costtotal=0;
	for(i=0;i<N;++i) {
		ASSERT_GX( isResultDataOK(s_gx.result[i],__FILE__,__LINE__) );

		GravDataResult[i].u.Acc[0] = s_gx.result[i].acc_x;
		GravDataResult[i].u.Acc[1] = s_gx.result[i].acc_y;
		GravDataResult[i].u.Acc[2] = s_gx.result[i].acc_z;

		GravDataResult[i].w.Ninteractions = s_gx.result[i].ninteractions;
		costtotal += GravDataResult[i].w.Ninteractions ;

		#ifdef CUDA_GX_DEBUG_MEMORY_ERROR
			ASSERT_GX(s_gx.result[i].temp2==i-100000);
		#endif
	}

	p_gx.export_P_type=-1;

	return costtotal;
}

#ifdef SPH
struct parameters_hydro_gx FillParameters_hydra_gx(const int ngas,const int sz_Ngblist,const int threads,const FLOAT_GX hubble_a2,const FLOAT_GX fac_mu,const FLOAT_GX fac_vsic_fix
	#ifdef PERIODIC
		,const FLOAT_GX boxSize,const FLOAT_GX boxHalf
	#endif
)
{
	FUN_MESSAGE(4,"FillParameters_hydra_gx(%d,...)",ngas);

	struct parameters_hydro_gx h;
	memset(&h,0,sizeof(struct parameters_hydro_gx));

	h.N_gas=ngas;
	ASSERT_GX( threads>=1 && sz_Ngblist%threads==0 );
	h.szngb=sz_Ngblist/threads;

	h.hubble_a2=hubble_a2;
	h.fac_mu=fac_mu;
	h.fac_vsic_fix=fac_vsic_fix;
	h.ArtBulkViscConst=All.ArtBulkViscConst;

	#ifdef PERIODIC
		h.boxSize = All.BoxSize;
		h.boxHalf = 0.5 * All.BoxSize;
		#ifdef LONG_X
			h.boxHalf_X = boxHalf * LONG_X;
			h.boxSize_X = boxSize * LONG_X;
		#endif
		#ifdef LONG_Y
			h.boxHalf_Y = boxHalf * LONG_Y;
			h.boxSize_Y = boxSize * LONG_Y;
		#endif
		#ifdef LONG_Z
			h.boxHalf_Z = boxHalf * LONG_Z;
			h.boxSize_Z = boxSize * LONG_Z;
		#endif
	#endif

	ValidateParameters_hydra(h,__FILE__,__LINE__);

	return h;
}

void UpdateSphParticleData_gx(const int ngas,const struct sph_particle_data*const sph,const int extra)
{
	FUN_MESSAGE(3,"UpdateSphParticleData_gx(%d)",ngas);

	// ASSERT_GX( N==s_gx.sz_result && N==s_gx.sz_etc && (s_gx.sz_Exportflag>0 || s_gx.NTask==1) );
	ALLOCATOR_WITH_HEADROOM_GX(ngas,extra,SphP,sizeof(struct sph_particle_data_gx),0,0,"sph particle mirror data");

	s_gx.sz_SphP=ngas;
	s_gx.N_gas=ngas;

	int i;
	for(i=0;i<ngas;++i){
		s_gx.SphP[i].Entropy   =sph[i].Entropy;
		s_gx.SphP[i].Density   =sph[i].Density;
		s_gx.SphP[i].Hsml      =sph[i].Hsml;
		s_gx.SphP[i].Pressure  =sph[i].Pressure;
		s_gx.SphP[i].VelPred[0]=sph[i].VelPred[0];
		s_gx.SphP[i].VelPred[1]=sph[i].VelPred[1];
		s_gx.SphP[i].VelPred[2]=sph[i].VelPred[2];
		s_gx.SphP[i].DivVel    =sph[i].DivVel;
		s_gx.SphP[i].CurlVel   =sph[i].CurlVel;
		s_gx.SphP[i].DhsmlDensityFactor=sph[i].DhsmlDensityFactor;
		s_gx.SphP[i].pad1=i;
		s_gx.SphP[i].pad2=-i;
	}
}
#endif

void UpdateNgbData_gx(const int N,const int extra)
{
	FUN_MESSAGE(3,"UpdateNgbData_gx(%d,%d)",N,extra);
	ALLOCATOR_WITH_HEADROOM_GX(N,extra,Ngblist,sizeof(int),s_gx.cudamode>=2,0,"NGB list data");
}

#ifdef SPH
void UpdateResultHydroData_gx(const int N,const int extra)
{
	FUN_MESSAGE(3,"UpdateResultHydroData_gx(%d)",N);

	//ALLOCATOR_WITH_HEADROOM_GX(N,extra,result_hydro,sizeof(struct result_hydro_gx),s_gx.cudamode>=2,0,"result_hydro_gx list data");
	//ALLOCATOR_WITH_HEADROOM_GX(N,extra,hydrodata_in,sizeof(struct hydrodata_in_gx),s_gx.cudamode>=2,0,"hydrodata_in list data");

	if (s_gx.result_hydro==NULL || N>s_gx.sz_result_hydro) {
		if (N>s_gx.sz_max_result_hydro){
			const size_t bz=sizeof(struct result_hydro_gx) * (N+extra);
			const size_t bz2=sizeof(struct hydrodata_in_gx) * (N+extra);
			if (s_gx.result_hydro!=NULL) {
				ASSERT_GX(s_gx.hydrodata_in!=NULL);
				Free_gx((void*)&s_gx.result_hydro);
				Free_gx((void*)&s_gx.hydrodata_in);
			}
			s_gx.result_hydro=Malloc_gx(bz,"result_hydro data",__FILE__,__LINE__);
			s_gx.hydrodata_in=Malloc_gx(bz2,"hydrodata_in data",__FILE__,__LINE__);
			s_gx.sz_max_result_hydro=N+extra;
		}
	}
	s_gx.sz_result_hydro=N;

	int i;
	for(i=0;i<N;++i) {
		s_gx.result_hydro[i].realtarget=i;
	}

	#if CUDA_DEBUG_GX > 0
		for(i=0;i<N;++i) {
			s_gx.result_hydro[i].pad0=-i;
			s_gx.result_hydro[i].pad1=i;
		}
		for(i=N;i<s_gx.sz_max_result_hydro;++i) {
			s_gx.result_hydro[i].pad0=0;
			s_gx.result_hydro[i].pad1=0;
		}
	#endif
}
#endif

void UpdateExtNode_gx(const int N,const struct extNODE*const s,const int All_MaxPart_gx,const int extra)
{
	FUN_MESSAGE(3,"UpdateExtNode_gx(%d)",N);

	ASSERT_GX( s_gx.MaxPart==All_MaxPart_gx );
	if (sizeof(struct extNODE)!=sizeof(struct extNODE_gx)) ERROR("extNode size mismatch");

	ALLOCATOR_WITH_HEADROOM_GX(N,extra,extNodes_base,sizeof(struct extNODE_gx),s_gx.cudamode>=2,0,"extnode mirror data");

	s_gx.sz_extNodes_base=N;
	ASSERT_GX( s_gx.sz_extNodes_base==s_gx.sz_Nodes_base);

	const size_t bz=sizeof(struct extNODE_gx) * N;
	if (s_gx.cudamode<2) memcpy(s_gx.extNodes_base,s,bz);
	else                 Memcpy_cuda_gx(s_gx.extNodes_base,s,bz,0);

	s_gx.extNodes=s_gx.extNodes_base-s_gx.MaxPart;
}

#ifdef SPH
int InitializeHydraCalculation_gx(const int N,const struct particle_data*const p,const struct sph_particle_data*const sph,const int ngas,const FLOAT_GX hubble_a2,const FLOAT_GX fac_mu,const FLOAT_GX fac_vsic_fix
	#ifdef PERIODIC
		,const FLOAT_GX boxSize,const FLOAT_GX boxHalf
 	#endif
)
{
	FUN_MESSAGE(3,"InitializeHydraCalculation_gx()");

	ASSERT_GX( ngas<=N && p!=NULL && sph!=NULL);

	p_gx=FillParameters_gx();
	s_gx.MaxPart=All.MaxPart;

	Trunckernelsize_gx(ngas,s_gx.NTask,&s_gx);
	const int threads=s_gx.blocks*s_gx.grids;

	#ifndef CUDA_GX_SHARED_NGBLIST
		UpdateNgbData_gx(MAX_NGB_GX * threads,g_alloc_extra);
		ASSERT_GX(threads*MAX_NGB_GX==s_gx.sz_Ngblist);
	#endif
	UpdateExtNode_gx(s_gx.sz_Nodes_base,Extnodes_base,All.MaxPart,10*g_alloc_extra);

	h_gx=FillParameters_hydra_gx(ngas,s_gx.sz_Ngblist,threads,hubble_a2,fac_mu,fac_vsic_fix
	#ifdef PERIODIC
		,boxSize, boxHalf
	#endif
	);

	const int Np=CountParticlesInTimeStep_gx(ngas,p,All.Ti_Current,1);

	#if CUDA_DEBUG_GX>2
		const int M=SortParticles_Init(ngas,p,All.Ti_Current,1);
		// SortParticles_Sort(Np,0);

		ASSERT_GX( Np==M || M==ngas);
		ASSERT_GX( M<=N && M<=ngas);
	#endif

	//UpdateParticleData_gx   (N,p,g_alloc_extra);
	UpdateSphParticleData_gx (ngas,sph,g_alloc_extra);
	UpdateEtcData_gx         (ngas,p,sph,g_alloc_extra);
	//UpdatePsortedDataHydro_gx(ngas,Np,g_alloc_extra);
	UpdateResultHydroData_gx (ngas,g_alloc_extra);

	const int Np2=InitializeCalculation_gx(N,p,1);
	s_gx.Np=Np;

	ASSERT_GX( Np<=Np2 );
	ASSERT_GX( Np<=ngas); // NOTE: calc only for all gas particles
	ASSERT_GX( N==s_gx.NumPart);

//MESSAGE("Np=%d, Np2=%d, ngas=%d",Np,Np2,ngas);
//MESSAGE("DistRMS(gas)=%g",DistRMS(ngas,P,All.Ti_Current));

	InitializeHydraCalculation_cuda_gx(s_gx);

	ASSERT_GX( ngas==s_gx.N_gas);
	ASSERT_GX( (ngas==s_gx.N_gas && ngas==s_gx.sz_SphP) );

	ValidateParameters(p_gx,__FILE__,__LINE__);
	ValidateParameters_hydra(h_gx,__FILE__,__LINE__);
	ValidateState(s_gx,p_gx.MaxNodes,1,__FILE__,__LINE__);

	CopyAuxData_cuda_gx(s_gx);

	ASSERT_GX( N==s_gx.NumPart);
	ASSERT_GX( ngas==s_gx.N_gas);
	ASSERT_GX( (ngas==s_gx.N_gas && ngas==s_gx.sz_SphP) );
	ASSERT_GX( (NumPart==s_gx.NumPart && NumPart==s_gx.sz_P));
	ASSERT_GX( ngas==p_gx.typesizes[0] );

	AssertsOnhasGadgetDataBeenModified_gx(0,1,1);
	return Np;
}

void UpdateHydroDataIn_gx(const int N,const int extra,struct hydrodata_in*const s)
{
	FUN_MESSAGE(3,"UpdateHydroDataIn_gx()");
	ERROR("UpdateHydroDataIn_gx() not tested");

	if (s_gx.hydrodata_in==NULL || N>s_gx.sz_hydrodata_in) {
		if (N>s_gx.sz_max_hydrodata_in){
			const size_t bz=sizeof(struct hydrodata_in_gx) * (N+extra);
			if (s_gx.hydrodata_in!=NULL) Free_gx((void*)&s_gx.hydrodata_in);
			s_gx.hydrodata_in=Malloc_gx(bz,"hydrodata_in data",__FILE__,__LINE__);
			s_gx.sz_max_hydrodata_in=N+extra;
		}
	}
	s_gx.sz_hydrodata_in=N;

	#if CUDA_DEBUG_GX > 0
		int i;
		for(i=0;i<N;++i) s[i].pad_gx=-i; // write dummy data into struct
	#endif

	const size_t bz=sizeof(struct hydrodata_in_gx) * N;
	if (s_gx.cudamode<2) memcpy(s_gx.hydrodata_in,s,bz);
	else                 Memcpy_cuda_gx(s_gx.hydrodata_in,s,bz,0);

	ASSERT_GX( s_gx.sz_result_hydro==s_gx.sz_hydrodata_in);
}

void InitializeHydraExportCalculation_gx(const int N,const struct hydrodata_in*const s)
{
	FUN_MESSAGE(3,"InitializeHydraExportCalculation_gx(%d)",N);
	ERROR("InitializeHydraExportCalculation_gx() not tested");

	AssertsOnhasGadgetDataBeenModified_gx(1,1,0);

	ASSERT_GX( s_gx.cudamode==2 && N<=s_gx.sz_result_hydro );
	ASSERT_GX( s_gx.mode==0 && s_gx.sphmode==1);
	s_gx.mode=1;

	UpdateResultHydroData_gx(N,g_alloc_extra);
	UpdateHydroDataIn_gx    (N,g_alloc_extra,HydroDataIn);

	ASSERT_GX( N==s_gx.sz_hydrodata_in);
	ASSERT_GX( N==s_gx.sz_result_hydro);

	ValidateParameters(p_gx,__FILE__,__LINE__);
	if (s_gx.cudamode>=2) InitializeResults_cuda_gx(s_gx);
}

void FinalizeHydraExportCalculation_gx(const int N)
{
	FUN_MESSAGE(3,"FinalizeHydraExportCalculation_gx(%d)",N);
	ERROR("FinalizeHydraExportCalculation_gx() not tested");

	ASSERT_GX( s_gx.cudamode==2 && N<=s_gx.sz_result_hydro);
	ASSERT_GX( s_gx.sz_result_hydro==s_gx.sz_hydrodata_in);
	ASSERT_GX( s_gx.mode==1 && s_gx.sphmode==1);
	s_gx.mode=0;

	int i;
	for(i=0;i<N;++i) {
		ASSERT_GX( isResultHydraDataOK(s_gx.result_hydro[i],__FILE__,__LINE__) );
		ASSERT_GX( s_gx.result_hydro[i].pad0==i );
		ASSERT_GX( s_gx.result_hydro[i].pad1==N-i );

		HydroDataResult[i].Acc[0] = s_gx.result_hydro[i].Acc[0];
		HydroDataResult[i].Acc[1] = s_gx.result_hydro[i].Acc[1];
		HydroDataResult[i].Acc[2] = s_gx.result_hydro[i].Acc[2];
		HydroDataResult[i].DtEntropy = s_gx.result_hydro[i].DtEntropy;
		HydroDataResult[i].MaxSignalVel = s_gx.result_hydro[i].MaxSignalVel;
	}
}

#endif

int GetID(const int target)
{
	if (target>=NumPart) ERROR("target(%d)>NumPart(%d) in GetID()",target,NumPart);
	return P[target].ID;
}

// from http://www.cse.yorku.ca/~oz/hash.html
int stringhash(unsigned char *str)
{
        unsigned long hash = 5381;
        int c;

        while (c = *str++)
            hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

        return abs((int) hash);
}
