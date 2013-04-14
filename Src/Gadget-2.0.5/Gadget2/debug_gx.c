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
#include <string.h> // for memcpy
#include <math.h>   // for fabs

#include "allvars.h"
#include "interface_gx.h"
#include "timer_gx.h"

#if CUDA_DEBUG_GX > 0

int isNAN      (const FLOAT_GX x,const char*const file,const int line){return x!=x;}

// NOTE: fun below not implemeted, problems finding a common function set btw GCC and NCC
int isNORMAL   (const FLOAT_GX x,const char*const file,const int line){return 1;}

int isINF      (const FLOAT_GX x,const char*const file,const int line)
{
	if (fabs(x)>CUDA_GX_INFINITY) return 1; //WARNING("float value close to inf: x=%g, in %s:%d",x,file,line);
	return 0;
}

int isSUBNORMAL(const FLOAT_GX x,const char*const file,const int line)
{
	if (fabs(x)<CUDA_GX_SUBNORMAL && fabs(x)>0) return 1; //WARNING("float value close to subnormal: x=%g, in %s:%d",x,file,line);
	return 0;
}

int isFloatOK(const FLOAT_GX x,const char*const file,const int line)
{
	const int floatok=!isNAN(x,file,line) && !isINF(x,file,line) && !isSUBNORMAL(x,file,line) && isNORMAL(x,file,line);
	//if (!floatok) WARNING("float value not ok: x=%g, in %s:%d (isNAN(x)=%d, isINF(x)=%d, isSUBNORMAL(x)=%d, isNORMAL(x)=%d)",x,file,line,isNAN(x,file,line),isINF(x,file,line),isSUBNORMAL(x,file,line),isNORMAL(x,file,line));
	return floatok;
}

FLOAT_GX Proxymy(const FLOAT_GX x,const FLOAT_GX x0)
{
	if (fabs(x0)<fabs(x)) return Proxymy(x0,x);
	const FLOAT_GX d= x0==0 ? fabs(x) : (x==0 ? fabs(x0) : fabs((x-x0)/x0));
	//const float d=fabs((x-x0)/x0);
	return d;
}

int isParticleDataOK(const struct particle_data_gx p,const char*const file,const int line)
{
	ASSERT_GX(isFloatOK(p.Pos[0],file,line));
	ASSERT_GX(isFloatOK(p.Pos[1],file,line));
	ASSERT_GX(isFloatOK(p.Pos[2],file,line));
	ASSERT_GX(isFloatOK(p.Mass,file,line));
	return 1;
}

int isSphParticleDataOK(const struct sph_particle_data_gx p,const char*const file,const int line)
{
	ASSERT_GX( isFloatOK( p.Entropy,file,line));
	ASSERT_GX( isFloatOK( p.Density,file,line));
	ASSERT_GX( isFloatOK( p.Hsml,file,line));
	ASSERT_GX( isFloatOK( p.Pressure,file,line));
	ASSERT_GX( isFloatOK( p.VelPred[0],file,line));
	ASSERT_GX( isFloatOK( p.VelPred[1],file,line));
	ASSERT_GX( isFloatOK( p.VelPred[2],file,line));
	ASSERT_GX( isFloatOK( p.DivVel,file,line));
	ASSERT_GX( isFloatOK( p.CurlVel,file,line));
	ASSERT_GX( isFloatOK( p.DhsmlDensityFactor,file,line));
	return 1;
}

int isEtcDataOK(const struct etc_gx r,const char*const file,const int line)
{
	ASSERT_GX(r.Type>=0 && r.Type<6);
	ASSERT_GX(r.Ti_endstep>=r.Ti_begstep);
	ASSERT_GX(isFloatOK(r.Hsml,file,line));

	return 1;
}

void PrintResult(const struct result_gx r,const char*const msg)
{
	MESSAGE("result_gx: %s",msg);
	MESSAGE("  acc=(%g|%g|%g)",r.acc_x,r.acc_y,r.acc_z);
	MESSAGE("  ninteractions=%d",r.ninteractions);
	#ifdef CUDA_GX_DEBUG_MEMORY_ERROR
		MESSAGE("   oldacc=%g, temp2=%d, realtarget=%d, type=%d",r.oldacc,r.temp2,r.realtarget,r.type);
	#endif
}

int isResultDataOK(const struct result_gx r,const char*const file,const int line)
{
	const int ok=isFloatOK(r.acc_x,file,line) && isFloatOK(r.acc_y,file,line) && isFloatOK(r.acc_z,file,line) && r.ninteractions>=0;
	if (!ok) {
		PrintResult(r,"result not OK!");
		return 0;
	} else{
		// dual check!
		ASSERT_GX(isFloatOK(r.acc_x,file,line));
		ASSERT_GX(isFloatOK(r.acc_y,file,line));
		ASSERT_GX(isFloatOK(r.acc_z,file,line));
		ASSERT_GX(r.ninteractions>=0);
		return 1;
	}
}

int isResultHydraDataOK(const struct result_hydro_gx r,const char*const file,const int line)
{
	ASSERT_GX(isFloatOK(r.Acc[0],file,line));
	ASSERT_GX(isFloatOK(r.Acc[1],file,line));
	ASSERT_GX(isFloatOK(r.Acc[2],file,line));
	ASSERT_GX(isFloatOK(r.DtEntropy,file,line));
	ASSERT_GX(isFloatOK(r.MaxSignalVel,file,line));
	return 1;
}

void PrintNode(const struct NODE_gx r,const char*const msg)
{
	MESSAGE("NODE_gx: %s",msg);
	MESSAGE("  len=%g, center=(%g|%g|%g)",r.len,r.center[0],r.center[1],r.center[2]);
	MESSAGE("  s=(%g|%g|%g)",r.u.d.s[0],r.u.d.s[1],r.u.d.s[2]);
	MESSAGE("  mass=%g",r.u.d.mass);
	MESSAGE("  bitflags=%d, sibling=%d, nextnode=%d, father=%d",r.u.d.bitflags,r.u.d.sibling,r.u.d.nextnode,r.u.d.father);
}

void PrintParameters(const struct parameters_gx*const p,const char*const msg)
{
	MESSAGE("Parameters: %s",msg);

	if (p==NULL) {
		MESSAGE("  p==NULL");
		return;
	}

	MESSAGE("  MaxPart              =%d",p->MaxPart);
	MESSAGE("  MaxNodes             =%d",p->MaxNodes);
	MESSAGE("  Ti_Current           =%d",p->Ti_Current);
	MESSAGE("  Asmth                =[%g;%g]",p->Asmth[0],p->Asmth[1]);
	MESSAGE("  Rcut                 =[%g;%g]",p->Rcut[0],p->Rcut[1]);
	MESSAGE("  BoxSize              =%g",p->BoxSize);
	MESSAGE("  ErrTolTheta;BoxSize  =%g",p->ErrTolTheta);
	MESSAGE("  ErrTolForceAccBoxSize=%g",p->ErrTolForceAcc);
	MESSAGE("  ForceSoftening       =[%g;%g;%g;%g;%g;%g]",p->ForceSoftening[0],p->ForceSoftening[1],p->ForceSoftening[2],p->ForceSoftening[3],p->ForceSoftening[4],p->ForceSoftening[5]);

	#ifdef ADAPTIVE_GRAVSOFT_FORGAS
		MESSAGE("  soft                 =%g",soft);
	#endif
	MESSAGE("  Masses               =[%g;%g;%g;%g;%g;%g]",p->Masses[0],p->Masses[1],p->Masses[2],p->Masses[3],p->Masses[4],p->Masses[5]);
	MESSAGE("  typesizes            =[%d;%d;%d;%d;%d;%d]",p->typesizes[0],p->typesizes[1],p->typesizes[2],p->typesizes[3],p->typesizes[4],p->typesizes[5]);
	MESSAGE("  nonsortedtypes       =%g",p->nonsortedtypes);
	MESSAGE("  ComovingIntegrationOn=%d",p->ComovingIntegrationOn);
	MESSAGE("  Timebase_interval    =%g",p->Timebase_interval);
	MESSAGE("  export_P_type        =%d",p->export_P_type);
}

void PrintParametersHydro(const struct parameters_hydro_gx*const h,const char*const msg)
{
	MESSAGE("ParametersHydro: %s",msg);

	if (h==NULL) {
		MESSAGE("  h==NULL");
		return;
	}
	MESSAGE("  N_gas                =%d",h->N_gas);
	MESSAGE("  szngb                =%d",h->szngb);
	MESSAGE("  hubble_a2            =%g",h->hubble_a2);
	MESSAGE("  fac_mu               =%g",h->fac_mu);
	MESSAGE("  fac_vsic_fix         =%g",h->fac_vsic_fix);
	MESSAGE("  ArtBulkViscConst     =%g",h->ArtBulkViscConst);

	#ifdef PERIODIC
		MESSAGE("  boxSize              =%g",h->boxSize);
		MESSAGE("  boxHalf              =%g",h->boxHalf);
		if (h->boxSize_X>0){
			MESSAGE("  boxSize_X            =%g",h->boxSize_X);
			MESSAGE("  boxHalf_X            =%g",h->boxHalf_X);
		}
		if (h->boxSize_Y>0){
			MESSAGE("  boxSize_Y                 =%g",h->boxSize_Y);
			MESSAGE("  boxHalf_Y                 =%g",h->boxHalf_Y);
		}
		if (h->boxSize_Z>0){
			MESSAGE("  boxSize_Z                 =%g",h->boxSize_Z);
			MESSAGE("  boxHalf_Z                 =%g",h->boxHalf_Z);
		}
	#endif
};

void PrintState(const struct state_gx*const s,const char*const msg)
{
	MESSAGE("State: %s",msg);

	if (s==NULL) {
		MESSAGE("  s==NULL");
		return;
	}
	MESSAGE("  P                    =%p",s->P);
	MESSAGE("  Nodes_base           =%p",s->Nodes_base);
	MESSAGE("  Nodes                =%p",s->Nodes);
	MESSAGE("  Nextnode             =%p",s->Nextnode);
	MESSAGE("  DomainTask           =%p",s->DomainTask);
	MESSAGE("  shortrange_table     =%p",s->shortrange_table);
	MESSAGE("  Exportflag           =%p",s->Exportflag);
	MESSAGE("  Nextnode             =%p",s->Nextnode);
	MESSAGE("  result               =%p",s->result);
	MESSAGE("  result_buffer        =%p",s->result_buffer);
	MESSAGE("  scratch              =%p",s->scratch);

	MESSAGE("  SphP                 =%p",s->SphP);
	MESSAGE("  etc                  =%p",s->etc);
	MESSAGE("  extNodes_base        =%p",s->extNodes_base);
	MESSAGE("  extNodes             =%p",s->extNodes);
	MESSAGE("  Ngblist              =%p",s->Ngblist);

	MESSAGE("  ThisTask             =%d",s->ThisTask);
	MESSAGE("  NTask                =%d",s->NTask);
	MESSAGE("  NumPart              =%d",s->NumPart);
	MESSAGE("  N_gas                =%d",s->N_gas);
	MESSAGE("  Np                   =%d",s->Np);
	MESSAGE("  segment              =%d",s->segment);
	MESSAGE("  sz_segments          =%d",s->sz_segments);
	MESSAGE("  MaxPart              =%d",s->MaxPart);
	MESSAGE("  sz_memory_limit      =%u",s->sz_memory_limit);

	MESSAGE("  sz_P                 =%u",s->sz_P);
	MESSAGE("  sz_SphP              =%u",s->sz_SphP);
	MESSAGE("  sz_Nodes_base        =%u",s->sz_Nodes_base);
	MESSAGE("  sz_Nextnode          =%u",s->sz_Nextnode);
	MESSAGE("  sz_DomainTask        =%u",s->sz_DomainTask);
	MESSAGE("  sz_shortrange_table  =%u",s->sz_shortrange_table);
	MESSAGE("  sz_Exportflag        =%u",s->sz_Exportflag);
	MESSAGE("  sz_result            =%u",s->sz_result);
	MESSAGE("  sz_result_buffer     =%u",s->sz_result_buffer);
	MESSAGE("  sz_scratch           =%u",s->sz_scratch);
	MESSAGE("  sz_etc               =%u",s->sz_etc);
	MESSAGE("  sz_extNodes_base     =%u",s->sz_extNodes_base);
	MESSAGE("  sz_Ngblist           =%u",s->sz_Ngblist);
	MESSAGE("  sz_result_hydro      =%u",s->sz_result_hydro);
	MESSAGE("  sz_Psorted           =%u",s->sz_Psorted);

	MESSAGE("  sz_max_P             =%u",s->sz_max_P);
	MESSAGE("  sz_max_SphP          =%u",s->sz_max_SphP);
	MESSAGE("  sz_max_Exportflag    =%u",s->sz_max_Exportflag);
	MESSAGE("  sz_max_result        =%u",s->sz_max_result);
// 	MESSAGE("  sz_max_result_buffer =%u",s->sz_max_result_buffer);
	MESSAGE("  sz_max_scratch       =%u",s->sz_max_scratch);
	MESSAGE("  sz_max_etc           =%u",s->sz_max_etc);
	MESSAGE("  sz_max_result_hydro  =%u",s->sz_max_result_hydro);
	MESSAGE("  sz_max_Ngblist       =%u",s->sz_max_Ngblist);
	MESSAGE("  sz_max_Psorted       =%u",s->sz_max_Psorted);

	MESSAGE("  mode                 =%d",s->mode);
	MESSAGE("  cudamode             =%d",s->cudamode);
	MESSAGE("  debugval             =%d",s->debugval);
	MESSAGE("  iteration            =%d",s->iteration);

	MESSAGE("  kernelsignals        =%p",s->kernelsignals);
	MESSAGE("  debug_msg            =%p",s->debug_msg);
	MESSAGE("  debug_sz_msg         =%d",s->debug_sz_msg);
}

void ValidateParameters(const struct parameters_gx p,const char*const file,const int line)
{
	// MESSAGE("ValidateParameters():%s:%d",file,line);
	int i;

	ASSERT_GX( p.MaxPart>0 );
	ASSERT_GX( p.MaxNodes>0 );
	ASSERT_GX( p.Ti_Current>=0 );
	#ifdef PMGRID
	ASSERT_GX( p.Asmth[0]!=0 );
	ASSERT_GX( p.Asmth[1]>=0 );
	ASSERT_GX( p.Rcut[0]>0 );
	ASSERT_GX( p.Rcut[1]>=0 );
	#endif
	#ifdef PERIODIC
		ASSERT_GX( p.BoxSize>0 );
	#endif
	ASSERT_GX( p.ErrTolTheta>=0 ); // does not hold?
	ASSERT_GX( p.ErrTolForceAcc>0 ); // does not hold?
	for(i=0;i<6;++i) ASSERT_GX( p.ForceSoftening[i]>=0 ); // maybe to restrictive?
	#ifdef ADAPTIVE_GRAVSOFT_FORGAS
		ASSERT_GX( p.soft>=0 ); // maybe to restrictive?
	#endif
	for(i=0;i<6;++i) ASSERT_GX( p.Masses[i]>0 || p.typesizes[i]==0 || p.nonsortedtypes==1);
/*
	int n=0;
	// N==p.typesizes[0] indicates N_gas/SPH mode
	for(i=0;i<6;++i) {
		ASSERT_GX( p.typesizes[i]>=0 );
		n+=p.typesizes[i];
		if (!(n<=N|| N==p.typesizes[0])) MESSAGE("xn=%d, N=%d, %d,%d,%d,%d,%d,%d",n,N,p.typesizes[0],p.typesizes[1],p.typesizes[2],p.typesizes[3],p.typesizes[4],p.typesizes[5]);

		ASSERT_GX(n<=N || N==p.typesizes[0]);
	}
	if (!( n==N  )) MESSAGE("n=%d, N=%d, %d,%d,%d,%d,%d,%d",n,N,p.typesizes[0],p.typesizes[1],p.typesizes[2],p.typesizes[3],p.typesizes[4],p.typesizes[5]);
	ASSERT_GX( n==N  || N==p.typesizes[0]); // all or sph mode
*/

	ASSERT_GX( p.ComovingIntegrationOn==0 || p.ComovingIntegrationOn==1);
	ASSERT_GX( p.Timebase_interval>0 );
	ASSERT_GX( (p.export_P_type>=0 && p.export_P_type<6) || p.export_P_type==-1 );

	ASSERT_GX(isFloatOK(p.Asmth[0],file,line));
	ASSERT_GX(isFloatOK(p.Asmth[1],file,line));
	ASSERT_GX(isFloatOK(p.Rcut[0],file,line));
	ASSERT_GX(isFloatOK(p.Rcut[1],file,line));
	ASSERT_GX(isFloatOK(p.BoxSize,file,line));
	ASSERT_GX(isFloatOK(p.ErrTolTheta,file,line));
	ASSERT_GX(isFloatOK(p.ErrTolForceAcc,file,line));
	for(i=0;i<6;++i) ASSERT_GX(isFloatOK(p.ForceSoftening[i],file,line));
	#ifdef ADAPTIVE_GRAVSOFT_FORGAS
		ASSERT_GX(isFloatOK(p.soft,file,line));
	#endif
	for(i=0;i<6;++i) ASSERT_GX(isFloatOK(p.Masses[i],file,line));
	ASSERT_GX ( isFloatOK(p.Timebase_interval,file,line) );
}

void ValidateParameters_hydra(const struct parameters_hydro_gx p,const char*const file,const int line)
{
	//MESSAGE("ValidateParameters_hydra():%s:%d",file,line);

	ASSERT_GX( p.N_gas>0 );
	#ifndef CUDA_GX_SHARED_NGBLIST
		ASSERT_GX( p.szngb>0 );
	#endif
	ASSERT_GX( p.hubble_a2>0);
	ASSERT_GX( p.fac_mu>0 );
	ASSERT_GX( p.fac_vsic_fix>0 );
	ASSERT_GX( p.ArtBulkViscConst>0 );

	ASSERT_GX ( isFloatOK(p.hubble_a2,file,line) );
	ASSERT_GX ( isFloatOK(p.fac_mu,file,line) );
	ASSERT_GX ( isFloatOK(p.fac_vsic_fix,file,line) );
	ASSERT_GX ( isFloatOK(p.ArtBulkViscConst,file,line) );

	#ifdef PERIODIC
		ASSERT_GX( p.boxSize>0 );
		ASSERT_GX( p.boxHalf>0 );
		#ifdef LONG_X
			ASSERT_GX( p.boxSize_X>0 );
			ASSERT_GX( p.boxHalf_X>0 );
		#else
			ASSERT_GX( p.boxSize_X==0 );
			ASSERT_GX( p.boxHalf_X==0 );
		#endif
		#ifdef LONG_Y
			ASSERT_GX( p.boxSize_Y>0 );
			ASSERT_GX( p.boxHalf_Y>0 );
		#else
			ASSERT_GX( p.boxSize_Y==0 );
			ASSERT_GX( p.boxHalf_Y==0 );
		#endif
		#ifdef LONG_Z
			ASSERT_GX( p.boxSize_Z>0 );
			ASSERT_GX( p.boxHalf_Z>0 );
		#else
			ASSERT_GX( p.boxSize_Z==0 );
			ASSERT_GX( p.boxHalf_Z==0 );
		#endif
		ASSERT_GX ( isFloatOK(p.boxSize,file,line) );
		ASSERT_GX ( isFloatOK(p.boxHalf,file,line) );
		ASSERT_GX ( isFloatOK(p.boxSize_X,file,line) );
		ASSERT_GX ( isFloatOK(p.boxHalf_X,file,line) );
		ASSERT_GX ( isFloatOK(p.boxSize_Y,file,line) );
		ASSERT_GX ( isFloatOK(p.boxHalf_Y,file,line) );
		ASSERT_GX ( isFloatOK(p.boxSize_Z,file,line) );
		ASSERT_GX ( isFloatOK(p.boxHalf_Z,file,line) );
	#else
		/* Disables these checks, since hydro is disabled in current version */
		/* ASSERT_GX( p.boxSize==0 );
		   ASSERT_GX( p.boxHalf==0 ); */
	#endif
}

void ValidateState(const struct state_gx s,const int maxnodes,const int checkparticledata,const char*const file,const int line)
{
	//MESSAGE("ValidateState(%d):%s:%d,",maxnodes,file,line);
	int i;

	// pointers
	if (checkparticledata){
		ASSERT_GX( s.P!=NULL || s.SphP!=NULL);
		ASSERT_GX( s.shortrange_table!=NULL);
		ASSERT_GX( s.result!=NULL);
		ASSERT_GX( s.Exportflag!=NULL || s.NTask==1 );
	}

	ASSERT_GX( s.Nodes_base!=NULL );
	ASSERT_GX( s.Nodes!=NULL );
	ASSERT_GX( s.Nextnode!=NULL );
	ASSERT_GX( s.DomainTask!=NULL || s.NTask==1 );

	// sizes
	if (checkparticledata){
		ASSERT_GX( s.sz_P>0 );
		ASSERT_GX( s.sz_shortrange_table>0 );
		ASSERT_GX( s.sz_result>0 );
	}

	// max sizes
	ASSERT_GX( s.sz_P<=s.sz_max_P );
	ASSERT_GX( s.sz_SphP<=s.sz_max_SphP );
	ASSERT_GX( s.sz_Exportflag<=s.sz_max_Exportflag );
	ASSERT_GX( s.sz_result<=s.sz_max_result );
	ASSERT_GX( s.sz_result_buffer<=s.sz_max_result_buffer );
	ASSERT_GX( s.sz_scratch<=s.sz_max_scratch );
	ASSERT_GX( s.sz_etc<=s.sz_max_etc );
	ASSERT_GX( s.sz_result_hydro<=s.sz_max_result_hydro );
	ASSERT_GX( s.sz_hydrodata_in<=s.sz_max_hydrodata_in );
	ASSERT_GX( s.sz_Ngblist<=s.sz_max_Ngblist );
	ASSERT_GX( s.sz_Psorted<=s.sz_max_Psorted );

	// values
	ASSERT_GX( s.sz_memory_limit>0 );
	ASSERT_GX( s.MaxPart>0 );
	ASSERT_GX( s.ThisTask>=0 );
	ASSERT_GX( s.NTask>0 );

	ASSERT_GX( s.sz_P<=s.MaxPart );
	ASSERT_GX( s.sz_SphP<=s.MaxPart );
	ASSERT_GX( (s.segment==0 && s.sz_segments==0) || s.segment<s.sz_segments );

	// compare to GADGET2 vars
	if (checkparticledata){
		ASSERT_GX( s.sz_P==s.NumPart  || s.NumPart==0); // slightly inconsistet state, but only intermediate
		ASSERT_GX( s.sz_SphP==s.N_gas || s.N_gas==0);   // -
		ASSERT_GX( s.NumPart==NumPart || s.NumPart==0  ||  s.external_node>=0 );
		//ASSERT_GX( s.N_gas==N_gas     || s.N_gas==0 );
	}

	//ASSERT_GX( s.NumPart>=0 );   // NOTE: unsigned anyway so pointless compare
	//ASSERT_GX( s.N_gas>=0 );     // -
	//ASSERT_GX( s.Np>=0 );        // -
	ASSERT_GX( s.Np<=s.NumPart || s.mode==1 );
	ASSERT_GX( s.Np<=s.sz_P    || s.mode==1 );

	ASSERT_GX( s.mode==0 || s.mode==1 );
	ASSERT_GX( s.cudamode>=1 && s.cudamode<3 );
	ASSERT_GX( s.debugval>=0 && s.debugval<5 ); // somewhat abritrary upper limit
	//ASSERT_GX( s.iteration>=0 ); // NOTE: unsigned anyway so pointless compare

	ASSERT_GX( s.sz_result==s.Np ||  s.sz_result_hydro==s.Np );
	ASSERT_GX( s.sz_result==s.sz_Psorted || s.sz_Psorted==0 || s.mode==1 );
	ASSERT_GX( s.sz_scratch==s.sz_result || s.mode==0 );
	ASSERT_GX( s.sz_result_buffer==s.sz_Psorted );
	ASSERT_GX( s.sz_max_result_buffer==s.sz_max_Psorted );

	ASSERT_GX( s.sphmode==0 || s.sphmode==1 );
	ASSERT_GX( s.blocks>=0 );
	ASSERT_GX( s.grids>=0 );
	ASSERT_GX( s.external_node>=-1 );

	// multicpu mode or single mode states
	ASSERT_GX( s.NTask>=1 );
	if (s.NTask==1){
		ASSERT_GX( s.sz_Exportflag==0 && s.Exportflag==NULL);
		ASSERT_GX( s.sz_DomainTask==0 && s.DomainTask==NULL);
	} else {
		ASSERT_GX( s.sz_DomainTask>0  && s.DomainTask!=NULL);
		if (checkparticledata) ASSERT_GX( s.sz_Exportflag>0  && s.Exportflag!=NULL);
		if (s.cudamode<=1){
			for(i=0;i<s.sz_DomainTask;++i) ASSERT_GX( s.DomainTask[i]>=0 );
			if (checkparticledata){
		for(i=0;i<s.sz_Exportflag;++i) ASSERT_GX( s.Exportflag[i]==0);
			}
		}
	}

	// non-sph or sph mode
	if (s.sz_SphP>0){
		ASSERT_GX(s.SphP!=NULL && s.sz_SphP>0);
		ASSERT_GX(s.result_hydro!=NULL && s.sz_result_hydro>0);
	} else {
		ASSERT_GX(s.SphP==NULL && s.sz_SphP==0);
		ASSERT_GX(s.result_hydro==NULL && s.sz_result_hydro==0);
	}

	// test pointer array limits, make access violation if not valid
	if (maxnodes>0 && s.cudamode<=1 && s.sz_P>0) {
		const float t1=s.P[0].Pos[0];
		const float t2=s.P[s.MaxPart-1].Pos[0];

		const int   t3=s.Nextnode[0];
		const int   t4=s.Nextnode[maxnodes-1];

		if (t1!=t2) {}; // avoid un-referenced vars
		if (t3!=t4) {};
	}
}

struct state_gx ClearStatepointers(const struct state_gx r)
{
	struct state_gx s=r;

	s.P=0;
	s.SphP=0;
	s.Nodes_base=0;
	s.Nodes=0;
	s.Nextnode=0;
	s.DomainTask=0;
	s.shortrange_table=0;
	s.Exportflag=0;
	s.result=0;
	s.result_buffer=0;
	s.scratch=0;
	s.etc=0;
	s.extNodes_base=0;
	s.extNodes=0;
	s.Ngblist=0;
	s.Psorted=0;

	// NOTE: these values will change btw s1 and s2
	s.Psorted=0;
	s.sz_Psorted=0;
	s.sz_max_Psorted=0;
	s.iteration=0;

	return s;
}

int EqualState(struct state_gx s1,struct state_gx s2,const int cmppointers,const int cmpall,const char*const file,const int line)
{
	//MESSAGE("EqualState() %s:%d:",file,line);
	if (!cmppointers){
		s1=ClearStatepointers(s1);
		s2=ClearStatepointers(s2);
	}

	// pointers
	ASSERT_GX( s1.P==s2.P );
	ASSERT_GX( s1.SphP==s2.SphP );
	ASSERT_GX( s1.Nodes_base==s2.Nodes_base);
	ASSERT_GX( s1.Nodes==s2.Nodes);
	ASSERT_GX( s1.Nextnode==s2.Nextnode);
	ASSERT_GX( s1.DomainTask==s2.DomainTask );
	ASSERT_GX( s1.shortrange_table==s2.shortrange_table);
	ASSERT_GX( s1.Exportflag==s2.Exportflag );
	ASSERT_GX( s1.result==s2.result);
	ASSERT_GX( s1.result_buffer==s2.result_buffer);
	ASSERT_GX( s1.scratch==s2.scratch);
	ASSERT_GX( s1.extNodes_base==s2.extNodes_base);
	ASSERT_GX( s1.extNodes==s2.extNodes);
	ASSERT_GX( s1.Ngblist==s2.Ngblist);
	ASSERT_GX( s1.Psorted==s2.Psorted);

	// sizes
	ASSERT_GX( s1.sz_P==s2.sz_P );
	ASSERT_GX( s1.sz_SphP==s2.sz_SphP );
	ASSERT_GX( s1.sz_Nodes_base==s2.sz_Nodes_base );
	ASSERT_GX( s1.sz_Nextnode==s2.sz_Nextnode );
	ASSERT_GX( s1.sz_DomainTask==s2.sz_DomainTask );

	if (cmpall){
		ASSERT_GX( s1.sz_shortrange_table==s2.sz_shortrange_table );
		ASSERT_GX( s1.sz_Exportflag==s2.sz_Exportflag );
		ASSERT_GX( s1.sz_result==s2.sz_result );

		ASSERT_GX( s1.sz_scratch==s2.sz_scratch );
		ASSERT_GX( s1.sz_etc==s2.sz_etc );
		ASSERT_GX( s1.sz_extNodes_base==s2.sz_extNodes_base);
		ASSERT_GX( s1.sz_Ngblist==s2.sz_Ngblist );
		ASSERT_GX( s1.sz_Psorted==s2.sz_Psorted );

		ASSERT_GX( s1.sz_max_P==s2.sz_max_P );
		ASSERT_GX( s1.sz_max_SphP==s2.sz_max_SphP );
		ASSERT_GX( s1.sz_max_Exportflag==s2.sz_max_Exportflag);
		ASSERT_GX( s1.sz_max_result==s2.sz_max_result );
		ASSERT_GX( s1.sz_max_scratch==s2.sz_max_scratch );
		ASSERT_GX( s1.sz_max_etc==s2.sz_max_etc );
		ASSERT_GX( s1.sz_max_result_hydro==s2.sz_max_result_hydro );
		ASSERT_GX( s1.sz_max_hydrodata_in==s2.sz_max_hydrodata_in );
		ASSERT_GX( s1.sz_max_Ngblist==s2.sz_max_Ngblist );
		ASSERT_GX( s1.sz_max_Psorted==s2.sz_max_Psorted );

		// values
		ASSERT_GX( s1.ThisTask==s2.ThisTask );
		ASSERT_GX( s1.NTask==s2.NTask );
		ASSERT_GX( s1.NumPart==s2.NumPart );
		ASSERT_GX( s1.N_gas==s2.N_gas );
		ASSERT_GX( s1.Np==s2.Np );
		ASSERT_GX( s1.sz_memory_limit==s2.sz_memory_limit );
		ASSERT_GX( s1.mode==s2.mode );
		ASSERT_GX( s1.cudamode==s2.cudamode );
		ASSERT_GX( s1.debugval==s2.debugval );

		ASSERT_GX( s1.iteration==s2.iteration );
	}

	return 1;
}

#ifndef __CUDACC__

int hasParticleDataBeenModified_gx(const size_t N,const struct particle_data *const s,const int print)
{
	if (s_gx.P==NULL || s_gx.sz_P!=N) {
		if (print) {
			if (s_gx.P==NULL) WARNING("particles has null pointer");
			if (s_gx.sz_P!=N) WARNING("particles changed on size");
		}
		return 1;
	}

	int changed=0;
	size_t m=0;
	while(m<N && !changed) {
		//if (memcmp(&s_gx.P[m],&s[m],sizeof(struct particle_data_gx))!=0) changed=m+1;
		if (s_gx.P[m].Pos[0]!=s[m].Pos[0]) changed=m+1;
		if (s_gx.P[m].Pos[1]!=s[m].Pos[1]) changed=m+1;
		if (s_gx.P[m].Pos[2]!=s[m].Pos[2]) changed=m+1;
		if (s_gx.P[m].Mass  !=s[m].Mass  ) changed=m+1;
		++m;
	}
	if (changed && print){
		const int c=changed-1;
		WARNING("particle_data changed on c=%d",c);
		if (s_gx.P[c].Pos[0]!=s[c].Pos[0]) MESSAGE("x: %g!=%g",s_gx.P[c].Pos[0],s[c].Pos[0]);
		if (s_gx.P[c].Pos[1]!=s[c].Pos[1]) MESSAGE("y: %g!=%g",s_gx.P[c].Pos[1],s[c].Pos[1]);
		if (s_gx.P[c].Pos[2]!=s[c].Pos[2]) MESSAGE("z: %g!=%g",s_gx.P[c].Pos[2],s[c].Pos[2]);
		if (s_gx.P[c].Mass  !=s[c].Mass  ) MESSAGE("Mass: %g!=%g",s_gx.P[c].Mass,s[c].Mass);
	}
	return changed;
}

int hasShortrangeTableDataBeenModified_gx(const size_t N,const float *const s,const int print)
{
	if (s_gx.shortrange_table==NULL || s_gx.sz_shortrange_table!=N) {
		if (print) {
			if (s_gx.shortrange_table==NULL) WARNING("shortrange_table has null pointer");
			if (s_gx.sz_shortrange_table!=N) WARNING("shortrange_table changed on size");
		}
		return 1;
	}

	if (s_gx.cudamode>=2) return 0;

	int changed=0;
	size_t m=0;
	while(m<N && !changed) {
		if (s_gx.shortrange_table[m]!=s[m]) return m;
		++m;
	}
	return changed;
}

int hasNodeDataBeenModified_gx(const size_t N,const struct NODE *const s,const int print)
{
	if (s_gx.Nodes_base==NULL || s_gx.sz_Nodes_base!=N) {
		if (print) {
			if (s_gx.Nextnode==NULL) MESSAGE("node has null pointer");
			if (s_gx.sz_Nextnode!=N) MESSAGE("node changed on size");
		}
		return 1;
	}

	if (s_gx.cudamode>=2) return 0;

	int changed=0;
	size_t m=0;
	while(m<N && !changed) {
		if (memcmp(&s_gx.Nodes_base[m],&s[m],sizeof(struct NODE))!=0) changed=m+1;
		++m;
	}

	if (changed && print) MESSAGE("node changed on c=%d",changed-1);
	return changed;
}

int hasNextnodeDataBeenModified_gx(const size_t N,const int*const s,const int print)
{
	if (s_gx.Nextnode==NULL || s_gx.sz_Nextnode!=N) {
		if (print) {
			if (s_gx.Nextnode==NULL) MESSAGE("nextnode has null pointer");
			if (s_gx.sz_Nextnode!=N) MESSAGE("nextnode changed on size");
		}
		return 1;
	}

	if (s_gx.cudamode>=2) return 0;

	int changed=0;
	size_t m=0;
	while(m<N && !changed) {
		if (s_gx.Nextnode[m]!=s[m]) {
			if (print) MESSAGE("s_gx.Nextnode[m]=%d, s[m]=%d",s_gx.Nextnode[m],s[m]);
			changed=m+1;
		}
		++m;
	}

	if (changed && print) MESSAGE("nextnode changed on c=%d",changed-1);
	return changed;
}

int hasDomainTaskDataBeenModified_gx(const size_t N,const int*const s,const int print)
{
	if (s_gx.DomainTask==NULL || s_gx.sz_DomainTask!=N) {
		if (print) {
			if (s_gx.DomainTask==NULL) MESSAGE("DomainTask has null pointer");
			if (s_gx.sz_DomainTask!=N) MESSAGE("DomainTask changed on size");
		}
		return 1;
	}

	int changed=0;
	size_t m=0;
	while(m<N && !changed) {
		if (s_gx.DomainTask[m]!=s[m]) changed=m+1;
		++m;
	}

	if (changed && print) MESSAGE("DomainTask changed on c=%d",changed-1);
	return changed;
}

int hasParameterDataBeenModified_gx(const int N,const int print,const int ignore_ti_current)
{
	struct parameters_gx p=FillParameters_gx();

	int i;
	for(i=0;i<6;++i) p.typesizes[i]=p_gx.typesizes[i]; // NOTE: typesizes are set in a poststep
	p.nonsortedtypes=p_gx.nonsortedtypes; // YYY NOTE: still strange!

	if (ignore_ti_current) {
		//ASSERT_GX(p.Ti_Current!=p_gx.Ti_Current);
		p.Ti_Current=p_gx.Ti_Current; // Ti current increased in sph calc
	}

	const int change=memcmp(&p,&p_gx,sizeof(struct parameters_gx))!=0;

	if (change && print){
		MESSAGE("parameters_gx structure changed:");

		if (p.MaxPart!=p_gx.MaxPart)               MESSAGE("  MaxPart:  %d!=%d",p.MaxPart,p_gx.MaxPart);
		if (p.MaxNodes!=p_gx.MaxNodes)             MESSAGE("  MaxNodes: %d!=%d",p.MaxNodes,p_gx.MaxNodes);

		if (p.Ti_Current!=p_gx.Ti_Current)         MESSAGE("  Ti_Current: %g!=%g",p.Ti_Current,p_gx.Ti_Current);
		if (p.Asmth[0]!=p_gx.Asmth[0])             MESSAGE("  Asmth[0]: %g!=%g",p.Asmth[0],p_gx.Asmth[0]);
		if (p.Asmth[1]!=p_gx.Asmth[1])             MESSAGE("  Asmth[1]: %g!=%g",p.Asmth[1],p_gx.Asmth[1]);
		if (p.Rcut[0]!=p_gx.Rcut[0])               MESSAGE("  Rcut[0]:  %g!=%g",p.Rcut[0],p_gx.Rcut[0]);
		if (p.Rcut[1]!=p_gx.Rcut[1])               MESSAGE("  Rcut[1]:  %g!=%g",p.Rcut[1],p_gx.Rcut[1]);
		if (p.BoxSize!=p_gx.BoxSize)               MESSAGE("  BoxSize:  %g!=%g",p.BoxSize,p_gx.BoxSize);
		if (p.ErrTolTheta!=p_gx.ErrTolTheta)       MESSAGE("  ErrTolTheta:    %g!=%g",p.ErrTolTheta,p_gx.ErrTolTheta);
		if (p.ErrTolForceAcc!=p_gx.ErrTolForceAcc) MESSAGE("  ErrTolForceAcc: %g!=%g",p.ErrTolForceAcc,p_gx.ErrTolForceAcc);
		for(i=0;i<6;++i){
			if (p.ForceSoftening[i]!=p_gx.ForceSoftening[i]) MESSAGE("  ForceSoftening[%d]: %g!=%g",i,p.ForceSoftening[i],p_gx.ForceSoftening[i]);
		}
		#ifdef ADAPTIVE_GRAVSOFT_FORGAS
		float soft;
		if (p.soft!=p_gx.soft)                      MESSAGE("  soft:     %g!=%g",p.soft,p_gx.soft);
		#endif
		for(i=0;i<6;++i){
			if (p.Masses[i]!=p_gx.Masses[i])      MESSAGE("  Masses[%d]: %g!=%g",i,p.Masses[i],p_gx.Masses[i]);
			if (p.typesizes[i]!=p_gx.typesizes[i])MESSAGE("  typesizes[%d]: %g!=%g",i,p.typesizes[i],p_gx.typesizes[i]);
		}
		//YYY if (p.nonsortedtypes!=p_gx.nonsortedtypes)               MESSAGE("  nonsortedtypes:  %d!=%d",p.nonsortedtypes,p_gx.nonsortedtypes);
		if (p.ComovingIntegrationOn!=p_gx.ComovingIntegrationOn) MESSAGE("  ComovingIntegrationOn:  %g!=%g",p.ComovingIntegrationOn,p_gx.ComovingIntegrationOn);
		if (p.Timebase_interval!=p_gx.Timebase_interval)         MESSAGE("  Timebase_interval:  %g!=%g",p.Timebase_interval,p_gx.Timebase_interval);
		if (p.export_P_type!=p_gx.export_P_type)                 MESSAGE("  export_P_type:  %g!=%g",p.export_P_type,p_gx.export_P_type);
	}
	return change;
}

#if CUDA_DEBUG_GX>1
	void AssertsOnhasGadgetDataBeenModified_gx(const int no_particle_change,const int print,const int ignore_ti_current)
	{
		if (isEmulationMode()){
			ASSERT_GX(!no_particle_change || !hasParticleDataBeenModified_gx(NumPart,P,print) );
			// ASSERT_GX( !hasShortrangeTableDataBeenModified_gx        (1000,shortrange_table,print) ); // NOTE: not possible, NTAB and shortrange_table is private in forcetree.c
			ASSERT_GX( !hasNodeDataBeenModified_gx                   (MaxNodes+1,Nodes_base,print) );
			ASSERT_GX( !hasNextnodeDataBeenModified_gx               (All.MaxPart + MAXTOPNODES,Nextnode,print));
			ASSERT_GX( NTask==1 || !hasDomainTaskDataBeenModified_gx (MAXTOPNODES,DomainTask,print) );
			ASSERT_GX( !hasParameterDataBeenModified_gx              (NumPart,print,ignore_ti_current) );
		} else{
			static int warn=1;
			if (warn){
				WARNING("AssertsOnhasGadgetDataBeenModified_gx() can only run in emulation mode");
				warn=0;
			}
		}
	}
#endif // CUDA_DEBUG_GX>1

double DistRMS(const size_t N,const struct particle_data *const p,const int Ti_Current)
{
	int i=0,n=0;
	double sum2=0;

	while(i<N && p[i].Ti_endstep!=Ti_Current) ++i;
	if (i>=N) {
		MESSAGE("no partiles with Ti_Current=%d",Ti_Current);
		return 0;
	}

	while(1){
		int j=i+1;

		while(j<N && p[j].Ti_endstep!=Ti_Current) ++j;

		if (i>=N || j>=N){
			if (n<2) return 0;
			else     return sqrt(sum2/(n-1));
		}

		ASSERT_GX( i<N && j<N);
		int k;
		for(k=0;k<3;++k) {
			const double t=p[i].Pos[k]-p[j].Pos[k];
			sum2 += t*t;
			++n;
		}
		i=j;
	}

	return 0;
}

double DistRMSGravdata(const size_t N,const struct gravdata_in *const s)
{
	int i,j;
	double sum2=0;
	if (N<=1) return 0;
	for(i=0;i<N-1;++i){
		for(j=0;j<3;++j) {
			const double t=s[i].u.Pos[j]-s[i+1].u.Pos[j];
			sum2 += t*t;
		}
	}
	return sqrt(sum2/(N-1));
}

int GetParticleTypeX(const int n)
{
	ASSERT_GX(n<s_gx.sz_etc);
	return s_gx.etc[n].Type;
}

int TestGetAuxData(const int N)
{
	int target;
	const struct state_gx g_const_state=s_gx;

// 	MESSAGE("TestGetAuxData: N=%d, s_gx.Np=%d, g_const_state.sz_result=%d, g_const_state.sz_etc=%d",N,s_gx.Np,g_const_state.sz_result,g_const_state.sz_etc);
//
// 	int j;
// 	const int tt=18318;
// 	for(j=tt-10;j<tt+10;++j) {
// 		int t=-1;
// 		if (j<g_const_state.sz_etc) t=GetParticleTypeX(j);
// 		if (j<g_const_state.sz_result) {
// 			const struct result_gx d=g_const_state.result[j];
// 			MESSAGE("GetParticleTypeX(%d))=%d, result: (d.ninteractions & 7)=%d, (d.ninteractions >> 3)=%d, d.Type=%d",j,t,(d.ninteractions & 7),(d.ninteractions >> 3),d.type);
// 		}
// 	}

	for(target=0;target<g_const_state.sz_result;target++){
		ASSERT_GX( target<g_const_state.sz_result);
		ASSERT_GX( g_const_state.mode==0 || g_const_state.mode==1);

		const struct result_gx d=g_const_state.result[target];

		ASSERT_GX( g_const_state.mode==1 || d.acc_y==0 );
		ASSERT_GX( g_const_state.mode==1 || d.acc_z==0 );
		//ASSERT_GX( !isNAN_device(d.acc_x) && !isNAN_device(d.acc_y) && !isNAN_device(d.acc_z) );
		ASSERT_GX( (d.ninteractions & 7)>=0 && (d.ninteractions& 7)<6 );

		if (!( g_const_state.mode==1 || (d.ninteractions & 7)==GetParticleTypeX(d.ninteractions>>3)))
			MESSAGE("g_const_state.mode=%d, target=%d, (d.ninteractions & 7)=%d, (d.ninteractions>>3)=%d, GetParticleType(d.ninteractions>>3)=%d",g_const_state.mode,target,(d.ninteractions & 7),(d.ninteractions>>3),GetParticleTypeX(d.ninteractions>>3));

		ASSERT_GX( g_const_state.mode==1 || (d.ninteractions & 7)==GetParticleTypeX(d.ninteractions>>3) );

		#ifdef CUDA_GX_DEBUG_MEMORY_ERROR
			ASSERT_GX ( (d.ninteractions>>3)==d.realtarget );
			if(!((g_const_state.mode==1 || GetParticleTypeX(d.ninteractions>>3)==d.type) && (d.ninteractions & 7)==d.type ))
				MESSAGE("mode=%d, target=%d, (d.ninteractions & 7)=%d, (d.ninteractions>>3)=%d, d.type=%d, GetParticleType(d.ninteractions>>3)=%d",g_const_state.mode,target,(d.ninteractions & 7),(d.ninteractions>>3), d.type, GetParticleTypeX(d.ninteractions>>3));

			ASSERT_GX( (d.ninteractions & 7)==d.type );
			ASSERT_GX( g_const_state.mode==1 || GetParticleTypeX(d.ninteractions>>3)==d.type );

// 			if (!((g_const_state.mode==0 && d.temp2==target) || (g_const_state.mode==1 && d.temp2==-target)))
// 				exit_device_info(-5,target,g_const_state.ThisTask,d.temp2,(size_t)&d.temp2,target>0 ? g_const_state.result[target-1].temp2 : 0,target+1<g_const_state.sz_result ? g_const_state.result[target+1].temp2 : 0,-42.5);

			ASSERT_GX( (g_const_state.mode==0 && d.temp2==target) || (g_const_state.mode==1 && d.temp2==-target) ); // YYY while non local target list in result data

			// a variation of error 4
/*			if (!((d.ninteractions & 7)>=0 && (d.ninteractions & 7)<6))
				exit_device_info(-4,d.ninteractions,(size_t)&d.ninteractions,d.temp2,g_const_state.ThisTask,0,0,-42.4);*/
		#endif

	}

	return 1;
}

#endif // __CUDACC__
#endif // CUDA_DEBUG_GX > 1

#ifndef __CUDACC__

int PrintInfoInitialize(const int N,const int cudamode,const int sphmode)
{
	FUN_MESSAGE(3,"PrintInfoInitialize(%d)",N);
	// if (s_gx.cudamode>0 && All.MaxPart>1400000) TimersSleep(10); // GPU card runs hot on large sims, this is around N_p=1404928
	// if (s_gx.cudamode>0) TimersSleep(10);

	if(cudamode>0) return CountParticlesInTimeStep_gx(N,P,All.Ti_Current,sphmode);
	else return -1;
}

void PrintInfoFinalize(const struct state_gx s,const int ndone,const int Np,const double starttime,const double cpytime,const double subtime,const int printmode,const int iter,const int lev,const int not_timestepped_gx,const int count_exported_gx,const int nexport,const int nexportsum,const int exporthash_gx,const double costtotal)
{
	// printmode: 0=forcetree, 1=hydro, 2=forcetree export, 3=hydro export
	FUN_MESSAGE(5,"PrintInfoFinalize()");
	ASSERT_GX( printmode>=0 && printmode<4 );

	const int orgmode=subtime<0;
	const char c=(s.cudamode>0 && orgmode) ? '*' : ' ';

	char iteration[16],level[16];

	ASSERT_GX(iter>=1 && lev>=-1);
	if (iter==1) iteration[0]=0;
	else sprintf(iteration,"[%d]",iter-1);
	if (lev<2) level[0]=0;
	else sprintf(level,"{%d}",lev);

	//if(!( Np==s_gx.Np || orgmode || printmode!=0 )) MESSAGE("ndone=%d, Np=%d, s_gx.Np=%d, s.mode=%d, orgmode=%d, printmode=%d",ndone,Np,s_gx.Np,s.mode,orgmode,printmode);
	//ASSERT_GX( Np==s_gx.Np || orgmode || printmode!=0 );

	#if CUDA_DEBUG_GX < 1
		//if (orgmode) return; // NOTE: not too much debug info in production mode
	#endif

	const double  totaltime=GetTime()-starttime;

	static double totaltime_sum [8]={0,0,0,0,0,0,0,0};
	static double totaltime_sum2[8]={0,0,0,0,0,0,0,0};
	static int    totaltime_n   [8]={0,0,0,0,0,0,0,0};

	const int tidex= printmode + orgmode*4;
	ASSERT_GX( tidex>=0 && tidex< 8);

	++totaltime_n[tidex];
	totaltime_sum [tidex] += totaltime;
	totaltime_sum2[tidex] += totaltime*totaltime;

	const double m=totaltime_sum[tidex]/totaltime_n[tidex];
	const double v=sqrt(totaltime_sum2[tidex]/totaltime_n[tidex] - m*m) ;

	if (printmode==0) {
//		ASSERT_GX(ndone==Np || Np==-1);
//		ASSERT_GX(ndone+not_timestepped_gx==NumPart || s_gx.cudamode==0);
		ASSERT_GX(ndone>=count_exported_gx || s_gx.cudamode==0);
		ASSERT_GX(NTask>1 || (nexport==0 && nexportsum==0));
		ASSERT_GX(iter==1 || orgmode);

		if (orgmode) MESSAGE("force_treeevaluate  [%9d,%9d,t=%.2f <%.2f,%.2f> s]%c"                ,NumPart,ndone,totaltime,m,v,c);
		else         MESSAGE("force_treeevaluate  [%9d,%9d,t={a=%.2f;c=%.2f;s=%.2f} <%.2f,%.2f> s]",NumPart,ndone,totaltime,cpytime,subtime,m,v);

		#if CUDA_DEBUG_GX>1
			MESSAGE("INFO: ndone=%d, export=%d=%.1f%c, nottimestep=%d=%.1f%c, hash=%d",ndone,count_exported_gx,100.0*count_exported_gx/ndone,'%',not_timestepped_gx,not_timestepped_gx!=0 ? 100.0*not_timestepped_gx/NumPart : 0.0,'%',exporthash_gx);
			MESSAGE("INFO: eGFLOPS/s=%g, costtotal=%g, DistRMS=%g",(double)costtotal*20/1000.0/1000.0/1000.0/totaltime,costtotal,DistRMS(NumPart,P,All.Ti_Current));
		#endif
	} else if (printmode==1){
		ASSERT_GX(iter==1 || orgmode);

		if (orgmode) MESSAGE("hydro_evaluate      [%9d,%9d,t=%.2f <%.2f,%.2f> s]%c"                ,N_gas,ndone,totaltime,m,v,c);
		else         MESSAGE("hydro_evaluate      [%9d,%9d,t={a=%.2f;c=%.2f;s=%.2f} <%.2f,%.2f> s]",N_gas,ndone,totaltime,cpytime,subtime,m,v);
	} else if (printmode==2){
		ASSERT_GX(ndone==0);

		if (orgmode) MESSAGE("force_treeevaluate,E[%9d,%8.1f%c,t=%.2f <%.2f,%.2f> s]%c %s%s"                ,Np,100.0*Np/NumPart,'%',totaltime,m,v,c,iteration,level);
		else         MESSAGE("force_treeevaluate,E[%9d,%8.1f%c,t={a=%.2f;c=%.2f;s=%.2f} <%.2f,%.2f> s] %s%s",Np,100.0*Np/NumPart,'%',totaltime,cpytime,subtime,m,v,iteration,level);
	} else if (printmode==3){
		ASSERT_GX(ndone==0);

		if (orgmode) MESSAGE("hydro_evaluate,E    [%9d,%8.1f%c,t=%.2f <%.2f,%.2f> s]%c %s%s"                ,Np,100.0*Np/NumPart,'%',totaltime,m,v,c,iteration,level);
		else         MESSAGE("hydro_evaluate,E    [%9d,%8.1f%c,t={a=%.2f;c=%.2f;s=%.2f} <%.2f,%.2f> s] %s%s",Np,100.0*Np/NumPart,'%',totaltime,cpytime,subtime,m,v,iteration,level);
	} else {
		ERROR("bad printmode, %d",printmode);
	}

	#if CUDA_DEBUG_GX>1
		MESSAGE("INFO: ndone=%d, export=%d=%.1f%c",ndone,nexport,100.0*nexport/ndone,'%',not_timestepped_gx);
	#endif
}

#endif // __CUDACC__

