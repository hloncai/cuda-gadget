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

#define NEW_PERIODIC_FUN_GX

// NOTE: these defines are pasted from the org GADGET2 source
#ifndef GAMMA
	#ifdef ISOTHERM_EQS
		#define  GAMMA         (1.0)     //  index for isothermal gas
	#else
		#define  GAMMA         (5.0/3)   // adiabatic index of simulated gas
	#endif
#endif

// NOTE: these defines are pasted from the org GADGET2 source
#ifndef NUMDIMS
	#ifndef  TWODIMS
		#define  NUMDIMS 3                                      // For 3D-normalized kernel
		#define  KERNEL_COEFF_1  2.546479089470                 // Coefficients for SPH spline kernel and its derivative
		#define  KERNEL_COEFF_2  15.278874536822
		#define  KERNEL_COEFF_3  45.836623610466
		#define  KERNEL_COEFF_4  30.557749073644
		#define  KERNEL_COEFF_5  5.092958178941
		#define  KERNEL_COEFF_6  (-15.278874536822)
		#define  NORM_COEFF      4.188790204786                 // Coefficient for kernel normalization. Note:  4.0/3 * PI = 4.188790204786
	#else
		#define  NUMDIMS 2                                      // For 2D-normalized kernel
		#define  KERNEL_COEFF_1  (5.0/7*2.546479089470)         // Coefficients for SPH spline kernel and its derivative
		#define  KERNEL_COEFF_2  (5.0/7*15.278874536822)
		#define  KERNEL_COEFF_3  (5.0/7*45.836623610466)
		#define  KERNEL_COEFF_4  (5.0/7*30.557749073644)
		#define  KERNEL_COEFF_5  (5.0/7*5.092958178941)
		#define  KERNEL_COEFF_6  (5.0/7*(-15.278874536822))
		#define  NORM_COEFF      M_PI                           // Coefficient for kernel normalization.
	#endif
#endif

#ifdef boxSize_X
	ERROR: boxSize_X should not be defined here
#endif
#ifdef boxSize_Y
	ERROR: boxSize_Y should not be defined here
#endif
#ifdef boxSize_Z
	ERROR: boxSize_Z should not be defined here
#endif

#ifdef LONG_X
	#define NGB_PERIODIC_X_GX(x) (xtmp=(x),(xtmp>g_const_parameters_hydro.boxHalf_X)?(xtmp-g_const_parameters_hydro.boxSize_X):((xtmp<-g_const_parameters_hydro.boxHalf_X)?(xtmp+g_const_parameters_hydro.boxSize_X):xtmp))
#else
	#define NGB_PERIODIC_X_GX(x) (xtmp=(x),(xtmp>g_const_parameters_hydro.boxHalf)?(xtmp-g_const_parameters_hydro.boxSize):((xtmp<-g_const_parameters_hydro.boxHalf)?(xtmp+g_const_parameters_hydro.boxSize):xtmp))
#endif
#ifdef LONG_Y
	#define NGB_PERIODIC_Y_GX(x) (xtmp=(x),(xtmp>g_const_parameters_hydro.boxHalf_Y)?(xtmp-g_const_parameters_hydro.boxSize_Y):((xtmp<-g_const_parameters_hydro.boxHalf_Y)?(xtmp+g_const_parameters_hydro.boxSize_Y):xtmp))
#else
	#define NGB_PERIODIC_Y_GX(x) (xtmp=(x),(xtmp>g_const_parameters_hydro.boxHalf)?(xtmp-g_const_parameters_hydro.boxSize):((xtmp<-g_const_parameters_hydro.boxHalf)?(xtmp+g_const_parameters_hydro.boxSize):xtmp))
#endif
#ifdef LONG_Z
	#define NGB_PERIODIC_Z_GX(x) (xtmp=(x),(xtmp>g_const_parameters_hydro.boxHalf_Z)?(xtmp-g_const_parameters_hydro.boxSize_Z):((xtmp<-g_const_parameters_hydro.boxHalf_Z)?(xtmp+g_const_parameters_hydro.boxSize_Z):xtmp))
#else
	#define NGB_PERIODIC_Z_GX(x) (xtmp=(x),(xtmp>g_const_parameters_hydro.boxHalf)?(xtmp-g_const_parameters_hydro.boxSize):((xtmp<-g_const_parameters_hydro.boxHalf)?(xtmp+g_const_parameters_hydro.boxSize):xtmp))
#endif

/*
__device__
FLOAT_INTERNAL_GX PeriodicDist1D_signed(const FLOAT_INTERNAL_GX& p0,const FLOAT_INTERNAL_GX& p1,const FLOAT_INTERNAL_GX& boxsize)
{
	const FLOAT_INTERNAL_GX d=fmin(fabs(p0-p1),boxsize-fabs(p0-p1));
	const FLOAT_INTERNAL_GX s=fabs(p0-p1)/(p0-p1);
	return d*s;
}
*/

#ifdef NEW_PERIODIC_FUN_GX
	#if CUDA_DEBUG_GX > 1
		__device__
		FLOAT_GX PeriodicDist1D(const FLOAT_GX& p0,const FLOAT_GX& p1,const FLOAT_GX& boxsize)
		{
			const FLOAT_GX d=fmin(fabs(p0-p1),boxsize-fabs(p0-p1));

			#if CUDA_DEBUG_GX > 0
				FLOAT_GX xtmp;
				xtmp=NGB_PERIODIC_X_GX(p0-p1);
				ASSERT_DEVICE_GX( isFloatinVicinity_device(d,xtmp,1E-9) || isFloatinVicinity_device(d,-xtmp,1E-9) );

				#if !(defined LONG_X || defined LONG_Y || defined LONG_Z)
					ASSERT_DEVICE_GX( d<=g_const_parameters_hydro.boxHalf );
				#endif
				ASSERT_DEVICE_GX( d<=boxsize/2 );
			#endif
			return d;
		}

		#ifdef LONG_X
			#define PeriodicDist1D_X(p0,p1) PeriodicDist1D(p0,p1,g_const_parameters_hydro.boxSize_X,devmsg)
		#else
			#define PeriodicDist1D_X(p0,p1) PeriodicDist1D(p0,p1,g_const_parameters_hydro.boxSize,devmsg)
		#endif
		#ifdef LONG_Y
			#define PeriodicDist1D_Y(p0,p1) PeriodicDist1D(p0,p1,g_const_parameters_hydro.boxSize_Y,devmsg)
		#else
			#define PeriodicDist1D_Y(p0,p1) PeriodicDist1D(p0,p1,g_const_parameters_hydro.boxSize,devmsg)
		#endif
		#ifdef LONG_Z
			#define PeriodicDist1D_Z(p0,p1) PeriodicDist1D(p0,p1,g_const_parameters_hydro.boxSize_Z,devmsg)
		#else
			#define PeriodicDist1D_Z(p0,p1) PeriodicDist1D(p0,p1,g_const_parameters_hydro.boxSize,devmsg)
		#endif
	#else
		#ifdef LONG_X
			#define PeriodicDist1D_X(p0,p1) fmin(fabs(p0-p1),g_const_parameters_hydro.boxSize_X-fabs(p0-p1))
		#else
			#define PeriodicDist1D_X(p0,p1) fmin(fabs(p0-p1),g_const_parameters_hydro.boxSize-fabs(p0-p1))
		#endif
		#ifdef LONG_Y
			#define PeriodicDist1D_Y(p0,p1) fmin(fabs(p0-p1),g_const_parameters_hydro.boxSize_Y-fabs(p0-p1))
		#else
			#define PeriodicDist1D_Y(p0,p1) fmin(fabs(p0-p1),g_const_parameters_hydro.boxSize-fabs(p0-p1))
		#endif
		#ifdef LONG_Z
			#define PeriodicDist1D_Z(p0,p1) fmin(fabs(p0-p1),g_const_parameters_hydro.boxSize_Z-fabs(p0-p1))
		#else
			#define PeriodicDist1D_Z(p0,p1) fmin(fabs(p0-p1),g_const_parameters_hydro.boxSize-fabs(p0-p1))
		#endif
	#endif

#endif

__device__
FLOAT_INTERNAL_GX dmin_gx(const FLOAT_INTERNAL_GX& x,const FLOAT_INTERNAL_GX& y)
{
	//if(x < y)
	//	return x;
	//else
	//	return y;
	return fmin(x,y);
}

__device__
int imax_gx(const int& x,const int& y)
{
	//if(x > y)
	//	return x;
	//else
	//	return y;
	return max(x,y);
}

__device__
struct sph_particle_data_gx GetSphParticle(int n)
{
	ASSERT_DEVICE_GX( g_const_state.mode==0);
	ASSERT_DEVICE_GX( g_const_state.SphP!=NULL && n<g_const_state.sz_SphP );
	ASSERT_DEVICE_GX( g_const_state.sz_SphP==g_const_state.N_gas );

	struct sph_particle_data_gx d;

 	#ifdef CUDA_GX_USE_TEXTURES
 		n *= 3;
 		int4* i4=(int4*)(&d);

 		*i4=tex1Dfetch(tex_SphP,n);
 		++i4;
 		++n;
 		*i4=tex1Dfetch(tex_SphP,n);
 		++i4;
 		++n;
 		*i4=tex1Dfetch(tex_SphP,n);
 	#else
		d=g_const_state.SphP[n];
	#endif

	#ifdef __DEVICE_EMULATION__
		#if CUDART_VERSION < 3 // NOTE: error in nvcc 3.0, will not pass parameters to GCC
			ASSERT_DEVICE_GX(isSphParticleDataOK(d,__FILE__,__LINE__));
		#endif
	#endif

	return d;
}

__device__
int isGasType(const int& n)
{
	ASSERT_DEVICE_GX( g_const_state.mode==0);
	return GetParticleType(n
		#if CUDA_DEBUG_GX > 1
			,devmsg
		#endif
	)==0;
}

__device__
int GetTimeStepDiff(const int& n)
{
	ASSERT_DEVICE_GX( g_const_state.mode==0);
	struct etc_gx d=GetEtc(n
		#if CUDA_DEBUG_GX > 1
			,devmsg
		#endif
	);

	return d.Ti_endstep-d.Ti_begstep;
}

__device__
FLOAT_GX GetSphHsml(const int& n)
{
	ASSERT_DEVICE_GX( g_const_state.mode==0);
	return GetSphParticle(n
		#if CUDA_DEBUG_GX > 1
			,devmsg
		#endif
	).Hsml;
}

__device__
FLOAT_GX GetExtNodeHmax(int n)
{
	ASSERT_DEVICE_GX( g_const_state.mode==0);
	ASSERT_DEVICE_GX( g_const_state.extNodes!=NULL );
	ASSERT_DEVICE_GX( n>=g_const_state.MaxPart );
	ASSERT_DEVICE_GX( n-g_const_state.MaxPart<g_const_state.sz_extNodes_base );
	ASSERT_DEVICE_GX( isFloatOK_device(g_const_state.extNodes[n].hmax,__FILE__,__LINE__) );

	#ifdef CUDA_GX_USE_TEXTURES
		ASSERT_DEVICE_GX( sizeof(struct extNODE_gx)==4*4 ) ;
		ASSERT_DEVICE_GX( tex1Dfetch(tex_extNodes_base,4*(n-g_const_state.MaxPart))==g_const_state.extNodes[n].hmax );

 		return tex1Dfetch(tex_extNodes_base,4*(n-g_const_state.MaxPart));
 	#else
		return g_const_state.extNodes[n].hmax;
	#endif
}

#ifndef CUDA_GX_SHARED_NGBLIST
	__device__
	int GetSizeNgblist(
		#if CUDA_DEBUG_GX > 1
			struct DevMsg& devmsg
		#endif
		)
	{
		return g_const_parameters_hydro.szngb;
	}

	__device__
	int AddToNgblist(const int& p,const int& n
		#if CUDA_DEBUG_GX > 1
			,struct DevMsg& devmsg
		#endif
		)
	{
		ASSERT_DEVICE_GX( g_const_state.sz_Ngblist/GetThreads()==g_const_parameters_hydro.szngb );
		ASSERT_DEVICE_GX( n<g_const_parameters_hydro.szngb );
		// if (!( n<szngb )) MESSAGE("g_const_state.sz_Ngblist=%d, GetThreads()=%d, szngb=%d, n=%d, szngb*GetTid()+n=%d",g_const_state.sz_Ngblist,GetThreads(),szngb,n,szngb*GetTid()+n);

		const int m=g_const_parameters_hydro.szngb*GetTid()+n;
		ASSERT_DEVICE_GX(g_const_state.Ngblist!=NULL && m<g_const_state.sz_Ngblist);

		g_const_state.Ngblist[m]=p;

		return n+1;
	}

	__device__
	int GetNgblist(const int n
		#if CUDA_DEBUG_GX > 1
			,struct DevMsg& devmsg
		#endif
		)
	{
		ASSERT_DEVICE_GX( g_const_state.sz_Ngblist/GetThreads()==g_const_parameters_hydro.szngb );
		ASSERT_DEVICE_GX( n<g_const_parameters_hydro.szngb );
		const int m=g_const_parameters_hydro.szngb*GetTid()+n;
		ASSERT_DEVICE_GX(g_const_state.Ngblist!=NULL && m<g_const_state.sz_Ngblist);

		return g_const_state.Ngblist[m];
	}
#endif

__device__
struct hydrodata_in_gx GetHydroDataIn(const int& n
	#if CUDA_DEBUG_GX > 1
		,struct DevMsg& devmsg
	#endif
	)
{
	ASSERT_DEVICE_GX(g_const_state.mode==1);
	ASSERT_DEVICE_GX(g_const_state.sz_hydrodata_in);

	const struct hydrodata_in_gx d=g_const_state.hydrodata_in[n];
	ASSERT_DEVICE_GX(d.pad==-n);

	return d;
}

// This routine finds all neighbours `j' that can interact with the
//  particle `i' in the communication buffer.
//
//  Note that an interaction can take place if
//  \f$ r_{ij} < h_i \f$  OR if  \f$ r_{ij} < h_j \f$.
//
//  In the range-search this is taken into account, i.e. it is guaranteed that
//  all particles are found that fulfil this condition, including the (more
//  difficult) second part of it. For this purpose, each node knows the
//  maximum h occuring among the particles it represents.

__device__
int ngb_treefind_pairs_gx(const FLOAT*const searchcenter,const FLOAT& hsml,int& no,const int& target
	#if CUDA_DEBUG_GX > 1
		,struct DevMsg& devmsg
	#endif
	)
{
	#ifndef PERIODIC
		FLOAT searchmin[3], searchmax[3];

		// cube-box window
		searchmin[0] = searchcenter[0] - hsml;
		searchmax[0] = searchcenter[0] + hsml;
		searchmin[1] = searchcenter[1] - hsml;
		searchmax[1] = searchcenter[1] + hsml;
		searchmin[2] = searchcenter[2] - hsml;
		searchmax[2] = searchcenter[2] + hsml;
	#endif

	#ifndef CUDA_GX_SHARED_NGBLIST
		int numngb = 0;
	#endif

	while(no >= 0)
	{
		if(no < g_const_parameters.MaxPart) // single particle
		{
			const int p = no;

			no = GetNextnode(no
				#if CUDA_DEBUG_GX > 1
					,devmsg
				#endif
			);

			if (!isGasType(p
				#if CUDA_DEBUG_GX > 1
					,devmsg
				#endif
				))  // if(P[p].Type > 0)
				continue;

			const FLOAT_INTERNAL_GX hdiff = fmax(GetSphHsml(p
				#if CUDA_DEBUG_GX > 1
					,devmsg
				#endif
				) - hsml,0);

			//if(hdiff < 0) hdiff = 0;

			{
				const struct particle_data_gx p_local=GetParticle(p
						#if CUDA_DEBUG_GX > 1
							,devmsg
						#endif
					);

				#ifdef PERIODIC
					#ifdef NEW_PERIODIC_FUN_GX
						if (PeriodicDist1D_X(p_local.Pos[0],searchcenter[0]) > (hsml + hdiff))
							continue;
						if (PeriodicDist1D_Y(p_local.Pos[1],searchcenter[1]) > (hsml + hdiff))
							continue;
						if (PeriodicDist1D_Z(p_local.Pos[2],searchcenter[2]) > (hsml + hdiff))
							continue;
					#else
						FLOAT_INTERNAL_GX xtmp;

						if(NGB_PERIODIC_X_GX(p_local.Pos[0] - searchcenter[0]) < (-hsml - hdiff))
							continue;
						if(NGB_PERIODIC_X_GX(p_local.Pos[0] - searchcenter[0]) > (hsml + hdiff))
							continue;
						if(NGB_PERIODIC_Y_GX(p_local.Pos[1] - searchcenter[1]) < (-hsml - hdiff))
							continue;
						if(NGB_PERIODIC_Y_GX(p_local.Pos[1] - searchcenter[1]) > (hsml + hdiff))
							continue;
						if(NGB_PERIODIC_Z_GX(p_local.Pos[2] - searchcenter[2]) < (-hsml - hdiff))
							continue;
						if(NGB_PERIODIC_Z_GX(p_local.Pos[2] - searchcenter[2]) > (hsml + hdiff))
							continue;
					#endif
				#else
					if(p_local.Pos[0] < (searchmin[0] - hdiff))
						continue;
					if(p_local.Pos[0] > (searchmax[0] + hdiff))
						continue;
					if(p_local.Pos[1] < (searchmin[1] - hdiff))
						continue;
					if(p_local.Pos[1] > (searchmax[1] + hdiff))
						continue;
					if(p_local.Pos[2] < (searchmin[2] - hdiff))
						continue;
					if(p_local.Pos[2] > (searchmax[2] + hdiff))
						continue;
				#endif
			}
			#ifdef CUDA_GX_SHARED_NGBLIST
				return p;
			#else
				numngb=AddToNgblist(p,numngb
						#if CUDA_DEBUG_GX > 1
							,devmsg
						#endif
					); // Ngblist[numngb++] = p;

				if(numngb == GetSizeNgblist(
						#if CUDA_DEBUG_GX > 1
							devmsg
						#endif
					))
				{
					#ifdef __DEVICE_EMULATION__
						// MESSAGE("ThisTask=%d: Need to do a second neighbour loop in hydro-force for (%g|%g|%g) hsml=%g no=%d\n",g_const_state.ThisTask, searchcenter[0], searchcenter[1], searchcenter[2], hsml, no);
					#endif
					return numngb;
				}
			#endif
		}
		else
		{
			if(no >= g_const_parameters.MaxPart + g_const_parameters.MaxNodes) // pseudo particle
			{
				//Exportflag[DomainTask[no - (All.MaxPart + MaxNodes)]] = 1;
				SetExportflag_gx(no,target
					#if CUDA_DEBUG_GX > 1
						,devmsg
					#endif
				);
				no = GetNextnode(no - g_const_parameters.MaxNodes
						#if CUDA_DEBUG_GX > 1
							,devmsg
						#endif
					);
				continue;
			}
/*
			{
				const struct NODE_gx node_local=GetNode(no //&Nodes[no];
					#if CUDA_DEBUG_GX > 1
						,devmsg
					#endif
				);

				const FLOAT_INTERNAL_GX hdiff = fmax(GetExtNodeHmax(no
					#if CUDA_DEBUG_GX > 1
						,devmsg
					#endif
					)-hsml,0); //Extnodes[no].hmax - hsml;
				//if(hdiff < 0) hdiff = 0;

				no = node_local.u.d.sibling; // in case the node can be discarded

				#ifdef PERIODIC
					#ifdef NEW_PERIODIC_FUN_GX
						if ((PeriodicDist1D_X(node_local.center[0],searchcenter[0]) - 0.5f * node_local.len) > (hsml + hdiff))
							continue;
						if ((PeriodicDist1D_Y(node_local.center[1],searchcenter[1]) - 0.5f * node_local.len) > (hsml + hdiff))
							continue;
						if ((PeriodicDist1D_Z(node_local.center[2],searchcenter[2]) - 0.5f * node_local.len) > (hsml + hdiff))
							continue;
					#else
						FLOAT_INTERNAL_GX xtmp;

						if((NGB_PERIODIC_X_GX(node_local.center[0] - searchcenter[0]) + 0.5 * node_local.len) < (-hsml - hdiff))
							continue;
						if((NGB_PERIODIC_X_GX(node_local.center[0] - searchcenter[0]) - 0.5 * node_local.len) > (hsml + hdiff))
							continue;
						if((NGB_PERIODIC_Y_GX(node_local.center[1] - searchcenter[1]) + 0.5 * node_local.len) < (-hsml - hdiff))
							continue;
						if((NGB_PERIODIC_Y_GX(node_local.center[1] - searchcenter[1]) - 0.5 * node_local.len) > (hsml + hdiff))
							continue;
						if((NGB_PERIODIC_Z_GX(node_local.center[2] - searchcenter[2]) + 0.5 * node_local.len) < (-hsml - hdiff))
							continue;
						if((NGB_PERIODIC_Z_GX(node_local.center[2] - searchcenter[2]) - 0.5 * node_local.len) > (hsml + hdiff))
							continue;
					#endif
				#else
					if((node_local.center[0] + 0.5 * node_local.len) < (searchmin[0] - hdiff))
						continue;
					if((node_local.center[0] - 0.5 * node_local.len) > (searchmax[0] + hdiff))
						continue;
					if((node_local.center[1] + 0.5 * node_local.len) < (searchmin[1] - hdiff))
						continue;
					if((node_local.center[1] - 0.5 * node_local.len) > (searchmax[1] + hdiff))
						continue;
					if((node_local.center[2] + 0.5 * node_local.len) < (searchmin[2] - hdiff))
						continue;
					if((node_local.center[2] - 0.5 * node_local.len) > (searchmax[2] + hdiff))
						continue;
				#endif
				no = node_local.u.d.nextnode; // ok, we need to open the node
			}
*/
			{

				FLOAT_INTERNAL_GX node_local_center[4];

				ASSERT_DEVICE_GX( no>=g_const_state.MaxPart );
				ASSERT_DEVICE_GX( no-g_const_state.MaxPart<g_const_state.sz_Nodes_base );

				#ifdef CUDA_GX_USE_TEXTURES
					ASSERT_DEVICE_GX( no-g_const_state.MaxPart<g_const_state.sz_Nodes_base );
					ASSERT_DEVICE_GX( sizeof(struct NODE_gx)==12*4 ) ;

					*((int4*)(&node_local_center))=tex1Dfetch(tex_Nodes_base,3*(no-g_const_state.MaxPart));
				#else

					node_local_center[0]=g_const_state.Nodes[no].len;
					node_local_center[1]=g_const_state.Nodes[no].center[0];
					node_local_center[2]=g_const_state.Nodes[no].center[1];
					node_local_center[3]=g_const_state.Nodes[no].center[2];
				#endif

				const FLOAT_INTERNAL_GX hdiff = fmax(GetExtNodeHmax(no
					#if CUDA_DEBUG_GX > 1
						,devmsg
					#endif
					)-hsml,0); //Extnodes[no].hmax - hsml;
				//if(hdiff < 0) hdiff = 0;

				#ifdef PERIODIC
					#ifdef NEW_PERIODIC_FUN_GX
						if ((PeriodicDist1D_X(node_local_center[1],searchcenter[0]) - 0.5f * node_local_center[0]) > (hsml + hdiff))
							{
								no = g_const_state.Nodes[no].u.d.sibling; // in case the node can be discarded
								continue;
							}
						if ((PeriodicDist1D_Y(node_local_center[2],searchcenter[1]) - 0.5f * node_local_center[0]) > (hsml + hdiff))
							{
								no = g_const_state.Nodes[no].u.d.sibling; // in case the node can be discarded
								continue;
							}
						if ((PeriodicDist1D_Z(node_local_center[3],searchcenter[2]) - 0.5f * node_local_center[0]) > (hsml + hdiff))
							{
								no = g_const_state.Nodes[no].u.d.sibling; // in case the node can be discarded
								continue;
							}
					#else
						FLOAT_INTERNAL_GX xtmp;

						if((NGB_PERIODIC_X_GX(node_local_center[1] - searchcenter[0]) + 0.5 * node_local_center[0]) < (-hsml - hdiff))
							{
								no = g_const_state.Nodes[no].u.d.sibling; // in case the node can be discarded
								continue;
							}
						if((NGB_PERIODIC_X_GX(node_local_center[1] - searchcenter[0]) - 0.5 * node_local_center[0]) > (hsml + hdiff))
							{
								no = g_const_state.Nodes[no].u.d.sibling; // in case the node can be discarded
								continue;
							}
						if((NGB_PERIODIC_Y_GX(node_local_center[2] - searchcenter[1]) + 0.5 * node_local_center[0]) < (-hsml - hdiff))
							{
								no = g_const_state.Nodes[no].u.d.sibling; // in case the node can be discarded
								continue;
							}
						if((NGB_PERIODIC_Y_GX(node_local_center[2] - searchcenter[1]) - 0.5 * node_local_center[0]) > (hsml + hdiff))
							{
								no = g_const_state.Nodes[no].u.d.sibling; // in case the node can be discarded
								continue;
							}
						if((NGB_PERIODIC_Z_GX(node_local_center[3] - searchcenter[2]) + 0.5 * node_local_center[0]) < (-hsml - hdiff))
							{
								no = g_const_state.Nodes[no].u.d.sibling; // in case the node can be discarded
								continue;
							}
						if((NGB_PERIODIC_Z_GX(node_local_center[3] - searchcenter[2]) - 0.5 * node_local_center[0]) > (hsml + hdiff))
							{
								no = g_const_state.Nodes[no].u.d.sibling; // in case the node can be discarded
								continue;
							}
					#endif
				#else
					if((node_local_center[1] + 0.5 * node_local_center[0]) < (searchmin[0] - hdiff))
						{
							no = g_const_state.Nodes[no].u.d.sibling; // in case the node can be discarded
							continue;
						}
					if((node_local_center[1] - 0.5 * node_local_center[0]) > (searchmax[0] + hdiff))
						{
							no = g_const_state.Nodes[no].u.d.sibling; // in case the node can be discarded
							continue;
						}
					if((node_local_center[2] + 0.5 * node_local_center[0]) < (searchmin[1] - hdiff))
						{
							no = g_const_state.Nodes[no].u.d.sibling; // in case the node can be discarded
							continue;
						}
					if((node_local_center[2] - 0.5 * node_local_center[0]) > (searchmax[1] + hdiff))
						{
							no = g_const_state.Nodes[no].u.d.sibling; // in case the node can be discarded
							continue;
						}
					if((node_local_center[3] + 0.5 * node_local_center[0]) < (searchmin[2] - hdiff))
						{
							no = g_const_state.Nodes[no].u.d.sibling; // in case the node can be discarded
							continue;
						}
					if((node_local_center[3] - 0.5 * node_local_center[0]) > (searchmax[2] + hdiff))
						{
							no = g_const_state.Nodes[no].u.d.sibling; // in case the node can be discarded
							continue;
						}
				#endif
				no = g_const_state.Nodes[no].u.d.nextnode; // ok, we need to open the node
			}
		}
	}

	no = -1;
	#ifdef CUDA_GX_SHARED_NGBLIST
		return -1;
	#else
		return numngb;
	#endif
}


// This function is the 'core' of the SPH force computation. A target
//  particle is specified which may either be local, or reside in the
//  communication buffer.

__device__
void hydro_evaluate_gx(const int& target
	#if CUDA_DEBUG_GX > 1
		,struct DevMsg& devmsg
	#endif
	)
{
	ASSERT_DEVICE_GX( g_const_state.mode==0 || g_const_state.mode==1 );
	ASSERT_DEVICE_GX( target>=0 && target<g_const_parameters_hydro.N_gas );
	ASSERT_DEVICE_GX( target<g_const_state.sz_SphP );

	#ifdef __DEVICE_EMULATION__
		#if CUDART_VERSION < 3 // NOTE: error in nvcc 3.0, will not pass parameters to GCC
			ValidateState(g_const_state,g_const_parameters.MaxNodes,1,__FILE__,__LINE__);
			ValidateParameters(g_const_parameters,__FILE__,__LINE__);
			ValidateParameters_hydra(g_const_parameters_hydro,__FILE__,__LINE__);
		#endif
	#endif

	int timestep, startnode;
	FLOAT_INTERNAL_GX pos[3], vel[3];
	FLOAT_INTERNAL_GX mass, h_i, dhsmlDensityFactor, rho, pressure, f1;
	FLOAT_INTERNAL_GX acc[3], dtEntropy, maxSignalVel;
	FLOAT_INTERNAL_GX dx, dy, dz, dvx, dvy, dvz;
	FLOAT_INTERNAL_GX h_i2, hinv, hinv4;
	FLOAT_INTERNAL_GX p_over_rho2_i, p_over_rho2_j, soundspeed_i, soundspeed_j;
	FLOAT_INTERNAL_GX hfc, dwk_i, vdotr, vdotr2, visc;
	FLOAT_INTERNAL_GX h_j, dwk_j, u, hfc_visc;

	#ifndef NOVISCOSITYLIMITER
		FLOAT_INTERNAL_GX dt;
	#endif

	if(g_const_state.mode == 0)
	{

		{
			const struct etc_gx etc_local=GetEtc(target
				#if CUDA_DEBUG_GX > 1
					,devmsg
				#endif
			);

			if (etc_local.Ti_endstep!=g_const_parameters.Ti_Current) return;
			timestep = etc_local.Ti_endstep-etc_local.Ti_begstep; // P[target].Ti_endstep - P[target].Ti_begstep;
		}

		{
			const struct particle_data_gx p_local=GetParticle(target
				#if CUDA_DEBUG_GX > 1
					,devmsg
				#endif
			);

			pos[0] = p_local.Pos[0];
			pos[1] = p_local.Pos[1];
			pos[2] = p_local.Pos[2];
			mass = p_local.Mass;
		}

		{
			const struct sph_particle_data_gx sph_local=GetSphParticle(target
				#if CUDA_DEBUG_GX > 1
					,devmsg
				#endif
			);

			vel[0] = sph_local.VelPred[0];
			vel[1] = sph_local.VelPred[1];
			vel[2] = sph_local.VelPred[2];
			h_i = sph_local.Hsml;
			dhsmlDensityFactor = sph_local.DhsmlDensityFactor;
			rho = sph_local.Density;
			pressure = sph_local.Pressure;
			soundspeed_i = sqrt(GAMMA * pressure / rho);
			f1 = fabs(sph_local.DivVel) /
			(fabs(sph_local.DivVel) + sph_local.CurlVel +
			0.0001 * soundspeed_i / sph_local.Hsml / g_const_parameters_hydro.fac_mu);
		}
	}
	else
	{
		const struct hydrodata_in_gx datain=GetHydroDataIn(target
								#if CUDA_DEBUG_GX > 1
									,devmsg
								#endif
								);
		pos[0] = datain.Pos[0];
		pos[1] = datain.Pos[1];
		pos[2] = datain.Pos[2];
		vel[0] = datain.Vel[0];
		vel[1] = datain.Vel[1];
		vel[2] = datain.Vel[2];
		h_i    = datain.Hsml;
		mass   = datain.Mass;
		dhsmlDensityFactor = datain.DhsmlDensityFactor;
		rho          =  datain.Density;
		pressure     = datain.Pressure;
		timestep     = datain.Timestep;
		soundspeed_i = sqrt(GAMMA * pressure / rho);
		f1           = datain.F1;
	}

	// initialize variables before SPH loop is started
	acc[0] = acc[1] = acc[2] = dtEntropy = 0;
	maxSignalVel = 0;

	p_over_rho2_i = pressure / (rho * rho) * dhsmlDensityFactor;
	h_i2 = h_i * h_i;

	// Now start the actual SPH computation for this particle
	startnode = g_const_state.MaxPart;
	do
	{
		const int numngb = ngb_treefind_pairs_gx(&pos[0],h_i,startnode,target
			#if CUDA_DEBUG_GX > 1
				,devmsg
			#endif
		);

		#ifdef CUDA_GX_SHARED_NGBLIST
			if (numngb>=0)
		#else
			int n;
			for(n = 0; n < numngb; n++)
		#endif
		{
			#ifdef CUDA_GX_SHARED_NGBLIST
				const int j=numngb;
			#else
				const int j= GetNgblist(n
					#if CUDA_DEBUG_GX > 1
						,devmsg
					#endif
					); // j = Ngblist[n];
			#endif

			#if CUDA_DEBUG_GX > 0
				const int org_j=j;
			#endif

			FLOAT_INTERNAL_GX p_local_Mass;
			{
				const struct particle_data_gx p_local=GetParticle(j
					#if CUDA_DEBUG_GX > 1
						,devmsg
					#endif
				);

				dx = pos[0] - p_local.Pos[0];
				dy = pos[1] - p_local.Pos[1];
				dz = pos[2] - p_local.Pos[2];

				p_local_Mass=p_local.Mass;
			}


			#ifdef PERIODIC //  find the closest image in the given box size
				#ifdef LONG_X
					if(dx > g_const_parameters_hydro.boxHalf_X)
						dx -=  g_const_parameters_hydro.boxSize_X;
					if(-dx > g_const_parameters_hydro.boxHalf_X)
						dx += g_const_parameters_hydro.boxSize_X;
				#else
					if(dx > g_const_parameters_hydro.boxHalf)
						dx -=  g_const_parameters_hydro.boxSize;
					if(-dx > g_const_parameters_hydro.boxHalf)
						dx += g_const_parameters_hydro.boxSize;
				#endif
				#ifdef LONG_Y
					if(dy > g_const_parameters_hydro.boxHalf_Y)
						dy -= g_const_parameters_hydro.boxSize_Y;
					if(-dy > g_const_parameters_hydro.boxHalf_Y)
						dy += g_const_parameters_hydro.boxSize_Y;
				#else
					if(dy > g_const_parameters_hydro.boxHalf)
						dy -= g_const_parameters_hydro.boxSize;
					if(-dy > g_const_parameters_hydro.boxHalf)
						dy += g_const_parameters_hydro.boxSize;
				#endif
				#ifdef LONG_Z
					if(dz > g_const_parameters_hydro.boxHalf_Z)
						dz -= g_const_parameters_hydro.boxSize_Z;
					if(-dz > g_const_parameters_hydro.boxHalf_Z)
						dz += g_const_parameters_hydro.boxSize_Z;
				#else
					if(dz > g_const_parameters_hydro.boxHalf)
						dz -= g_const_parameters_hydro.boxSize;
					if(-dz > g_const_parameters_hydro.boxHalf)
						dz += g_const_parameters_hydro.boxSize;
				#endif
			#endif
			const FLOAT_INTERNAL_GX r2 = dx * dx + dy * dy + dz * dz;

			h_j = GetSphHsml(j
				#if CUDA_DEBUG_GX > 1
					,devmsg
				#endif
			);
			if(r2 < h_i2 || r2 < h_j * h_j)
			{
				const FLOAT_INTERNAL_GX r = sqrt(r2);
				if(r > 0)
				{
					const struct sph_particle_data_gx sph_local=GetSphParticle(j
							#if CUDA_DEBUG_GX > 1
								,devmsg
							#endif
						);

					p_over_rho2_j = sph_local.Pressure / (sph_local.Density * sph_local.Density);
					soundspeed_j = sqrt(GAMMA * p_over_rho2_j * sph_local.Density);
					dvx = vel[0] - sph_local.VelPred[0];
					dvy = vel[1] - sph_local.VelPred[1];
					dvz = vel[2] - sph_local.VelPred[2];
					vdotr = dx * dvx + dy * dvy + dz * dvz;

					if(g_const_parameters.ComovingIntegrationOn)
						vdotr2 = vdotr + g_const_parameters_hydro.hubble_a2 * r2;
					else
						vdotr2 = vdotr;

					if(r2 < h_i2)
					{
						hinv = 1.0 / h_i;
						#ifndef  TWODIMS
						hinv4 = hinv * hinv * hinv * hinv;
						#else
						hinv4 = hinv * hinv * hinv / boxSize_Z;
						#endif
						u = r * hinv;
						if(u < 0.5)
							dwk_i = hinv4 * u * (KERNEL_COEFF_3 * u - KERNEL_COEFF_4);
						else
							dwk_i = hinv4 * KERNEL_COEFF_6 * (1.0 - u) * (1.0 - u);
					}
					else
					{
						dwk_i = 0;
					}

					if(r2 < h_j * h_j)
					{
						hinv = 1.0 / h_j;
						#ifndef  TWODIMS
						hinv4 = hinv * hinv * hinv * hinv;
						#else
						hinv4 = hinv * hinv * hinv / boxSize_Z;
						#endif
						u = r * hinv;
						if(u < 0.5)
							dwk_j = hinv4 * u * (KERNEL_COEFF_3 * u - KERNEL_COEFF_4);
						else
							dwk_j = hinv4 * KERNEL_COEFF_6 * (1.0 - u) * (1.0 - u);
					}
					else
					{
						dwk_j = 0;
					}

					//if(soundspeed_i + soundspeed_j > maxSignalVel)
					//	maxSignalVel = soundspeed_i + soundspeed_j;
					maxSignalVel = fmax(maxSignalVel, soundspeed_i + soundspeed_j);

					if(vdotr2 < 0) // ... artificial viscosity
					{
						const FLOAT_INTERNAL_GX mu_ij = g_const_parameters_hydro.fac_mu * vdotr2 / r;	// note: this is negative!
						const FLOAT_INTERNAL_GX vsig = soundspeed_i + soundspeed_j - 3 * mu_ij;

						//if(vsig > maxSignalVel)
						//	maxSignalVel = vsig;
						maxSignalVel =  fmax(maxSignalVel,vsig);

						const FLOAT_INTERNAL_GX rho_ij = 0.5 * (rho + sph_local.Density);
						const FLOAT_INTERNAL_GX f2=fabs(sph_local.DivVel) / (fabs(sph_local.DivVel) + sph_local.CurlVel + 0.0001 * soundspeed_j / g_const_parameters_hydro.fac_mu / sph_local.Hsml);

						visc = 0.25 * g_const_parameters_hydro.ArtBulkViscConst * vsig * (-mu_ij) / rho_ij * (f1 + f2);

						// .... end artificial viscosity evaluation
						#ifndef NOVISCOSITYLIMITER
							// make sure that viscous acceleration is not too large
							//dt = imax_gx(timestep, (p_local.Ti_endstep - p_local.Ti_begstep)) * All.Timebase_interval;
							dt = imax_gx(timestep, GetTimeStepDiff(j
								#if CUDA_DEBUG_GX > 1
									,devmsg
								#endif
								)) * g_const_parameters.Timebase_interval;
							if(dt > 0 && (dwk_i + dwk_j) < 0)
							{
								visc = dmin_gx(visc, 0.5 * g_const_parameters_hydro.fac_vsic_fix * vdotr2 /
								(0.5 * (mass + p_local_Mass) * (dwk_i + dwk_j) * r * dt));
							}
						#endif
					}
					else
						visc = 0;

					p_over_rho2_j *= sph_local.DhsmlDensityFactor;

					hfc_visc = 0.5 * p_local_Mass * visc * (dwk_i + dwk_j) / r;

					hfc = hfc_visc + p_local_Mass * (p_over_rho2_i * dwk_i + p_over_rho2_j * dwk_j) / r;

					acc[0] -= hfc * dx;
					acc[1] -= hfc * dy;
					acc[2] -= hfc * dz;
					dtEntropy += 0.5 * hfc_visc * vdotr2;
				}
			}
			ASSERT_DEVICE_GX( j==org_j );
		}
	}
	while(startnode >= 0);

	{
		// Now collect the result at the right place
		struct result_hydro_gx result;

		result.Acc[0] = acc[0];
		result.Acc[1] = acc[1];
		result.Acc[2] = acc[2];

		result.DtEntropy = dtEntropy;
		result.MaxSignalVel = maxSignalVel;

		ASSERT_DEVICE_GX( target<g_const_state.sz_result_hydro );

		#if CUDA_DEBUG_GX > 0
			ASSERT_DEVICE_GX( g_const_state.result_hydro[target].pad0==-target );
			ASSERT_DEVICE_GX( g_const_state.result_hydro[target].pad1==target );
			result.realtarget=target;
			result.pad0=target;
			result.pad1=g_const_state.sz_result_hydro-target;
		#endif

		g_const_state.result_hydro[target]=result;
	}

// 	if(g_const_state.mode == 0)
// 	{
// 		for(k = 0; k < 3; k++)
// 			SphP[target].HydroAccel[k] = acc[k];
// 		SphP[target].DtEntropy = dtEntropy;
// 		SphP[target].MaxSignalVel = maxSignalVel;
//
// 		static int NN=0;
// 	}
// 	else
// 	{
// 		ERROR("mode=1 not supported yet");
// 		for(k = 0; k < 3; k++)
// 			HydroDataResult[target].Acc[k] = acc[k];
// 		HydroDataResult[target].DtEntropy = dtEntropy;
// 		HydroDataResult[target].MaxSignalVel = maxSignalVel;
// 	}

}

__global__
void hydro_evaluate_shortrange_cuda_gx_kernel()
{
	#if CUDA_DEBUG_GX > 1
		struct DevMsg devmsg=ResetMsg(g_const_state.debug_msg,GetTid(),g_const_state.debug_sz_msg);
		#if CUDA_DEBUG_GX > 3
			PrintDevMsg(&devmsg,"hydra kernel",1);
			//Debug_PrintVars(tid,N_gas,devmsg);
		#endif
	#endif

	KernelPreinit(1
		#if CUDA_DEBUG_GX > 1
			,devmsg
		#endif
	);
	
	dim3 tid;
	tid.x = threadIdx.x+blockIdx.x*blockDim.x;
	tid.y = g_const_state.Np;
	tid.z = blockDim.x*gridDim.x;

	#if CUDA_DEBUG_GX > 3
		PrintParametersHydro(devmsg);
		PrintParameters(devmsg);
	#endif

	for(int target=tid.x;target<tid.y;target+=tid.z)
	{
		#if CUDA_DEBUG_GX > 1
			ASSERT_DEVICE_GX( isGasType(target,devmsg) );
		#else
			ASSERT_DEVICE_GX( isGasType(target) );
		#endif
		//MESSAGE("tid.x=%d, target=%d",tid.x,target);

		hydro_evaluate_gx(target
			#if CUDA_DEBUG_GX > 1
				,devmsg
			#endif
			);
	}

	SET_DEVICE_THREAD_STATE('D'); // done
}
