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

// register struct result_gx d;
// register float f;

__forceinline__ __device__
struct result_gx GetAuxData(const int target
	#if CUDA_DEBUG_GX > 1
		,struct DevMsg& devmsg
	#endif
)
{
	ASSERT_DEVICE_GX( target<g_const_state.sz_result);
	ASSERT_DEVICE_GX( g_const_state.mode==0 || g_const_state.mode==1);

	const struct result_gx d = g_const_state.result[target];

	ASSERT_DEVICE_GX( g_const_state.mode==1 || d.acc_y==0 );
	ASSERT_DEVICE_GX( g_const_state.mode==1 || d.acc_z==0 );
	ASSERT_DEVICE_GX( !isNAN_device(d.acc_x) && !isNAN_device(d.acc_y) && !isNAN_device(d.acc_z) );
	ASSERT_DEVICE_GX( (d.ninteractions & 7)>=0 && (d.ninteractions& 7)<6 );
	#if CUDA_DEBUG_GX > 1
		ASSERT_DEVICE_GX( g_const_state.mode==1 || (d.ninteractions & 7)==GetParticleType(d.ninteractions>>3,devmsg) );
	#else
		// This check fails when using multiple particle types??
		ASSERT_DEVICE_GX( g_const_state.mode==1 || (d.ninteractions & 7)==GetParticleType(d.ninteractions>>3) );
	#endif

	#ifdef CUDA_GX_DEBUG_MEMORY_ERROR
		ASSERT_DEVICE_GX( (d.ninteractions >>3)==d.realtarget );
		ASSERT_DEVICE_GX( (d.ninteractions & 7)==d.type );
		#if CUDA_DEBUG_GX > 1
			ASSERT_DEVICE_GX( g_const_state.mode==1 || GetParticleType(d.ninteractions>>3,devmsg)==d.type );
		#else
			ASSERT_DEVICE_GX( g_const_state.mode==1 || GetParticleType(d.ninteractions>>3)==d.type );
		#endif

		if (!((g_const_state.mode==0 && d.temp2==target) || (g_const_state.mode==1 && d.temp2==-target)))
			exit_device_info(-5,target,g_const_state.ThisTask,d.temp2,(size_t)&d.temp2,target>0 ? g_const_state.result[target-1].temp2 : 0,target+1<g_const_state.sz_result ? g_const_state.result[target+1].temp2 : 0,-42.5);

		ASSERT_DEVICE_GX( (g_const_state.mode==0 && d.temp2==target) || (g_const_state.mode==1 && d.temp2==-target) ); // YYY while non local target list in result data

		// a variation of error 4
		if (!((d.ninteractions & 7)>=0 && (d.ninteractions & 7)<6))
			exit_device_info(-4,d.ninteractions,(size_t)&d.ninteractions,d.temp2,g_const_state.ThisTask,0,0,-42.4);
	#endif

	return d;
}

__forceinline__ __device__
float Getshortrange_table(const int tabindex
	#if CUDA_DEBUG_GX  > 1
		,struct DevMsg& devmsg
	#endif
)
{	
	ASSERT_DEVICE_GX( g_const_state.mode==0 || g_const_state.mode==1);
	ASSERT_DEVICE_GX(tabindex>=0 && tabindex<g_const_state.sz_shortrange_table);
	#ifdef CUDA_GX_USE_TEXTURES
		const float f=tex1Dfetch(tex_shortrange_table,tabindex);
		#if CUDA_DEBUG_GX > 1
			ASSERT_DEVICE_GX( f==g_const_state.shortrange_table[tabindex] );
		#endif
	#else
		const float f=g_const_state.shortrange_table[tabindex];
	#endif
	return f;
}

__global__
void force_treeevaluate_shortrange_cuda_gx_kernel()
{
	// Algorithm
	// 1a: initialize variables
	// 2: loop over all particles, for(i=0;i<Np;++i)
	// (2: for cuda threads, divide [0;numpart[ into subpartitions, linear or fragmeted)
	//  2a: reset loop variables, acc[3], niterations
	//  2b: get particle(i) data, pos[3]=P(i).pos (and the same for mass, ptype and aold),  {Global mem access: P[i]}
	//  2c: set various variables
	//  3: loop over all nodes, while(no >= 0)
	//    3a:  is no a particle index or a node
	//      3b1: the index of the node is the index of the particle, get particle,  {Global mem access: P[no]}
	//      3b2: find distance
	//      3b3: get next node, no=GetNextnode(no), {Global mem access: GetNextnode[no]}
	//      3c1: the index is a node
	//      (3c2: is it a psudo particle, then do some export functionality (will not happen on a 1-CPU sys))
	//      (3c3: and get next node, no=GetNextnode(no), and continue loop {Global mem access: GetNextnode[no]})
	//      3d1: get node data, GetNode(no), {Global mem access: GetNode[no]}
	//      (3d2: check for various bitflags, will not happen in mode=0)
	//      3d2: find dist to node
	//      3d3: check stop criteria, r2>rcut2,
	//         3e1: x, check eff. dist, set no=sibling and continue if dist<-eff_dist || dist>eff_dist
	//         3e2: y,...
	//         3e3: z,...
	//      3f1: check Barnes-Hut opening criterion
	//         3f2: if(g_const_parameters.ErrTolTheta) and more then open cell, no=nextnode, continue loop
	//      3g: check relative opening criterion, if fullfilled,  no=nextnode, continue loop
	//      3h: ok, node can be used, no = NODE_gx_sibling
	//      (3i: check various bitflags)
	//      3j: find distance and fac
	//      3k: find tabindex and make shortrange table lookup
	//        3k1: if(tabindex < g_const_state.sz_shortrange_table) do lookup, {Global mem access: Getshortrange_table[tabindex]}
	//        3k2: add to acc[3] and update niterations
	//    3l: end loop
	//    2d: write acc to global results, {Global mem access: result[target]}
	// 2e: end for
	// 1b: do various debug maintainance

	// 1a:
	KernelPreinit(0
		#if CUDA_DEBUG_GX > 1
			,devmsg
		#endif
	);
	
	dim3 tid;
	tid.x = threadIdx.x+blockIdx.x*blockDim.x;
	tid.y = g_const_state.Np;
	tid.z = blockDim.x*gridDim.x;
	
	#if CUDA_DEBUG_GX > 1
		struct DevMsg devmsg=ResetMsg(g_const_state.debug_msg,GetTid(),g_const_state.debug_sz_msg);
		#if CUDA_DEBUG_GX > 2
			PrintDevMsg(&devmsg,"forcetree kernel",1);
			Debug_PrintVars(tid,s_gx.Np,devmsg);
		#endif
	#endif


	int ptype,target,target_x;
	double r2, dx, dy, dz, mass, h, h_inv, h3_inv;
	double pos_x, pos_y, pos_z, aold;
	#ifdef PMGRID
	double rcut, asmth;
	#endif

	#if defined(UNEQUALSOFTENINGS) && !defined(ADAPTIVE_GRAVSOFT_FORGAS)
		int maxsofttype;
	#endif
	#ifdef ADAPTIVE_GRAVSOFT_FORGAS
		double soft = 0;
	#endif
	#ifdef PERIODIC
		const double boxsize = g_const_parameters.BoxSize;
		const double boxhalf = 0.5 * g_const_parameters.BoxSize;
	#endif

	// 2:
	for(target_x=tid.x;target_x<tid.y;target_x+=tid.z)
	{
		int target_id;
		
		// __syncthreads(); // NOTE: won't add any speed
				// WARNING: also will cause barrier deadlock
				// if not all threads are assigned a particle

		ASSERT_DEVICE_GX( target_x<g_const_state.Np );
		ASSERT_DEVICE_GX( target_x<g_const_state.sz_result );

		// 2a:
		double acc_x = 0;
		double acc_y = 0;
		double acc_z = 0;
		int ninteractions = 0;

		// 2b:
		{
			const struct result_gx a=GetAuxData(target_x
				#if CUDA_DEBUG_GX > 1
					,devmsg
				#endif
			);
			ASSERT_DEVICE_GX( a.ninteractions>=0 );
			target=a.ninteractions >> 3;
			ptype =a.ninteractions &  7;

			ASSERT_DEVICE_GX( ptype>=0 && ptype<6 );
			#if CUDA_DEBUG_GX > 1
				ASSERT_DEVICE_GX( g_const_state.mode==1 || ptype==GetParticleType(target,devmsg) );
			#else
				// This fails with multiple particle types
				// Is it allowed s.t. the export particles are greater than the node particles?
				if (g_const_state.mode==0) {
					int ptype2 = GetParticleType(target);
					if (ptype!=ptype2) {
						printf("particle: %d bitwise particle type: %d variable particle type: %d\n",target,ptype,ptype2);
					}
					ASSERT_DEVICE_GX( g_const_state.mode==1 || ptype==GetParticleType(target) );
				}
			#endif

			if (g_const_state.mode==0){
				aold=a.acc_x;
				const struct particle_data_gx p_local=GetParticle(target
					#if CUDA_DEBUG_GX > 1
						,devmsg
					#endif
				);

				pos_x=p_local.Pos[0];
				pos_y=p_local.Pos[1];
				pos_z=p_local.Pos[2];
				// target_id=p_local.ID;
			} else {
				aold=GetScratch(target_x
					#if CUDA_DEBUG_GX > 1
						,devmsg
					#endif
				);

				pos_x=a.acc_x; // reuse of result data, acc.x=pos.x etc.
				pos_y=a.acc_y;
				pos_z=a.acc_z;
			}

			#ifdef CUDA_GX_DEBUG_MEMORY_ERROR
				ASSERT_DEVICE_GX( aold==a.oldacc );
			#endif

			aold *= g_const_parameters.ErrTolForceAcc;
		}

		// 2c:
		#ifdef ADAPTIVE_GRAVSOFT_FORGAS
			not supported yet
			if(ptype == 0) soft = SphP[target].Hsml;
		#endif

		#ifdef PMGRID
			rcut = g_const_parameters.Rcut[0];
			asmth =  g_const_parameters.Asmth[0];
			#ifdef PLACEHIGHRESREGION
				if(((1 << ptype) & (PLACEHIGHRESREGION))) {
					rcut = g_const_parameters.Rcut[1];
					asmth = g_const_parameters.Asmth[1];
				}
			#endif
			const double rcut2 = rcut * rcut;
			const double asmthfac = 0.5 / asmth * (g_const_state.sz_shortrange_table / 3.0);
		#endif
		
		#ifndef UNEQUALSOFTENINGS
			h = g_const_parameters.ForceSoftening[ptype];
			h_inv = 1.0 / h;
			h3_inv = h_inv * h_inv * h_inv;
		#endif

		int no = g_const_parameters.MaxPart; // root node

		#if CUDA_DEBUG_GX > 1
			Debug_PrintValues(devmsg,pos_x,pos_y,pos_z,target_x,rcut,rcut2,asmth,asmthfac,h3_inv,no);
		#endif

		// 3:
		while(no >= 0)
		{
			// 3a:
			if(no < g_const_parameters.MaxPart) {
				// 3b1:
				// the index of the node is the index of the particle
				{
					const struct particle_data_gx p_local=GetParticle(no
						#if CUDA_DEBUG_GX > 1
							,devmsg
						#endif
					);

					mass = p_local.Mass;

					dx = p_local.Pos[0] - pos_x;
					dy = p_local.Pos[1] - pos_y;
					dz = p_local.Pos[2] - pos_z;
				}

				// 3b2
				#ifdef PERIODIC
					dx = NEAREST(dx);
					dy = NEAREST(dy);
					dz = NEAREST(dz);
				#endif

				r2 = dx * dx + dy * dy + dz * dz;

				#ifdef UNEQUALSOFTENINGS
					const int pnotype=GetParticleType(no
						#if CUDA_DEBUG_GX > 1
							,devmsg
						#endif
					);

					#ifdef ADAPTIVE_GRAVSOFT_FORGAS
						const double Hsml=GetHsml(no); // SphP[no].Hsml
						if(ptype == 0) h = soft;
						else           h = g_const_parameters.ForceSoftening[ptype];

						if(pnotype==0) { // if(P[no].Type == 0)
							if(h < Hsml) h = Hsml;
						} else {
							if(h < g_const_parameters.ForceSoftening[pnotype]) h = g_const_parameters.ForceSoftening[pnotype];
						}
					#else
						h = g_const_parameters.ForceSoftening[ptype];
						if(h < g_const_parameters.ForceSoftening[pnotype]) h = g_const_parameters.ForceSoftening[pnotype];
					#endif
				#endif

				// 3b3:
				no=GetNextnode(no
					#if CUDA_DEBUG_GX > 1
						,devmsg
					#endif
					);
			} else {
				// 3c1:
				// we have an  internal node
				if(no >= g_const_parameters.MaxPart + g_const_parameters.MaxNodes) { // pseudo particle
					// 3c2:
					if(g_const_state.mode == 0) {
						if(target_id == 58896)
							printf("pseudoparticle: %d %d %d\n",g_const_parameters.Ti_Current,g_const_state.ThisTask,no);
						//Exportflag[DomainTask[no - (g_const_parameters.MaxPart + g_const_parameters.MaxNodes)]] = 1;
						SetExportflag_gx(no,target
						#if CUDA_DEBUG_GX > 1
							,devmsg
						#endif
						);
					}

					// 3c3:
					no=GetNextnode(no - g_const_parameters.MaxNodes
						#if CUDA_DEBUG_GX > 1
							,devmsg
						#endif
						);

					continue;
				}

				ASSERT_DEVICE_GX( no>=g_const_state.MaxPart );

				// 3d1:
				double NODE_gx_len;
				double NODE_gx_center0;
				double NODE_gx_center1;
				double NODE_gx_center2;

				double NODE_gx_s0;
				double NODE_gx_s1;
				double NODE_gx_s2;
				double NODE_gx_mass;

				int NODE_gx_bitflags;
				int NODE_gx_sibling;
				int NODE_gx_nextnode;
				//int NODE_gx_father;

					{
						const struct NODE_gx node_local=GetNode(no
						#if CUDA_DEBUG_GX > 1
							,devmsg
						#endif
						);

						NODE_gx_len    = node_local.len;
						NODE_gx_center0= node_local.center[0];
						NODE_gx_center1= node_local.center[1];
						NODE_gx_center2= node_local.center[2];

						NODE_gx_s0     = node_local.u.d.s[0];
						NODE_gx_s1     = node_local.u.d.s[1];
						NODE_gx_s2     = node_local.u.d.s[2];
						NODE_gx_mass   = node_local.u.d.mass;

						NODE_gx_bitflags = node_local.u.d.bitflags;
						NODE_gx_sibling  = node_local.u.d.sibling;
						NODE_gx_nextnode = node_local.u.d.nextnode;
						// NODE_gx_father   = node_local.u.d.father;
					}

				// 3d2:
				if(g_const_state.mode == 1) {
					if((NODE_gx_bitflags & 3) == 1) {
						// if it's a top-level node
						// which does not contain
						// local particles we can
						// continue at this point

						no = NODE_gx_sibling;
						continue;
					}
				}

				// 3d2:
				mass = NODE_gx_mass;

				dx = NODE_gx_s0 - pos_x;
				dy = NODE_gx_s1 - pos_y;
				dz = NODE_gx_s2 - pos_z;

				/* For debugging */
				if(target_id == 58896)
					printf("Cell opened: %d %d %d %g %g %g\n",g_const_parameters.Ti_Current,g_const_state.ThisTask,no,pos_x,pos_y,pos_z);

				#ifdef PERIODIC
					dx = NEAREST(dx);
					dy = NEAREST(dy);
					dz = NEAREST(dz);
				#endif

				r2 = dx * dx + dy * dy + dz * dz;

				// 3d3:
				#ifdef PMGRID
				if(r2 > rcut2) {
					// check whether we can stop walking along this branch
					const double eff_dist = rcut + 0.5 * NODE_gx_len;
					double dist;

					// 3e1:
					#ifdef PERIODIC
						dist = NEAREST(NODE_gx_center0 - pos_x);
					#else
						dist = NODE_gx_center0 - pos_x;
					#endif
					if(dist < -eff_dist || dist > eff_dist) {
						no = NODE_gx_sibling;
						continue;
					}

					// 3e2:
					#ifdef PERIODIC
						dist = NEAREST(NODE_gx_center1 - pos_y);
					#else
						dist = NODE_gx_center1 - pos_y;
					#endif

					if(dist < -eff_dist || dist > eff_dist) {
						no = NODE_gx_sibling;
						continue;
					}

					// 3e3:
					#ifdef PERIODIC
						dist = NEAREST(NODE_gx_center2 - pos_z);
					#else
						dist = NODE_gx_center2 - pos_z;
					#endif

					if(dist < -eff_dist || dist > eff_dist) {
						no = NODE_gx_sibling;
						continue;
					}
				}
				#endif

				// 3f:
				if(g_const_parameters.ErrTolTheta) { // check Barnes-Hut opening criterion
					// 3f2:
					if(NODE_gx_len * NODE_gx_len > r2 * g_const_parameters.ErrTolTheta * g_const_parameters.ErrTolTheta) {
						// open cell
						no = NODE_gx_nextnode;
						continue;
					}
				}
				// 3g:
				else // check relative opening criterion
				{
					if(mass * NODE_gx_len * NODE_gx_len > r2 * r2 * aold ){
						// open cell
						no = NODE_gx_nextnode;
						continue;
					}

					// check in addition whether we lie inside the cell
					if(fabs(NODE_gx_center0 - pos_x) < 0.60 * NODE_gx_len) {
						if(fabs(NODE_gx_center1 - pos_y) < 0.60 * NODE_gx_len) {
							if(fabs(NODE_gx_center2 - pos_z) < 0.60 * NODE_gx_len) {
								no = NODE_gx_nextnode;
								continue;
							}
						}
					}
				}

				#ifdef UNEQUALSOFTENINGS
					#ifndef ADAPTIVE_GRAVSOFT_FORGAS
						h = g_const_parameters.ForceSoftening[ptype];
						maxsofttype = (NODE_gx_bitflags >> 2) & 7;
						if(maxsofttype == 7) { // may only occur for zero mass top-level nodes
							if(mass > 0) exit_device(-987); // simulate endrun(987)
							no = NODE_gx_nextnode;
							continue;
						} else 	{
							if(h < g_const_parameters.ForceSoftening[maxsofttype]) {
								h = g_const_parameters.ForceSoftening[maxsofttype];
								if(r2 < h * h) {
									if(((NODE_gx_bitflags >> 5) & 1)) {	// bit-5 signals that there are particles of different softening in the node
										no = NODE_gx_nextnode;
										continue;
									}
								}
							}
						}
					#else
						if(ptype == 0) h = soft;
						else           h =g_const_parameters.ForceSoftening[ptype];

						if(h < nop->maxsoft) {
							h = nop->maxsoft;
							if(r2 < h * h) {
								no = NODE_gx_nextnode;
								continue;
							}
						}
					#endif
				#endif

				no = NODE_gx_sibling;	// ok, node can be used

				// 3i_
				if(g_const_state.mode == 1) {
					if(((NODE_gx_bitflags) & 1)) continue;	// Bit 0 signals that this node belongs to top-level tree
				}
			}


			{ //  local scope for variables r, fac and tabindex
				// 3j
				const double r = sqrt(r2);
				double fac;

				if(r >= h) fac = mass / (r2 * r);
				else {
					#ifdef UNEQUALSOFTENINGS
						h_inv = 1.0 / h;
						h3_inv = h_inv * h_inv * h_inv;
					#endif
					const double u = r * h_inv;
					if(u < 0.5)	fac = mass * h3_inv * (10.666666666667 + u * u * (32.0 * u - 38.4));
					else		fac = mass * h3_inv * (21.333333333333 - 48.0 * u + 38.4 * u * u - 10.666666666667 * u * u * u - 0.066666666667 / (u * u * u));
				}

				// 3k:
				/* If we're only doing short-range, multiply by erfc factor;
					if not, then don't need any extra factor. */

				#ifdef PMGRID
				const int tabindex = (int) (asmthfac * r);

				if( tabindex < g_const_state.sz_shortrange_table) {
					// 3k1:
					fac *= Getshortrange_table(tabindex
						#if CUDA_DEBUG_GX > 1
							,devmsg
						#endif
						);
				#endif
			
					// 3k2:
					acc_x += dx * fac;
					acc_y += dy * fac;
					acc_z += dz * fac;

					ninteractions++;
			
				#ifdef PMGRID
				}
				#endif
			}

		} // 3l: end while

		// 2d:
		ASSERT_DEVICE_GX( g_const_state.mode==0 || g_const_state.mode==1 );
		ASSERT_DEVICE_GX( target_x<g_const_state.sz_result );

		// #ifdef CUDA_GX_DEBUG_MEMORY_ERROR
			if (!( ((g_const_state.result[target_x].ninteractions & 7)==ptype && g_const_state.result[target_x].acc_z==0) || g_const_state.mode==1 ))
				exit_device_info(-4,target_x,g_const_state.result[target_x].ninteractions,ptype,g_const_state.mode,0,g_const_state.result[target_x].acc_z,-42.4);
		// #endif

		// This check appears to fail when multiple particle types are present??
		//ASSERT_DEVICE_GX( (g_const_state.result[target_x].ninteractions & 7)==ptype || g_const_state.mode==1 );
		ASSERT_DEVICE_GX( g_const_state.result[target_x].acc_z==0 || g_const_state.mode==1 );

		// NOTE: check if result is not overwritten, dummy check on interactions, real check in fun
		#if CUDA_DEBUG_GX > 1
			ASSERT_DEVICE_GX( GetAuxData(target_x,devmsg).ninteractions+1 > 0 );
		#else
			ASSERT_DEVICE_GX( GetAuxData(target_x).ninteractions+1 > 0 );
		#endif

		g_const_state.result[target_x].acc_x=acc_x;
		g_const_state.result[target_x].acc_y=acc_y;
		g_const_state.result[target_x].acc_z=acc_z;
		g_const_state.result[target_x].ninteractions=ninteractions;
		#ifdef CUDA_GX_DEBUG_MEMORY_ERROR
			g_const_state.result[target_x].temp2=(g_const_state.mode==0 ? target_x+100000 : target_x-100000);
		#endif

		#ifdef __DEVICE_EMULATION__
			#if CUDART_VERSION < 3 // NOTE: error in nvcc 3.0, will not pass parameters to GCC
				ASSERT_DEVICE_GX( isResultDataOK(g_const_state.result[target_x],__FILE__,__LINE__) );
			#endif
		#endif
	} // 2e: end for

	// 1b:
	SET_DEVICE_THREAD_STATE('D'); // done
}

