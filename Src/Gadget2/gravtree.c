#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <mpi.h>

#include "allvars.h"
#include "proto.h"
#include "interface_gx.h"
#include "timer_gx.h"
#include "debug_gx.h"
#include "chunckmanager_gx.h"
#include "../gadget_cuda_gx.h"

/*! \file gravtree.c
 *  \brief main driver routines for gravitational (short-range) force computation
 *
 *  This file contains the code for the gravitational force computation by
 *  means of the tree algorithm. To this end, a tree force is computed for
 *  all active local particles, and particles are exported to other
 *  processors if needed, where they can receive additional force
 *  contributions. If the TreePM algorithm is enabled, the force computed
 *  will only be the short-range part.
 */

/*! This function computes the gravitational forces for all active
 *  particles.  If needed, a new tree is constructed, otherwise the
 *  dynamically updated tree is used.  Particles are only exported to other
 *  processors when really needed, thereby allowing a good use of the
 *  communication buffer.
 */
void gravity_tree(void)
{
	int tim=20; // GX mod, timer to profile calls

	TimerBeg(29);
	TimerBeg(tim);

	long long ntot;
	int numnodes, nexportsum = 0;
	int i, j, iter = 0;
	int *numnodeslist, maxnumnodes, nexport, *numlist, *nrecv, *ndonelist;
	double tstart, tend, timetree = 0, timecommsumm = 0, timeimbalance = 0, sumimbalance;
	double ewaldcount;
	double costtotal, ewaldtot, *costtreelist, *ewaldlist;
	double maxt, sumt, *timetreelist, *timecommlist;
	double fac, plb, plb_max, sumcomm;

	#ifndef NOGRAVITY
		int *noffset, *nbuffer, *nsend, *nsend_local;
		long long ntotleft;
		int ndone,maxfill, ngrp;
		int k, place;
		int level, sendTask, recvTask;
		double ax, ay, az;
		MPI_Status status;
	#endif

	///////////////// GX //////////////////////
	int totdone=0;
	#if CUDA_DEBUG_GX>0
		int not_timestepped_gx=0;
		int exporthash_gx=0;
		int count_exported_gx=0;
	#endif
	///////////////// GX //////////////////////

	/* set new softening lengths */
	if(All.ComovingIntegrationOn)
		set_softenings();

	/* contruct tree if needed */
	tstart = second();
	if(TreeReconstructFlag)
	{
		if(ThisTask == 0)
		printf("Tree construction.\n");

		force_treebuild(NumPart);

		TreeReconstructFlag = 0;

		if(ThisTask == 0)
		printf("Tree construction done.\n");
	}
	tend = second();
	All.CPU_TreeConstruction += timediff(tstart, tend);

	costtotal = ewaldcount = 0;

	/* Note: 'NumForceUpdate' has already been determined in find_next_sync_point_and_drift() */
	numlist = malloc(NTask * sizeof(int) * NTask);

	MPI_Allgather(&NumForceUpdate, 1, MPI_INT, numlist, 1, MPI_INT, MPI_COMM_WORLD);

	for(i = 0, ntot = 0; i < NTask; i++)
		ntot += numlist[i];
	free(numlist);

	#ifndef NOGRAVITY
	if(ThisTask == 0)
		printf("Begin tree force.\n");

	#ifdef SELECTIVE_NO_GRAVITY
		for(i = 0; i < NumPart; i++)
			if(((1 << P[i].Type) & (SELECTIVE_NO_GRAVITY)))
				P[i].Ti_endstep = -P[i].Ti_endstep - 1;
	#endif

	noffset = malloc(sizeof(int) * NTask);	/* offsets of bunches in common list */
	nbuffer = malloc(sizeof(int) * NTask);
	nsend_local = malloc(sizeof(int) * NTask);
	nsend = malloc(sizeof(int) * NTask * NTask);
	ndonelist = malloc(sizeof(int) * NTask);

	i = 0;           /* begin with this index */
	ntotleft = ntot; /* particles left for all tasks together */

	TimerEnd(tim++);

	///////////////// GX //////////////////////
	// if (s_gx.cudamode>0 && All.MaxPart>1400000) TimersSleep(10); // GPU card runs hot on large sims, this is around N_p=1404928
	// if (s_gx.cudamode>0) TimersSleep(10);
	TimerBeg(tim);

	double starttime,subtime=-1,cpytime=-1;
	int Np=-1;
	int buffered=0;

	if(s_gx.cudamode>0)
	{
		FUN_MESSAGE(2,"gravity_tree()");

		TimerBeg(50);
		cpytime=GetTime();

		Np=InitializeProlog_gx(NumPart);

		TimerEnd(50);
		cpytime=GetTime()-cpytime;
	}
	///////////////// GX //////////////////////

	while(ntotleft > 0)
	{
		TimerBeg(31);
		starttime=GetTime();

		iter++;

		for(j = 0; j < NTask; j++)
			nsend_local[j] = 0;

		/* do local particles and prepare export list */
		tstart = second();

		if (s_gx.cudamode==0 || Np<MIN_FORCE_PARTICLES_FOR_GPU_GX) {
			ASSERT_GX( !buffered );

			ReLaunchChunkManager();

			for(nexport = 0, ndone = 0; i < NumPart && nexport < All.BunchSizeForce - NTask; i++) {
				if(P[i].Ti_endstep == All.Ti_Current)
				{
					ndone++;

					for(j = 0; j < NTask; j++)
						Exportflag[j] = 0;

					TimerUpdateCounter(31,1);
					#ifndef PMGRID
						costtotal += force_treeevaluate(i, 0, &ewaldcount);
					#else
						costtotal += force_treeevaluate_shortrange(i, 0 );
					#endif

					#if CUDA_DEBUG_GX>0
						int flagexported_gx=0;
					#endif
					for(j = 0; j < NTask; j++)
					{
						if(Exportflag[j])
						{
							ASSERT_GX( NTask>1 );
							#if CUDA_DEBUG_GX>0
								flagexported_gx=1;
								exporthash_gx += (i-j)*(j+ThisTask+1);
							#endif

							for(k = 0; k < 3; k++)
							GravDataGet[nexport].u.Pos[k] = P[i].Pos[k];
							#ifdef UNEQUALSOFTENINGS
								GravDataGet[nexport].Type = P[i].Type;
								#ifdef ADAPTIVE_GRAVSOFT_FORGAS
									if(P[i].Type == 0)
									GravDataGet[nexport].Soft = SphP[i].Hsml;
								#endif
							#endif
							GravDataGet[nexport].w.OldAcc = P[i].OldAcc;
							GravDataIndexTable[nexport].Task = j;
							GravDataIndexTable[nexport].Index = i;
							GravDataIndexTable[nexport].SortIndex = nexport;
							nexport++;
							nexportsum++;
							nsend_local[j]++;
						}
					}
					#if CUDA_DEBUG_GX>0
						if (flagexported_gx) ++count_exported_gx;
					#endif
				}
				#if CUDA_DEBUG_GX>0
					else ++not_timestepped_gx;
				#endif
			}
			ManageChuncks(0);
		} else {
			///////////////// GX //////////////////////
			// cudamode>0
			///////////////// GX //////////////////////
			#ifndef PMGRID
				// WARNING Attemping to run in tree-only mode, examine results carefully
				// ERROR cannot run in non PMGRID mode
			#endif

			if (iter==1){
				const double tx=GetTime();
				TimerBeg(51);

				ASSERT_GX(NumPart>=i);
				ASSERT_GX(!buffered);

				if (iter!=1) ERROR("cuda mode does not support iterations in gravtree calc, try to increasing the 'BufferSize' in the parameter file to surcomevent this problem");

				const int Np2=InitializeCalculation_gx(NumPart,P,0);
				ASSERT_GX( Np2==Np );
				if (Np2==0) WARNING("no particles participate in this timestep");

				TimerEnd(51);

				cpytime += GetTime() - tx;
				subtime=GetTime();
				TimerBeg(52);

				force_treeevaluate_shortrange_range_gx(0, Np);
				buffered=1;

				TimerUpdateCounter(31,NumPart-i);
				TimerEnd(52);

				subtime = GetTime() - subtime;
			} else {
				cpytime=-1;
				subtime=-1;
				ASSERT_GX(buffered);
			}

			for(nexport = 0, ndone = 0; i < NumPart &&  nexport < All.BunchSizeForce - NTask; i++) {
				if(P[i].Ti_endstep == All.Ti_Current)
				{
					ndone++;

					ASSERT_GX( i<NumPart );
					ASSERT_GX( buffered );

					const struct result_gx r=GetTarget(totdone++,i); // s_gx.result[target];

					P[i].GravAccel[0] = r.acc_x;
					P[i].GravAccel[1] = r.acc_y;
					P[i].GravAccel[2] = r.acc_z;
					P[i].GravCost = r.ninteractions;
					costtotal += r.ninteractions;

					if (s_gx.NTask>1) {
						#if CUDA_DEBUG_GX>0
							int flagexported_gx=0;
						#endif
						for(j = 0; j < NTask; j++) {
							if (GetExportflag_gx(&s_gx,i,NTask,j)){
								ASSERT_GX( NTask>1 );
								#if CUDA_DEBUG_GX>0
									flagexported_gx=1;
									exporthash_gx += (i-j)*(j+ThisTask+1);
								#endif

								for(k = 0; k < 3; k++) GravDataGet[nexport].u.Pos[k] = P[i].Pos[k];
								#ifdef UNEQUALSOFTENINGS
								GravDataGet[nexport].Type = P[i].Type;
									#ifdef ADAPTIVE_GRAVSOFT_FORGAS
										if(P[i].Type == 0) GravDataGet[nexport].Soft = SphP[i].Hsml;
									#endif
								#endif
								GravDataGet[nexport].w.OldAcc = P[i].OldAcc;
								GravDataIndexTable[nexport].Task = j;
								GravDataIndexTable[nexport].Index = i;
								GravDataIndexTable[nexport].SortIndex = nexport;
								nexport++;
								nexportsum++;
								nsend_local[j]++;
							}
						}
						#if CUDA_DEBUG_GX>0
							if (flagexported_gx) ++count_exported_gx;
						#endif
					}
				}
				#if CUDA_DEBUG_GX>0
					else ++not_timestepped_gx;
				#endif
			}
			AssertsOnhasGadgetDataBeenModified_gx(0,1,0);
		}
		TimerEnd(31);

		///////////////// GX //////////////////////
		if (iter==1 || !buffered){
				PrintInfoFinalize(s_gx,ndone,Np,starttime,cpytime,subtime,0,iter,-1
				#if CUDA_DEBUG_GX>0
					,not_timestepped_gx,count_exported_gx,nexport,nexportsum,exporthash_gx,costtotal
				#else
					,0,0,0,0,0,0
				#endif
				);
			subtime=-1;
		}

		TimerBeg(39);
		///////////////// GX //////////////////////

		tend = second();
		timetree += timediff(tstart, tend);

		qsort(GravDataIndexTable, nexport, sizeof(struct gravdata_index), grav_tree_compare_key);

		for(j = 0; j < nexport; j++)
			GravDataIn[j] = GravDataGet[GravDataIndexTable[j].SortIndex];

		for(j = 1, noffset[0] = 0; j < NTask; j++)
			noffset[j] = noffset[j - 1] + nsend_local[j - 1];

		tstart = second();

		MPI_Allgather(nsend_local, NTask, MPI_INT, nsend, NTask, MPI_INT, MPI_COMM_WORLD);

		tend = second();
		timeimbalance += timediff(tstart, tend);

		/* now do the particles that need to be exported */

		for(level = 1; level < (1 << PTask); level++)
		{
			tstart = second();
			for(j = 0; j < NTask; j++)
				nbuffer[j] = 0;

			for(ngrp = level; ngrp < (1 << PTask); ngrp++)
			{
				maxfill = 0;
				for(j = 0; j < NTask; j++)
				{
					if((j ^ ngrp) < NTask)
						if(maxfill < nbuffer[j] + nsend[(j ^ ngrp) * NTask + j])
							maxfill = nbuffer[j] + nsend[(j ^ ngrp) * NTask + j];
				}
				if(maxfill >= All.BunchSizeForce)
				break;

				sendTask = ThisTask;
				recvTask = ThisTask ^ ngrp;

				if(recvTask < NTask)
				{
					if(nsend[ThisTask * NTask + recvTask] > 0 || nsend[recvTask * NTask + ThisTask] > 0)
					{
						/* get the particles */
						MPI_Sendrecv(&GravDataIn[noffset[recvTask]],
						nsend_local[recvTask] * sizeof(struct gravdata_in), MPI_BYTE,
						recvTask, TAG_GRAV_A,
						&GravDataGet[nbuffer[ThisTask]],
						nsend[recvTask * NTask + ThisTask] * sizeof(struct gravdata_in), MPI_BYTE,
						recvTask, TAG_GRAV_A, MPI_COMM_WORLD, &status);
					}
				}

				for(j = 0; j < NTask; j++)
					if((j ^ ngrp) < NTask)
						nbuffer[j] += nsend[(j ^ ngrp) * NTask + j];
			}
			tend = second();
			timecommsumm += timediff(tstart, tend);

			TimerBeg(30);
			TimerUpdateCounter(30,nbuffer[ThisTask]);

			tstart = second();
			///////////////// GX //////////////////////
			// Do exported particles on the CPU/GPU
			{
				AssertsOnhasGadgetDataBeenModified_gx(1,1,0);

				#if CUDA_DEBUG_GX>1
					MESSAGE("INFO: DistRMSGrav=%g",DistRMSGravdata(nbuffer[ThisTask],GravDataGet));
				#endif

				starttime=GetTime();
				const int N=nbuffer[ThisTask];

				if (N>0){
					if (s_gx.cudamode==0 || N<MIN_FORCE_PARTICLES_FOR_GPU_GX || Np<MIN_FORCE_PARTICLES_FOR_GPU_GX) {
						ReLaunchChunkManager();
						for(j = 0; j<N ; j++)
						{
							#ifndef PMGRID
								costtotal += force_treeevaluate(j, 1, &ewaldcount);
							#else
								costtotal += force_treeevaluate_shortrange(j, 1);
							#endif
						}
						ManageChuncks(0);
					} else {
						ASSERT_GX( buffered );

						cpytime=GetTime();
						InitializeExportCalculation_gx(N,P[0].Type);
						ASSERT_GX( N==s_gx.Np );

						subtime=GetTime();
						force_treeevaluate_shortrange_range_gx(1, N);
						subtime=GetTime()-subtime;

						costtotal += FinalizeExportCalculation_gx(N);
						cpytime=GetTime()-cpytime-subtime;

						ASSERT_GX( N==s_gx.Np );
					}

					PrintInfoFinalize(s_gx,0,N,starttime,cpytime,subtime,2,iter,level,0,0,nexport,0,0,0);
					subtime=-1;
				} else {
					ReLaunchChunkManager();
					ManageChuncks(0);
				}
			}
			///////////////// GX //////////////////////
			if (nbuffer[ThisTask]>0) TimerUpdateCounter(30,-1);
			TimerEnd(30);
			tend = second();
			timetree += timediff(tstart, tend);

			TimerBeg(33);
			tstart = second();

			MPI_Barrier(MPI_COMM_WORLD);
			tend = second();
			timeimbalance += timediff(tstart, tend);
			TimerEnd(33);

			/* get the result */
			tstart = second();
			for(j = 0; j < NTask; j++)
				nbuffer[j] = 0;
			for(ngrp = level; ngrp < (1 << PTask); ngrp++)
			{
				maxfill = 0;
				for(j = 0; j < NTask; j++)
				{
					if((j ^ ngrp) < NTask)
						if(maxfill < nbuffer[j] + nsend[(j ^ ngrp) * NTask + j])
						maxfill = nbuffer[j] + nsend[(j ^ ngrp) * NTask + j];
				}
				if(maxfill >= All.BunchSizeForce)
					break;

				sendTask = ThisTask;
				recvTask = ThisTask ^ ngrp;
				if(recvTask < NTask)
				{
					if(nsend[ThisTask * NTask + recvTask] > 0 || nsend[recvTask * NTask + ThisTask] > 0)
					{
						/* send the results */
						MPI_Sendrecv(&GravDataResult[nbuffer[ThisTask]],
						nsend[recvTask * NTask + ThisTask] * sizeof(struct gravdata_in),
						MPI_BYTE, recvTask, TAG_GRAV_B,
						&GravDataOut[noffset[recvTask]],
						nsend_local[recvTask] * sizeof(struct gravdata_in),
						MPI_BYTE, recvTask, TAG_GRAV_B, MPI_COMM_WORLD, &status);

						/* add the result to the particles */
						for(j = 0; j < nsend_local[recvTask]; j++)
						{
							place = GravDataIndexTable[noffset[recvTask] + j].Index;
/* // disable export forces for debugging
							for(k = 0; k < 3; k++)
								P[place].GravAccel[k] += GravDataOut[j + noffset[recvTask]].u.Acc[k];
*/
							P[place].GravCost += GravDataOut[j + noffset[recvTask]].w.Ninteractions;
						}
					}
				}

				for(j = 0; j < NTask; j++)
					if((j ^ ngrp) < NTask)
						nbuffer[j] += nsend[(j ^ ngrp) * NTask + j];

			}
			tend = second();
			timecommsumm += timediff(tstart, tend);

			level = ngrp - 1;
		}

		MPI_Allgather(&ndone, 1, MPI_INT, ndonelist, 1, MPI_INT, MPI_COMM_WORLD);

		for(j = 0; j < NTask; j++)
			ntotleft -= ndonelist[j];

		TimerEnd(39);
	}

	TimerEnd(tim++);
	TimerBeg(tim);

	free(ndonelist);
	free(nsend);
	free(nsend_local);
	free(nbuffer);
	free(noffset);

	/* now add things for comoving integration */

	#ifndef PERIODIC
		#ifndef PMGRID
			if(All.ComovingIntegrationOn)
			{
				fac = 0.5 * All.Hubble * All.Hubble * All.Omega0 / All.G;

				for(i = 0; i < NumPart; i++)
					if(P[i].Ti_endstep == All.Ti_Current)
						for(j = 0; j < 3; j++)
							P[i].GravAccel[j] += fac * P[i].Pos[j];
			}
		#endif
	#endif

	for(i = 0; i < NumPart; i++)
		if(P[i].Ti_endstep == All.Ti_Current)
		{
			#ifdef PMGRID
				ax = P[i].GravAccel[0] + P[i].GravPM[0] / All.G;
				ay = P[i].GravAccel[1] + P[i].GravPM[1] / All.G;
				az = P[i].GravAccel[2] + P[i].GravPM[2] / All.G;
			#else
				ax = P[i].GravAccel[0];
				ay = P[i].GravAccel[1];
				az = P[i].GravAccel[2];
			#endif
			P[i].OldAcc = sqrt(ax * ax + ay * ay + az * az);
		}

	if(All.TypeOfOpeningCriterion == 1)
		All.ErrTolTheta = 0;	/* This will switch to the relative opening criterion for the following force computations */

	/*  muliply by G */
	for(i = 0; i < NumPart; i++)
		if(P[i].Ti_endstep == All.Ti_Current)
		for(j = 0; j < 3; j++)
			P[i].GravAccel[j] *= All.G;


	/* Finally, the following factor allows a computation of a cosmological simulation
		with vacuum energy in physical coordinates */
	#ifndef PERIODIC
		#ifndef PMGRID
		if(All.ComovingIntegrationOn == 0)
		{
			fac = All.OmegaLambda * All.Hubble * All.Hubble;

			for(i = 0; i < NumPart; i++)
				if(P[i].Ti_endstep == All.Ti_Current)
				for(j = 0; j < 3; j++)
					P[i].GravAccel[j] += fac * P[i].Pos[j];
		}
		#endif
	#endif

	#ifdef SELECTIVE_NO_GRAVITY
		for(i = 0; i < NumPart; i++)
			if(P[i].Ti_endstep < 0)
				P[i].Ti_endstep = -P[i].Ti_endstep - 1;
	#endif

	if(ThisTask == 0)
		printf("tree is done.\n");

	#else /* gravity is switched off */

	for(i = 0; i < NumPart; i++)
		if(P[i].Ti_endstep == All.Ti_Current)
		for(j = 0; j < 3; j++)
			P[i].GravAccel[j] = 0;

	#endif

	/* Now the force computation is finished */

	/*  gather some diagnostic information */

	timetreelist = malloc(sizeof(double) * NTask);
	timecommlist = malloc(sizeof(double) * NTask);
	costtreelist = malloc(sizeof(double) * NTask);
	numnodeslist = malloc(sizeof(int) * NTask);
	ewaldlist = malloc(sizeof(double) * NTask);
	nrecv = malloc(sizeof(int) * NTask);

	numnodes = Numnodestree;

	MPI_Gather(&costtotal, 1, MPI_DOUBLE, costtreelist, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&numnodes, 1, MPI_INT, numnodeslist, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&timetree, 1, MPI_DOUBLE, timetreelist, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&timecommsumm, 1, MPI_DOUBLE, timecommlist, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&NumPart, 1, MPI_INT, nrecv, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&ewaldcount, 1, MPI_DOUBLE, ewaldlist, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Reduce(&nexportsum, &nexport, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&timeimbalance, &sumimbalance, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if(ThisTask == 0)
	{
		All.TotNumOfForces += ntot;

		fprintf(FdTimings, "Step= %d  t= %g  dt= %g \n", All.NumCurrentTiStep, All.Time, All.TimeStep);
		fprintf(FdTimings, "Nf= %d%09d  total-Nf= %d%09d  ex-frac= %g  iter= %d\n",
			(int) (ntot / 1000000000), (int) (ntot % 1000000000),
			(int) (All.TotNumOfForces / 1000000000), (int) (All.TotNumOfForces % 1000000000),
			nexport / ((double) ntot), iter);
		/* note: on Linux, the 8-byte integer could be printed with the format identifier "%qd", but doesn't work on AIX */

		fac = NTask / ((double) All.TotNumPart);

		for(i = 0, maxt = timetreelist[0], sumt = 0, plb_max = 0,
		maxnumnodes = 0, costtotal = 0, sumcomm = 0, ewaldtot = 0; i < NTask; i++)
		{
			costtotal += costtreelist[i];

			sumcomm += timecommlist[i];

			if(maxt < timetreelist[i])
				maxt = timetreelist[i];
			sumt += timetreelist[i];

			plb = nrecv[i] * fac;

			if(plb > plb_max)
				plb_max = plb;

			if(numnodeslist[i] > maxnumnodes)
				maxnumnodes = numnodeslist[i];

			ewaldtot += ewaldlist[i];
		}
		fprintf(FdTimings, "work-load balance: %g  max=%g avg=%g PE0=%g\n",
			maxt / (sumt / NTask), maxt, sumt / NTask, timetreelist[0]);
		fprintf(FdTimings, "particle-load balance: %g\n", plb_max);
		fprintf(FdTimings, "max. nodes: %d, filled: %g\n", maxnumnodes,
			maxnumnodes / (All.TreeAllocFactor * All.MaxPart));
		fprintf(FdTimings, "part/sec=%g | %g  ia/part=%g (%g)\n", ntot / (sumt + 1.0e-20),
			ntot / (maxt * NTask), ((double) (costtotal)) / ntot, ((double) ewaldtot) / ntot);
		fprintf(FdTimings, "\n");

		fflush(FdTimings);

		All.CPU_TreeWalk += sumt / NTask;
		All.CPU_Imbalance += sumimbalance / NTask;
		All.CPU_CommSum += sumcomm / NTask;
	}

	free(nrecv);
	free(ewaldlist);
	free(numnodeslist);
	free(costtreelist);
	free(timecommlist);
	free(timetreelist);

	ASSERT_GX( tim==22 );
	TimerEnd(tim++);
	TimerEnd(29);

	//MESSAGE("%6.2f, %6.2f, %6.2f, %6.2f, %6.2f  -  %5.1f, %5.1f, %5.1f, %5.1f %c force timers d 29,31,30,33,net",TimerGet(29),TimerGet(31),TimerGet(30),TimerGet(33),TimerGet(29)-TimerGet(31)-TimerGet(30),100.0*TimerGet(31)/TimerGet(29),100.0*TimerGet(30)/TimerGet(29),100.0*TimerGet(33)/TimerGet(29),100.0*(TimerGet(29)-TimerGet(31)-TimerGet(30))/TimerGet(29),'%');
	//MESSAGE("%6.2f, %6.2f, %6.2f, %6.2f, %6.2f  -  %5.1f, %5.1f, %5.1f, %5.1f %c force timers a 29,31,30,33,net",TimerGetAccumulated(29),TimerGetAccumulated(31),TimerGetAccumulated(30),TimerGetAccumulated(33),TimerGetAccumulated(29)-TimerGetAccumulated(31)-TimerGetAccumulated(30),100.0*TimerGetAccumulated(31)/TimerGetAccumulated(29),100.0*TimerGetAccumulated(30)/TimerGetAccumulated(29),100.0*TimerGetAccumulated(33)/TimerGetAccumulated(29),100.0*(TimerGetAccumulated(29)-TimerGetAccumulated(31)-TimerGetAccumulated(30))/TimerGetAccumulated(29),'%');
}

/*! This function sets the (comoving) softening length of all particle
 *  types in the table All.SofteningTable[...].  We check that the physical
 *  softening length is bounded by the Softening-MaxPhys values.
 */
void set_softenings(void)
{
  int i;

  if(All.ComovingIntegrationOn)
    {
      if(All.SofteningGas * All.Time > All.SofteningGasMaxPhys)
        All.SofteningTable[0] = All.SofteningGasMaxPhys / All.Time;
      else
        All.SofteningTable[0] = All.SofteningGas;

      if(All.SofteningHalo * All.Time > All.SofteningHaloMaxPhys)
        All.SofteningTable[1] = All.SofteningHaloMaxPhys / All.Time;
      else
        All.SofteningTable[1] = All.SofteningHalo;

      if(All.SofteningDisk * All.Time > All.SofteningDiskMaxPhys)
        All.SofteningTable[2] = All.SofteningDiskMaxPhys / All.Time;
      else
        All.SofteningTable[2] = All.SofteningDisk;

      if(All.SofteningBulge * All.Time > All.SofteningBulgeMaxPhys)
        All.SofteningTable[3] = All.SofteningBulgeMaxPhys / All.Time;
      else
        All.SofteningTable[3] = All.SofteningBulge;

      if(All.SofteningStars * All.Time > All.SofteningStarsMaxPhys)
        All.SofteningTable[4] = All.SofteningStarsMaxPhys / All.Time;
      else
        All.SofteningTable[4] = All.SofteningStars;

      if(All.SofteningBndry * All.Time > All.SofteningBndryMaxPhys)
        All.SofteningTable[5] = All.SofteningBndryMaxPhys / All.Time;
      else
        All.SofteningTable[5] = All.SofteningBndry;
    }
  else
    {
      All.SofteningTable[0] = All.SofteningGas;
      All.SofteningTable[1] = All.SofteningHalo;
      All.SofteningTable[2] = All.SofteningDisk;
      All.SofteningTable[3] = All.SofteningBulge;
      All.SofteningTable[4] = All.SofteningStars;
      All.SofteningTable[5] = All.SofteningBndry;
    }

  for(i = 0; i < 6; i++)
    All.ForceSoftening[i] = 2.8 * All.SofteningTable[i];

  All.MinGasHsml = All.MinGasHsmlFractional * All.ForceSoftening[0];
}


/*! This function is used as a comparison kernel in a sort routine. It is
 *  used to group particles in the communication buffer that are going to
 *  be sent to the same CPU.
 */
int grav_tree_compare_key(const void *a, const void *b)
{
  if(((struct gravdata_index *) a)->Task < (((struct gravdata_index *) b)->Task))
    return -1;

  if(((struct gravdata_index *) a)->Task > (((struct gravdata_index *) b)->Task))
    return +1;

  return 0;
}
