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

#ifdef CUDA_GX_SEGMENTED
	unsigned int GetSegmentSize(const struct state_gx& s)
	{
		FUN_MESSAGE(5,"GetSegmentSize()");
		ASSERT_GX( CUDA_GX_SEGMENTED>0 && s.blocks*s.grids>0 );
		return s.blocks*s.grids*CUDA_GX_SEGMENTED;
	}

	unsigned int GetSegmentOffset(const struct state_gx& s)
	{
		FUN_MESSAGE(5,"GetSegmentOffset()");
		ASSERT_GX( s.segment*GetSegmentSize(s)<s.Np );
		return s.segment*GetSegmentSize(s);
	}

	unsigned int GetSegmentEnd(const struct state_gx& s)
	{
		FUN_MESSAGE(5,"GetSegmentEnd()");
		return min(s.Np,GetSegmentOffset(s) + GetSegmentSize(s));
	}
#else
	#ifndef CUDA_GX_SEGMENTED
		extern struct state_gx s_gx_cuda;
		void SetSegments()
		{
			FUN_MESSAGE(5,"SetSegments()");
			s_gx_cuda.sz_segments=1;
		}
	#endif
#endif

#ifdef CUDA_GX_CHUNCK_MANAGER
	#ifndef 
		ERROR: cannot define CUDA_GX_CHUNCK_MANAGER without CUDA_GX_SEGMENTED
	#endif

void SyncChunckData(const struct state_gx& s_gx,const int sphmode)
{
	FUN_MESSAGE(3,"SyncChunckData()");
	ASSERT_GX( s_gx.external_node>=0 && s_gx_cuda.external_node>=0 );

	InitializeCalculation_cuda_gx(s_gx,sphmode);

	size_t t;

	ASSERT_GX(s_gx.Nodes_base!=NULL && s_gx.sz_Nodes_base>0 );
	ALLOCATE_DATA_WITH_HEADROOM(Nodes_base,struct NODE_gx,"NODE_gx data");
	Memcpy_cuda_gx(s_gx_cuda.Nodes_base,s_gx.Nodes_base,t,0);
	s_gx_cuda.Nodes   =s_gx_cuda.Nodes_base-s_gx.MaxPart; // reassign with corret base

	ASSERT_GX(s_gx.Nextnode!=NULL && s_gx.sz_Nextnode>0 );
	ALLOCATE_DATA(Nextnode,int,"Nextnode data");
	Memcpy_cuda_gx(s_gx_cuda.Nextnode,s_gx.Nextnode,t,0);

	ASSERT_GX(s_gx.DomainTask!=NULL && s_gx.sz_DomainTask>0 );
	ALLOCATE_DATA(DomainTask,int,"DomainTask data");
	Memcpy_cuda_gx(s_gx_cuda.DomainTask,s_gx.DomainTask,t,0);

	if (sphmode) {

		ASSERT_GX(s_gx.extNodes_base!=NULL && s_gx.sz_extNodes_base>0 );
		ALLOCATE_DATA_WITH_HEADROOM(extNodes_base,struct extNODE_gx,"extNodes_base data");
		Memcpy_cuda_gx(s_gx_cuda.extNodes_base,s_gx.extNodes_base,t,0);

		ASSERT_GX(s_gx.Ngblist!=NULL && s_gx.sz_Ngblist>0 );
		ALLOCATE_DATA_WITH_HEADROOM(Ngblist,int,"Ngblist data");

		s_gx_cuda.extNodes=s_gx_cuda.extNodes_base-s_gx.MaxPart; // reassign with corret base
	}

	SyncStaticData(s_gx);

	s_gx_cuda.external_node=s_gx.external_node;
	ASSERT_GX(s_gx_cuda.external_node>=0);
}

void SetSegments()
{
	FUN_MESSAGE(5,"SetSegments()");

	#ifdef CUDA_GX_SEGMENTED
		const unsigned int sz=GetSegmentSize(s_gx_cuda);
		s_gx_cuda.sz_segments=s_gx_cuda.Np/sz + (s_gx_cuda.Np%sz==0 ? 0 : 1);
		ASSERT_GX( s_gx_cuda.Np<=s_gx_cuda.sz_segments*sz );
	#else
		s_gx_cuda.sz_segments=1;
	#endif
}


//#define CHUNCK_TIMERS

	#ifdef CHUNCK_TIMERS
		#define GET_TIME(tx)\
			tx += GetTime() - t;\
			t = GetTime()
		#define SET_TIME t = GetTime()
	#else
		#define GET_TIME(tx)
		#define SET_TIME
	#endif

	#if CUDA_DEBUG_GX > 0
		void ValidateSegments_fun(const char*const calcsegments,const int final,const char*const file,const int line)
		{
			unsigned int j;
			if (final==0){
				if (!( s_gx_cuda.segment<s_gx_cuda.sz_segments && calcsegments[s_gx_cuda.segment]==final )){
					MESSAGE("ValidateSegments failed: s_gx_cuda.segment=%d, s_gx_cuda.sz_segments=%d, calcsegments[s_gx_cuda.segment]=%d, at %s:%d",s_gx_cuda.segment,s_gx_cuda.sz_segments,calcsegments[s_gx_cuda.segment],file,line);
					for(j=0;j<s_gx_cuda.segment;++j) MESSAGE("  calcsegments[%d]=%d",j,calcsegments[j]);
				}
			}

			if (final==0)       ASSERT_GX( s_gx_cuda.segment<s_gx_cuda.sz_segments && calcsegments[s_gx_cuda.segment]==0 );
			else if (final==2)  ASSERT_GX( s_gx_cuda.segment==s_gx_cuda.sz_segments || s_gx_cuda.segment<s_gx_cuda.sz_segments && calcsegments[s_gx_cuda.segment]==0 );
			else if (final==1)  {
				ASSERT_GX( s_gx_cuda.segment==s_gx_cuda.sz_segments );
				for(j=0;j<s_gx_cuda.sz_segments;++j) ASSERT_GX( calcsegments[j]>=1 && calcsegments[j]<=3 );
			}
		}
		#define ValidateSegments(calcsegments,final) ValidateSegments_fun(calcsegments,final,__FILE__,__LINE__)
	#else
		#define ValidateSegments(calcsegments,final)
	#endif


int SendSegmentToAvailNodes(struct state_gx s,const struct parameters_gx& p,const struct parameters_hydro_gx& h,int*const availnodes,int& repack,char* calcsegments,const int minleft)
{
	#ifdef CHUNCK_TIMERS
		double t0=GetTime(),t1=0,t2=0,t;
	#endif

	FUN_MESSAGE(2,"SendSegmentToAvailNodes()");
	ASSERT_GX( repack==0 || repack==1 );

	if (s_gx_cuda.segment+minleft<s_gx_cuda.sz_segments){
		#ifdef CHUNCK_TIMERS
			t = GetTime();
		#endif
		const int a=RecvChunckAvailNodes(CHUNK_MANAGER_ALLOCATE_NODES,availnodes);
		ASSERT_GX(a<CHUNK_MANAGER_SIZE_GX);
		#ifdef CHUNCK_TIMERS
			t1 += GetTime()-t;
		#endif

		for(int i=0;i<a;++i){
			#ifdef CHUNCK_TIMERS
				t = GetTime();
			#endif
			ASSERT_GX( availnodes[i]>=0 );

			s.segment      =s_gx_cuda.segment;
			s.sz_segments  =s_gx_cuda.sz_segments;
			s.iteration    =-1;

			SendKernelDataToNode(availnodes[i],&s,&p,&h,repack);

			repack=0;

			#if CUDA_DEBUG_GX > 0
				ValidateSegments(calcsegments,0);
				calcsegments[s_gx_cuda.segment]=2;
			#endif

			++s_gx_cuda.segment;

			ValidateSegments(calcsegments,0);

			#ifdef CHUNCK_TIMERS
				t2 += GetTime()-t;
			#endif
		}
		#ifdef CHUNCK_TIMERS
			t0 = GetTime() -t0;
			if(t0>1E-3) MESSAGE("SendSegmentToAvailNodes(): timing: %4.3f sec,%3.1f,%3.1f",t0,t1/t0*100,t2/t0*100);
		#endif

		return a;
	}

	#ifdef CHUNCK_TIMERS
		t0 = GetTime() -t0;
		if(t0>1E-3) MESSAGE("SendSegmentToAvailNodes(): timing: %4.3f sec,%3.1f,%3.1f",t0,t1/t0*100,t2/t0*100);
	#endif
	return 0;
}

int RecvSegmentsFromNodes(const int a,int*const assignednodes,const struct state_gx*const s,const int sphmode,char* calcsegments,const int minleft)
{
	#ifdef CHUNCK_TIMERS
		double t0=GetTime(),t1=0,t2=0,t;
	#endif

	FUN_MESSAGE(2,"RecvSegmentsFromNodes()");
	ASSERT_GX(a>=0 && a<CHUNK_MANAGER_SIZE_GX);
	ASSERT_GX( s->sphmode==sphmode );
	int j;

	for(j=0;j<a;++j) ASSERT_GX(assignednodes[j]>=0);
	int m=a;

	while(PendingResults()) {
		#ifdef CHUNCK_TIMERS
			t = GetTime();
		#endif
		int relaunch=-1;

		if (s_gx_cuda.segment+minleft<s_gx_cuda.sz_segments) {
			#if CUDA_DEBUG_GX > 0
				ASSERT_GX( s_gx_cuda.segment<s_gx_cuda.sz_segments && calcsegments[s_gx_cuda.segment]==0 );
				calcsegments[s_gx_cuda.segment]=3;
			#else
				ASSERT_GX( calcsegments==NULL );
			#endif

			relaunch=++s_gx_cuda.segment;

			#if CUDA_DEBUG_GX > 0
				ASSERT_GX( s_gx_cuda.segment<s_gx_cuda.sz_segments && calcsegments[s_gx_cuda.segment]==0 );
			#endif
		}

		const int node=RecvKernelResultFromNode(s,sphmode,relaunch);
		#ifdef CHUNCK_TIMERS
			t1 += GetTime()-t;
			t = GetTime();
		#endif

		//MESSAGE( "RecvSegmentsFromNodes: %d<-|%d",s->ThisTask,node);
		//if (relaunch>=0) MESSAGE("RecvSegmentsFromNodes: relaunch=%d, sz=%d, %d|->%d",relaunch,s_gx_cuda.sz_segments,s_gx_cuda.ThisTask,node);

		int n=-1;
		for(j=0;j<a;++j) if (assignednodes[j]==node) {ASSERT_GX(n==-1); n=j; break;}
		ASSERT_GX( n>=0 && assignednodes[n]==node );

		if (relaunch<0) {
			assignednodes[n]=-1;
			for(int j=n;j<a-1;++j) assignednodes[j]=assignednodes[j+1];
			--m;
		}
		#ifdef CHUNCK_TIMERS
			t2 += GetTime()-t;
		#endif
	}

	ASSERT_GX( m<=a );
	for(j=0;j<m;++j) ASSERT_GX((j<m && assignednodes[j]>=0) || (j>=m && assignednodes[j]==-1) );

	#ifdef CHUNCK_TIMERS
		t0 = GetTime()-t0;
		if (m<a && t0>1E-3) MESSAGE("RecvSegmentsFromNodes(): timing: %4.3f sec,%3.1f,%3.1f",t0,t1/t0*100,t2/t0*100);
	#endif
	return m;
}

void ManageChuncks(const int sphmode)
{
	#ifdef CHUNCK_TIMERS
		double t0=GetTime(),t1=0,t2=0,t3=0,t4=0,t;
	#endif

	FUN_MESSAGE(2,"ManageChuncks(sphmode=%d)",sphmode);
	ASSERT_GX( sphmode==0 || sphmode==1 );

	#ifdef CHUNCK_TIMERS
		t = GetTime();
	#endif

	SendChunckDone(0);

	#ifdef CHUNCK_TIMERS
		t1 = GetTime()-t;
		t = GetTime();
	#endif
	int n=WaitAllChunkNodesDone();

	#ifdef CHUNCK_TIMERS
		t4 = GetTime()-t;
	#endif

	while(n!=-2){
		if(n>=0){
			#ifdef CHUNCK_TIMERS
				t = GetTime();
			#endif
			struct state_gx* s=NULL;
			struct parameters_gx* p=NULL;
			struct parameters_hydro_gx* h=NULL;

			unsigned int sz;

			const char*const v=RecvKernelDataFromNode(n,&s,&p,&h,&sz);
			ASSERT_GX( n==s->external_node );
			s_gx_cuda.external_node=s->external_node;


			#ifndef CUDA_GX_USE_TEXTURES
				static char* c=NULL;
				static unsigned int sz_c=0;

				if (sz>sz_c){
					sz_c=sz+CHUNK_MANAGER_MALLOC_EXTRA_GX;
					if (c!=NULL) Free_cuda_gx((void**)&c);
					c=(char*)Malloc_cuda_gx(sz_c+CHUNK_MANAGER_ALIGN_GX,"ChunckManager kernel data",__FILE__,__LINE__);
				}

				struct state_gx s_cuda;
				struct parameters_gx p_cuda;
				struct parameters_hydro_gx h_cuda;
				char* ca=AdjustKernelData(v,c,&s_cuda,&p_cuda,&h_cuda,sz);


				Memcpy_cuda_gx(ca,v,sz,0);

// if (sphmode){
// 	const unsigned int offset=GetSegmentOffset(s_cuda);
// 	const unsigned int rsz=GetSegmentEnd(s_cuda)-offset;
//
// 	MESSAGE("sph chunck 0: s_cuda.segment=%d, s_cuda.sz_segments=%d, offset=%d, rsz=%d, s_cuda.sz_result_hydro=%d",s_cuda.segment,s_cuda.sz_segments,offset,rsz,s_cuda.sz_result_hydro);
// 	for(unsigned int j=offset;j<rsz;++j) {
// 		ASSERT_GX( j<s_cuda.sz_result_hydro );
// 		struct result_hydro_gx r=s->result_hydro[j];
// 		MESSAGE("  [%d] real=%d, acc={%f,%f,%f}, DtE=%f, MSVel=%f",j,r.realtarget,r.Acc[0],r.Acc[1],r.Acc[2],r.DtEntropy,r.MaxSignalVel);
// 	}
// }

				s_cuda.kernelsignals=s_gx_cuda.kernelsignals;
				s_cuda.debug_msg    =s_gx_cuda.debug_msg=NULL;
				s_cuda.debug_sz_msg =s_gx_cuda.debug_sz_msg=0;

				double time_kernel=0;
				while (s_cuda.segment<s_cuda.sz_segments){
					ASSERT_GX(s_cuda.segment==s->segment );
					ASSERT_GX(s_cuda.sz_segments==s->sz_segments);

					MESSAGE("ManageChuncks: seg=%d, segs=%d, %d<-|%d",s_cuda.segment,s_cuda.sz_segments,s_gx_cuda.ThisTask,n);

					ASSERT_GX( sphmode==s_cuda.sphmode );
					time_kernel += CallKernel(s_cuda,*s,p_cuda,sphmode==1 ? &h_cuda : NULL,1,sphmode);

					const unsigned int offset=GetSegmentOffset(s_cuda);
					const unsigned int rsz=GetSegmentEnd(s_cuda)-offset;

/*if (sphmode){
	const unsigned int offset=GetSegmentOffset(s_cuda);
	const unsigned int rsz=GetSegmentEnd(s_cuda)-offset;

	MESSAGE("sph chunck 1: s_cuda.segment=%d, s_cuda.sz_segments=%d, offset=%d, rsz=%d, s_cuda.sz_result_hydro=%d",s_cuda.segment,s_cuda.sz_segments,offset,rsz,s_cuda.sz_result_hydro);
	for(unsigned int j=offset;j<rsz;++j) {
		ASSERT_GX( j<s_cuda.sz_result_hydro );
		struct result_hydro_gx r=s->result_hydro[j];
		MESSAGE("  [%d] real=%d, acc={%f,%f,%f}, DtE=%f, MSVel=%f",j,r.realtarget,r.Acc[0],r.Acc[1],r.Acc[2],r.DtEntropy,r.MaxSignalVel);
	}
}*/

					s_cuda.segment=SendKernelResultToNode(n,s,rsz,offset,sphmode);
					if (s_cuda.segment<s_cuda.sz_segments)  {
						s->segment=s_cuda.segment;
						//MESSAGE("relauncing: s_cuda.segment=%d, s_cuda.sz_segments=%d, %d<-|%d",s_cuda.segment,s_cuda.sz_segments,s_gx_cuda.ThisTask,n);
					}

				}
			#else
				ASSERT_GX( s!=NULL && p!=NULL );

				SyncChunckData(*s,sphmode);

				double time_kernel=0;
				while (s_gx_cuda.segment<s_gx_cuda.sz_segments){
					ASSERT_GX(s_gx_cuda.segment==s->segment );
					ASSERT_GX(s_gx_cuda.sz_segments==s->sz_segments);

					ASSERT_GX( sphmode==s_gx_cuda.sphmode );
					time_kernel += CallKernel(s_gx_cuda,*s,*p,sphmode==1 ? h : NULL,1,sphmode);

					const unsigned int offset=GetSegmentOffset(s_gx_cuda);
					const unsigned int rsz=GetSegmentEnd(s_gx_cuda)-offset;

					s_gx_cuda.segment=SendKernelResultToNode(n,s,rsz,offset,sphmode);
					if (s_gx_cuda.segment<s_gx_cuda.sz_segments)  {
						s->segment=s_gx_cuda.segment;
						//MESSAGE("relauncing: s_cuda.segment=%d, s_cuda.sz_segments=%d, %d<-|%d",s_cuda.segment,s_cuda.sz_segments,s_gx_cuda.ThisTask,n);
					}
				}
			#endif

			#ifdef CHUNCK_TIMERS
				t2 += GetTime()-t;
				t = GetTime();
			#endif

			SendChunckDone(time_kernel);

			#ifdef CHUNCK_TIMERS
				t3 += GetTime()-t;
			#endif
		}
		#ifdef CHUNCK_TIMERS
			t = GetTime();
		#endif

		n=WaitAllChunkNodesDone();

		#ifdef CHUNCK_TIMERS
			t4 += GetTime()-t;
		#endif
	}
	ASSERT_GX( PendingMessages()==0 );
	#ifdef CHUNCK_TIMERS
		t0 = GetTime()-t0;
		if (t0>1E-3) MESSAGE("ManageChuncks(): timing: %4.3f sec,%3.1f,%3.1f,%3.1f,%3.1f",t0,t1/t0*100,t2/t0*100,t3/t0*100,t4/t0*100);
	#endif
	MESSAGE("ManageChuncks(): done");
}

void force_treeevaluate_shortrange_range_cuda_gx(const int mode,const unsigned int Np,const struct state_gx s,const struct parameters_gx p)
{
	FUN_MESSAGE(1,"force_treeevaluate_shortrange_range_cuda_gx range=%d cudamode=[%d;%d;%d]",Np,s.cudamode,s.debugval,CUDA_DEBUG_GX);

	ASSERT_GX( mode==0 || mode==1);
	ASSERT_GX( s.cudamode>=1 || s.cudamode<=3 );
	ASSERT_GX( s.mode==mode );
	ASSERT_GX( s.NumPart==s.sz_P);
	ASSERT_GX( Np>0 && Np==s.Np );
	ASSERT_GX( mode==1 || (Np<=s.MaxPart && Np<=s.sz_P) );
	ASSERT_GX( mode==0 || Np==s.sz_result );
	ASSERT_GX( s_gx_cuda.sz_result=s.sz_result );

	ValidateState(s,p.MaxNodes,1,__FILE__,__LINE__);

	if (mode==0) for (unsigned int i=0;i<s.sz_Exportflag;++i) ASSERT_GX( s.Exportflag[i]==0 );

	if (s.cudamode==1){
		// call host-mode code, that is Cuda compiled C seriel code
		ERROR("force_treeevaluate_shortrange_cuda_gx_host not included");
	 } else {
		#ifdef CHUNCK_TIMERS
			double t0=GetTime(),t1=0,t2=0,t3=0,t4=0,t5=0,t6=0,t=GetTime();
		#endif

		// call device-mode code, that is Cuda compiled C parallel code
		ASSERT_GX( s_gx_cuda.NumPart==s.NumPart );
		ASSERT_GX( Np<=s_gx_cuda.NumPart || mode==1);
		ASSERT_GX( s_gx_cuda.sz_result==s.sz_result && Np==s_gx_cuda.sz_result );
		//ASSERT_GX( (mode==0 && s_gx_cuda.sz_result==s.sz_result) || (mode==1 && s_gx_cuda.sz_result>=s.sz_result) );

		if (mode==1){
			// copy new aux data to GPU
			ASSERT_GX( s_gx_cuda.sz_result==s.sz_result );
			const size_t t=sizeof(struct result_gx)*Np;
			Memcpy_cuda_gx(s_gx_cuda.result,s.result,t,0);
		}

		SyncRunTimeData(s,1);
		s_gx_cuda.kernelsignals=Initialize_KernelSignals();

		// create a kernel
		INIT_DEBUG_DATA;
		SetSegments();
		ReLaunchChunkManager();

		ASSERT_GX( s_gx_cuda.sz_segments>=1 );
		#ifdef CUDA_GX_CHUNCK_MANAGER
			unsigned int j,m=0;
			int assignednodes[CHUNK_MANAGER_SIZE_GX],availnodes[CHUNK_MANAGER_SIZE_GX];
			for(j=0;j<CHUNK_MANAGER_SIZE_GX;++j) assignednodes[j]=-1;
			#if CUDA_DEBUG_GX > 0
				char* calcsegments=(char*)malloc(s_gx_cuda.sz_segments);
				for(j=0;j<s_gx_cuda.sz_segments;++j) calcsegments[j]=0;
			#else
				char* calcsegments=NULL;
			#endif
			int repack=1;
		#endif

		GET_TIME(t1);

		s_gx_cuda.segment=0;
		while(s_gx_cuda.segment<s_gx_cuda.sz_segments){
			SET_TIME;

			ValidateSegments(calcsegments,0);

			#ifdef CUDA_GX_CHUNCK_MANAGER
				unsigned int n=0;
				if (m<CHUNK_MANAGER_MAX_ALLOC_NODES) {
					struct parameters_hydro_gx h_dummy;
					n=SendSegmentToAvailNodes(s,p,h_dummy,availnodes,repack,calcsegments,CHUNK_MANAGER_MIN_SEG_LEFT_FORCE);
					for(j=0;j<n;++j) assignednodes[m++]=availnodes[j];
				}
				GET_TIME(t2);

				ValidateSegments(calcsegments,0);

				//if (m<=4){
			#endif
					ASSERT_GX( 0==s_gx_cuda.sphmode );

					CallKernel(s_gx_cuda,s,p,NULL,0,0);

					GetTime();

					#ifdef CUDA_GX_CHUNCK_MANAGER
						#if CUDA_DEBUG_GX > 0
							ValidateSegments(calcsegments,0);
							calcsegments[s_gx_cuda.segment]=1;
						#endif
					#endif

					++s_gx_cuda.segment;

			#ifdef CUDA_GX_CHUNCK_MANAGER
					ValidateSegments(calcsegments,2);

					//} else {
					//	//MESSAGE("redistrib: m=%d",m);
					//}
				//}

				SET_TIME;
				ASSERT_GX( m>=n );
				for(j=0;j<m;++j) ASSERT_GX(assignednodes[j]>=0);

				m=RecvSegmentsFromNodes(m,assignednodes,&s,0,calcsegments,CHUNK_MANAGER_MIN_SEG_LEFT_FORCE);

				for(j=0;j<m;++j) ASSERT_GX(assignednodes[j]>=0);

				GET_TIME(t4);

			#endif

			ValidateSegments(calcsegments,2);
			//if (s_gx_cuda.ThisTask==1) usleep(1000000);
		}

		#ifdef CUDA_GX_CHUNCK_MANAGER
			SET_TIME;

			while(m>0) m=RecvSegmentsFromNodes(m,assignednodes,&s,0,calcsegments,CHUNK_MANAGER_MIN_SEG_LEFT_FORCE);

			#if CUDA_DEBUG_GX > 0
				ValidateSegments(calcsegments,1);
				free(calcsegments);
			#endif

			GET_TIME(t5);

			ManageChuncks(0);

			GET_TIME(t6);
			#ifdef CHUNCK_TIMERS
				t0 = GetTime()-t0;
				if (t0>1E-3) MESSAGE("force_eval...(): timing: %4.3f sec, kernel: %4.3f\n##    init  =%3.1f,\n##    send  =%3.1f,\n##    kernel=%3.1f,\n##    recv1 =%3.1f,\n##    recv2 =%3.1f,\n##    manage=%3.1f",t0,t3,t1/t0*100,t2/t0*100,t3/t0*100,t4/t0*100,t5/t0*100,t6/t0*100);
			#endif
		#endif
	}
}

void hydro_evaluate_range_cuda_gx(const int mode,const unsigned int N_gas,const struct state_gx s,const struct parameters_gx p,const struct parameters_hydro_gx h)
{
	FUN_MESSAGE(1,"hydro_evaluate_range_cuda_gx range=%d cudamode=[%d;%d;%d]",N_gas,s.cudamode,s.debugval,CUDA_DEBUG_GX);

	ASSERT_GX( mode==0 || mode==1);
	ASSERT_GX( s.cudamode>=1 || s.cudamode<=3 );
	ASSERT_GX( s.mode==mode );
	ASSERT_GX( N_gas<=s.MaxPart && N_gas>0 && s.N_gas==s.sz_SphP);
	ASSERT_GX( mode==1 || (N_gas==s.N_gas && N_gas==s.sz_SphP));
	ASSERT_GX( s_gx_cuda.sz_result=s.sz_result );

	ValidateState(s,p.MaxNodes,1,__FILE__,__LINE__);

	if (mode==0) for (unsigned int i=0;i<s.sz_Exportflag;++i) ASSERT_GX( s.Exportflag[i]==0 );

	if (s.cudamode==1){
		// call host-mode code, that is Cuda compiled C seriel code
		ERROR("hydro_evaluate_shortrange_cuda_gx_host not included");
	 } else {
		#ifdef CHUNCK_TIMERS
			double t0=GetTime(),t1=0,t2=0,t3=0,t4=0,t5=0,t6=0,t=GetTime();
		#endif
		// call device-mode code, that is Cuda compiled C parallel code

		SyncRunTimeData(s,1);
		s_gx_cuda.kernelsignals=Initialize_KernelSignals();

		// create a kernel
		INIT_DEBUG_DATA;
		SetSegments();
		#ifdef CUDA_GX_CHUNCK_MANAGER_SPH
			ReLaunchChunkManager();
		#endif

//MESSAGE("SPH segments...s_gx_cuda.segment=%d, s_gx_cuda.sz_segments=%d",s_gx_cuda.segment,s_gx_cuda.sz_segments);

		ASSERT_GX( s_gx_cuda.sz_segments>=1 );
		#ifdef CUDA_GX_CHUNCK_MANAGER_SPH
			unsigned int j,m=0;
			int assignednodes[CHUNK_MANAGER_SIZE_GX],availnodes[CHUNK_MANAGER_SIZE_GX];
			for(j=0;j<CHUNK_MANAGER_SIZE_GX;++j) assignednodes[j]=-1;
			#if CUDA_DEBUG_GX > 0
				char* calcsegments=(char*)malloc(s_gx_cuda.sz_segments);
				for(j=0;j<s_gx_cuda.sz_segments;++j) calcsegments[j]=0;
			#else
				char* calcsegments=NULL;
			#endif
			int repack=1;
		#endif

		GET_TIME(t1);

		s_gx_cuda.segment=0;
		while(s_gx_cuda.segment<s_gx_cuda.sz_segments){

			SET_TIME;

			ASSERT_GX( N_gas==s_gx_cuda.N_gas );
			ASSERT_GX( s_gx_cuda.Np==s_gx_cuda.N_gas );

			#ifdef CUDA_GX_CHUNCK_MANAGER_SPH
				ValidateSegments(calcsegments,0);

				unsigned int n=0;
				if (m<CHUNK_MANAGER_MAX_ALLOC_NODES) {
					//n=SendSegmentToAvailNodes(s,p,h,availnodes,repack,calcsegments,CHUNK_MANAGER_MIN_SEG_LEFT_SPH);
					//for(j=0;j<n;++j) assignednodes[m++]=availnodes[j];
				}

				ASSERT_GX( s_gx_cuda.segment<s_gx_cuda.sz_segments );
				ValidateSegments(calcsegments,0);
			#endif

			GET_TIME(t2);

			ASSERT_GX( 1==s_gx_cuda.sphmode );

			CallKernel(s_gx_cuda,s,p,&h,0,1);

			GET_TIME(t3)

			#ifdef CUDA_GX_CHUNCK_MANAGER_SPH
				#if CUDA_DEBUG_GX > 0
					ValidateSegments(calcsegments,0);
					calcsegments[s_gx_cuda.segment]=1;
				#endif
			#endif

			++s_gx_cuda.segment;

			#ifdef CUDA_GX_CHUNCK_MANAGER_SPH
				ValidateSegments(calcsegments,2);

				SET_TIME;

				ASSERT_GX( m>=n );
				for(j=0;j<m;++j) ASSERT_GX(assignednodes[j]>=0);

				m=RecvSegmentsFromNodes(m,assignednodes,&s,1,calcsegments,CHUNK_MANAGER_MIN_SEG_LEFT_SPH);

				for(j=0;j<m;++j) ASSERT_GX(assignednodes[j]>=0);

				GET_TIME(t4);

				ValidateSegments(calcsegments,2);
				//if (s_gx_cuda.ThisTask==7) usleep(1000000);
			#endif
		}

		#ifdef CUDA_GX_CHUNCK_MANAGER_SPH
			SET_TIME;

			while(m>0) m=RecvSegmentsFromNodes(m,assignednodes,&s,1,calcsegments,CHUNK_MANAGER_MIN_SEG_LEFT_SPH);

			GET_TIME(t5);

			#if CUDA_DEBUG_GX > 0
				ValidateSegments(calcsegments,1);
				free(calcsegments);
			#endif

			ManageChuncks(1);

			#ifdef CHUNCK_TIMERS
				GET_TIME(t6);
				t0 = GetTime()-t0;
				if (t0>1E-3) MESSAGE("hydro_eval...(): timing: %4.3f sec, kernel: %4.3f\n##    init  =%3.1f,\n##    send  =%3.1f,\n##    kernel=%3.1f,\n##    recv1 =%3.1f,\n##    recv2 =%3.1f,\n##    manage=%3.1f",t0,t3,t1/t0*100,t2/t0*100,t3/t0*100,t4/t0*100,t5/t0*100,t6/t0*100);
			#endif
		#endif
	}
}

#endif
