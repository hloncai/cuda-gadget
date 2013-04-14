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
#include <unistd.h> // for usleep
#include <pthread.h>
#include <mpi.h>

#include "allvars.h"
#include "interface_gx.h"
#include "chunckmanager_gx.h"
#include "debug_gx.h"

#ifdef CUDA_GX_CHUNCK_MANAGER

#define TAG_MESG_GX          100000
#define TAG_ALL_DONE_GX      100001
#define TAG_AVAIL_GX         100002
#define TAG_NEW_JOB_GX       100003
#define TAG_KERNEL_DATA_GX   100004
#define TAG_KERNEL_RESULT_GX 100005
#define TAG_TIME_GX          100006

#define CHUNK_MANAGER_DSIZE_GX 32
//#define CHUNCK_MANAGER_COMM_GX  MPI_COMM_WORLD
static MPI_Comm CHUNCK_MANAGER_COMM_GX;

#define CHECKMPICALL_GX(arg)   CheckMPI(arg,__STRING(arg),__FILE__,__LINE__)

static volatile unsigned int g_barrier=1;
static const int g_dbg=-1;

static pthread_mutex_t g_mpi_sync = PTHREAD_MUTEX_INITIALIZER;
static int             g_mpi_sync_n = 0;

/*

static pthread_mutex_t g_mpi_sync_x[4] = {PTHREAD_MUTEX_INITIALIZER,PTHREAD_MUTEX_INITIALIZER,PTHREAD_MUTEX_INITIALIZER,PTHREAD_MUTEX_INITIALIZER};
static int             g_mpi_sync_n_x[4] = {0,0,0,0};

void Lock_x(const int n)
{
	FUN_MESSAGE(4,"Lock_x()");
	ASSERT_GX(n>=0 && n<4);
	if (0!=pthread_mutex_lock(&g_mpi_sync_x[n])) ERROR("pthread_mutex_lock failed");
	ASSERT_GX( g_mpi_sync_n_x[n]==0 );
	++g_mpi_sync_n_x[n];
}

void Unlock_x(const int n)
{
	FUN_MESSAGE(4,"Unlock_x()");
	ASSERT_GX(n>=0 && n<4);
	ASSERT_GX( g_mpi_sync_n_x[n]==1 );
	--g_mpi_sync_n_x[n];
	if (0!=pthread_mutex_unlock(&g_mpi_sync_x[n])) ERROR("pthread_mutex_lock failed");
}
*/

void Lock()
{
	FUN_MESSAGE(4,"Lock()");
	if (0!=pthread_mutex_lock(&g_mpi_sync)) ERROR("pthread_mutex_lock failed");
	ASSERT_GX( g_mpi_sync_n==0 );
	++g_mpi_sync_n;
}

void Unlock()
{
	FUN_MESSAGE(4,"Unlock()");
	ASSERT_GX( g_mpi_sync_n==1 );
	--g_mpi_sync_n;
	if (0!=pthread_mutex_unlock(&g_mpi_sync)) ERROR("pthread_mutex_lock failed");
}

void CheckMPI(const int e,const char*const msg,const char*const file,const int line)
{
	FUN_MESSAGE(5,"Unlock()");
	if (e!=MPI_SUCCESS) ERROR("MPI error encounted, error=%d, %s:%d: %s",e,file,line,msg);
}

int Pending(const int src,const int tag)
{
	FUN_MESSAGE(4,"PendingMessages()");

	int r;
	MPI_Status s;

	CHECKMPICALL_GX( MPI_Iprobe(src,tag,CHUNCK_MANAGER_COMM_GX,&r,&s) );
	return r;
}

int PendingResults()
{
	FUN_MESSAGE(4,"PendingResults()");
	if (ThisTask==0) return 0;

	return Pending(MPI_ANY_SOURCE,TAG_KERNEL_RESULT_GX);
}

int PendingMessages()
{
	FUN_MESSAGE(4,"PendingMessages()");
	return Pending(MPI_ANY_SOURCE,MPI_ANY_TAG);
}

void ChunckManager_dump(const int ndone,const char*const nodes,const int*const allocnodes)
{
	FUN_MESSAGE(4,"ChunckManager_dump()");
	if (g_dbg<1) return;

	int i,n;
	char t[1024];

	MESSAGE("chunck: dump: sz=%d, ndone=%d",NTask,ndone);
	for(i=0;i<NTask;++i) MESSAGE("chunck: dump: %d=(%c,%d)",i,nodes[i]+'0',allocnodes[i]);
}

void SendChunckDone(double time_kernel)
{
	FUN_MESSAGE(2,"SendChunckDone()");

	int r=ThisTask+1;

	if (g_dbg>2) MESSAGE("chunck: SendChunckDone(), sending r=%d, time_kernel=%f",r,time_kernel);
	if (ThisTask!=0) {
		CHECKMPICALL_GX( MPI_Bsend(&r          ,1,MPI_INT   ,0,TAG_MESG_GX,CHUNCK_MANAGER_COMM_GX) );
		CHECKMPICALL_GX( MPI_Bsend(&time_kernel,1,MPI_DOUBLE,0,TAG_TIME_GX,CHUNCK_MANAGER_COMM_GX) );
	}
}

int WaitAllChunkNodesDone()
{
	FUN_MESSAGE(2,"WaitAllChunkNodesDone()");
	if (g_dbg>5) MESSAGE("chunck: WaitAllChunkNodesDone()");

	if (ThisTask!=0)
	{
		int r;
		MPI_Status s;

		if (g_dbg>2) MESSAGE("chunck: peeking done");
		CHECKMPICALL_GX( MPI_Iprobe(0,TAG_ALL_DONE_GX,CHUNCK_MANAGER_COMM_GX,&r,&s) );

		if (r) {
			CHECKMPICALL_GX( MPI_Recv(&r,1,MPI_INT,0,TAG_ALL_DONE_GX,CHUNCK_MANAGER_COMM_GX,&s) );
			ASSERT_GX( r==ThisTask );
			if (g_dbg>2) MESSAGE("chunck: WaitAllChunkNodesDone()...done");

			return -2;
		} else  {
			if (g_dbg>1) MESSAGE("chunck: peeking job");
			CHECKMPICALL_GX( MPI_Iprobe(MPI_ANY_SOURCE,TAG_NEW_JOB_GX,CHUNCK_MANAGER_COMM_GX,&r,&s) );

			if (r){
				CHECKMPICALL_GX( MPI_Recv(&r,1,MPI_INT,MPI_ANY_SOURCE,TAG_NEW_JOB_GX,CHUNCK_MANAGER_COMM_GX,&s) );
				ASSERT_GX( r>=0 && r!=ThisTask);
				if (g_dbg>1) MESSAGE("chunck: peeked node=%d",r);

				return r;
			}
		}

		usleep(CHUNK_MANAGER_SLEEP_GX);
		if (g_dbg>0) MESSAGE("chunck: WaitAllChunkNodesDone()...waiting");

		return -1;
	} else {
		if (g_dbg>0) MESSAGE("chunck: WaitAllChunkNodesDone()...waiting");
		while(g_barrier%2==0){
			usleep(10*CHUNK_MANAGER_SLEEP_GX);
			if (g_dbg>2) MESSAGE("chunck: sleep");
		}
		if (g_dbg>0) MESSAGE("chunck: WaitAllChunkNodesDone()...done");

		return -2;
	}
}

int RecvChunckAvailNodes(const int nodeneeded,int*const availnodes)
{
	FUN_MESSAGE(2,"RecvChunckAvailNodes()");
	if (g_dbg>1) MESSAGE("chunck: RecvChunckAvailNodes()");

	if (ThisTask==0 || nodeneeded==0) return 0;
	ASSERT_GX( nodeneeded>0 && availnodes!=NULL );

	MPI_Status s;
	int i,r=-nodeneeded;

	CHECKMPICALL_GX( MPI_Bsend(&r,1,MPI_INT,0,TAG_MESG_GX ,CHUNCK_MANAGER_COMM_GX) );
	CHECKMPICALL_GX( MPI_Recv (&r,1,MPI_INT,0,TAG_AVAIL_GX,CHUNCK_MANAGER_COMM_GX,&s) );

	ASSERT_GX( r<CHUNK_MANAGER_SIZE_GX );
	if (r>0){
		for(i=0;i<CHUNK_MANAGER_SIZE_GX;++i) availnodes[i]=-1;
		CHECKMPICALL_GX( MPI_Recv(availnodes,r,MPI_INT,0,TAG_AVAIL_GX,CHUNCK_MANAGER_COMM_GX,&s) );
	}

	if (g_dbg>=0) MESSAGE("chunck: got %d availble nodes",r);
	for(i=0;i<r;++i) {
		if (g_dbg>=0) MESSAGE("chunck: availnodes[i]=%d",i,availnodes[i]);
		int m=ThisTask;
		CHECKMPICALL_GX( MPI_Bsend(&m,1,MPI_INT,availnodes[i],TAG_NEW_JOB_GX,CHUNCK_MANAGER_COMM_GX) );
	}

	return r;
}

int ResetChunckManager(const int init,char*const nodes,int*const allocnodes,const int relocated,const int realocated_total,int*const relocatestat,double*const timestat)
{
	FUN_MESSAGE(2,"ResetChunckManager()");
	ASSERT_GX( (init==0 || init==1) && nodes!=NULL );

	if (g_dbg>2) MESSAGE("chunck: ResetChunckManager()");
	if (NTask>CHUNK_MANAGER_SIZE_GX) ERROR("please increase chunck size in ChunkManager");

	int i;

	if (init==0) {
		double savedtime=0;
		for(i=0;i<NTask;++i) savedtime += timestat[i];
		MESSAGE("ChunckMaster: relocated=%d, total relocated=%d, total saved time=%.2f sec",relocated,realocated_total,savedtime);
		if (g_dbg>0) for(i=0;i<NTask;++i) if (relocatestat[i]>0) MESSAGE("   reallocation on node[%2d]=%3d, time=%f sec",i,relocatestat[i],timestat[i]);
	}

	for(i=0;i<NTask;++i) {
		ASSERT_GX( init || nodes[i]==1);
		nodes[i]=0;
		allocnodes[i]=-1;

		if(init) {relocatestat[i]=0; timestat[i]=0;}
	}
	nodes[0]=1;
	return 1;
}

void* ChunckManager(void* p)
{
	FUN_MESSAGE(1,"ChunckManager()");

	if (g_dbg>0) MESSAGE("chunck: starting ChunckManager...");

	int i,j,r;
	unsigned int relocated=0,relocated_total=0;
	MPI_Status s;

	char   nodes       [CHUNK_MANAGER_SIZE_GX];
	int    allocnodes  [CHUNK_MANAGER_SIZE_GX];
	int    relocatestat[CHUNK_MANAGER_SIZE_GX];
	double timestat    [CHUNK_MANAGER_SIZE_GX];

	int  ndone=ResetChunckManager(1,nodes,allocnodes,relocated,relocated_total,relocatestat,timestat);

	while(1){
		if (g_dbg>0) MESSAGE("chunck: master: receive...");
		CHECKMPICALL_GX( MPI_Recv(&r,1,MPI_INT,MPI_ANY_SOURCE,TAG_MESG_GX,CHUNCK_MANAGER_COMM_GX,&s) );

		const int src=s.MPI_SOURCE;
		if (g_dbg>0) MESSAGE("chunck: master: received message...r=%d, src=%d, ndone=%d",r,src,ndone);

		ASSERT_GX( src<NTask );
		ASSERT_GX( ndone>=0 && ndone<NTask );

		ChunckManager_dump(ndone,nodes,allocnodes);

		if(r>0){
			if (g_dbg>0) MESSAGE("chunck: master: set done: %d",src);
			ASSERT_GX( r-1==src && ((nodes[src]==0 && allocnodes[src]==-1) || (nodes[src]==2 && allocnodes[src]>=0)) );

			nodes[src]=1;
			allocnodes[src]=-1;
			++ndone;

			double time_kernel;
			CHECKMPICALL_GX( MPI_Recv(&time_kernel,1,MPI_DOUBLE,src,TAG_TIME_GX,CHUNCK_MANAGER_COMM_GX,&s) );
			timestat[src] += time_kernel;

			ChunckManager_dump(ndone,nodes,allocnodes);

			if(ndone==NTask){
				if (g_dbg>0) MESSAGE("chunck: master done");

				for(i=1;i<NTask;++i) {
					ASSERT_GX( nodes[i]==1 && allocnodes[i]==-1 );
					if (g_dbg>2) MESSAGE("chunck: master: sync to %d",i);
					CHECKMPICALL_GX( MPI_Bsend(&i,1,MPI_INT,i,TAG_ALL_DONE_GX,CHUNCK_MANAGER_COMM_GX) );
				}

				relocated_total += relocated;
				ndone=ResetChunckManager(0,nodes,allocnodes,relocated,relocated_total,relocatestat,timestat);
				relocated=0;

				ASSERT_GX( g_barrier%2==0 );
				++g_barrier;
				while(g_barrier%2==1){
					usleep(CHUNK_MANAGER_SLEEP_GX);
					if (g_dbg>2) MESSAGE("chunck: master sleep");
				}
			}
		}
		else if (r<0){
			if (g_dbg>0) MESSAGE("chunck: master: get avalible nodes");
			ASSERT_GX( -r<NTask );

			r = -r;
			int alloc[CHUNK_MANAGER_SIZE_GX];

			for(i=0,j=0;i<NTask;++i){
				alloc[i]=0;
				if (i!=0 && nodes[i]==1 && j<r) {
					ASSERT_GX( nodes[i]==1 );
					ASSERT_GX( allocnodes[i]==-1  );
					ASSERT_GX( alloc[j]==0 );
					//ASSERT_GX( nodes[i]==1 && allocnodes[i]==-1 && alloc[j]==0 );
					ASSERT_GX( src!=i );

					alloc[j++]=i;
					nodes[i]=2;
					allocnodes[i]=src;
					relocatestat[i] += 1;

					--ndone;
					++relocated;
				}
			}

			ChunckManager_dump(ndone,nodes,allocnodes);

			CHECKMPICALL_GX( MPI_Bsend(&j,1,MPI_INT,src,TAG_AVAIL_GX,CHUNCK_MANAGER_COMM_GX) );
			if (j>0) CHECKMPICALL_GX( MPI_Bsend(&alloc,j,MPI_INT,src,TAG_AVAIL_GX,CHUNCK_MANAGER_COMM_GX) );
		}
		else ERROR("bad command in ChunckManager");
	}
	return NULL;
}

void ReLaunchChunkManager()
{
	FUN_MESSAGE(1,"ReLaunchChunkManager()");

	static int launched=0;

	if (launched==0) {
		const unsigned buffersize=1024*16;
		MPI_Buffer_attach( malloc(buffersize), buffersize);

		MPI_Group orig_group;
		CHECKMPICALL_GX( MPI_Comm_group (MPI_COMM_WORLD, &orig_group) );
		CHECKMPICALL_GX( MPI_Comm_create(MPI_COMM_WORLD, orig_group, &CHUNCK_MANAGER_COMM_GX) );
	}

	const int p=PendingMessages();
	//ASSERT_GX( p==0 );
	ASSERT_GX( p==0 || (p==1 && ThisTask==0) ); // NOTE: dunno why node 0 sometimes leaves one messages

	if (launched==0) {
		launched=1;

		if (ThisTask==0) {
			pthread_t thread;
			if(0!=pthread_create(&thread,NULL,&ChunckManager,NULL)) ERROR("pthread_create() failed");
		}
	}

	if (ThisTask==0) {
		ASSERT_GX( g_barrier%2==1 );
		++g_barrier;
		ASSERT_GX( g_barrier%2==0 );
	}
}

int Align(const char*const p)
{
	FUN_MESSAGE(5,"Align()");

	ASSERT_GX(p!=NULL);

	if (CHUNK_MANAGER_ALIGN_GX<=1) return 0;

	const size_t pi=(size_t)p;
	int a=0;

	while((pi+a)%CHUNK_MANAGER_ALIGN_GX!=0) ++a;

	ASSERT_GX( a<=CHUNK_MANAGER_ALIGN_GX && (pi+a)%CHUNK_MANAGER_ALIGN_GX==0 );
	return a;
}

int AlignInt(const int p)
{
	FUN_MESSAGE(5,"AlignInt()");
	ASSERT_GX(p>0);
	return Align((char*)p);
}

void AddSize(const unsigned int s,unsigned int*const sz,unsigned int*const sza,unsigned int*const n)
{
	FUN_MESSAGE(5,"AddSize()");
	ASSERT_GX( sz!=NULL && sza!=NULL && n!=NULL && *n<1024);
	*sz += s;
	*sz += AlignInt(*sz);
	sza[*n]=*sz;
	++(*n);
}

void AddData(const unsigned int s,const void*const d,char*const v,unsigned int*const m,const unsigned int sz)
{
	FUN_MESSAGE(5,"AddData()");
	ASSERT_GX( s==0 || (s>0 && d!=NULL) );
	ASSERT_GX( v!=NULL && m!=NULL );
	ASSERT_GX( Align(&v[*m])==0 );
	ASSERT_GX( *m+s<sz );

	if (g_dbg>2) MESSAGE("chunck: AddData, %p<-%p, sz=%d, m+=%d, newp=%p",&(v[*m]),d,s,*m+s+Align(&v[*m+s]),&(v[*m+s+Align(&v[*m+s])]) );

	memcpy(&(v[*m]),d,s);
	*m += s;
	*m += Align(&v[*m]);

	ASSERT_GX( *m<=sz );
}

unsigned int AlignIntSize()
{
	FUN_MESSAGE(5,"AlignIntSize()");
	ASSERT_GX( CHUNK_MANAGER_ALIGN_GX>=sizeof(unsigned int) );
	const unsigned int uintalign=sizeof(unsigned int) + CHUNK_MANAGER_ALIGN_GX-sizeof(unsigned int);
	return uintalign;
}

char* MallocBLOB(const unsigned int sz)
{
	FUN_MESSAGE(2,"MallocBLOB()");
	ASSERT_GX( sz>0 );

	static char* v=NULL;
	static unsigned int sz_v=0;
	if (sz>=sz_v) {
		if (g_dbg>2) MESSAGE("chunck: MallocBLOB(%d)",sz);

		sz_v=sz+CHUNK_MANAGER_MALLOC_EXTRA_GX;
		if (v!=NULL) free(v);
		v=(char*)malloc(sz_v+CHUNK_MANAGER_ALIGN_GX);
		if (v==NULL) ERROR("could not malloc structure for internode communication");
	} else if (sz==0){
		sz_v=sz;
		if (v!=NULL) free(v);
		v=NULL;
	}

	ASSERT_GX( v!=NULL && sz<=sz_v );
	memset(v,0,sz);
	const unsigned m = Align(&v[0]);

	return v+m;
}

unsigned int DataBLOB_check(const char*const v,const unsigned int sz,const int checkdata,const int iscudamem)
{
	FUN_MESSAGE(2,"DataBLOB_check()");

	ASSERT_GX( sizeof(unsigned int)<= CHUNK_MANAGER_ALIGN_GX );
	ASSERT_GX( v!=NULL && (checkdata==0 || checkdata==1) && (iscudamem==0 || iscudamem==1) );

	const unsigned int uintalign=AlignIntSize();

	if (sz>0) {
		const int t0=*(int*)(v+uintalign*0);
		const int t1=*(int*)(v+uintalign*1);
		const int t2=*(int*)(v+sz-uintalign);
		const int*const sza=(int*)(v+uintalign*2);

		unsigned int i;
		if (g_dbg>1) MESSAGE("chunck: BLOB dump: uintalign=%d, t0=%d, t1=%d, t2=%d, sz=%d, align=%d, v=%u, v mod align=%d",uintalign,t0,t1,t2,sz,CHUNK_MANAGER_ALIGN_GX,(size_t)v,(size_t)v%CHUNK_MANAGER_ALIGN_GX);
		for(i=0;i<CHUNK_MANAGER_DSIZE_GX;++i) {
			if (g_dbg>1) MESSAGE("  sza[%d]=%d, elem sz=%d",i,sza[i],sza[i]==0 ? 0 : sza[i]-(i>0 ? sza[i-1] : 0));
			ASSERT_GX( i<21 || sza[i]==0 );
		}

		ASSERT_GX( Align(v)==0 );
		ASSERT_GX( t0==42 );
		ASSERT_GX( t1==sz );
		ASSERT_GX( t2==-42 );

		if (checkdata) {
			int n=1;

			struct state_gx*            s=(struct state_gx*)           (v+sza[++n]);
			struct parameters_gx*       p=(struct parameters_gx*)      (v+sza[++n]);
			struct parameters_hydro_gx* h=(struct parameters_hydro_gx*)(v+sza[++n]);

			ASSERT_GX( s->sz_P               <=sza[n+2]-sza[n] && ++n );
			ASSERT_GX( s->sz_Nodes_base      <=sza[n+2]-sza[n] && ++n );
			ASSERT_GX( s->sz_Nextnode        <=sza[n+2]-sza[n] && ++n );
			ASSERT_GX( s->sz_DomainTask      <=sza[n+2]-sza[n] && ++n );
			ASSERT_GX( s->sz_shortrange_table<=sza[n+2]-sza[n] && ++n );
			ASSERT_GX( s->sz_Exportflag      <=sza[n+2]-sza[n] && ++n );
			ASSERT_GX( s->sz_result          <=sza[n+2]-sza[n] && ++n );
			ASSERT_GX( s->sz_scratch         <=sza[n+2]-sza[n] && ++n );
			ASSERT_GX( s->sz_SphP            <=sza[n+2]-sza[n] && ++n );
			ASSERT_GX( s->sz_etc             <=sza[n+2]-sza[n] && ++n );
			ASSERT_GX( s->sz_extNodes_base   <=sza[n+2]-sza[n] && ++n );
			ASSERT_GX( s->sz_Ngblist         <=sza[n+2]-sza[n] && ++n );
			ASSERT_GX( s->sz_result_hydro    <=sza[n+2]-sza[n] && ++n );
			ASSERT_GX( s->sz_hydrodata_in    <=sza[n+2]-sza[n] && ++n );

			ValidateParameters(*p,__FILE__,__LINE__);
			//if (!iscudamem) ValidateState(*s,p->MaxNodes,1,__FILE__,__LINE__);
		}
	}

	return uintalign;
}

unsigned int DataBLOB_getsize(const char*const v)
{
	FUN_MESSAGE(2,"DataBLOB_getsize()");

	unsigned int sz=0;
	const unsigned int uintalign=DataBLOB_check(v,sz,0,0);

	int t=*(int*)(v+uintalign*0);
	if (t!=42) ERROR("mismatch in BLOB prolog");

	sz=*(int*)(v+uintalign*1);

	t=*(int*)(v+sz-uintalign);
	if (t!=-42) ERROR("mismatch in BLOB epilog");

	return sz;
}

void DataBLOB_unpack(struct state_gx** s,struct parameters_gx** p,struct parameters_hydro_gx** h,const char*const v,const unsigned int sz)
{
	FUN_MESSAGE(2,"DataBLOB_unpack()");

	ASSERT_GX( s!=NULL && *s==NULL && p!=NULL && *p==NULL && h!=NULL && *h==NULL && v!=NULL && sz>0 );

	if (g_dbg>0) MESSAGE("chunck: DataBLOB_unpack()");
	const unsigned int uintalign=DataBLOB_check(v,sz,0,0);

	int t=*(int*)(v+uintalign*0);
	if (t!=42) ERROR("mismatch in BLOB prolog");

	t=*(int*)(v+uintalign*1);
	if (t!=sz) ERROR("mismatch in BLOB sizes");

	t=*(int*)(v+sz-uintalign);
	if (t!=-42) ERROR("mismatch in BLOB epilog");

	const int*const sza=(int*)(v+uintalign*2);

	*s=(struct state_gx*)           (v+sza[2]);
	*p=(struct parameters_gx*)      (v+sza[3]);
	*h=(struct parameters_hydro_gx*)(v+sza[4]);

	(*s)->sz_max_P              =(*s)->sz_P;
	(*s)->sz_max_SphP           =(*s)->sz_SphP;
	(*s)->sz_max_Exportflag     =(*s)->sz_Exportflag;
	(*s)->sz_max_Nodes_base     =(*s)->sz_Nodes_base;
	(*s)->sz_max_result         =(*s)->sz_result;
	(*s)->sz_max_scratch        =(*s)->sz_scratch;
	(*s)->sz_max_etc            =(*s)->sz_etc;
	(*s)->sz_max_Ngblist        =(*s)->sz_Ngblist;
	(*s)->sz_max_result_hydro   =(*s)->sz_result_hydro;
	(*s)->sz_max_hydrodata_in   =(*s)->sz_hydrodata_in;
	(*s)->sz_max_Psorted        =(*s)->sz_Psorted;

	ASSERT_GX( (*s)->iteration==-1 );
}

char* DataBLOB_pack(const struct state_gx*const s,const struct parameters_gx*const p,const struct parameters_hydro_gx*const h,unsigned int* ret_sz)
{
	FUN_MESSAGE(2,"DataBLOB_pack()");

	ASSERT_GX( s!=NULL && p!=NULL );
	if (g_dbg>0) MESSAGE("chunck: DataBLOB_pack()");

	ValidateState(*s,p->MaxNodes,1,__FILE__,__LINE__);
	ASSERT_GX( ret_sz!=NULL );
	ASSERT_GX( s->iteration==-1 );

	struct state_gx c;
	memset(&c,0,sizeof(struct state_gx));

	c.ThisTask            =s->ThisTask;
	c.NTask               =s->NTask;
	c.NumPart             =s->NumPart;
	c.N_gas               =s->N_gas;
	c.Np                  =s->Np;

	c.segment             =s->segment;
	c.sz_segments         =s->sz_segments;

	c.MaxPart             =s->MaxPart;
	c.sz_memory_limit     =s->sz_memory_limit;

	c.mode                =s->mode;
	c.cudamode            =s->cudamode;
	c.debugval            =s->debugval;
	c.iteration           =s->iteration;

	// pointers
	c.sz_P                =s->sz_P;
	c.sz_Nodes_base       =s->sz_Nodes_base;
	c.sz_Nextnode         =s->sz_Nextnode;
	c.sz_DomainTask       =s->sz_DomainTask;
	c.sz_shortrange_table =s->sz_shortrange_table;
	c.sz_Exportflag       =s->sz_Exportflag;
	c.sz_result           =s->sz_result;
	c.sz_scratch          =s->sz_scratch;
	c.sz_etc              =s->sz_etc;

	if (s->sphmode==1){
		c.sz_SphP             =s->sz_SphP;
		c.sz_extNodes_base    =s->sz_extNodes_base;
		c.sz_Ngblist          =s->sz_Ngblist;
		c.sz_result_hydro     =s->sz_result_hydro;
	}

	c.blocks              =s->blocks;
	c.grids               =s->grids;
	c.sphmode             =s->sphmode;

	unsigned int sz=0,n=0;
	unsigned int sza[CHUNK_MANAGER_DSIZE_GX];

	memset(sza,0,sizeof(unsigned int)*CHUNK_MANAGER_DSIZE_GX);

	AddSize(sizeof(int)                         * 1                     ,&sz,sza,&n); // prolog
	AddSize(sizeof(unsigned int)                * 1                     ,&sz,sza,&n); // total size
	AddSize(sizeof(unsigned int)                * CHUNK_MANAGER_DSIZE_GX,&sz,sza,&n); // size data
	AddSize(sizeof(struct state_gx)             * 1                     ,&sz,sza,&n);
	AddSize(sizeof(struct parameters_gx)        * 1                     ,&sz,sza,&n);
	AddSize(sizeof(struct parameters_hydro_gx)  * 1                     ,&sz,sza,&n);
	AddSize(sizeof(struct particle_data_gx)     * c.sz_P                ,&sz,sza,&n);
	AddSize(sizeof(struct NODE_gx)              * c.sz_Nodes_base       ,&sz,sza,&n);
	AddSize(sizeof(int)                         * c.sz_Nextnode         ,&sz,sza,&n);
	AddSize(sizeof(int)                         * c.sz_DomainTask       ,&sz,sza,&n);
	AddSize(sizeof(float)                       * c.sz_shortrange_table ,&sz,sza,&n);
	AddSize(sizeof(char)                        * c.sz_Exportflag       ,&sz,sza,&n);
	AddSize(sizeof(struct result_gx)            * c.sz_result           ,&sz,sza,&n);
	AddSize(sizeof(FLOAT_GX)                    * c.sz_scratch          ,&sz,sza,&n);
	AddSize(sizeof(struct sph_particle_data_gx) * c.sz_SphP             ,&sz,sza,&n);
	AddSize(sizeof(struct etc_gx)               * c.sz_etc              ,&sz,sza,&n);
	AddSize(sizeof(struct extNODE_gx)           * c.sz_extNodes_base    ,&sz,sza,&n);
	AddSize(sizeof(int)                         * c.sz_Ngblist          ,&sz,sza,&n);
	AddSize(sizeof(struct result_hydro_gx)      * c.sz_result_hydro     ,&sz,sza,&n);
	AddSize(sizeof(struct hydrodata_in_gx)      * c.sz_result_hydro     ,&sz,sza,&n);
	AddSize(sizeof(int)                         * 1                     ,&sz,sza,&n); // epilog

	unsigned int m=0,t;
	char*const v=MallocBLOB(sz);

	t=42; // prolog
	AddData(sizeof(int)                         * 1                      ,&t                 ,v,&m,sz);
	AddData(sizeof(unsigned int)                * 1                      ,&sz                ,v,&m,sz);
	AddData(sizeof(unsigned int)                * CHUNK_MANAGER_DSIZE_GX ,sza                ,v,&m,sz);
	AddData(sizeof(struct state_gx)             * 1                      ,&c                 ,v,&m,sz);
	AddData(sizeof(struct parameters_gx)        * 1                      ,p                  ,v,&m,sz);
	AddData(sizeof(struct parameters_hydro_gx)  * 1                      ,h                  ,v,&m,sz);
	AddData(sizeof(struct particle_data_gx)     * c.sz_P                 ,s->P               ,v,&m,sz);
	AddData(sizeof(struct NODE_gx)              * c.sz_Nodes_base        ,Nodes_base         ,v,&m,sz);
	AddData(sizeof(int)                         * c.sz_Nextnode          ,Nextnode           ,v,&m,sz);
	AddData(sizeof(int)                         * c.sz_DomainTask        ,DomainTask         ,v,&m,sz);
	AddData(sizeof(float)                       * c.sz_shortrange_table  ,s->shortrange_table,v,&m,sz);
	AddData(sizeof(char)                        * c.sz_Exportflag        ,s->Exportflag      ,v,&m,sz);
	AddData(sizeof(struct result_gx)            * c.sz_result            ,s->result          ,v,&m,sz);
	AddData(sizeof(FLOAT_GX)                    * c.sz_scratch           ,s->scratch         ,v,&m,sz);
	AddData(sizeof(struct sph_particle_data_gx) * c.sz_SphP              ,s->SphP            ,v,&m,sz);
	AddData(sizeof(struct etc_gx)               * c.sz_etc               ,s->etc             ,v,&m,sz);
	AddData(sizeof(struct extNODE_gx)           * c.sz_extNodes_base     ,Extnodes_base      ,v,&m,sz);
	AddData(sizeof(int)                         * c.sz_Ngblist           ,s->Ngblist         ,v,&m,sz);
	AddData(sizeof(struct result_hydro_gx)      * c.sz_result_hydro      ,s->result_hydro    ,v,&m,sz);
	AddData(sizeof(struct hydrodata_in_gx)      * c.sz_result_hydro      ,s->hydrodata_in    ,v,&m,sz);

	t=-42; // epilog
	AddData(sizeof(int)                         * 1                      ,&t                 ,v,&m,sz);

	ASSERT_GX( m==sz );
	if (g_dbg>0) MESSAGE("chunck: packed size: %d bytes = %.1f Mb",sz,sz/1024.0/1024.0);

	DataBLOB_check(v,sz,0,0);

	*ret_sz=sz;
	return v;
}

void PrintData(const char* msg,const char*const v,const unsigned int sz,const struct state_gx* sold)
{
	FUN_MESSAGE(4,"PrintData()");

	ASSERT_GX( msg!=NULL && v!=NULL && sz>0 );
	if (g_dbg>2){
		MESSAGE("chunck: PrintData(), %s,sz=%d",msg,sz);

		int i;
		struct state_gx* s=NULL;
		struct parameters_gx* p=NULL;
		struct parameters_hydro_gx *h=NULL;

		DataBLOB_unpack(&s,&p,&h,v,sz);

		if (s->P!=NULL || (sold!=NULL && sold->P!=NULL)){
			for(i=0;i<10;++i){
				const struct particle_data_gx p_local=s->P!=NULL ? s->P[i]: sold->P[i];
				MESSAGE("chunck: p%c[%d]=(%f,%f,%f;%f)",s->P!=NULL ? 'c' : 'o',i,p_local.Pos[0],p_local.Pos[1],p_local.Pos[2],p_local.Mass);
			}
			for(i=s->sz_P-10;i<s->sz_P;++i){
				const struct particle_data_gx p_local=s->P!=NULL ? s->P[i]: sold->P[i];
				MESSAGE("chunck: p%c[%d]=(%f,%f,%f;%f)",s->P!=NULL ? 'c' : 'o',i,p_local.Pos[0],p_local.Pos[1],p_local.Pos[2],p_local.Mass);
			}
			for(i=0;i<10;++i){
				const float f=s->shortrange_table!=NULL ? s->shortrange_table[i]: sold->shortrange_table[i];
				MESSAGE("chunck: shortrange_table%c[%d]=%f",s->shortrange_table!=NULL ? 'c' : 'o',i,f);
			}
			for(i=0;i<10;++i){
				const int f=s->Nextnode!=NULL ? s->Nextnode[i]: sold->Nextnode[i];
				MESSAGE("chunck: Nextnode%c[%d]=%d",s->Nextnode!=NULL ? 'c' : 'o',i,f);
			}
		}
	}
}

void  AdjustKernelDataPointers(const char*const v,const char* ca,struct state_gx* s,struct parameters_gx* p,struct parameters_hydro_gx* h,const unsigned int sz,const int iscudamem)
{
	FUN_MESSAGE(3,"AdjustKernelDataPointers()");

	ASSERT_GX( v!=NULL && ca!=NULL && sz>0 && (iscudamem==0 || iscudamem==1));
	ASSERT_GX( s!=NULL && p!=NULL && h!=NULL );

	const unsigned int uintalign=DataBLOB_check(v,sz,0,iscudamem);
	const int*const sza=(const int*)(v+uintalign*2);
	ASSERT_GX( sz==*(const int*)(v+uintalign*1) );

	int n=4;

	s->P               =(struct particle_data_gx*)     (ca+sza[++n]);
	s->Nodes_base      =(struct NODE_gx*)              (ca+sza[++n]);
	s->Nextnode        =(int*)                         (ca+sza[++n]);
	s->DomainTask      =(int*)                         (ca+sza[++n]);
	s->shortrange_table=(float*)                       (ca+sza[++n]);
	s->Exportflag      =(char*)                        (ca+sza[++n]);
	s->result          =(struct result_gx*)            (ca+sza[++n]);
	s->scratch         =(FLOAT_GX*)                    (ca+sza[++n]);
	s->SphP            =(struct sph_particle_data_gx*) (ca+sza[++n]);
	s->etc             =(struct etc_gx*)               (ca+sza[++n]);
	s->extNodes_base   =(struct extNODE_gx*)           (ca+sza[++n]);
	s->Ngblist         =(int*)                         (ca+sza[++n]);
	s->result_hydro    =(struct result_hydro_gx*)      (ca+sza[++n]);
	s->hydrodata_in    =(struct hydrodata_in_gx*)      (ca+sza[++n]);

 	ASSERT_GX( n==18 );
 	ASSERT_GX( sza[++n]>0 );
 	ASSERT_GX( sza[++n]>0 );
	ASSERT_GX( sza[++n]==0 );

	s->Nodes   =s->Nodes_base   -s->MaxPart;
	s->extNodes=s->extNodes_base-s->MaxPart;

	#if CUDA_DEBUG_GX>0
		if (g_dbg>3) PrintParameters(p,"chunck: recv");
		if (g_dbg>3) PrintState(s,"chunck: recv");
	#endif

	DataBLOB_check(v,sz,1,iscudamem);
}

char* AdjustKernelData(const char*const v,char* c,struct state_gx* s,struct parameters_gx* p,struct parameters_hydro_gx* h,const unsigned int sz)
{
	FUN_MESSAGE(3,"AdjustKernelData()");

	ASSERT_GX( v!=NULL && c!=NULL && sz>0 && s!=NULL && p!=NULL && h!=NULL );
	if (g_dbg>0) MESSAGE("chunck: AdjustKernelData(), sz=%d",sz);

	char*ca=c+Align(c);

	struct state_gx* ps=NULL;
	struct parameters_gx* pp=NULL;
	struct parameters_hydro_gx* ph=NULL;

	DataBLOB_unpack(&ps,&pp,&ph,v,sz);

	*s=*ps;
	*p=*pp;
	*h=*ph;

	AdjustKernelDataPointers(v,ca,s,p,h,sz,1);

	return ca;
}

void SendKernelDataToNode(const int node,const struct state_gx*const s,const struct parameters_gx*const p,const struct parameters_hydro_gx*const h,const int repack)
{
	FUN_MESSAGE(2,"SendKernelDataToNode()");

	ASSERT_GX(node>=0 && node<NTask && s!=NULL && p!=NULL && h!=NULL && (repack==0 || repack==1));

	unsigned int sz;
	static char* v=NULL;
	static MPI_Request req;
	static int req_init=1;

	if (req_init==1) req_init=0;
	else {
		req_init=0;
		MPI_Status st;
		CHECKMPICALL_GX( MPI_Wait(&req,&st) );
	}

	if (repack==1) v=DataBLOB_pack(s,p,h,&sz);
	else {
		sz=DataBLOB_getsize(v);

		struct state_gx* s2=NULL;
		struct parameters_gx* p2=NULL;
		struct parameters_hydro_gx *h2=NULL;

		DataBLOB_unpack(&s2,&p2,&h2,v,sz);

		ASSERT_GX(s2->sz_segments==s->sz_segments );
		s2->segment=s->segment;
	}

	if (g_dbg>=0) MESSAGE("chunck: SendKernelDataToNode(), %d|->%d, sz=%d bytes=%.1f Mb",ThisTask,node,sz,sz/1024.0/1024.0);
	ASSERT_GX( v!=NULL );

	#if CUDA_DEBUG_GX>0
		if (g_dbg>3) PrintParameters(p,"chunck: send");
		if (g_dbg>3) PrintState(s,"chunck: send");
	#endif
	PrintData("chunck: send",v,sz,s);

	//AdjustKernelDataPointers(v,v,sz,0);
	//PrintData("chunck: send",v,sz,&s);

	int seg=s->segment;
	CHECKMPICALL_GX( MPI_Bsend(&sz ,1 ,MPI_INT ,node,TAG_KERNEL_DATA_GX,CHUNCK_MANAGER_COMM_GX) );
	CHECKMPICALL_GX( MPI_Bsend(&seg,1 ,MPI_INT ,node,TAG_KERNEL_DATA_GX,CHUNCK_MANAGER_COMM_GX) );
	CHECKMPICALL_GX( MPI_Isend(v   ,sz,MPI_CHAR,node,TAG_KERNEL_DATA_GX,CHUNCK_MANAGER_COMM_GX,&req) );
}

char* RecvKernelDataFromNode(const int node,struct state_gx** s,struct parameters_gx** p,struct parameters_hydro_gx** h,unsigned int*const sz)
{
	FUN_MESSAGE(2,"RecvKernelDataFromNode()");
	ASSERT_GX( node>=0 && node<NTask && s!=NULL && *s==NULL && p!=NULL && *p==NULL && h!=NULL && *h==NULL && sz!=NULL );

	MPI_Status st;
	int seg=-1;

	CHECKMPICALL_GX( MPI_Recv(sz  ,1,MPI_INT,node,TAG_KERNEL_DATA_GX,CHUNCK_MANAGER_COMM_GX,&st) );
	CHECKMPICALL_GX( MPI_Recv(&seg,1,MPI_INT,node,TAG_KERNEL_DATA_GX,CHUNCK_MANAGER_COMM_GX,&st) );

	if (g_dbg>=0) MESSAGE("chunck: RecvKernelDataFromNode(), %d<-|%d, sz=%d bytes=%.1f Mb",ThisTask,node,*sz,*sz/1024.0/1024.0);

	char*const v=MallocBLOB(*sz);

	CHECKMPICALL_GX( MPI_Recv(v,*sz,MPI_CHAR,node,TAG_KERNEL_DATA_GX,CHUNCK_MANAGER_COMM_GX,&st) );

	DataBLOB_unpack(s,p,h,v,*sz);
	AdjustKernelDataPointers(v,v,*s,*p,*h,*sz,0);
	(*s)->external_node=node;
	(*s)->segment=seg;

	ASSERT_GX(seg>=0 && seg<(*s)->sz_segments);

	PrintData("chunck: recv",v,*sz,NULL);

	return v;
}

int SendKernelResultToNode(const int node,const struct state_gx*const s,unsigned int sz,unsigned int offset,const int sphmode)
{
	FUN_MESSAGE(2,"SendKernelResultToNode()");

	ASSERT_GX( s!=NULL && node>=0 && node<NTask && node==s->external_node && (sphmode==0 || sphmode==1));
	ASSERT_GX( s->NTask==NTask );
	ASSERT_GX( s->sphmode==sphmode );

	if (g_dbg>=0) MESSAGE("chunck: SendKernelResultToNode(), %d|->%d, offset=%d, sz=%d, sphmode=%d",ThisTask,node,offset,sz,sphmode);

	CHECKMPICALL_GX( MPI_Bsend(&sz         ,1,MPI_INT   ,node,TAG_KERNEL_RESULT_GX,CHUNCK_MANAGER_COMM_GX) );
	CHECKMPICALL_GX( MPI_Bsend(&offset     ,1,MPI_INT   ,node,TAG_KERNEL_RESULT_GX,CHUNCK_MANAGER_COMM_GX) );

if (!((!sphmode && offset+sz<=s->sz_result) || (sphmode && offset+sz<=s->sz_result_hydro)))
	MESSAGE("sphmode=%d, offset+sz=%d, s->sz_result_hydro=%d",sphmode,offset+sz,s->sz_result_hydro);

	ASSERT_GX((!sphmode && offset+sz<=s->sz_result) || (sphmode && offset+sz<=s->sz_result_hydro));
	ASSERT_GX( s->mode!=0 || s->NTask*offset+NTask*sz<s->sz_Exportflag );

	if (!sphmode)   CHECKMPICALL_GX( MPI_Send(s->result+offset             ,sizeof(struct result_gx)*sz      ,MPI_CHAR,node,TAG_KERNEL_RESULT_GX,CHUNCK_MANAGER_COMM_GX) );
	else            CHECKMPICALL_GX( MPI_Send(s->result_hydro+offset       ,sizeof(struct result_hydro_gx)*sz,MPI_CHAR,node,TAG_KERNEL_RESULT_GX,CHUNCK_MANAGER_COMM_GX) );
	if (s->mode==0) CHECKMPICALL_GX( MPI_Send(s->Exportflag+s->NTask*offset,sizeof(char)*s->NTask*sz         ,MPI_CHAR,node,TAG_KERNEL_RESULT_GX,CHUNCK_MANAGER_COMM_GX) );

	MPI_Status st;
	unsigned int relaunch;
	CHECKMPICALL_GX( MPI_Recv(&relaunch,1,MPI_INT,node,TAG_KERNEL_RESULT_GX,CHUNCK_MANAGER_COMM_GX,&st) );

	return relaunch;
}

int RecvKernelResultFromNode(const struct state_gx*const s,const int sphmode,int relaunch)
{
	FUN_MESSAGE(2,"RecvKernelResultFromNode()");
	if (g_dbg>=0) MESSAGE("chunck: RecvKernelResultFromNode() 00");

	ASSERT_GX( s!=NULL && (sphmode==0 || sphmode==1) );
	ASSERT_GX( sphmode==s->sphmode );

	int sz,offset;
	double time_kernel;

	MPI_Status st;

	CHECKMPICALL_GX( MPI_Recv(&sz    ,1,MPI_INT,MPI_ANY_SOURCE,TAG_KERNEL_RESULT_GX,CHUNCK_MANAGER_COMM_GX,&st) );
	CHECKMPICALL_GX( MPI_Recv(&offset,1,MPI_INT,st.MPI_SOURCE ,TAG_KERNEL_RESULT_GX,CHUNCK_MANAGER_COMM_GX,&st) );

	const int node=st.MPI_SOURCE;

	if (g_dbg>=0) MESSAGE("chunck: RecvKernelResultFromNode(),  %d<-|%d, offset=%d, sz=%d",ThisTask,node,offset,sz);

	ASSERT_GX((!sphmode && offset+sz<=s->sz_result) || (sphmode && offset+sz<=s->sz_result_hydro));
	ASSERT_GX( s->mode!=0 || s->NTask*offset+NTask*sz<s->sz_Exportflag );

	if (!sphmode)   CHECKMPICALL_GX( MPI_Recv(s->result+offset             ,sizeof(struct result_gx)*sz      ,MPI_CHAR,node,TAG_KERNEL_RESULT_GX,CHUNCK_MANAGER_COMM_GX,&st) );
	else            CHECKMPICALL_GX( MPI_Recv(s->result_hydro+offset       ,sizeof(struct result_hydro_gx)*sz,MPI_CHAR,node,TAG_KERNEL_RESULT_GX,CHUNCK_MANAGER_COMM_GX,&st) );

	if (s->mode==0) CHECKMPICALL_GX( MPI_Recv(s->Exportflag+s->NTask*offset,sizeof(char)*s->NTask*sz         ,MPI_CHAR,node,TAG_KERNEL_RESULT_GX,CHUNCK_MANAGER_COMM_GX,&st) );

	CHECKMPICALL_GX( MPI_Bsend(&relaunch,1,MPI_INT,node,TAG_KERNEL_RESULT_GX,CHUNCK_MANAGER_COMM_GX) );

/*if (sphmode){
	const unsigned int rsz=offset+sz;
	unsigned int j;

	MESSAGE("sph chunck recv: segment=%d, sz_segments=%d, offset=%d, rsz=%d, sz_result_hydro=%d",s->segment,s->sz_segments,offset,rsz,s->sz_result_hydro);
	for(j=offset;j<rsz;++j) {
		ASSERT_GX( j<s->sz_result_hydro );
		struct result_hydro_gx r=s->result_hydro[j];
		MESSAGE("  [%d] real=%d, acc={%f,%f,%f}, DtE=%f, MSVel=%f",j,r.realtarget,r.Acc[0],r.Acc[1],r.Acc[2],r.DtEntropy,r.MaxSignalVel);
	}
	exit(223);
}*/


	return node;
}

#endif
