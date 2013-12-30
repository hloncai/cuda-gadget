#ifdef CUDA_GX_CONSTANT_VARS_IN_SHARED_MEM
	__shared__ __device__ struct state_gx            g_const_state;
	__shared__ __device__ struct parameters_gx       g_const_parameters;
#ifdef SPH
	__shared__ __device__ struct parameters_hydro_gx g_const_parameters_hydro;
#endif
	// global constants, placed into cuda constant memory
	__constant__ struct state_gx             g_const_state_copy;
	__constant__ struct parameters_gx        g_const_parameters_copy;
#ifdef SPH
	__constant__ struct parameters_hydro_gx g_const_parameters_hydro_copy;
#endif
#else
	// global constants, placed into cuda constant memory
	__constant__  __device__ struct state_gx            g_const_state;
	__constant__  __device__ struct parameters_gx       g_const_parameters;
#ifdef SPH
	__constant__  __device__ struct parameters_hydro_gx g_const_parameters_hydro;
#endif
#endif

// split node structure into three parts to avoid it getting placed in slow local memory
// inspection of asm indicates no problem, at least for now
struct NODE_gx_0 {
	float len;
	float center0;
	float center1;
	float center2;
};

struct NODE_gx_1 {
	float s0;
	float s1;
	float s2;
	float mass;
};

struct NODE_gx_2 {
	int bitflags;
	int sibling;
	int nextnode;
	int father;
};

#ifdef CUDA_GX_USE_TEXTURES
	texture<int4 ,1,cudaReadModeElementType> tex_P;
	texture<int  ,1,cudaReadModeElementType> tex_Nextnode;
	texture<float,1,cudaReadModeElementType> tex_shortrange_table;
	texture<int4 ,1,cudaReadModeElementType> tex_Nodes_base;
	texture<int  ,1,cudaReadModeElementType> tex_DomainTask;
	texture<char ,1,cudaReadModeElementType> tex_Exportflag;
	texture<int4 ,1,cudaReadModeElementType> tex_SphP;
	texture<float,1,cudaReadModeElementType> tex_extNodes_base;
	// NOTE: no need to cache result, read once, write once only
#endif

// register size_t tdim;
// register dim3 result;
// __device__ dim3 tid;
// __device__ struct particle_data_gx getParticle_d;
// __device__ struct NODE_gx node;
// __device__ struct etc_gx getEtc_d;

__forceinline__ __device__
void Assert_fail_device_gx(const char* expr,const char* file,const int line,const char* msg,struct DevMsg* pdevmsg)
{
	#ifdef __DEVICE_EMULATION__
		fprintf(stderr,"\n%s:%d: ASSERT_DEVICE_GX, expression '%s' failed\n",file,line,expr);
		if (msg!=0) fprintf(stderr,"\tmessage : %s\n",msg);
		fprintf(stderr,"\taborting now...\n\n");
		fflush(stdout);
		fflush(stderr);
		abort();
		if (pdevmsg); // avoid compiler warning
	#else
	/*
		#if CUDA_DEBUG_GX > 1
			if (pdevmsg!=0){
				struct DevMsg& devmsg=*pdevmsg;
				PRINT_DEV_MSG_S("Assert_fail_device_gx");
				if (file) {
					PRINT_DEV_MSG_S("file:");
					PRINT_DEV_MSG_S(file);
				}
				PRINT_DEV_MSG_S("line:")
				PRINT_DEV_MSG_I(line);
				if (expr) {
					PRINT_DEV_MSG_S("expression:");
					PRINT_DEV_MSG_S(expr);
				}
				if (msg){
					PRINT_DEV_MSG_S("message:")
					PRINT_DEV_MSG_S(msg);
				}
				SET_DEVICE_THREAD_STATE('F'); // Failed
			}
		#endif
		*/
		exit_device_fun(-42,file,line,expr);
	#endif
}

#if CUDA_DEBUG_GX > 0
	__device__ int isNAN_device    (const FLOAT_INTERNAL_GX x){return x!=x;}
	__device__ int isNORMAL_device (const FLOAT_INTERNAL_GX x){
		if (x); // avoid compiler warning
		return 1;
	}
	__device__ int isINF_device    (const FLOAT_INTERNAL_GX x)
	{
		if (fabs(x)>=CUDA_GX_INFINITY) return 1;
		else return 0;
	}

	__device__ int isSUBNORMAL_device(const FLOAT_INTERNAL_GX x)
	{
		if (fabs(x)<=CUDA_GX_SUBNORMAL && fabs(x)>0) return 1;
		else return 0;
	}

	__device__ int isFloatOK_fast_device(const FLOAT_INTERNAL_GX x,const char*const file,const int line)
	{
		const int floatok=!isNAN_device(x) &&  !isINF_device(x) && !isSUBNORMAL_device(x) && isNORMAL_device(x);
		if (file || line); // avoid compiler warning
		return floatok;
	}

	__device__ int isFloatOK_device(const FLOAT_INTERNAL_GX x,const char*const file,const int line)
	{
		const int floatok=isFloatOK_fast_device(x,file,line);
		if  (!floatok) exit_device_info(-100,line,isNAN_device(x),isINF_device(x),isSUBNORMAL_device(x),isNORMAL_device(x),0,x);
		//ASSERT_DEVICE_GX(floatok);
		return floatok;
	}

	__device__ int isFloatinVicinity_device(const FLOAT_INTERNAL_GX x0,const FLOAT_INTERNAL_GX x1,const FLOAT_INTERNAL_GX e=1E-6)
	{
		//ASSERT_DEVICE_GX( e>0.0 );
		if (x1<x0-x0/e || x1>x0+x0*e) {
			if (x0<x1-x1/e || x0>x1+x1/e) return 0;
			else return 1;
		}
		else return 1;
	}
#endif

/* 
__forceinline__ __device__
int GetThreads()
{
	// ASSERT_DEVICE_GX( blockDim.y==1 && blockDim.z==1 && gridDim.y==1 && gridDim.z==1 ); // not possible, devmsg not defined yet
	tdim=blockDim.x * gridDim.x;
	return tdim;
}

__forceinline__ __device__
size_t GetTid()
{
	// 1D tid calculation
	// ASSERT_DEVICE_GX( blockDim.y==1 && blockDim.z==1 && gridDim.y==1 && gridDim.z==1 );
	return threadIdx.x+blockIdx.x*blockDim.x;
}


__forceinline__ __device__
dim3 GetThreadIndex(const int Np)
{	
	#ifdef CUDA_GX_SEGMENTED
		result.x=GetTid() + GetThreads()*g_const_state.segment*CUDA_GX_SEGMENTED;
		result.y=min(Np,result.x + GetThreads()*CUDA_GX_SEGMENTED);
		result.z=GetThreads();
	#else
		result.x = GetTid();
		result.y = Np;
		result.z = GetThreads();
 	#endif

	return dim3(threadIdx.x+blockIdx.x*blockDim.x, Np, blockDim.x*gridDim.x);
}

*/

__forceinline__ __device__
void KernelPreinit(const int sphmode
	#if CUDA_DEBUG_GX > 1
		,struct DevMsg& devmsg
	#endif
)
{
	// g_kernelsignals_shared=g_const_state.kernelsignals;
	
	#ifdef CUDA_GX_CONSTANT_VARS_IN_SHARED_MEM
		__syncthreads();
		if(threadIdx.x==0) {
			g_const_state=g_const_state_copy;
			g_const_parameters=g_const_parameters_copy;
		}
		__syncthreads();
	#endif

	dim3 tid;
	tid.x = threadIdx.x+blockIdx.x*blockDim.x;
	tid.y = g_const_state.Np;
	tid.z = blockDim.x*gridDim.x;
	

// If thread index exceeds the number of particles, then we can't compute anything on this thread and we goofed up initialization
if (!(tid.x<=tid.y || g_const_state.segment+1==g_const_state.sz_segments)) {
	//printf("** Np=%d, tid=(%d,%d,%d), g_const_state.segment=%d, g_const_state.sz_segments=%d\n",g_const_state.Np,tid.x,tid.y,tid.z,g_const_state.segment,g_const_state.sz_segments);
	exit_device_info(-44,g_const_state.Np,tid.x,tid.y,tid.z,g_const_state.segment,g_const_state.sz_segments,0);
}

	ASSERT_DEVICE_GX( tid.x<=tid.y || g_const_state.segment+1==g_const_state.sz_segments );
	
	if (tid.y > g_const_state.Np) {
			printf("** Np=%d, tid=(%d,%d,%d), g_const_state.segment=%d, g_const_state.sz_segments=%d\n",g_const_state.Np,tid.x,tid.y,tid.z,g_const_state.segment,g_const_state.sz_segments);
	}
	ASSERT_DEVICE_GX( tid.y<=g_const_state.Np  );
	
	ASSERT_DEVICE_GX( tid.z>0  );

	ASSERT_DEVICE_GX( g_const_state.mode==0 || g_const_state.mode==1 );
	ASSERT_DEVICE_GX( sphmode==0 || sphmode==1 );
	ASSERT_DEVICE_GX( g_const_state.cudamode>=2 );
	ASSERT_DEVICE_GX( g_const_state.Np>0 );

	ASSERT_DEVICE_GX( blockDim.x>=1 && blockDim.y==1 && blockDim.z==1); // 1D thread grid
	ASSERT_DEVICE_GX( gridDim.x>=1  && gridDim.y==1 && gridDim.z==1 );

	ASSERT_DEVICE_GX( (sphmode==0 && g_const_state.Np==g_const_state.sz_result) || (sphmode==1 && g_const_state.Np==g_const_state.sz_result_hydro) );
	ASSERT_DEVICE_GX( g_const_state.sz_P==g_const_state.NumPart );
	ASSERT_DEVICE_GX( g_const_state.mode==1 || g_const_state.Np<=g_const_state.sz_P );

	ASSERT_DEVICE_GX( g_const_state.MaxPart==g_const_parameters.MaxPart );
	ASSERT_DEVICE_GX( (size_t)(&g_const_state.Nodes_base[1])-(size_t)(&g_const_state.Nodes_base[0])==48 );
	ASSERT_DEVICE_GX( sizeof(struct NODE_gx)==3*4*4 && sizeof(struct NODE_gx_0)==4*4 && sizeof(struct NODE_gx_1)==4*4 && sizeof(struct NODE_gx_2)==4*4 && sizeof(int4[3])==3*4*4 );
#ifdef SPH
	if (sphmode) {
// if (!( g_const_state.mode==1 || (g_const_state.Np==g_const_state.sz_SphP && g_const_state.Np==g_const_parameters_hydro.N_gas) ))
// 	printf("** g_const_state.mode=%d, g_const_state.Np=%d, g_const_state.sz_SphP=%d, g_const_parameters_hydro.N_gas=%d",g_const_state.mode,g_const_state.Np,g_const_state.sz_SphP,g_const_parameters_hydro.N_gas);

		ASSERT_DEVICE_GX( g_const_state.mode==1 || (g_const_state.Np==g_const_state.sz_SphP && g_const_state.Np==g_const_parameters_hydro.N_gas) );
		ASSERT_DEVICE_GX( g_const_state.mode==1 || g_const_state.sz_result_hydro==g_const_state.N_gas );
		ASSERT_DEVICE_GX( g_const_state.mode==0 || g_const_state.sz_result_hydro==g_const_state.sz_hydrodata_in );
	}
#endif

	ASSERT_DEVICE_GX( isFloatOK_fast_device(g_const_parameters.ErrTolTheta,__FILE__,__LINE__) );
	ASSERT_DEVICE_GX( isFloatOK_fast_device(g_const_parameters.ErrTolTheta,__FILE__,__LINE__) );
	ASSERT_DEVICE_GX( isFloatOK_fast_device(g_const_parameters.Asmth[0],__FILE__,__LINE__) );
	ASSERT_DEVICE_GX( isFloatOK_fast_device(g_const_parameters.Asmth[1],__FILE__,__LINE__) );
	ASSERT_DEVICE_GX( isFloatOK_fast_device(g_const_parameters.Rcut[0],__FILE__,__LINE__) );
	ASSERT_DEVICE_GX( isFloatOK_fast_device(g_const_parameters.Rcut[1],__FILE__,__LINE__) );
	ASSERT_DEVICE_GX( isFloatOK_fast_device(g_const_parameters.BoxSize,__FILE__,__LINE__) );
	ASSERT_DEVICE_GX( isFloatOK_fast_device(g_const_parameters.ErrTolTheta,__FILE__,__LINE__) );
	ASSERT_DEVICE_GX( isFloatOK_fast_device(g_const_parameters.ErrTolForceAcc,__FILE__,__LINE__) );
	for(int i=0;i<6;++i) {
		ASSERT_DEVICE_GX( isFloatOK_fast_device(g_const_parameters.ForceSoftening[i],__FILE__,__LINE__) );
		ASSERT_DEVICE_GX( isFloatOK_fast_device(g_const_parameters.Masses[i],__FILE__,__LINE__) );
	}
	#ifdef ADAPTIVE_GRAVSOFT_FORGAS
		ASSERT_DEVICE_GX( isFloatOK_fast_device(g_const_parameters.soft,__FILE__,__LINE__) );
	#endif
	ASSERT_DEVICE_GX( isFloatOK_fast_device(g_const_parameters.Timebase_interval,__FILE__,__LINE__) );

	// return tid;
	return;
}

__forceinline__ __device__
struct particle_data_gx GetParticle(const int n
	#if CUDA_DEBUG_GX > 1
		,struct DevMsg& devmsg
	#endif
)
{
	ASSERT_DEVICE_GX( g_const_state.mode==0 || g_const_state.mode==1);
	ASSERT_DEVICE_GX( n>=0 && n<g_const_state.sz_P);

	struct particle_data_gx getParticle_d;

	#ifdef CUDA_GX_USE_TEXTURES
		int4* i4=(int4*)(&getParticle_d);
		*i4=tex1Dfetch(tex_P,n);
	#else
		getParticle_d=g_const_state.P[n];
	#endif

	#ifdef __DEVICE_EMULATION__
		#if CUDART_VERSION < 3 // NOTE: error in nvcc 3.0, will not pass parameters to GCC
			ASSERT_DEVICE_GX(isParticleDataOK(getParticle_d,__FILE__,__LINE__));
		#endif
	#endif

	return getParticle_d;
}

// union NODE_gx_cache
// {
// 	struct NODE_gx n;
// 	int4           i[3];
// };

__forceinline__ __device__
struct NODE_gx GetNode(int n
	#if CUDA_DEBUG_GX > 1
		,struct DevMsg& devmsg
	#endif
)
{
	ASSERT_DEVICE_GX( g_const_state.mode==0 || g_const_state.mode==1);
	ASSERT_DEVICE_GX( n>=g_const_state.MaxPart );
	ASSERT_DEVICE_GX( n-g_const_state.MaxPart<g_const_state.sz_Nodes_base );

	#ifdef CUDA_GX_USE_TEXTURES

		struct NODE_gx node;

		n -= g_const_state.MaxPart;
		ASSERT_DEVICE_GX( n<g_const_state.sz_Nodes_base );
		ASSERT_DEVICE_GX( sizeof(struct NODE_gx)==12*4 ) ;

		n *= 3;
		int4* i4=(int4*)(&node);

		*i4=tex1Dfetch(tex_Nodes_base,n);
		++i4;
		++n;
		*i4=tex1Dfetch(tex_Nodes_base,n);
		++i4;
		++n;
		*i4=tex1Dfetch(tex_Nodes_base,n);

		return node;
	#else
		return g_const_state.Nodes[n];
	#endif
}

__forceinline__ __device__
int GetNextnode(const int n
	#if CUDA_DEBUG_GX > 1
		,struct DevMsg& devmsg
	#endif
)
{
	ASSERT_DEVICE_GX(g_const_state.mode==0 || g_const_state.mode==1);
	ASSERT_DEVICE_GX(n>=0 && n<g_const_state.sz_Nextnode);

	#ifdef CUDA_GX_USE_TEXTURES
		return tex1Dfetch(tex_Nextnode,n);
	#else
		return g_const_state.Nextnode[n];
	#endif
}

__forceinline__ __device__
struct etc_gx GetEtc(const int n
	#if CUDA_DEBUG_GX > 1
		,struct DevMsg& devmsg
	#endif
	)
{
	ASSERT_DEVICE_GX( g_const_state.mode==0 || g_const_state.mode==1 );
	ASSERT_DEVICE_GX( g_const_state.etc!=NULL && n<g_const_state.sz_etc );

	struct etc_gx getEtc_d=g_const_state.etc[n];
	ASSERT_DEVICE_GX(getEtc_d.Type>=0 && getEtc_d.Type<6);
	ASSERT_DEVICE_GX(getEtc_d.Ti_endstep>=getEtc_d.Ti_begstep);

	#ifdef TRANS_DEBUG_GX
		ASSERT_DEVICE_GX( g_const_state.P!=NULL && n<g_const_state.sz_P );
		ASSERT_DEVICE_GX( g_const_state.etc[n].Ti_begstep==P[n].Ti_begstep );
		ASSERT_DEVICE_GX( g_const_state.etc[n].Ti_endstep==P[n].Ti_endstep );
	#endif

	#ifdef __DEVICE_EMULATION__
		#if CUDART_VERSION < 3 // NOTE: error in nvcc 3.0, will not pass parameters to GCC
			ASSERT_DEVICE_GX( isEtcDataOK(getEtc_d,__FILE__,__LINE__) );
		#endif
	#endif
	return getEtc_d;
}

__forceinline__ __device__
float GetScratch(const int n
	#if CUDA_DEBUG_GX > 1
		,struct DevMsg& devmsg
	#endif
	)
{
	ASSERT_DEVICE_GX( g_const_state.mode==0 || g_const_state.mode==1 );
	ASSERT_DEVICE_GX( n<g_const_state.sz_scratch );

	return g_const_state.scratch[n];
}

/*
__forceinline__ __device__
int GetParticleType(const int target,const int mode
	#if CUDA_DEBUG_GX > 1
		,struct DevMsg& devmsg
	#endif
	)
{
	ASSERT_DEVICE_GX( target<g_const_state.sz_P );
	ASSERT_DEVICE_GX( mode==1 || (g_const_parameters.typesizes[0]+g_const_parameters.typesizes[1]+g_const_parameters.typesizes[2]+g_const_parameters.typesizes[3]+g_const_parameters.typesizes[4]+g_const_parameters.typesizes[5] <= g_const_state.sz_P) );

	int n=g_const_parameters.typesizes[0];
	if (target<n) return 0;
	n += g_const_parameters.typesizes[1];
	if (target<n) return 1;
	n += g_const_parameters.typesizes[2];
	if (target<n) return 2;
	n += g_const_parameters.typesizes[3];
	if (target<n) return 3;
	n += g_const_parameters.typesizes[4];
	if (target<n) return 4;
	n += g_const_parameters.typesizes[5];
	if (target<n) return 5;
	else {
		exit_device(-1);
		return 0; // dummy return
	}
}
*/

__forceinline__ __device__
int GetParticleType(const int n
	#if CUDA_DEBUG_GX > 1
		,struct DevMsg& devmsg
	#endif
)
{
	ASSERT_DEVICE_GX( g_const_state.mode==0 || g_const_state.mode==1);
	if (!(n<g_const_state.sz_P))
		printf("n: %d sz_P: %d\n",n,g_const_state.sz_P);
	ASSERT_DEVICE_GX( n<g_const_state.sz_P );

	struct etc_gx d=GetEtc(n
		#if CUDA_DEBUG_GX > 1
			,devmsg
		#endif
	);

	const int type=d.Type;  // alias for P[no].Type
	ASSERT_DEVICE_GX(type>=0 && type<6 );
	return type;
}

#ifdef ADAPTIVE_GRAVSOFT_FORGAS
	__forceinline__ __device__
	FLOAT_INTERNAL_GX GetHsml(const int target)
	{
		ASSERT_DEVICE_GX( g_const_state.mode==0);
		ASSERT_DEVICE_GX( target<g_const_state.sz_SphP && p!=NULL);
		// YYY should this apply for Type==0 particles only??
		ASSERT_DEVICE_GX( GetParticleType(target)==0 );

		return g_const_state.SphP[target].Hsml; // alias for SphP[no].Hsml
	}
#endif


// NOTE: cannot mix device and host functions well (apart from using __global__)
//       so implement functionality both places
#ifdef CUDA_GX_BITWISE_EXPORTFLAGS
	__forceinline__ __device__
	int GetByteSizeForBitArray(const int bitelemsize)
	{
		return bitelemsize/(sizeof(char)*8)+(bitelemsize%(sizeof(char)*8)!=0);
	}

	int GetByteSizeForBitArray_host(const int bitelemsize)
	{
		return bitelemsize/(sizeof(char)*8)+(bitelemsize%(sizeof(char)*8)!=0);
	}

	__forceinline__ __device__
	int GetByteAndBitIndex(const int n,const int bitelemsize,const int index,int* byte)
	{
		ASSERT_DEVICE_GX(byte!=NULL && index<bitelemsize);
		*byte=n*GetByteSizeForBitArray(bitelemsize)+index/(sizeof(char)*8);
		const int bit=index%(sizeof(char)*8);
		return bit;
	}

	int GetByteAndBitIndex_host(const int n,const int bitelemsize,const int index,int* byte)
	{
		ASSERT_GX(byte!=NULL && index<bitelemsize);
		*byte=n*GetByteSizeForBitArray_host(bitelemsize)+index/(sizeof(char)*8);
		static const int bit=index%(sizeof(char)*8);
		return bit;
	}
#endif

__forceinline__ __device__
void SetExportflag_gx(const int no,const int target
	#if CUDA_DEBUG_GX > 1
		,struct DevMsg& devmsg
	#endif
	)
{
	ASSERT_DEVICE_GX( g_const_state.mode==0 );
	if (g_const_state.NTask==1) return;

	const int d=no - (g_const_parameters.MaxPart + g_const_parameters.MaxNodes);

#ifdef CUDA_GX_DEBUG_MEMORY_ERROR
if (!(d>=0 && d<g_const_state.sz_DomainTask))
	exit_device_info(-6,no,target,g_const_parameters.MaxPart,g_const_parameters.MaxNodes,g_const_state.sz_DomainTask,d,-42.6);
#endif

	ASSERT_DEVICE_GX( d>=0 && d<g_const_state.sz_DomainTask );

	#ifdef CUDA_GX_USE_TEXTURES
		const int dt=tex1Dfetch(tex_DomainTask,d);
	#else
		const int dt=g_const_state.DomainTask[d];
	#endif

	#ifdef CUDA_GX_BITWISE_EXPORTFLAGS
		// bitwise set
		int byte;
		const int bit=GetByteAndBitIndex(target,g_const_state.NTask,dt,&byte);
		ASSERT_DEVICE_GX( byte<g_const_state.sz_Exportflag );

		#ifdef CUDA_GX_USE_TEXTURES
			const char c=tex1Dfetch(tex_Exportflag,byte);
			ASSERT_DEVICE_GX( c==g_const_state.Exportflag[byte] );
		#else
			const char c=g_const_state.Exportflag[byte];
		#endif

		if (!((c >> bit) & 0x1)){
			// write only if note already written/setted
			g_const_state.Exportflag[byte] = c | (1 << bit);
		}

	#else
		const int t=g_const_state.NTask*(target)+dt;
		ASSERT_DEVICE_GX( t>=0 && t<g_const_state.sz_Exportflag );
//if (g_const_state.Exportflag[t]==0) MESSAGE("SetExportflag_gx: no=%d, target=%d, dg=%d, t=%d",no,target,dt,t);
		g_const_state.Exportflag[t]=1;
	#endif
}

#if CUDA_DEBUG_GX > 1
	__forceinline__ __device__
	void Debug_PrintVars(const dim3 tid,struct DevMsg& devmsg)
	{
		PRINT_DEV_MSG_I(tid.x);
		if(g_const_state.debugval) {
			PRINT_DEV_MSG_I(tid.x);
			PRINT_DEV_MSG_I(tid.y);
			PRINT_DEV_MSG_I(tid.z);
		}

		#ifdef CUDA_GX_USE_TEXTURES
			#if CUDA_DEBUG_GX > 1
			if (tid.x==0 ) {
				int sum=0;
				FLOAT_INTERNAL_GX sum2=0;
				int sz=g_const_state.sz_Nextnode;
				for(int ii=0;ii<sz;++ii) sum += tex1Dfetch(tex_Nextnode,ii);
				sz=g_const_state.sz_shortrange_table;
				for(int ii=0;ii<sz;++ii) sum2 += tex1Dfetch(tex_shortrange_table,ii);
				//printf("\n** Nextnode kernel (sz=%d,sz2=%d), sum=%d  sum2=%g\n",g_const_state.sz_Nextnode,g_const_state.sz_shortrange_table,sum,sum2);
				PRINT_DEV_MSG_I(sum);
				PRINT_DEV_MSG_D(sum2);
			}
			#endif
		#endif
	}

	__forceinline__ __device__
	void Debug_PrintValues(struct DevMsg& devmsg,const FLOAT_INTERNAL_GX pos_x,const FLOAT_INTERNAL_GX pos_y,const FLOAT_INTERNAL_GX pos_z,const int target,const FLOAT_INTERNAL_GX rcut,const FLOAT_INTERNAL_GX rcut2,const FLOAT_INTERNAL_GX asmth,const FLOAT_INTERNAL_GX asmthfac,const FLOAT_INTERNAL_GX h3_inv,const int no)
	{
		if(target<g_const_state.debugval) {
			PRINT_DEV_MSG_I(g_const_state.sz_P);
			PRINT_DEV_MSG_I(g_const_state.sz_Nodes_base);
			PRINT_DEV_MSG_I(g_const_state.sz_Nextnode);
			PRINT_DEV_MSG_I(g_const_state.sz_DomainTask);
			PRINT_DEV_MSG_I(g_const_state.sz_shortrange_table);
			PRINT_DEV_MSG_I(g_const_state.sz_Exportflag);
			PRINT_DEV_MSG_I(g_const_state.sz_result);

			PRINT_DEV_MSG_I(target);
			PRINT_DEV_MSG_I(g_const_state.sz_shortrange_table);
			PRINT_DEV_MSG_D(pos_x);
			PRINT_DEV_MSG_D(pos_y);
			PRINT_DEV_MSG_D(pos_z);
			PRINT_DEV_MSG_D(rcut);
			PRINT_DEV_MSG_D(rcut2);
			PRINT_DEV_MSG_I(g_const_state.sz_shortrange_table);
			PRINT_DEV_MSG_D(asmth);
			PRINT_DEV_MSG_D(asmthfac);
			PRINT_DEV_MSG_D(h3_inv);
			PRINT_DEV_MSG_I(no);

			PRINT_DEV_MSG_I(g_const_parameters.MaxPart);
		}
	}

	__forceinline__ __device__
	void PrintParameters(struct DevMsg& devmsg)
	{
		PRINT_DEV_MSG_I(g_const_parameters.MaxPart);
		PRINT_DEV_MSG_I(g_const_parameters.MaxNodes);
		PRINT_DEV_MSG_I(g_const_parameters.Ti_Current);
		PRINT_DEV_MSG_D(g_const_parameters.Asmth[0]);
		PRINT_DEV_MSG_D(g_const_parameters.Asmth[1]);
		PRINT_DEV_MSG_D(g_const_parameters.Rcut[0]);
		PRINT_DEV_MSG_D(g_const_parameters.Rcut[1]);
		PRINT_DEV_MSG_D(g_const_parameters.BoxSize);
		PRINT_DEV_MSG_D(g_const_parameters.ErrTolTheta);
		PRINT_DEV_MSG_D(g_const_parameters.ErrTolForceAcc);
		for(int i=0;i<6;++i) PRINT_DEV_MSG_D(g_const_parameters.ForceSoftening[i]);
		#ifdef ADAPTIVE_GRAVSOFT_FORGAS
			PRINT_DEV_MSG_D(g_const_parameters.soft);
		#endif
		for(int i=0;i<6;++i) PRINT_DEV_MSG_D(g_const_parameters.Masses[i]);
		for(int i=0;i<6;++i) PRINT_DEV_MSG_I(g_const_parameters.typesizes[i]);
		PRINT_DEV_MSG_I(g_const_parameters.nonsortedtypes);
		PRINT_DEV_MSG_I(g_const_parameters.ComovingIntegrationOn);
		PRINT_DEV_MSG_D(g_const_parameters.Timebase_interval);
		PRINT_DEV_MSG_I(g_const_parameters.export_P_type);
	}

#ifdef SPH
	__forceinline__ __device__
	void PrintParametersHydro(struct DevMsg& devmsg)
	{
		PRINT_DEV_MSG_I(g_const_parameters_hydro.N_gas);
		PRINT_DEV_MSG_I(g_const_parameters_hydro.szngb);

		PRINT_DEV_MSG_D(g_const_parameters_hydro.hubble_a2);
		PRINT_DEV_MSG_D(g_const_parameters_hydro.fac_mu);
		PRINT_DEV_MSG_D(g_const_parameters_hydro.fac_vsic_fix);
		PRINT_DEV_MSG_D(g_const_parameters_hydro.ArtBulkViscConst);
		PRINT_DEV_MSG_D(g_const_parameters_hydro.boxSize);
		PRINT_DEV_MSG_D(g_const_parameters_hydro.boxHalf);
	}
#endif

	__forceinline__ __device__
	void PrintState(struct DevMsg& devmsg,const int simplemode)
	{
		if (!simplemode){
			PRINT_DEV_MSG_I((int)g_const_state.P);
			PRINT_DEV_MSG_I((int)g_const_state.Nodes_base);
			PRINT_DEV_MSG_I((int)g_const_state.Nodes);
			PRINT_DEV_MSG_I((int)g_const_state.Nextnode);
			PRINT_DEV_MSG_I((int)g_const_state.DomainTask);
			PRINT_DEV_MSG_I((int)g_const_state.shortrange_table);
			PRINT_DEV_MSG_I((int)g_const_state.Exportflag);
			PRINT_DEV_MSG_I((int)g_const_state.result);

			PRINT_DEV_MSG_I((int)g_const_state.SphP);
			PRINT_DEV_MSG_I((int)g_const_state.etc);
			PRINT_DEV_MSG_I((int)g_const_state.extNodes_base);
			PRINT_DEV_MSG_I((int)g_const_state.extNodes);
			PRINT_DEV_MSG_I((int)g_const_state.Ngblist);
			PRINT_DEV_MSG_I((int)g_const_state.result_hydro);
		}

		PRINT_DEV_MSG_I(g_const_state.ThisTask);
		PRINT_DEV_MSG_I(g_const_state.NTask);
		PRINT_DEV_MSG_I(g_const_state.NumPart);
		PRINT_DEV_MSG_I(g_const_state.N_gas);

		PRINT_DEV_MSG_I(g_const_state.MaxPart);
		PRINT_DEV_MSG_I(g_const_state.sz_memory_limit);

		PRINT_DEV_MSG_I(g_const_state.sz_P);
		PRINT_DEV_MSG_I(g_const_state.sz_SphP);
		PRINT_DEV_MSG_I(g_const_state.sz_Nodes_base);
		PRINT_DEV_MSG_I(g_const_state.sz_Nextnode);
		PRINT_DEV_MSG_I(g_const_state.sz_DomainTask);
		PRINT_DEV_MSG_I(g_const_state.sz_shortrange_table);
		PRINT_DEV_MSG_I(g_const_state.sz_Exportflag);
		PRINT_DEV_MSG_I(g_const_state.sz_result);

		PRINT_DEV_MSG_I(g_const_state.sz_etc);
		PRINT_DEV_MSG_I(g_const_state.sz_extNodes_base);
		PRINT_DEV_MSG_I(g_const_state.sz_Ngblist);
		PRINT_DEV_MSG_I(g_const_state.sz_result_hydro);

		if (!simplemode){
			PRINT_DEV_MSG_I(g_const_state.sz_max_P);
			PRINT_DEV_MSG_I(g_const_state.sz_max_SphP);
			PRINT_DEV_MSG_I(g_const_state.sz_max_Exportflag);
			PRINT_DEV_MSG_I(g_const_state.sz_max_result);
			PRINT_DEV_MSG_I(g_const_state.sz_max_etc);
			PRINT_DEV_MSG_I(g_const_state.sz_max_result_hydro);
			PRINT_DEV_MSG_I(g_const_state.sz_max_hydrodata_in);
			PRINT_DEV_MSG_I(g_const_state.sz_max_Ngblist);
		}

		PRINT_DEV_MSG_I(g_const_state.mode);
		PRINT_DEV_MSG_I(g_const_state.cudamode);
		PRINT_DEV_MSG_I(g_const_state.debugval);
		PRINT_DEV_MSG_I(g_const_state.iteration);
	}
#endif

#include "cudautils.h"

void CheckCUDAError_fun(const char *msg,const char* const file,const int line)
{
	FUN_MESSAGE(5,"CheckCUDAError_fun(%s)",msg);

	const cudaError_t err = cudaGetLastError();
	if(cudaSuccess!=err) {
		if (msg==NULL)  ERROR("in %s:%d: CUDA ERROR: %s",file,line,cudaGetErrorString(err));
		else            ERROR("in %s:%d: CUDA ERROR: %s: %s",file,line,msg,cudaGetErrorString(err));
		exit(-1); // wil be done in ERROR, but play safe
	}
}

#define CHECKCUDAERROR(msg) CheckCUDAError_fun(msg,__FILE__,__LINE__)

// N.B. Moved from gadget_cuda_gx.cu to here
// file scope issues related to memCopying to __constant__ memory: http://stackoverflow.com/questions/2450556/allocate-constant-memory
void CopyStateAndParamToConstMem(const struct state_gx*const s,const struct parameters_gx*const p,const struct parameters_hydro_gx*const h)
{
	FUN_MESSAGE(2,"CopyStateAndParamToConstMem()");
	ASSERT_GX(s!=NULL && p!=NULL);
	ASSERT_GX((s->sphmode==0 && h==NULL) || (s->sphmode==1 && h!=NULL));

	#ifdef __DEVICE_EMULATION__
		// test that copy went ok, 0
		#ifdef CUDA_GX_CONSTANT_VARS_IN_SHARED_MEMs
			g_const_state_copy.sz_memory_limit=0;
		#else
			g_const_state.sz_memory_limit=0;
		#endif
	#endif

	// copy state and parametes to constant mem
	#ifdef CUDA_GX_CONSTANT_VARS_IN_SHARED_MEM
		// *VERY* IMPORTANT: Character string usage is deprecated as of CUDA 4.1 and (in my experience) causes kernel launch failure
		// g_const_state_copy",s,sizeof(struct state_gx),0,cudaMemcpyHostToDevice);
		//struct state_gx* addr_g_const_state_copy;
		//cudaGetSymbolAddress((void **)&addr_g_const_state_copy, g_const_state_copy);
		//CHECKCUDAERROR("cudaGetSymbolAddress");
		cudaMemcpyToSymbol(g_const_state_copy,s,sizeof(struct state_gx),0,cudaMemcpyHostToDevice);
		CHECKCUDAERROR("cudaMemcpyToSymbol");
		//cudaMemcpy(addr_g_const_state_copy,s,sizeof(struct state_gx),cudaMemcpyHostToDevice);
		//CHECKCUDAERROR("cudaMemcpy");
		
		//struct parameters_gx* addr_g_const_parameters_copy;
		//cudaGetSymbolAddress((void **)&addr_g_const_parameters_copy, g_const_parameters_copy);
		//CHECKCUDAERROR("cudaGetSymbolAddress");
		cudaMemcpyToSymbol(g_const_parameters_copy,p,sizeof(struct parameters_gx),0,cudaMemcpyHostToDevice);
		CHECKCUDAERROR("cudaMemcpyToSymbol");
		//cudaMemcpy(addr_g_const_state_copy,s,sizeof(struct state_gx),cudaMemcpyHostToDevice);
		//CHECKCUDAERROR("cudaMemcpy");
		
		#ifdef SPH
		if(h!=NULL){
			//struct parameters_hydro_gx* addr_g_const_parameters_hydro_copy;
			//cudaGetSymbolAddress((void **)&addr_g_const_parameters_hydro_copy, g_const_parameters_hydro_copy);
			//CHECKCUDAERROR("cudaGetSymbolAddress");

			cudaMemcpyToSymbol(g_const_parameters_hydro_copy,h,sizeof(struct parameters_hydro_gx),0,cudaMemcpyHostToDevice);
			CHECKCUDAERROR("cudaMemcpyToSymbol");

			//cudaMemcpy(addr_g_const_state_copy,s,sizeof(struct state_gx),cudaMemcpyHostToDevice);
			//CHECKCUDAERROR("cudaMemcpy");
		}
		#endif
	#else
		//struct parameters_gx *addr_g_const_parameters;
		//cudaGetSymbolAddress((void **)&addr_g_const_parameters, "g_const_parameters");
		//CHECKCUDAERROR("cudaGetSymbolAddress");
		cudaMemcpyToSymbol(g_const_parameters,p,sizeof(struct parameters_gx),0,cudaMemcpyHostToDevice);
		CHECKCUDAERROR("cudaMemcpyToSymbol");
		
		//struct state_gx *addr_g_const_state;
		//cudaGetSymbolAddress((void **)&addr_g_const_state, g_const_state);
		//CHECKCUDAERROR("cudaGetSymbolAddress");
		cudaMemcpyToSymbol(g_const_state,s,sizeof(struct state_gx),0,cudaMemcpyHostToDevice);
		CHECKCUDAERROR("cudaMemcpytoSymbol");
		#ifdef SPH
		if(h!=NULL){
			//struct parameters_hydro_gx* addr_g_const_parameters_hydro;
			//cudaGetSymbolAddress((void **)&addr_g_const_parameters_hydro, g_const_parameters_hydro);
			//CHECKCUDAERROR("cudaGetSymbolAddress");
			cudaMemcpyToSymbol(g_const_parameters_hydro,h,sizeof(struct parameters_hydro_gx),0,cudaMemcpyHostToDevice);
			CHECKCUDAERROR("cudaMemcpytoSymbol");
		}
		#endif
	#endif

	#ifdef __DEVICE_EMULATION__
		// strange enough cudaMemcpyToSymbol produces errorneous result when running in emu-mode, revert to old memcpy
		#ifdef CUDA_GX_CONSTANT_VARS_IN_SHARED_MEM
			memcpy(&g_const_state_copy,s,sizeof(struct state_gx));
			memcpy(&g_const_parameters_copy,p,sizeof(struct parameters_gx));
			#ifdef SPH
			if(h!=NULL) memcpy(&g_const_parameters_hydro_copy,h,sizeof(struct parameters_hydro_gx));
			#endif
			ASSERT_GX(memcmp(&g_const_state_copy,s,sizeof(struct state_gx))==0);
			ASSERT_GX(memcmp(&g_const_parameters_copy,p,sizeof(struct parameters_gx))==0);
			#ifdef SPH
			ASSERT_GX(h==NULL || memcmp(&g_const_parameters_hydro_copy,h,sizeof(struct parameters_hydro_gx))==0);
			#endif
		#else
			memcpy(&g_const_state,s,sizeof(struct state_gx));
			memcpy(&g_const_parameters,p,sizeof(struct parameters_gx));
			#ifdef SPH
			if(h!=NULL) memcpy(&g_const_parameters_hydro,h,sizeof(struct parameters_hydro_gx));
			#endif
			ASSERT_GX(memcmp(&g_const_state,s,sizeof(struct state_gx))==0);
			ASSERT_GX(memcmp(&g_const_parameters,p,sizeof(struct parameters_gx))==0);
			#ifdef SPH
			ASSERT_GX(h==NULL || memcmp(&g_const_parameters_hydro,h,sizeof(struct parameters_hydro_gx))==0);
			#endif
		#endif

		// test that copy went ok, 1
		#ifdef CUDA_GX_CONSTANT_VARS_IN_SHARED_MEMs
			ASSERT_GX(g_const_state_copy.sz_memory_limit==s->sz_memory_limit);
			ASSERT_GX(g_const_state_copy.iteration==s->iteration);
			ASSERT_GX(EqualState(g_const_state_copy,*s,1,1,__FILE__,__LINE__));
		#else
			ASSERT_GX(g_const_state.sz_memory_limit==s->sz_memory_limit);
			ASSERT_GX(g_const_state.iteration==s->iteration);
			ASSERT_GX(EqualState(g_const_state,*s,1,1,__FILE__,__LINE__));
		#endif
	#endif
}
