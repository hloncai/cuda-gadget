
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

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h> // for gettimeofday
#include <unistd.h>   // for gethostname
#include <pthread.h>

#include "defines.h"
#include "options.h"
#include "cudautils.h"

#include "Gadget2/interface_gx.h"
#include "Gadget2/handshake_gx.h"
#include "Gadget2/debug_gx.h"
#include "Gadget2/chunckmanager_gx.h"
#include "version.h"

#include "kernelsupport.h"
/* #include "hydrakernel.h" */
#include "forcekernel.h"
#include "chunck.h"
#include "test.h"

#include "gadget_cuda_gx.h"

struct state_gx s_gx_cuda;
size_t          g_cuda_mem=0;
int             g_kernelsize_blocks=128;
int             g_kernelsize_grid=4;

int isEmulationMode()
{
	FUN_MESSAGE(5,"isEmulationMode()");
	#ifdef __DEVICE_EMULATION__
		return 1;
	#else
		return 0;
	#endif
}

double GetTime()
{
	FUN_MESSAGE(5,"GetTime()");
	struct timeval tv;
	gettimeofday(&tv,NULL);
	return (double)tv.tv_sec+tv.tv_usec*1E-6;
}

double Rand()
{
	FUN_MESSAGE(5,"Rand()");
	static int rand_init=0;

	if (!rand_init){
		rand_init=1;
		struct timeval tv;
		gettimeofday(&tv,NULL);
		srand(tv.tv_usec);
	}
	return 1.0*rand()/RAND_MAX; // Rand: [0;1]
}

size_t CalcCudaMemory(const struct state_gx s)
{
	FUN_MESSAGE(5,"CalcCudaMemory()");

	const size_t allocmem=
		sizeof(float)                   * s.sz_shortrange_table +
		sizeof(struct NODE_gx)          * s.sz_Nodes_base +
		sizeof(int)                     * s.sz_Nextnode +
		sizeof(int)                     * s.sz_DomainTask +
		sizeof(int)                     * s.sz_Exportflag +
		sizeof(struct result_gx)        * s.sz_result +
		sizeof(struct particle_data_gx) * s.sz_P +
		sizeof(float)                       * s.sz_shortrange_table +
		sizeof(struct NODE_gx)              * s.sz_Nodes_base +
		sizeof(int)                         * s.sz_Nextnode +
		sizeof(int)                         * s.sz_DomainTask +
		sizeof(char)                        * s.sz_Exportflag +
		sizeof(struct result_gx)            * s.sz_result +
		sizeof(struct particle_data_gx)     * s.sz_P +
		#ifdef SPH
		sizeof(struct sph_particle_data_gx) * s.sz_SphP +
		#endif
		sizeof(struct etc_gx)               * s.sz_etc +
		sizeof(struct extNODE_gx)           * s.sz_extNodes_base +
		sizeof(int)                         * s.sz_Ngblist +
		#ifdef SPH
		sizeof(int)                         * s.sz_Ngblist +
		sizeof(struct result_hydro_gx)      * s.sz_result_hydro;
		#else
		sizeof(int)                         * s.sz_Ngblist;
		#endif

	return allocmem;
}

/*
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
*/
#define CHECKCUDAERROR(msg) CheckCUDAError_fun(msg,__FILE__,__LINE__)

void PrintDeviceInfo_gx(const cudaDeviceProp& deviceProp)
{
	FUN_MESSAGE(4,"PrintDevice_gx()");

	printf("Device info:\n");
	#if CUDART_VERSION >= 2020
		int driverVersion = 0, runtimeVersion = 0;
		cudaDriverGetVersion(&driverVersion);
		printf("  CUDA Driver Version:                           %d.%d\n", driverVersion/1000, driverVersion%100);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
	#endif
	printf("  Major revision number:                         %d\n", deviceProp.major);
	printf("  Minor revision number:                         %d\n", deviceProp.minor);
	printf("  Total amount of global memory:                 %lu bytes = %3.0f Mb\n",(long unsigned int)deviceProp.totalGlobalMem,deviceProp.totalGlobalMem/1024./1024+0.5);
	#if CUDART_VERSION >= 2000
		printf("  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
		printf("  Number of cores:                               %d\n", 8 * deviceProp.multiProcessorCount);
	#endif
	printf("  Total amount of constant memory:               %lu bytes\n", (long unsigned int)deviceProp.totalConstMem);
	printf("  Total amount of shared memory per block:       %lu bytes\n", (long unsigned int)deviceProp.sharedMemPerBlock);
	printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
	printf("  Warp size:                                     %d\n", deviceProp.warpSize);
	printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
	printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf("  Maximum memory pitch:                          %lu bytes\n", (long unsigned int)deviceProp.memPitch);
	printf("  Texture alignment:                             %lu bytes\n", (long unsigned int)deviceProp.textureAlignment);
	printf("  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
	#if CUDART_VERSION >= 2000
		printf("  Concurrent copy and execution:                 %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
		printf("  CanMapHostMemory:                              %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
	#endif
    #if CUDART_VERSION >= 2020
		printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  Compute mode:                                  %s\n", deviceProp.computeMode == cudaComputeModeDefault ? "Default (multiple host threads can use this device simultaneously)" :
			deviceProp.computeMode == cudaComputeModeExclusive  ? "Exclusive (only one host thread at a time can use this device)" :
			deviceProp.computeMode == cudaComputeModeProhibited ? "Prohibited (no host thread can use this device)" : "Unknown");
    #endif
}

void GetDriver(char*const d,const int len)
{
	FUN_MESSAGE(4,"GetDriver()");

	ASSERT_GX( d!=NULL && len>0);
	d[0]=0;

	FILE* file = fopen("/proc/driver/nvidia/version", "r");
	if (file == NULL) WARNING("open driver failed\n");
	else if ( fgets(d,len, file) == NULL) WARNING("read driver failed\n");

	int i=0;
	while(d[i]!=0 && d[i]!='\n' && i<len) ++i;
	if (i<len) d[i]=0;
}

void PrintConfig()
{
	FUN_MESSAGE(4,"PrintConfig()");

	// define config
	char defconfig[16*1024],temp[16*1024];
	MESSAGE("G2X version     =[%d]",G2X_VERSION);

	defconfig[0]=0;
	#ifdef CUDA_GX_USE_TEXTURES
	sprintf(defconfig,"%s CUDA_GX_USE_TEXTURES",defconfig);
	#endif
	#ifdef CUDA_GX_NO_SPH_SUPPORT
	sprintf(defconfig,"%s CUDA_GX_NO_SPH_SUPPORT",defconfig);
	#endif
	#ifdef CUDA_GX_DEBUG_MEMORY_ERROR
	sprintf(defconfig,"%s CUDA_GX_DEBUG_MEMORY_ERROR",defconfig);
	#endif
	#ifdef CUDA_GX_SHARED_NGBLIST
	sprintf(defconfig,"%s CUDA_GX_SHARED_NGBLIST",defconfig);
	#endif
	#ifdef CUDA_GX_CONSTANT_VARS_IN_SHARED_MEM
	sprintf(defconfig,"%s CUDA_GX_CONSTANT_VARS_IN_SHARED_MEM",defconfig);
	#endif
	#ifdef CUDA_GX_BITWISE_EXPORTFLAGS
	sprintf(defconfig,"%s CUDA_GX_BITWISE_EXPORTFLAGS",defconfig);
	#endif
	#ifdef CUDA_GX_BITWISE_EXPORTFLAGS
	sprintf(defconfig,"%s CUDA_GX_SEGMENTED",defconfig);
	#endif
	#ifdef CUDA_GX_BITWISE_EXPORTFLAGS
	sprintf(defconfig,"%s CUDA_GX_CHUNCK_MANAGER",defconfig);
	#endif
	#ifdef CUDA_DEBUG_GX
	sprintf(defconfig,"%s CUDA_DEBUG_GX=%d",defconfig,CUDA_DEBUG_GX);
	#endif
	#ifdef __DEVICE_EMULATION__
	sprintf(defconfig,"%s __DEVICE_EMULATION__",defconfig);
	#endif
	if (defconfig[0]==' ') {
		sprintf(temp,"%s",&defconfig[1]); // remove prefixing space
		sprintf(defconfig,"%s",temp);
	}
	MESSAGE("CUDA config     =[%s]",defconfig);

	defconfig[0]=0;
	#ifdef MAX_NGB_GX
	sprintf(defconfig,"%s MAX_NGB_GX=%d",defconfig,MAX_NGB_GX);
	#endif
	#ifdef MIN_FORCE_PARTICLES_FOR_GPU_GX
	sprintf(defconfig,"%s MIN_FORCE_PARTICLES_FOR_GPU_GX=%d",defconfig,MIN_FORCE_PARTICLES_FOR_GPU_GX );
	#endif
	#ifdef MIN_SPH_PARTICLES_FOR_GPU_GX
	sprintf(defconfig,"%s MIN_SPH_PARTICLES_FOR_GPU_GX=%d",defconfig,MIN_SPH_PARTICLES_FOR_GPU_GX );
	#endif
	if (defconfig[0]==' '){
		sprintf(temp,"%s",&defconfig[1]); // remove prefixing space
		sprintf(defconfig,"%s",temp);
	}
	MESSAGE("CUDA defines    =[%s]",defconfig);

	defconfig[0]=0;
	sprintf(defconfig,"GADGET2=%s"          ,sizeof(FLOAT_GX)==8 ? "double" : "float"); // XXX wrong float
	sprintf(defconfig,"%s G2X_interface=%s" ,defconfig,sizeof(FLOAT_GX)==8 ? "double" : "float");
	sprintf(defconfig,"%s G2X_internal=%s"  ,defconfig,sizeof(FLOAT_INTERNAL_GX)==8 ? "double" : "float");

	MESSAGE("float config    =[%s]",defconfig);

	defconfig[0]=0;
	#ifdef PERIODIC
	sprintf(defconfig,"%s PERIODIC",defconfig);
	#endif
	#ifdef UNEQUALSOFTENINGS
	sprintf(defconfig,"%s UNEQUALSOFTENINGS",defconfig);
	#endif
	#ifdef PEANOHILBERT
	sprintf(defconfig,"%s PEANOHILBERT",defconfig);
	#endif
	#ifdef WALLCLOCK
	sprintf(defconfig,"%s WALLCLOCK",defconfig);
	#endif
	#ifdef PMGRID
	sprintf(defconfig,"%s PMGRID=%d",defconfig,PMGRID);
	#endif
	#ifdef PLACEHIGHRESREGION
	sprintf(defconfig,"%s PLACEHIGHRESREGION=%d",defconfig,PLACEHIGHRESREGION);
	#endif
	#ifdef ENLARGEREGION
	sprintf(defconfig,"%s ENLARGEREGION=%d",defconfig,ENLARGEREGION);
	#endif
	#ifdef ASMTH
	sprintf(defconfig,"%s ASMTH=%g",defconfig,ASMTH);
	#endif
	#ifdef RCUT
	sprintf(defconfig,"%s RCUT=%g",defconfig,RCUT);
	#endif
	#ifdef DOUBLEPRECISION
	sprintf(defconfig,"%s DOUBLEPRECISION",defconfig);
	#endif
	#ifdef DOUBLEPRECISION_FFTW
	sprintf(defconfig,"%s DOUBLEPRECISION_FFTW",defconfig);
	#endif
	#ifdef SYNCHRONIZATION
	sprintf(defconfig,"%s SYNCHRONIZATION",defconfig);
	#endif
	#ifdef FLEXSTEPS
	sprintf(defconfig,"%s FLEXSTEPS",defconfig);
	#endif
	#ifdef PSEUDOSYMMETRIC
	sprintf(defconfig,"%s PSEUDOSYMMETRIC",defconfig);
	#endif
	#ifdef NOSTOP_WHEN_BELOW_MINTIMESTEP
	sprintf(defconfig,"%s NOSTOP_WHEN_BELOW_MINTIMESTEP",defconfig);
	#endif
	#ifdef NOPMSTEPADJUSTMENT
	sprintf(defconfig,"%s NOPMSTEPADJUSTMENT",defconfig);
	#endif
	#ifdef HAVE_HDF5
	sprintf(defconfig,"%s HAVE_HDF5",defconfig);
	#endif
	#ifdef OUTPUTPOTENTIAL
	sprintf(defconfig,"%s OUTPUTPOTENTIAL",defconfig);
	#endif
	#ifdef OUTPUTACCELERATION
	sprintf(defconfig,"%s OUTPUTACCELERATION",defconfig);
	#endif
	#ifdef OUTPUTCHANGEOFENTROPY
	sprintf(defconfig,"%s OUTPUTCHANGEOFENTROPY",defconfig);
	#endif
	#ifdef OUTPUTTIMESTEP
	sprintf(defconfig,"%s OUTPUTTIMESTEP",defconfig);
	#endif
	#ifdef NOGRAVITY
	sprintf(defconfig,"%s NOGRAVITY",defconfig);
	#endif
	#ifdef NOTREERND
	sprintf(defconfig,"%s NOTREERND",defconfig);
	#endif
	#ifdef NOTYPEPREFIX_FFTW
	sprintf(defconfig,"%s NOTYPEPREFIX_FFTW",defconfig);
	#endif
	#ifdef LONG_X
	sprintf(defconfig,"%s LONG_X=%g",defconfig,LONG_X);
	#endif
	#ifdef LONG_Y
	sprintf(defconfig,"%s LONG_X=%g",defconfig,LONG_Y);
	#endif
	#ifdef LONG_Z
	sprintf(defconfig,"%s LONG_X=%g",defconfig,LONG_Z);
	#endif
	#ifdef TWODIMS
	sprintf(defconfig,"%s TWODIMS",defconfig);
	#endif
	#ifdef SPH_BND_PARTICLES
	sprintf(defconfig,"%s SPH_BND_PARTICLES",defconfig);
	#endif
	#ifdef NOVISCOSITYLIMITER
	sprintf(defconfig,"%s NOVISCOSITYLIMITER",defconfig);
	#endif
	#ifdef COMPUTE_POTENTIAL_ENERGY
	sprintf(defconfig,"%s COMPUTE_POTENTIAL_ENERGY",defconfig);
	#endif
	#ifdef LONGIDS
	sprintf(defconfig,"%s LONGIDS",defconfig);
	#endif
	#ifdef ISOTHERMAL
	sprintf(defconfig,"%s ISOTHERMAL",defconfig);
	#endif
	#ifdef SELECTIVE_NO_GRAVITY
	sprintf(defconfig,"%s SELECTIVE_NO_GRAVITY=%d",defconfig,SELECTIVE_NO_GRAVITY);
	#endif
	#ifdef FORCETEST
	sprintf(defconfig,"%s FORCETEST=%g",defconfig,FORCETEST);
	#endif
	#ifdef MAKEGLASS
	sprintf(defconfig,"%s MAKEGLASS=%g",defconfig,MAKEGLASS);
	#endif
	#ifdef ADAPTIVE_GRAVSOFT_FORGAS
	sprintf(defconfig,"%s ADAPTIVE_GRAVSOFT_FORGAS",defconfig);
	#endif
	#ifdef FLOAT
	sprintf(defconfig,"%s FLOAT",defconfig);
	#endif
	if (defconfig[0]==' ') {
		sprintf(temp,"%s",&defconfig[1]); // remove prefixing space
		sprintf(defconfig,"%s",temp);
	}

	MESSAGE("gadget config   =[%s]",defconfig);

	defconfig[0]=0;
	#ifdef __CUDACC__
	sprintf(defconfig,"NVCC CUDART_VERSION=%d",CUDART_VERSION);

	#else
	sprintf(defconfig,"GCC version %d.%d.%d",__GNUC_MAJOR__,___GNUC_MINOR__,__GNUC_PATCHLEVEL__);
	#endif

	#ifdef NDEBUG
	sprintf(defconfig,"%s NDEBUG",defconfig);
	#endif
	#ifdef _NDEBUG
	sprintf(defconfig,"%s _NDEBUG",defconfig);
	#endif
	#ifdef PROFILE
	sprintf(defconfig,"%s PROFILE",defconfig);
	#endif
	#ifdef DEBUG
	sprintf(defconfig,"%s DEBUG",defconfig);
	#endif
	#ifdef _DEBUG
	sprintf(defconfig,"%s _DEBUG",defconfig);
	#endif

	if      (sizeof(void*)==4) sprintf(defconfig,"%s 32BIT",defconfig);
	else if (sizeof(void*)==8) sprintf(defconfig,"%s 64BIT",defconfig);
	else                       sprintf(defconfig,"%s XXBIT",defconfig);

	const long one= 1;
	const int big=!(*(const char *)(&one)); // actually, not tested on a bigendian system, hope it works!
	if (big) sprintf(defconfig,"%s BIGENDIAN",defconfig);
	else     sprintf(defconfig,"%s LITENDIAN",defconfig);

	MESSAGE("compiler config =[%s]",defconfig);

	defconfig[0]=0;
	GetDriver(defconfig,1024);
	#if CUDART_VERSION >= 2020
		int driverVersion = 0, runtimeVersion = 0;
		cudaDriverGetVersion(&driverVersion);
		sprintf(defconfig,"%s CUDA Driver=%d.%d",defconfig,driverVersion/1000, driverVersion%100);
		cudaRuntimeGetVersion(&runtimeVersion);
		sprintf(defconfig,"%s CUDA Runtime=%d.%d",defconfig,runtimeVersion/1000, runtimeVersion%100);
	#endif
	MESSAGE("CUDA version config=[%s]",defconfig);
}

int AssignDevice_gx(const int thistask,const int localrank,const char host[1024],const int deviceCount)
{
	FUN_MESSAGE(3,"AssignDevice_gx(thistask=%d,deviceCount=%d)",thistask,deviceCount);

	int dev=-1;

	// NOTE: Custom GPU device assignent, this is for my system:
	if  (strncmp(host,"nv",1024)==0)       dev=thistask % 2 ? 0 : 2; // Tesla-Tesla config
	else if (strncmp(host,"mann",1024)==0) dev=0;
	else if (strncmp(host,"jep",1024)==0)  dev=0;
	else if (strncmp(host,"comp0",5)==0)   dev=thistask % 4; // cseth config, 4 GPUs per node
	else if (strncmp(host,"uchu",4)==0)    dev=thistask % 2; // uchu config, 2 GPUs per node
	else if (strncmp(host,"node",4)==0)    dev=thistask % 4; // cph config, 4 GPUs per node
	else {
		dev = localrank;
		WARNING("Using MPI coloring by hostname to assign tasks to GPUs...");

		/* WARNING("host not configured in AssignDevice_gx() [%s:%d], defaulting to 'dev=thistask modulus deviceCount'",__FILE__,__LINE__);
		dev=thistask % deviceCount; */
	}

	if (dev<0) ERROR("device not properly configurated");
	return dev;
}

int Initialize_cuda_gx(const int argc,char*const*const argv,const int thistask,const int localrank)
{
	FUN_MESSAGE(4,"Initialize_cuda_gx(thistask=%d, localrank=%d)",thistask,localrank);
	SetThisTask(thistask);

	// initialize the requested device
	char host[1024];
	if (gethostname(host,1024)!=0) ERROR("could not call get hostname");

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	CHECKCUDAERROR("cudaGetDeviceCount()");

	const int dev = AssignDevice_gx(thistask,localrank,host,deviceCount);
	//const int dev = 0;

	if (deviceCount==0)  ERROR("cudaErrorInitializationError, no devices supporting CUDA");
	if (deviceCount<dev) ERROR("cudaErrorInitializationError, requested device number larger than present devices");

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	CHECKCUDAERROR("cudaGetDeviceProperties()");

	if (deviceProp.major<1) ERROR("cudaErrorInvalidDevice, device does not support CUDA v1.x");

	cudaSetDevice(dev);
	CHECKCUDAERROR("cudaSetDevice()");
	
	int cuda_driver_ver, cuda_runtime_ver;
	cudaDriverGetVersion(&cuda_driver_ver);
    cudaRuntimeGetVersion(&cuda_runtime_ver);
	MESSAGE("Initalizing GX cards: using device %2d on host %2d='%s', '%s' with driver %2d and CUDA runtime %2d\n",dev,thistask,host,deviceProp.name,cuda_driver_ver,cuda_runtime_ver);

	// Get properties and verify device 0 supports mapped memory
	#if CUDART_VERSION < 2020
		#error "This CUDART version does not support mapped memory!\n"
	#endif
	if(!deviceProp.canMapHostMemory) ERROR("Device %d cannot map host memory",dev);
	cudaSetDeviceFlags(cudaDeviceMapHost);
	CHECKCUDAERROR("cudaSetDeviceFlags");

	if (deviceProp.major == 9999 && deviceProp.minor == 9999) WARNING("There is no device supporting CUDA");
	printf("Device %d on host %d='%s': '%s'",dev,thistask,host,deviceProp.name);
	PrintDeviceInfo_gx(deviceProp);
	g_cuda_mem=deviceProp.totalGlobalMem;

	if ((sizeof(FLOAT_GX)>4 || sizeof(FLOAT_INTERNAL_GX)>4) && (deviceProp.major==1 && deviceProp.minor<3)) ERROR("double precision floats are not supported by this device");

	FILE* fp=fopen("cudakernelsize.txt","r");
	if (fp!=NULL){
		char s0[256],s1[256];
		fscanf(fp,"%s %s",s0,s1);
		g_kernelsize_blocks=atoi(s0);
		g_kernelsize_grid =atoi(s1);
		if (g_kernelsize_blocks<1 || g_kernelsize_blocks>1024) ERROR("bad kernel block size in <cudakernelsize.txt> file");
		if (g_kernelsize_grid  <1 || g_kernelsize_grid  >1024) ERROR("bad kernel grid size in <cudakernelsize.txt> file");

		if (thistask==0) MESSAGE("CUDA kernel size=[%d,%d] (block,grids) = %d (total threads)",g_kernelsize_blocks,g_kernelsize_grid,g_kernelsize_blocks*g_kernelsize_grid);
		fclose(fp);
	}

	TestAll();
	if (argc || argv); // avoid compiler warn

	return g_cuda_mem;
}

void FreeAllCudamem_gx()
{
	FUN_MESSAGE(4,"FreeAllCudamem_gx()");

	Free_cuda_gx((void**)&s_gx_cuda.P);
	Free_cuda_gx((void**)&s_gx_cuda.result);
	Free_cuda_gx((void**)&s_gx_cuda.Exportflag);

	ASSERT_GX(s_gx_cuda.shortrange_table==NULL &&  s_gx_cuda.P==NULL && s_gx_cuda.result==NULL);

	Free_cuda_gx((void**)&s_gx_cuda.Nodes_base);
	Free_cuda_gx((void**)&s_gx_cuda.Nodes);
	Free_cuda_gx((void**)&s_gx_cuda.Nextnode);
	Free_cuda_gx((void**)&s_gx_cuda.DomainTask);

	ASSERT_GX(s_gx_cuda.Nodes_base==NULL && s_gx_cuda.Nodes==NULL && s_gx_cuda.Nextnode==NULL && s_gx_cuda.DomainTask==NULL);

	s_gx_cuda.sz_shortrange_table=0;
	s_gx_cuda.sz_P=0;
	s_gx_cuda.sz_result=0;
	s_gx_cuda.sz_Exportflag=0;

	s_gx_cuda.sz_Nodes_base=0;
	s_gx_cuda.sz_Nextnode=0;
	s_gx_cuda.sz_DomainTask=0;
}

void FinalizeCalculation_cuda_gx()
{
	FUN_MESSAGE(2,"FinalizeCalculation_cuda_gx()");
	Free_KernelSignals();

	#if CUDA_DEBUG_GX>1
		// memory gets deallocate automatically at program end, but here we only check for any mem leaks		Free_cuda_gx((void**)&s_gx_cuda.shortrange_table);
		// FreeAllCudamem_gx();

		// check for leaks
		const size_t t1=DebugMem_Size();
		const size_t t2=CalcCudaMemory(s_gx_cuda);

		if (t1!=0 || t1!=0) WARNING("expected all memory to be deallocated, leak: allocated=%d, expected=%d bytes",t1,t2);
		if (t1!=t2)  WARNING("all cuda memory not deallocated, leak: allocated=%d, expected=%d bytes",t1,t2);
		else         MESSAGE("INFO: cuda mem leak check OK: allocated=%d, expected=%d bytes",t1,t2);
	#endif
}

#define ALLOCATE_DATA(NAME,TYPE,MSG) \
		t=sizeof(TYPE)*s_gx.sz_##NAME;\
		if (s_gx_cuda.sz_##NAME!=s_gx.sz_##NAME){\
			Free_cuda_gx((void**)&s_gx_cuda.NAME);\
			s_gx_cuda.NAME=(TYPE*)Malloc_cuda_gx(t,MSG,__FILE__,__LINE__);\
			s_gx_cuda.sz_##NAME=s_gx.sz_##NAME;\
		}

#define ALLOCATE_DATA_WITH_HEADROOM(NAME,TYPE,MSG) \
		t=sizeof(TYPE)*s_gx.sz_##NAME;\
		if (s_gx_cuda.sz_##NAME!=s_gx.sz_##NAME){\
			if (s_gx_cuda.sz_max_##NAME!=s_gx.sz_max_##NAME){\
				Free_cuda_gx((void**)&s_gx_cuda.NAME);\
				s_gx_cuda.NAME=(TYPE*)Malloc_cuda_gx(sizeof(TYPE)*s_gx.sz_max_##NAME,MSG,__FILE__,__LINE__);\
				s_gx_cuda.sz_max_##NAME=s_gx.sz_max_##NAME;\
			}\
			s_gx_cuda.sz_##NAME=s_gx.sz_##NAME;\
		}

void InitializeResults_cuda_gx(const struct state_gx s_gx)
{
	FUN_MESSAGE(4,"InitializeResults_cuda_gx()");

	ASSERT_GX( s_gx.mode==0 || s_gx.mode==1);
	ASSERT_GX( s_gx.Np==s_gx.sz_result );

	size_t t;

	ASSERT_GX(s_gx.result!=NULL && s_gx.sz_result>0 );
	ALLOCATE_DATA_WITH_HEADROOM(result,struct result_gx,"result data");
	Memcpy_cuda_gx(s_gx_cuda.result,s_gx.result,t,0);

	ASSERT_GX( s_gx_cuda.sz_result==s_gx.sz_result );
	#ifdef CUDA_GX_DEBUG_MEMORY_ERROR
		#if CUDA_DEBUG_GX>0
			for(int i=0;(unsigned int)i<s_gx.sz_result;++i) ASSERT_GX( (s_gx.mode==0 && s_gx.result[i].temp2==i) || (s_gx.mode==1 && s_gx.result[i].temp2==-i));
			for(int i=s_gx.sz_result;(unsigned int)i<s_gx.sz_max_result;++i) ASSERT_GX( s_gx.result[i].temp2==0);
		#endif
	#endif
}

void InitializeScratch_cuda_gx(const struct state_gx s_gx)
{
	if (s_gx.sz_scratch>0){
		size_t t;
		ASSERT_GX(s_gx.scratch!=NULL && s_gx.sz_scratch>0 );
		ALLOCATE_DATA_WITH_HEADROOM(scratch,FLOAT_GX,"scratch data");
		Memcpy_cuda_gx(s_gx_cuda.scratch,s_gx.scratch,t,0);
	} else {
		s_gx_cuda.sz_scratch=0;
	}

	ASSERT_GX( s_gx_cuda.sz_scratch==s_gx.sz_scratch );
	ASSERT_GX( s_gx_cuda.sz_result==s_gx_cuda.sz_scratch || s_gx.mode==0);
}

void SyncRunTimeData(const struct state_gx& s,const int updateinteration)
{
	FUN_MESSAGE(4,"SyncRunTimeData()");
	s_gx_cuda.NumPart=s.NumPart;
	s_gx_cuda.N_gas=s.N_gas;
	s_gx_cuda.Np=s.Np;
	s_gx_cuda.mode=s.mode;

	s_gx_cuda.segment=0;
	s_gx_cuda.sz_segments=0;

	s_gx_cuda.blocks =s.blocks;
	s_gx_cuda.grids  =s.grids;
	s_gx_cuda.sphmode=s.sphmode;

	if (updateinteration) s_gx_cuda.iteration++;

	ValidateState(s_gx_cuda,0,1,__FILE__,__LINE__);
	ASSERT_GX( s.external_node>=0 || EqualState(s,s_gx_cuda,0,updateinteration,__FILE__,__LINE__) );
}

void InitializeCalculation_cuda_gx(const struct state_gx s_gx,const int sphmode)
{
	FUN_MESSAGE(3,"InitializeCalculation_cuda_gx(sz_shortrange_table=%d,sz_P=%d,sphmode=%d)",s_gx.sz_shortrange_table,s_gx.sz_P,sphmode);

	ASSERT_GX( s_gx.mode==0 || s_gx.external_node>=0 );
	ASSERT_GX( s_gx.Np==s_gx.sz_result );
	ASSERT_GX( s_gx.sphmode==sphmode );

	if (s_gx.cudamode>=3) ERROR("cudamode>=3 not supported yet");

	size_t t;

	ASSERT_GX(s_gx.shortrange_table!=NULL && s_gx.sz_shortrange_table>0 );
	ALLOCATE_DATA(shortrange_table,float,"shortrange table");
	Memcpy_cuda_gx(s_gx_cuda.shortrange_table,s_gx.shortrange_table,t,0);

	ASSERT_GX(s_gx.etc!=NULL && s_gx.sz_etc>0 );
	ALLOCATE_DATA_WITH_HEADROOM(etc,struct etc_gx,"etc_gx data");
	Memcpy_cuda_gx(s_gx_cuda.etc,s_gx.etc,t,0);

	#ifdef SPH
	if(s_gx.mode==0 && s_gx.sphmode==1) InitializeResultsHydro_cuda_gx(s_gx);
	#endif
	InitializeResults_cuda_gx(s_gx);
	InitializeScratch_cuda_gx(s_gx);

	ASSERT_GX((s_gx.Exportflag!=NULL && s_gx.sz_Exportflag>0) || s_gx.NTask==1 );
	ALLOCATE_DATA_WITH_HEADROOM(Exportflag,char,"Exportflags data");

	ASSERT_GX(s_gx.NTask>=1);
	if (s_gx.NTask>1){
		ASSERT_GX(s_gx_cuda.sz_Exportflag==s_gx.sz_Exportflag);
		ASSERT_GX(s_gx_cuda.Exportflag && s_gx.Exportflag);

		t=sizeof(char)*s_gx.sz_Exportflag;
		memset(s_gx.Exportflag,0,t);
		Memcpy_cuda_gx(s_gx_cuda.Exportflag,s_gx.Exportflag,t,0);
	} else{
		ASSERT_GX(s_gx_cuda.sz_Exportflag==0 && s_gx.sz_Exportflag==0);
		ASSERT_GX(s_gx_cuda.Exportflag==0 && s_gx.Exportflag==0);
	}

	ASSERT_GX(s_gx.P!=NULL && s_gx.sz_P>0 );
	ALLOCATE_DATA_WITH_HEADROOM(P,struct particle_data_gx,"particle data");
	Memcpy_cuda_gx(s_gx_cuda.P,s_gx.P,t,0);

	s_gx_cuda.NumPart=s_gx.NumPart;
	ASSERT_GX( s_gx_cuda.NumPart==s_gx_cuda.sz_P);
	#ifdef SPH
	if (sphmode){
		ASSERT_GX((s_gx.SphP!=NULL && s_gx.sz_SphP>0) || (s_gx.SphP==NULL && s_gx.sz_SphP==0 && s_gx.result_hydro==NULL && s_gx.sz_result_hydro==0));
		ALLOCATE_DATA_WITH_HEADROOM(SphP,struct sph_particle_data_gx,"sph particle data");
		Memcpy_cuda_gx(s_gx_cuda.SphP,s_gx.SphP,t,0);

		s_gx_cuda.N_gas=s_gx.N_gas;
		ASSERT_GX( s_gx_cuda.N_gas==s_gx_cuda.sz_SphP  );
	}
	#endif
	SyncRunTimeData(s_gx,0);
	ASSERT_GX( s_gx.sphmode==sphmode );
}

void SyncStaticData(const struct state_gx& s)
{
	FUN_MESSAGE(4,"SyncStaticData()");

	#if CUDA_DEBUG_GX > 0
		const int oldext=s_gx_cuda.external_node;
	#endif

	s_gx_cuda.MaxPart         =s.MaxPart;
	s_gx_cuda.sz_memory_limit =s.sz_memory_limit;
	s_gx_cuda.MaxPart         =s.MaxPart;
	s_gx_cuda.ThisTask        =s.ThisTask;
	s_gx_cuda.NTask           =s.NTask;
	s_gx_cuda.NumPart         =s.NumPart;
	s_gx_cuda.N_gas           =s.N_gas;
	s_gx_cuda.Np              =s.Np;
	s_gx_cuda.cudamode        =s.cudamode;
	s_gx_cuda.debugval        =s.debugval;

	s_gx_cuda.segment         =0;
	s_gx_cuda.sz_segments     =0;
	s_gx_cuda.external_node   =-1;

	ValidateState(s_gx_cuda,0,0,__FILE__,__LINE__);
	ASSERT_GX( oldext>=0 || EqualState(s,s_gx_cuda,0,0,__FILE__,__LINE__) );
}

void CopyAuxData_cuda_gx(const struct state_gx s)
{
	FUN_MESSAGE(4,"CopyAuxData_cuda_gx(sz_Nodes_base=%d,sz_Nextnode=%d,sz_DomainTask=%d)",s.sz_Nodes_base,s.sz_Nextnode,s.sz_DomainTask);

	ASSERT_GX( s.mode==0 );

	s_gx_cuda.Nodes_base      =s.Nodes_base;
	s_gx_cuda.Nodes           =s_gx_cuda.Nodes_base-s.MaxPart; // reassign with corret base
	s_gx_cuda.Nextnode        =s.Nextnode;
	s_gx_cuda.DomainTask      =s.DomainTask;
	s_gx_cuda.extNodes_base   =s.extNodes_base;
	s_gx_cuda.extNodes        =s_gx_cuda.extNodes_base-s.MaxPart; // reassign with corret base
	s_gx_cuda.Ngblist         =s.Ngblist;

	s_gx_cuda.sz_Nodes_base   =s.sz_Nodes_base;
	s_gx_cuda.sz_Nextnode     =s.sz_Nextnode;
	s_gx_cuda.sz_DomainTask   =s.sz_DomainTask;
	s_gx_cuda.sz_extNodes_base=s.sz_extNodes_base;
	s_gx_cuda.sz_Ngblist      =s.sz_Ngblist;
	s_gx_cuda.sz_max_Ngblist  =s.sz_max_Ngblist;

	SyncStaticData(s);
}

void Trunckernelsize_gx(const int N,const int NTask,struct state_gx*const s)
{
	FUN_MESSAGE(4,"Trunckernelsize_gx(N=%d,NTask=%d)",N,NTask);
	ASSERT_GX( NTask>=1 && N>0 );

	//if (N<=32) {
	//	g_kernelsize_blocks=N;
	//	g_kernelsize_grid=1;
	//}
	//else if (g_kernelsize_blocks*g_kernelsize_grid>N){
	//	const int t=g_kernelsize_blocks*g_kernelsize_grid;
	//	int n=0;
	//	while (g_kernelsize_blocks*g_kernelsize_grid>N){
	//		if (++n%2==0 && g_kernelsize_blocks>2) g_kernelsize_blocks/=2;
	//		else if (g_kernelsize_grid>2)          g_kernelsize_grid/=2;
	//		else ERROR("could not find a suitable set of blocks and grids, such that g_kernelsize_blocks*g_kernelsize_grid<=N(=%d), NTask=%d",N,NTask);
	//	}
	//	WARNING("g_kernelsize_blocks*g_kernelsize_grid(=%d) > N(=%d), limiting the number of blocks and threads to (%d,%d)",t,N,g_kernelsize_blocks,g_kernelsize_grid);
	//}
	// //const int m=N%g_kernelsize_blocks*g_kernelsize_grid;
	// //if (m!=0 && NTask==1) WARNING("N(=%d) modulus g_kernelsize_blocks*g_kernelsize_grid(=%d) != 0 (%d)",N,g_kernelsize_blocks*g_kernelsize_grid,m);
	//
	//ASSERT_GX(g_kernelsize_blocks*g_kernelsize_grid<=N);

	ASSERT_GX(g_kernelsize_blocks>=1 &&  g_kernelsize_grid>=1 );
	ASSERT_GX(s!=NULL);

	s->blocks=g_kernelsize_blocks;
	s->grids =g_kernelsize_grid;

	#ifdef CHUNK_MANAGER_THREAD_DIV_SPH

	if(s->sphmode){
		s->blocks /= CHUNK_MANAGER_THREAD_DIV_SPH;
		s->grids  /= CHUNK_MANAGER_THREAD_DIV_SPH;
		if (s->blocks<=0) s->blocks=1;
		if (s->grids<=0)  s->grids=1;
	}
	#endif
}


#ifdef CUDA_GX_USE_TEXTURES
	void BindTexture(const int sz,const int elemsize,const cudaChannelFormatDesc*const channelDesc,const void* cuda_pointer,struct textureReference* tex)
	{
		FUN_MESSAGE(4,"BindTexture(sz=%d,elemsize=%d)",sz,elemsize);
		if (sz==0) return;

		ASSERT_GX( sz>0 && elemsize>0 && channelDesc!=NULL && cuda_pointer!=NULL && tex!=NULL );
		const int bsz=elemsize*sz;

		tex->addressMode[0] = cudaAddressModeClamp;
		tex->addressMode[1] = cudaAddressModeClamp;
		tex->filterMode = cudaFilterModePoint;
		tex->normalized = false;

		size_t offset=0;
		cudaBindTexture(&offset, tex, cuda_pointer, channelDesc, bsz);
		CHECKCUDAERROR("cudaBindTexture");
		if (offset!=0) ERROR("cudamem offset'ed %d bytes",offset);
	}
#endif

void BindTextures(const struct state_gx& s_cuda)
{
	FUN_MESSAGE(2,"BindTextures()");
	#ifdef CUDA_GX_USE_TEXTURES
		// bind data to texture

		if (sizeof(particle_data_gx)%sizeof(int4)!=0)   ERROR("sizeof(particle_data_gx)%sizeof(int4) must be zero to be used in texture cache");
		if (sizeof(struct NODE_gx)%(3*sizeof(int4))!=0) ERROR("if (sizeof(struct NODE_gx)%(3*sizeof(int4)) must be zero to be used in texture cache");
		if (sizeof(struct extNODE_gx)%(4*sizeof(float))!=0) ERROR("if (sizeof(struct extNODE_gx)%(4*sizeof(float)) must be zero to be used in texture cache");
		if (sizeof(struct sph_particle_data_gx)%(3*sizeof(int4))!=0) ERROR("if (sizeof(struct sph_particle_data_gx)%(3*sizeof(int4)) must be zero to be used in texture cache");

		const int pratio=sizeof(particle_data_gx)/sizeof(int4);

		ASSERT_GX( pratio==1 || pratio==2 );
		ASSERT_GX( sizeof(particle_data_gx)==4*4*pratio && sizeof(particle_data_gx)==pratio*sizeof(int4) );
		ASSERT_GX( sizeof(struct NODE_gx)  ==(8+4)*4 && sizeof(struct NODE_gx)==3*sizeof(int4) );
		ASSERT_GX( (size_t)(&s_cuda.Nodes_base[1])-(size_t)(&s_cuda.Nodes_base[0])==48 );
		ASSERT_GX( sizeof(struct sph_particle_data_gx)==12*4 && sizeof(sph_particle_data_gx)==3*sizeof(int4) );
		ASSERT_GX( (size_t)(&s_cuda.SphP[1])-(size_t)(&s_cuda.SphP[0])==48 );

		const cudaChannelFormatDesc channelDesc_int   = cudaCreateChannelDesc<int>();
		const cudaChannelFormatDesc channelDesc_float = cudaCreateChannelDesc<float>();
		const cudaChannelFormatDesc channelDesc_int4  = cudaCreateChannelDesc<int4>();
		const cudaChannelFormatDesc channelDesc_char  = cudaCreateChannelDesc<char>();
		const cudaChannelFormatDesc channelDesc_float4= cudaCreateChannelDesc<float4>();

		BindTexture(s_cuda.sz_P*pratio        ,sizeof(int4) ,&channelDesc_int4 ,s_cuda.P               ,&tex_P);
		BindTexture(s_cuda.sz_Nextnode        ,sizeof(int)  ,&channelDesc_int  ,s_cuda.Nextnode        ,&tex_Nextnode);
		BindTexture(s_cuda.sz_shortrange_table,sizeof(float),&channelDesc_float,s_cuda.shortrange_table,&tex_shortrange_table);
		BindTexture(s_cuda.sz_Nodes_base*3    ,sizeof(int4) ,&channelDesc_int4 ,s_cuda.Nodes_base      ,&tex_Nodes_base);
		BindTexture(s_cuda.sz_SphP*3          ,sizeof(int4) ,&channelDesc_int4 ,s_cuda.SphP            ,&tex_SphP);
		BindTexture(s_cuda.sz_DomainTask      ,sizeof(int)  ,&channelDesc_int  ,s_cuda.DomainTask      ,&tex_DomainTask);
		BindTexture(s_cuda.sz_Exportflag      ,sizeof(char) ,&channelDesc_char ,s_cuda.Exportflag      ,&tex_Exportflag);
		BindTexture(s_cuda.sz_extNodes_base*4 ,sizeof(float),&channelDesc_float,s_cuda.extNodes_base   ,&tex_extNodes_base);

		// NOTE: no need to chache result, read once, write once only
	#endif
}

void Finalizekernel(const double time_kernel,const struct state_gx& s,const struct state_gx& s_cuda,const int sphmode,const int chunckmode
	#if CUDA_DEBUG_GX > 1
		,struct printkernelstate& pks
	#endif
	)
{
	FUN_MESSAGE(2,"Finalizekernel(sphmode=%d,chunckmode=%d)",sphmode,chunckmode);

	ASSERT_GX(sphmode==0 || sphmode==1);
	ASSERT_GX(s.sphmode==sphmode);

	// cudaThreadSynchronize is deprecated, changed to use cudaDeviceSynchronize instead:
	cudaDeviceSynchronize(); // not really nessecary, will implicit be done in memcopy operation
	CHECKCUDAERROR("cudaDeviceSynchronize()");

	#if CUDA_DEBUG_GX > 1
		pks.ShowThreadInfo();
		if (s_cuda.debugval>3 || sphmode) pks.ShowThreadInfo(0);
	#endif

	Finalize_KernelSignals();

	//const FLOAT_GX mintime=s.Np>100000 ? 1E-2 : 1E-5;
	//if (time_kernel<(mode==0 ? mintime : 0.1*mintime)) ERROR("very low elapsed time for kernel T=%g sec, is the GPU running hot?",time_kernel);

	#ifdef CUDA_GX_SEGMENTED
		const unsigned int offset=GetSegmentOffset(s_cuda);
		const unsigned int sz=GetSegmentEnd(s_cuda)-offset;
	#endif

	if (!sphmode) {
		#ifdef CUDA_GX_DEBUG_MEMORY_ERROR
			ASSERT_GX( sizeof(result_gx)==32 );
		#else
			ASSERT_GX( sizeof(result_gx)==16 );
		#endif

		ASSERT_GX( s_cuda.sz_result==s.sz_result );
		#ifdef CUDA_GX_SEGMENTED
			ASSERT_GX( sz+offset<=s_cuda.sz_result );
			Memcpy_cuda_gx(s.result+offset,s_cuda.result+offset,sizeof(result_gx)*sz,1);
		#else
			Memcpy_cuda_gx(s.result,s_cuda.result,sizeof(result_gx)*s_cuda.sz_result,1);
		#endif

		#ifdef CUDA_GX_DEBUG_MEMORY_ERROR
			#if CUDA_DEBUG_GX>0
				for(int j=0;(unsigned int)j<s.sz_result;++j) {
					if  (!(((s.mode==0 && s.result[j].temp2==j+100000) || (s.mode==1 && s.result[j].temp2==j-100000)))) {
						MESSAGE("mode=%d, s.result[%d].temp2=%d, s.sz_result=%d, ",s.mode,j,s.result[j].temp2,s.sz_result);
						if (j>0)                         MESSAGE("mode=%d, s.result[%d-1].temp2=%d, s.sz_result=%d, ",s.mode,j,s.result[j-1].temp2,s.sz_result);
						if ((unsigned int)j<s.sz_result) MESSAGE("mode=%d, s.result[%d+1].temp2=%d, s.sz_result=%d, ",s.mode,j,s.result[j+1].temp2,s.sz_result);
					}
					ASSERT_GX( (s.mode==0 && s.result[j].temp2==j+100000) || (s.mode==1 && s.result[j].temp2==j-100000) );
				}
			#endif
		#endif
	} else {
	#ifdef SPH
		ASSERT_GX( s_cuda.sz_result_hydro==s.sz_result_hydro && sizeof(result_hydro_gx)==32  );
		#ifdef CUDA_GX_SEGMENTED
 			ASSERT_GX( sz+offset<=s_cuda.sz_result_hydro);
			Memcpy_cuda_gx(s.result_hydro+offset,s_cuda.result_hydro+offset,sizeof(result_hydro_gx)*sz,1);
		#else
			Memcpy_cuda_gx(s.result_hydro,s_cuda.result_hydro,sizeof(result_hydro_gx)*s_cuda.sz_result_hydro,1);
		#endif
	#endif
	}
	if (s_cuda.mode==0) {
		#ifdef CUDA_GX_SEGMENTED
 			ASSERT_GX( s_cuda.NTask*(sz+offset)<=s_cuda.sz_Exportflag);
			Memcpy_cuda_gx(s.Exportflag+s_cuda.NTask*offset,s_cuda.Exportflag+s_cuda.NTask*offset,sizeof(char)*s_cuda.NTask*sz,1);
		#else
			Memcpy_cuda_gx(s.Exportflag,s_cuda.Exportflag,sizeof(char)*s_cuda.sz_Exportflag,1);
		#endif
	}
}

char GetExportflag_gx(const struct state_gx* s,const unsigned int n,const unsigned int bitelemsize,const unsigned int index)
{
	FUN_MESSAGE(5,"GetExportflag_gx()");

	ASSERT_GX( s->mode==0 );
	ASSERT_GX( s->NTask>1 );
	ASSERT_GX( s!=NULL && bitelemsize==(unsigned int)s->NTask && n<s->sz_Exportflag && s->cudamode>0 );

	#ifdef CUDA_GX_BITWISE_EXPORTFLAGS
		// bitwise set
		int byte;
		const int bit=GetByteAndBitIndex_host(n,bitelemsize,index,&byte);
		ASSERT_GX( byte<s->sz_Exportflag );
		return (s->Exportflag[byte] >> bit) & 0x1;
	#else
		const int t=n*bitelemsize+index;
		return s->Exportflag[t];
	#endif
}

int GetExportflag_size_gx(const unsigned int N,const unsigned int bitelemsize)
{
	FUN_MESSAGE(5,"GetExportflag_size_gx()");

	#ifdef CUDA_GX_BITWISE_EXPORTFLAGS
		return N*GetByteSizeForBitArray_host(bitelemsize);
	#else
		return sizeof(char)*N*bitelemsize;
	#endif
}

#ifdef SPH
void InitializeResultsHydro_cuda_gx(const struct state_gx s_gx)
{
	FUN_MESSAGE(2,"InitializeResultsHydro_cuda_gx()");

	ASSERT_GX( s_gx.mode==0 );
	size_t t;

if (!((s_gx.result_hydro!=NULL && s_gx.sz_result_hydro>0) || (s_gx.SphP==NULL && s_gx.sz_SphP==0 && s_gx.result_hydro==NULL && s_gx.sz_result_hydro==0) ))
	MESSAGE("s_gx.result_hydro=%d, s_gx.sz_result_hydro=%d, s_gx.SphP=%p, s_gx.sz_SphP=%d, s_gx.result_hydro=%p, s_gx.sz_result_hydro=%d",s_gx.result_hydro,s_gx.sz_result_hydro,s_gx.SphP,s_gx.sz_SphP,s_gx.result_hydro,s_gx.sz_result_hydro);

	ASSERT_GX((s_gx.result_hydro!=NULL && s_gx.sz_result_hydro>0) || (s_gx.SphP==NULL && s_gx.sz_SphP==0 && s_gx.result_hydro==NULL && s_gx.sz_result_hydro==0) );
	ALLOCATE_DATA_WITH_HEADROOM(result_hydro,struct result_hydro_gx,"result hydro data");
	Memcpy_cuda_gx(s_gx_cuda.result_hydro,s_gx.result_hydro,t,0);
}

void InitializeHydraCalculation_cuda_gx(const struct state_gx s_gx)
{
	FUN_MESSAGE(2,"InitializeHydraCalculation_cuda_gx()");
	size_t t;

	ALLOCATE_DATA_WITH_HEADROOM(SphP,struct sph_particle_data_gx,"sph particle data");
	Memcpy_cuda_gx(s_gx_cuda.SphP,s_gx.SphP,t,0);

	ValidateState(s_gx_cuda,0,1,__FILE__,__LINE__);

	ASSERT_GX( s_gx.sz_SphP==s_gx_cuda.sz_SphP);
	ASSERT_GX( s_gx.SphP!=NULL && s_gx.sz_SphP>0 );
	ASSERT_GX( s_gx_cuda.SphP!=NULL && s_gx_cuda.sz_SphP>0 );
}
#endif

#if CUDA_DEBUG_GX > 1
	void InitDebugData(const struct& state_gx s,const struct printkernelstate*const pks)
	{
		FUN_MESSAGE(4,"InitDebugData()");

		ASSERT_GX( pks!=NULL );
		const size_t allocmem=CalcCudaMemory(s);

		ASSERT_GX( s.blocks>=1 && s.grids>=1 );
		const dim3 dimBlock(s.blocks,1);
		const dim3 dimGrid (s.grids,1);

		MESSAGE("calling kernel...[threads=%d;allocmem=%d MB,freemem=%d MB]",dimBlock.x*dimBlock.y*dimGrid.x*dimGrid.y,allocmem/1024/1024,(g_cuda_mem-allocmem)/1024/1024);
		s_gx_cuda.debug_msg=pks.msg;
		s_gx_cuda.debug_sz_msg=pks.sz_msg;
	}

	#define INIT_DEBUG_DATA\
			struct printkernelstate pks(dimGrid,dimBlock);\
			InitDebugData(s,pks)

#else
	#define INIT_DEBUG_DATA\
			s_gx_cuda.debug_msg=NULL;\
			s_gx_cuda.debug_sz_msg=0

#endif

double CallKernel(struct state_gx s_cuda,const struct state_gx& s,const struct parameters_gx& p,const struct parameters_hydro_gx*const h,const int chunckmode,const int sphmode)
{
	FUN_MESSAGE(1,"CallKernel(chunckmode=%d,sphmode=%d,seg=%d,segs=%d)",chunckmode,sphmode,s_cuda.segment,s_cuda.sz_segments);

	ASSERT_GX( (chunckmode==0 || chunckmode==1) && (sphmode==0 || sphmode==1) );
	ASSERT_GX( s_cuda.blocks==s.blocks && s_cuda.grids==s.grids );
	ASSERT_GX( s_cuda.segment<s_cuda.sz_segments );
	ASSERT_GX( sphmode==s_cuda.sphmode && sphmode==s.sphmode);
	ASSERT_GX( s_cuda.segment<s_cuda.Np/(s_cuda.blocks*s_cuda.grids) || s_cuda.segment+1==s_cuda.sz_segments );

	/* Prefer L1 cache to shared memory (since we don't use shared memory...) */
	cudaFuncSetCacheConfig(force_treeevaluate_shortrange_cuda_gx_kernel, cudaFuncCachePreferL1);

	CopyStateAndParamToConstMem(&s_cuda,&p,h);
	BindTextures(s_cuda);

	#if CUDA_DEBUG_GX > 1
		DebugMem_Dump(0);
	#endif

	ASSERT_GX( s_cuda.blocks>=1 && s_cuda.grids>=1 );

	const dim3 dimBlock(s_cuda.blocks,1);
	const dim3 dimGrid (s_cuda.grids,1);

	double time_kernel=GetTime();

	if (!sphmode) force_treeevaluate_shortrange_cuda_gx_kernel<<< dimGrid, dimBlock >>>();
	/* For compiling faster, we have temporarily disabled hydro functionality */
	/* else          hydro_evaluate_shortrange_cuda_gx_kernel    <<< dimGrid, dimBlock >>>(); */

	time_kernel=GetTime()-time_kernel;

	Finalizekernel(time_kernel,s,s_cuda,sphmode,chunckmode
		#if CUDA_DEBUG_GX > 1
			,pks
		#endif
	);

	return time_kernel;
}

#ifndef CUDA_GX_CHUNCK_MANAGER
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
		// call device-mode code, that is Cuda compiled C parallel code
		ASSERT_GX( s_gx_cuda.NumPart==s.NumPart );
		ASSERT_GX( Np<=s_gx_cuda.NumPart || mode==1);
		ASSERT_GX( s_gx_cuda.sz_result==s.sz_result && Np==s_gx_cuda.sz_result );
		//ASSERT_GX( (mode==0 && s_gx_cuda.sz_result==s.sz_result) || (mode==1 && s_gx_cuda.sz_result>=s.sz_result) );
		ASSERT_GX( s_gx_cuda.sphmode==0 );

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

		ASSERT_GX( s_gx_cuda.sz_segments>=1 );
		for(s_gx_cuda.segment=0;s_gx_cuda.segment<s_gx_cuda.sz_segments;++s_gx_cuda.segment){
			CallKernel(s_gx_cuda,s,p,NULL,0,0);
		}
	}
}

#ifdef SPH
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
		// call device-mode code, that is Cuda compiled C parallel code

		SyncRunTimeData(s,1);
		ASSERT_GX( s_gx_cuda.sphmode==1 );

		s_gx_cuda.kernelsignals=Initialize_KernelSignals();

		// create a kernel
		INIT_DEBUG_DATA;
		SetSegments();

		for(s_gx_cuda.segment=0;s_gx_cuda.segment<s_gx_cuda.sz_segments;++s_gx_cuda.segment){
			ASSERT_GX( N_gas==s_gx_cuda.N_gas );
			ASSERT_GX( s_gx_cuda.Np==s_gx_cuda.N_gas );
			CallKernel(s_gx_cuda,s,p,&h,0,1);
		}
	}
}
#endif

#endif
