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

#define CALC_ALIGNMENT_PADDING_GX(s,a)\
a=0;\
{\
        struct s p[17];\
        const size_t vp0=(size_t)(&p[0]);\
        const size_t vp1=(size_t)(&p[16]);\
        a=vp1-vp0;\
        a=a%(sizeof(struct s)*16);\
}

struct Gadget2CudaData_handshake_gx
{
	int    sizes  [16]; // 0=char, 1=int, 2=long, 3=size_t, 4=float, 5=double, 6=has extended types, 7=long long, 8=long double
	int    endian [64];
	float  floats [16];
	double doubles[16];
	int    structs[16]; // 0=particle_data_gx, 1=NODE_gx, 2=parameters_gx, 3=state_gx, 4=result_gx
};

#define TEST_HANDSHAKE_GX(expr) ((void)((expr) ? 0 : (Assert_fail_gx(__STRING(expr), __FILE__, __LINE__,0),0)));

#ifdef __CUDACC__
	struct Gadget2CudaData_handshake_gx FillHandshakedata_gx(void);
	extern "C" struct Gadget2CudaData_handshake_gx FillHandshakedata_cuda_gx(void)
#else
	struct Gadget2CudaData_handshake_gx FillHandshakedata_cuda_gx(void);
	struct Gadget2CudaData_handshake_gx FillHandshakedata_gx(void)
#endif
{
	unsigned int i;

	struct Gadget2CudaData_handshake_gx a;
	memset(&a,0,sizeof(struct Gadget2CudaData_handshake_gx));
	const int adjust32_to_64bit=sizeof(void*)==4 ? 0 : 1;

	// check primitive sizes
	ASSERT_GX( sizeof(int)==4 );
	ASSERT_GX( sizeof(long)==4*(adjust32_to_64bit+1) );
	ASSERT_GX( sizeof(size_t)==4*(adjust32_to_64bit+1) );
	ASSERT_GX( sizeof(void*)==4*(adjust32_to_64bit+1) );
	ASSERT_GX( sizeof(float)==4 );
	ASSERT_GX( sizeof(double)==8 );

	// check equal sizes non-cuda/cuda
	#ifndef __CUDACC__
		TEST_HANDSHAKE_GX( sizeof(struct NODE)==sizeof(struct NODE_gx) );
		TEST_HANDSHAKE_GX( sizeof(struct extNODE)==sizeof(struct extNODE_gx) );
		/* Temporarily disable hydro for grav testing... */
		/* TEST_HANDSHAKE_GX( sizeof(struct hydrodata_in)==sizeof(struct hydrodata_in_gx) ); */
	#endif

	// check alignment
	TEST_HANDSHAKE_GX( sizeof(struct particle_data_gx)    %16==0 );
	/* TEST_HANDSHAKE_GX( sizeof(struct sph_particle_data_gx)%16==0 ); */
	TEST_HANDSHAKE_GX( sizeof(struct NODE_gx)             %16==0 );
	TEST_HANDSHAKE_GX( sizeof(struct extNODE_gx)          %16==0 );
	TEST_HANDSHAKE_GX( sizeof(struct result_gx)           %16==0 );
	printf("etc_gx sizeof %d\n",sizeof(struct etc_gx));
	TEST_HANDSHAKE_GX( sizeof(struct etc_gx)              %16==0 );
	/*
	TEST_HANDSHAKE_GX( sizeof(struct result_hydro_gx)     %16==0 );
	TEST_HANDSHAKE_GX( sizeof(struct hydrodata_in_gx)     %16==0 );
	*/

	size_t n;
	CALC_ALIGNMENT_PADDING_GX(particle_data_gx,n);
	TEST_HANDSHAKE_GX( n==0 );

	/*
	CALC_ALIGNMENT_PADDING_GX(sph_particle_data_gx,n);
	TEST_HANDSHAKE_GX( n==0 );
	*/

	CALC_ALIGNMENT_PADDING_GX(NODE_gx,n);
	TEST_HANDSHAKE_GX( n==0 );

	CALC_ALIGNMENT_PADDING_GX(extNODE_gx ,n);
	TEST_HANDSHAKE_GX( n==0 );

	CALC_ALIGNMENT_PADDING_GX(result_gx,n);
	TEST_HANDSHAKE_GX( n==0 );

	CALC_ALIGNMENT_PADDING_GX(etc_gx,n);
	TEST_HANDSHAKE_GX( n==0 );

	/*
	CALC_ALIGNMENT_PADDING_GX(result_hydro_gx,n);
	TEST_HANDSHAKE_GX( n==0 );

	CALC_ALIGNMENT_PADDING_GX(hydrodata_in_gx,n);
	TEST_HANDSHAKE_GX( n==0 );
	*/

	// check alignment of floats and doubles in structs, test only a couple of double elements
	const int falign=4;
	#ifdef __CUDACC__
		const int dalign=8;
	#else
		const int dalign=4; // currently nvcc and gcc aligns different!
	#endif
	struct parameters_gx p;
	TEST_HANDSHAKE_GX( ((size_t)&p.Asmth[0])%falign==0 );
	TEST_HANDSHAKE_GX( ((size_t)&p.Asmth[1])%falign==0 );
	for(i=0;i<6;++i)	TEST_HANDSHAKE_GX( ((size_t)&p.Masses[i])%falign==0 );

	/*
	struct parameters_hydro_gx h;
	if (sizeof(h.fac_mu)==sizeof(double)){
		TEST_HANDSHAKE_GX( ((size_t)&h.hubble_a2)%dalign==0 );
		TEST_HANDSHAKE_GX( ((size_t)&h.fac_mu)%dalign==0 );
		TEST_HANDSHAKE_GX( ((size_t)&h.ArtBulkViscConst)%dalign==0 );
	}
	*/

	// set sizes
	a.sizes[0]=sizeof(char);
	a.sizes[1]=sizeof(int);
	a.sizes[2]=sizeof(long);
	a.sizes[3]=sizeof(size_t);
	a.sizes[4]=sizeof(float);
	a.sizes[5]=1;
	a.sizes[6]=sizeof(double);
	a.sizes[7]=sizeof(long long);
	a.sizes[8]=sizeof(long double);

	// check sizes
	TEST_HANDSHAKE_GX( a.sizes[0]==1 );
	TEST_HANDSHAKE_GX( a.sizes[1]==4 );
	TEST_HANDSHAKE_GX( a.sizes[2]==4+adjust32_to_64bit*4 );
	TEST_HANDSHAKE_GX( a.sizes[3]==4+adjust32_to_64bit*4 );
	TEST_HANDSHAKE_GX( a.sizes[4]==4 );
	if (a.sizes[5]){
		TEST_HANDSHAKE_GX( a.sizes[6]==8 );
		TEST_HANDSHAKE_GX( a.sizes[7]==8 );
		TEST_HANDSHAKE_GX( a.sizes[8]==12+adjust32_to_64bit*4 );
	}

	TEST_HANDSHAKE_GX( sizeof(int)<=64 );
	for(i=0;i<sizeof(int);++i){
		a.endian[i]=1<<i;
	}
	for(i=0;i<16;++i){
		a.floats [i]=(float)sin(i/16.);
		a.doubles[i]=sin(i/16.);
	}

	// structs
	a.structs[0]=sizeof(struct particle_data_gx[2]);
	// a.structs[1]=sizeof(struct sph_particle_data_gx[2]);
	a.structs[1]=0;
	a.structs[2]=sizeof(struct NODE_gx[2]);
	// printf("int %d float %d double %d node sizeof %d",sizeof(int),sizeof(float),sizeof(double),sizeof(struct NODE_gx[2]));
	a.structs[3]=sizeof(struct extNODE_gx[2]);
	a.structs[4]=sizeof(struct result_gx);
	printf("result_gx %d\n",sizeof(struct result_gx));
	// a.structs[5]=sizeof(struct result_hydro_gx[2]);
	a.structs[5]=0;
	a.structs[6]=sizeof(struct etc_gx[2]);
	a.structs[7]=sizeof(struct parameters_gx);
	printf("parameters_gx %d\n", a.structs[7]); // check for double alignment issues
	// a.structs[8]=sizeof(struct parameters_hydro_gx);
	a.structs[8]=0;
	a.structs[9]=sizeof(struct state_gx);
	// a.structs[10]=sizeof(struct hydrodata_in_gx);
	a.structs[10]=0;

	const int si =sizeof(int);
	const int sp =sizeof(void*);
	const int sf =sizeof(FLOAT);
	const int sfg=sizeof(FLOAT_GX);
	//const int sd =sizeof(double);

	#ifdef ADAPTIVE_GRAVSOFT_FORGAS
		const int agf=1;
	#else
		const int agf=0;
	#endif

	#ifdef CUDA_GX_DEBUG_MEMORY_ERROR
		const int rpad=1;
	#else
		const int rpad=0;
	#endif

	/* 
		// disable these tests -- switching from single/double makes this all change b/c of alignment

	TEST_HANDSHAKE_GX( a.structs[0]/2==sf*4);                  // particle
	// TEST_HANDSHAKE_GX( a.structs[1]/2==si*2+sfg*10);           // sph particle
	TEST_HANDSHAKE_GX( a.structs[2]/2==si*12+sf*(4) );           // node
	TEST_HANDSHAKE_GX( a.structs[3]/2==4*sfg);                 // extNode
	TEST_HANDSHAKE_GX( a.structs[4]==si*2+sfg*3+rpad*(si*3+sf*1) ); // result
	// TEST_HANDSHAKE_GX( a.structs[5]/2==sfg*(5+3));             // result hydro, padded with 12
	TEST_HANDSHAKE_GX( a.structs[6]/2==si*6+sfg*1);            // etc
	TEST_HANDSHAKE_GX( a.structs[7]  ==si*12+sfg*21);          // parameters
	// TEST_HANDSHAKE_GX( a.structs[8]  ==si*2+sfg*12);           // sph parameters
	TEST_HANDSHAKE_GX( a.structs[9]  ==sp*20+si*48);           // state
	// TEST_HANDSHAKE_GX( a.structs[10] ==sfg*12+si*4);           // hydrodata_in

		// end struct size tests
	*/

	return a;
}

#ifndef __CUDACC__
	int TestHandshakedata_gx(void)
	{
		const struct Gadget2CudaData_handshake_gx a=FillHandshakedata_gx();
		const struct Gadget2CudaData_handshake_gx b=FillHandshakedata_cuda_gx();

		int i;
		for(i=0;i<16;++i) if (a.sizes [i]!=b.sizes [i]) {MESSAGE("mismatch in sizes, i=%d",i); return -1;}
		for(i=0;i<64;++i) if (a.endian[i]!=b.endian[i]) {MESSAGE("mismatch in endianessc",i); return -2;}
		for(i=0;i<16;++i) {
			if(a.floats [i]!=b.floats [i]) {MESSAGE("mismatch in float data, i=%d",i);  return -3;}
			if(a.doubles[i]!=b.doubles[i]) {MESSAGE("mismatch in double data, i=%d",i); return -4;}
			if(a.structs [i]!=b.structs[i]) {MESSAGE("mismatch in struct sizes, i=%d",i); return -i;}
		}

		return sizeof(struct Gadget2CudaData_handshake_gx); // OK
	}
#endif


