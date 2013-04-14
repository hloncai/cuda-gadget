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

#define TEST(expr) Test(expr,#expr,__FILE__,__LINE__)

char g_test[256];

void Settestcase(const char* testcase)
{
	strncpy(g_test,testcase,256);
}

int Test(const int e,const char* expr,const char* file,const int line)
{
	if (!e){
		WARNING("%s:%d: test failed, '%s' failed in '%s'",file,line,g_test,expr);
		return 0;
	}
	else return 1;
}

/*
void GetThreadIndex(const int NumPart,const bool fragmented,int*const  tidx,int*const tidy,int*const tidz,const int threadIdxx,const int blockIdxx,const int blockDimx,const int gridDimx)
{
	*tidx=0;
	*tidy=0;
	*tidz=0;

	// 1D tid calculation
	// ASSERT_DEVICE_GX( blockDim.y==1 && blockDim.z==1 && gridDim.y==1 && gridDim.z==1 );

	// Thread index
	const int tx = threadIdxx;

	// Block index
	const int bx = blockIdxx;
	const size_t tdim=blockDimx * gridDimx;
	const size_t tid =tx*gridDimx + bx;
	//const int bs=NumPart/tdim;

	if (fragmented){
		*tidx=tid;
		*tidy=tid;
		*tidz=tdim;

		if (*tidy+*tidz>NumPart) *tidz=NumPart-*tidy;
		// ASSERT_DEVICE_GX( ty<NumPart && tz>0 && ty+tz<=NumPart);
	} else {
		//const int finalthread=(tx+1==blockDim.x);
		//const int target_beg=tid*bs;
		//const int target_end=finalthread ? NumPart : (tid+1)*bs;

		const int bs=NumPart/tdim + (NumPart%tdim==0 ? 0 : 1);
		const int target_beg=tid;
		const int target_end=tid+bs;

		*tidx=tid;
		*tidy=target_beg;
		*tidz=target_end;
		if (target_end>NumPart)	*tidz=NumPart;

		//ASSERT_DEVICE_GX( ty<tz && tz<=NumPart );
	}

	MESSAGE("NumPart=%d, tidx=%d, tidy=%d, tidz=%d, tidx+tidz=%d, threadIdxx=%d ,blockIdxx=%d, blockDimx%d, gridDimx=%d",NumPart,*tidx,*tidy,*tidz,*tidx+(*tidz),threadIdxx,blockIdxx,blockDimx,gridDimx);
	MESSAGE(" for(target_x=tid.y(%d);target_x<NumPart(%d);target_x+=tid.z(%d)",*tidy,NumPart,*tidz);

}

void TestAllTheadIndex()
{
	MESSAGE("TestAllTheadIndex...");
	int blockDimx,gridDimx,threadIdxx,blockIdxx,n,tx0,ty0,tz0,tx1,ty1,tz1;

	blockDimx=64;
	gridDimx=256;
	threadIdxx=0;
	blockIdxx=0;
	GetThreadIndex(55615,1,&tx0,&ty0,&tz0,threadIdxx,blockIdxx,blockDimx,gridDimx);

	threadIdxx=1;
	blockIdxx=0;
	GetThreadIndex(55615,1,&tx0,&ty0,&tz0,threadIdxx,blockIdxx,blockDimx,gridDimx);

	threadIdxx=0;
	blockIdxx=1;
	GetThreadIndex(55615,1,&tx0,&ty0,&tz0,threadIdxx,blockIdxx,blockDimx,gridDimx);

	threadIdxx=blockDimx-1;
	blockIdxx=0;
	GetThreadIndex(55615,1,&tx0,&ty0,&tz0,threadIdxx,blockIdxx,blockDimx,gridDimx);

	return ;


	for(blockDimx=1;blockDimx<1024;++blockDimx){
		for(gridDimx=1;gridDimx<1024;++gridDimx){
			for(threadIdxx=0;threadIdxx<blockDimx-1;++threadIdxx){
				for(blockIdxx=0;blockIdxx<gridDimx;++blockIdxx){
					for(n=1;n<1000000;n += 1123 ){
						GetThreadIndex(n,1,&tx0,&ty0,&tz0,threadIdxx,blockIdxx,blockDimx,gridDimx);
						TEST( tx0>=0 && tx0<n && tz0>0);

						GetThreadIndex(n,1,&tx1,&ty1,&tz1,threadIdxx+1,blockIdxx,blockDimx,gridDimx);
						TEST( tx1>=0 && tx1<n && tz1>0);

						TEST( tx0+tz0==tx1 );
					}
				}
			}
		}
	}
}
*/

void TestBitArray()
{
	#ifdef CUDA_GX_BITWISE_EXPORTFLAGS
		Settestcase("TestBitArray");

		char a[256*1024];
		char b[256*1024];

		for(int i=1;i<256;++i){
			int s=GetByteSizeForBitArray_host(i);

			for(int j=0;j<256*1024;++j) a[j]=b[j]=0;

			for(int n=0;n<16;++n) {
				for(int j=0;j<i;++j) {
					const int v=(j+i)%5==0;
					a[j]=v;

					const int bitelemsize=i;
					int byte;
					int bit=GetByteAndBitIndex_host(n,bitelemsize,j,&byte);
					if   (v) b[byte] |= 1 << bit;
				}
			}

			for(int n=0;n<16;++n) {
				for(int j=0;j<i;++j) {
					const int bitelemsize=i;
					int byte;
					int bit=GetByteAndBitIndex_host(n,bitelemsize,j,&byte);
					const int bi=(b[byte] >> bit) & 0x1;
					TEST(a[j]==bi);
					// MESSAGE("s=%d, n=%d, bitelemsize=%d, byte=%d, bit=%d",s,n,bitelemsize,byte,bit);
				}
			}
		}
	#endif
}

void TestAll()
{
	TestBitArray();
	//TestAllTheadIndex();
}
