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

#define TIMER_TYPE float

#ifdef CUDA_GX_TIMERS

#define TIMER_N 16
#define TIMER_THREAD_MONITOR 0

__shared__ TIMER_TYPE g_tim[2*TIMER_N+1];
__shared__ TIMER_TYPE g_clk;
__shared__ unsigned int g_clk_wraps;

#define START_CLOCK(i) if (GetTid()==TIMER_THREAD_MONITOR) g_tim[TIMER_N+i]=clock_gx();
#define STOP_CLOCK(i)  if (GetTid()==TIMER_THREAD_MONITOR) g_tim[i]+=clock_gx()-g_tim[TIMER_N+i];

__device__ void initclock_gx()
{
	if (GetTid()==0){
		g_clk_wraps=0;
		for(int i=0;i<2*TIMER_N;++i) g_tim[i]=0;
	}
}

__device__ void updateclock_gx()
{
	if (GetTid()==0){
		if (clock()<g_clk) ++g_clk_wraps;
		g_clk=clock();
	}
}

__device__ float clock_gx()
{
	updateclock_gx();
	return 1.0*g_clk+2147483648.0*g_clk_wraps;
}

__device__ void finalizeclock_gx(TIMER_TYPE* const tim)
{
	if (GetTid()==TIMER_THREAD_MONITOR) for(int i=0;i<TIMER_N;++i) tim[i]=g_tim[i];
	if (GetTid()==TIMER_THREAD_MONITOR) {tim[14]=clock_gx(); tim[15]=g_clk_wraps;}
}

TIMER_TYPE* Inittimers_gx()
{
	return (TIMER_TYPE*)Malloc_cuda_gx(sizeof(int)*TIMER_N,"timers",__FILE__,__LINE__);
}

void Printtimers_gx(const TIMER_TYPE*const tim)
{
	ASSERT_GX(tim!=0);
	const float speed=1.30E9/2;
	TIMER_TYPE tcpy[TIMER_N];
	Memcpy_cuda_gx(&tcpy[0],tim,sizeof(TIMER_TYPE)*TIMER_N,1);
	if (1.0*tcpy[0]/speed>8.0){
		for(int i=0;i<TIMER_N;++i){
			if (tcpy[i]>0) MESSAGE("timer[%2d]=%4.4f sec = %5.1f%c, ticks=%.0f, ",i,1.0*tcpy[i]/speed,100.0*tcpy[i]/tcpy[0],'%',tcpy[i]);
		}
		static int n=0;
		if (n>4) abort();
	}
	//Free_cuda_gx((void**)&tim);
}

#else
	#define initclock_gx()
	#define updateclock_gx()
	#define finalizeclock_gx(tim)
	#define Inittimers_gx() 0
	#define Printtimers_gx(tim)

	#define START_CLOCK(i)
	#define STOP_CLOCK(i)
#endif

// USAGE:
//
// static TIMER_TYPE* tim=Inittimers_gx();
//
//  (call kernel)
//
// Printtimers_gx(tim);
