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
#include <stdlib.h>
#include <time.h>
#include <sys/time.h> // for gettimeofday
#include <unistd.h>   // for sleep

#include "timer_gx.h"
#include "interface_gx.h"

//#define NO_TIMER_GX  // disable timer debug info calls
#define WALLCLOCK_GX   // use wall clock not cpu time
#define TIMER_CLOCKS_GX 256

#ifdef NO_TIMER_GX
	void TimerUpdateCounter(const int n,const int c) {}
	void TimerBeg(const int n) {}
	void TimerEnd(const int n) {}
	int  isTimerRunning(const int n);
	int  TimerReset(const int task) {return 0;}
	void TimerReport() {}
	void TimersSleep(const int sec) {}
	double TimerGet(const int n) {return 0;}
	double TimerGetAccumulated(const int n) {return 0;}
	double GetTime() {}
#else

#ifdef WALLCLOCK_GX
	#define CLOCK_TYPE_GX double
	CLOCK_TYPE_GX gettime(void)
	{
		struct timeval tv;
		gettimeofday(&tv,NULL);
		return (CLOCK_TYPE_GX)tv.tv_sec+tv.tv_usec*1E-6;
		//static CLOCK_TYPE_GX offset=0;
		//if (offset==0) offset=t;
		//return t-offset;
	}
	double GetTime() {return gettime();}
#else
	#define CLOCK_TYPE_GX clock_t
	CLOCK_TYPE_GX gettime() {return clock();}
	double GetTime() {return 1.0*gettime()/CLOCKS_PER_SEC;}
#endif

struct timeprofile {
	// struct timeval
	CLOCK_TYPE_GX t[TIMER_CLOCKS_GX];  // temp
	CLOCK_TYPE_GX a[TIMER_CLOCKS_GX];  // accum
	long long c[TIMER_CLOCKS_GX];      // count
	char d[TIMER_CLOCKS_GX];           // debug

	int task;
	CLOCK_TYPE_GX beg;
	unsigned int sleep;   // for sleeping, debug only
	int pad;
};

static struct timeprofile g_t;

void TimerUpdateCounter(const int n,const int c)
{
	if (n>TIMER_CLOCKS_GX) ERROR("Timer n=%d out of range",n);
	else if (g_t.d[n]!=1)  ERROR("Timer n=%d not started...aborting",n);
	else g_t.c[n] += c;
}

void TimerBeg(const int n)
{
	if (n>TIMER_CLOCKS_GX) ERROR("Timer n=%d out of range",n);
	else if (g_t.d[n]!=0)  ERROR("Timer n=%d already in use",n);
	else {
		g_t.d[n]=1;
		TimerUpdateCounter(n,1);
		g_t.t[n]=gettime();
	}
}

void TimerEnd(const int n)
{
	if (n>TIMER_CLOCKS_GX) ERROR("Timer n=%d out of range",n);
	else if (g_t.d[n]!=1)  ERROR("Timer n=%d not started",n);
	else{
		g_t.t[n]=gettime()-g_t.t[n];
		g_t.a[n]+=g_t.t[n];
		g_t.d[n]=0;
	}
}

int isTimerRunning(const int n)
{
	if (n>TIMER_CLOCKS_GX) ERROR("Timer n=%d out of range",n);
	return g_t.d[n];
}

int TimerReset(const int task)
{
	g_t.task=task;
	int i;
	for(i=0;i<TIMER_CLOCKS_GX;++i){
		g_t.t[i]=0;
		g_t.a[i]=0;
		g_t.d[i]=0;
	}
	g_t.beg=gettime();
	return 0;
}

double TimerGetAccumulated(const int n)
{
	if (n>TIMER_CLOCKS_GX) ERROR("Timer n=%d out of range",n);

	double t=g_t.a[n];
	if (g_t.d[n]!=0) t+=gettime()-g_t.t[n];

	#ifndef WALLCLOCK_GX
		t/=CLOCKS_PER_SEC;
	#endif

	return t;
}

double TimerGet(const int n)
{
	if (n>TIMER_CLOCKS_GX) ERROR("Timer n=%d out of range",n);

	double t=g_t.t[n];
	if (g_t.d[n]!=0) t+=gettime()-g_t.t[n];

	#ifndef WALLCLOCK_GX
		t/=CLOCKS_PER_SEC;
	#endif

	return t;
}

void TimerReport()
{
	if (g_t.task!=0) return;

	const double tnow=gettime();
	#ifdef WALLCLOCK_GX
		const double c=1.0*(tnow-g_t.beg);
	#else
		const double c=1.0*(tnow-g_t.beg)/CLOCKS_PER_SEC;
	#endif
	if (c<10) return;
	printf("\n## TimerReport: total time=%2.1f sec, task=%d",c,g_t.task);
	if (g_t.sleep>0) printf("  [sleeped for %d sec]",g_t.sleep);
	printf("\n");

	int i;
	for(i=0;i<TIMER_CLOCKS_GX;++i){
		double t=g_t.a[i];
		if (g_t.d[i]!=0) t+=tnow-g_t.t[i];

		if ((t>0 && c>0) || i==0){
			#ifndef WALLCLOCK_GX
				t/=CLOCKS_PER_SEC;
			#endif
			const double p=c==0 ? 0 : 100.*t/c;

			printf("##  n=%3d c=%8.0f t=%8.1f sec %5.1f %c",i,(double)g_t.c[i],t,p,37);

			if      (i==0) printf(" [main()]");
			else if (i==1) printf(" [run()]");
			else if (i==2) printf(" [run:find_next.../every_timestep...()]");
			else if (i==3) printf(" [run:domain_Decomposition()]");
			else if (i==4) printf(" [run:compute_accelerations()]");
			else if (i==5) printf(" [run:compute_potential()]");
			else if (i==6) printf(" [run:advance_and_find_timesteps()]");
			else if (i==7) printf(" [run:while-loop epilog]");
			else if (i==9) printf(" [force_treeevaluate_shortrange_gx()]");
			else if (i==10) printf(" [run:compute_accelerations:long_range_force]");
			else if (i==11) printf(" [run:compute_accelerations:gravity_tree()]");
			else if (i==12) printf(" [run:compute_accelerations:epilog()]");

			else if (i==20) printf(" [run:compute_accelerations:gravity_tree(), sector1]");
			else if (i==21) printf(" [run:compute_accelerations:gravity_tree(), sector2]");
			else if (i==22) printf(" [run:compute_accelerations:gravity_tree(), sector3]");
			else if (i==29) printf(" [run:compute_accelerations:gravity_tree(), total]");

			else if (i==30) printf(" [run:co.:grav.:force_treeevaluate_shortrange(),export]");
			else if (i==31) printf(" [run:co.:grav.:force_treeevaluate_shortrange()]");
			else if (i==39) printf(" [run:compute_accelerations:gravity_tree(), export.F]");

			else if (i==40) printf(" [while, total]");
			else if (i==41) printf(" [while, particles]");
			else if (i==42) printf(" [while, nodes]");

			else if (i==50) printf(" [run:co.:grav.:force.:GX UpdateNodeData]");
			else if (i==51) printf(" [run:co.:grav.:force.:GX InitializeCalculation_gx]");
			else if (i==52) printf(" [run:co.:grav.:force.:GX force_treeevaluate_shortrange,loop]");

			else if (i==60) printf(" [run:compute_accelerations(), sector1]");
			else if (i==61) printf(" [run:compute_accelerations(), sector2]");
			else if (i==62) printf(" [run:compute_accelerations(), sector3]");
			else if (i==63) printf(" [run:compute_accelerations(), sector4]");
			else if (i==64) printf(" [run:compute_accelerations(), sector5]");
			else if (i==65) printf(" [run:compute_accelerations(), sector6]");
			else if (i==66) printf(" [run:compute_accelerations(), sector7]");

			else if (i==70) printf(" [long_range_force(), all]");
			else if (i==71) printf(" [long_range_force(), sector1]");
			else if (i==72) printf(" [long_range_force(), sector2]");
			else if (i==73) printf(" [long_range_force(), sector3]");
			else if (i==74) printf(" [long_range_force(), sector4]");
			else if (i==75) printf(" [long_range_force(), sector5]");
			else if (i==76) printf(" [long_range_force(), sector6]");
			else if (i==77) printf(" [long_range_force(), sector7]");
			else if (i==78) printf(" [long_range_force(), sector8]");
			else if (i==79) printf(" [long_range_force(), sector9]");

			else if (i==80) printf(" [pmforce_periodic(), all]");
			else if (i==81) printf(" [pmforce_periodic(), sector1]");
			else if (i==82) printf(" [pmforce_periodic(), sector2]");
			else if (i==83) printf(" [pmforce_periodic(), sector3, FFT1]");
			else if (i==84) printf(" [pmforce_periodic(), sector4, Green's]");
			else if (i==85) printf(" [pmforce_periodic(), sector5, FFT2]");
			else if (i==86) printf(" [pmforce_periodic(), sector6]");
			else if (i==87) printf(" [pmforce_periodic(), sector7, finite differencing]");
			else if (i==88) printf(" [pmforce_periodic(), sector8]");
			else if (i==89) printf(" [pmforce_periodic(), sector9]");

			else if (i==90) printf(" [hydro_force(), all]");
			else if (i==91) printf(" [hydro_force(), core]");
			else if (i==92) printf(" [hydro_force(), export]");
			else if (i==93) printf(" [hydro_force(), hydro_evaluate()]");
			else if (i==94) printf(" [hydro_force(), hydro_evaluate(),export]");

			else if (i==100) printf(" [compute_accelerations(), all]");
			else if (i==101) printf(" [compute_accelerations(), long_range_force]");
			else if (i==102) printf(" [compute_accelerations(), gravity_tree]");
			else if (i==103) printf(" [compute_accelerations(), density]");
			else if (i==104) printf(" [compute_accelerations(), force_update_hmax]");
			else if (i==105) printf(" [compute_accelerations(), hydro_force]");

			//if (g_t.d!=0) printf(" WARN: timer still running ");
			if (g_t.d[i]!=0) printf(" * ");
			printf("\n");
		}
	}
}

void TimersSleep(int sec)
{
	#ifndef WALLCLOCK_GX
		Error, TimersSleep expect timer to be of type double
	#endif

	sleep(sec);
	g_t.beg += sec;
	g_t.sleep += sec;
	int i;
	for(i=0;i<TIMER_CLOCKS_GX;++i){
		if (g_t.d[i]!=0){
			if (g_t.t[i]<sec) ERROR("Strange timer value in TimersSleep(), value %d should be greater than %d",g_t.t[i],sec);
			g_t.t[i] += sec;
		}
	}
}

#endif
