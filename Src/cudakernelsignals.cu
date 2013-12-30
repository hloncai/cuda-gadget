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

#include "defines.h"
#include "options.h"
#include "cudautils.h"

static struct KernelSignals* g_kernelsignals_cuda=NULL;

struct KernelSignals* Initialize_KernelSignals()
{
	/* struct  KernelSignals s;
	const size_t t=sizeof(s);
	memset(&s,0,t);

	if (g_kernelsignals_cuda==NULL){
		g_kernelsignals_cuda=(struct KernelSignals*)Malloc_cuda_gx(t,"kernel signals",__FILE__,__LINE__);
	}
	Memcpy_cuda_gx(g_kernelsignals_cuda,&s,t,0);

	return g_kernelsignals_cuda; */
	return NULL;
}

void Finalize_KernelSignals()
{
	/*
	if (g_kernelsignals_cuda==NULL) ERROR("expected kernelsignals_cuda to be !NULL");

	struct  KernelSignals s;
	const size_t t=sizeof(s);

	Memcpy_cuda_gx(&s,g_kernelsignals_cuda,t,1);

	if (s.sig_exit!=0)   {
		if (s.sig_msg[0]) MESSAGE("ERROR: thread %d has signaled an exit(%d) condition, %s:%d",s.sig_tid,s.sig_exit,s.sig_msg,s.sig_line);
		else              MESSAGE("ERROR: thread %d has signaled an exit(%d) condition, line=%d",s.sig_tid,s.sig_exit,s.sig_line);
		if (s.m!=0)       MESSAGE("    DEBUG INFO: m=%d, o=%d, p=%d, q=%d, r=%d, s=%g, t=%g",s.m,s.o,s.p,s.q,s.r,s.s,s.t);
		exit(s.sig_exit);
	} else if (s.sig_abort!=0) {
		if (s.sig_msg[0]) MESSAGE("ERROR: thread %d has signaled an abort condition, %s:%d",s.sig_tid,s.sig_msg,s.sig_line);
		else              MESSAGE("ERROR: thread %d has signaled an abort condition, line=%d",s.sig_tid,s.sig_line);
		if (s.m!=0)       MESSAGE("    DEBUG INFO: m=%d, o=%d, p=%d, q=%d, r=%d, s=%g, t=%g",s.m,s.o,s.p,s.q,s.r,s.s,s.t);
		abort();
	}
	*/
	return;
}

void Free_KernelSignals()
{
	if (g_kernelsignals_cuda==NULL) WARNING("expected kernelsignals_cuda to be !NULL");
	else 	Free_cuda_gx((void**)&g_kernelsignals_cuda);
}
