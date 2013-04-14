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

extern "C" {
	#include "allvars.h"
	#include "interface_gx.h"
	#include "tools_gx.h"
}

#ifdef ENABLE_TOOLS_FUN_GX

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

using namespace std;

template<typename T> inline string tostring(const T& x)
{
	ostringstream os;
	os << x;
	return os.str();
}

class DebugMem
{
private:
	typedef pair<size_t,string>     t_pair;
	typedef map<const void*,t_pair> t_map;
	t_map m_mem;
	double m_accumulated;
	size_t m_calls;
	const bool m_reportleaks;

	static string HumanReadable(const size_t sz)
	{
		if      (sz<1024)           return tostring(sz) + " b";
		else if (sz<1024*1024)      return tostring(static_cast<int>(sz/1024.0+0.5)) + " Kb";
		else if (sz<1024*1024*1024) return tostring(static_cast<int>(sz/1024.0/1024.0+0.5)) + " Mb";
		else                        return tostring(static_cast<int>(sz/1024.0/1024.0/1024.0+0.5)) + " Gb";
	}

 public:
	DebugMem() :  m_accumulated(0), m_calls(0), m_reportleaks(false) {}

	~DebugMem()
	{
		if (m_reportleaks && size()>0) WARNING("All debug mapped memory not deallocated, bytes left=%d",size());
	}

	void Insert(const void* p,const size_t sz,const char* msg,const char* file,const int line)
	{
		if (m_mem.find(p)!=m_mem.end()) ERROR("pointer already in debug memory map");
		m_mem[p]=make_pair(sz,string(file) + ":" + tostring(line) + ": " + string(msg));
		m_accumulated += sz;
		++m_calls;
	}

	void Remove(const void* p)
	{
		t_map::iterator itt=m_mem.find(p);
		if (itt==m_mem.end()) ERROR("pointer not found in debug memory map");
		m_mem.erase(itt);
	}

	size_t size() const
	{
		size_t sz=0;
		for(t_map::const_iterator itt=m_mem.begin();itt!=m_mem.end();++itt) sz += itt->second.first;
		return sz;
	}

	void Dump(const int verbose) const
	{
		MESSAGE("Cuda memory dump:  allocated:%d bytes=%.1f Mb, accumulated:%.0f bytes=%.1f Mb, calls=%d",size(),size()/1024./1024.,m_accumulated,m_accumulated/1024.0/1024.0,m_calls);

		if (!verbose) return;
		for (t_map::const_iterator itt=m_mem.begin();itt!=m_mem.end();++itt){
			const size_t base=reinterpret_cast<size_t>(itt->first);
			string m = "p=" + tostring(base) + ":" + tostring(base+itt->second.first-1) + " s=" + tostring(itt->second.first) + "=" + HumanReadable(itt->second.first) + " msg=" + itt->second.second;
			MESSAGE(m.c_str());

			t_map::const_iterator itt2=itt;
			++itt2;
			if (itt2!=m_mem.end()){
				const size_t base2=reinterpret_cast<size_t>(itt2->first);
				ASSERT_GX(base2>=base+itt->second.first);
				const size_t v=base2-(base+itt->second.first);
				if (v>0) {
					m = "v=" + tostring(v) + " =" + HumanReadable(v);
					MESSAGE(m.c_str());
				}
			}
		}
	}
};

DebugMem g_mem;

void    DebugMem_Insert(const void* p,const size_t sz,const char*const msg,const char* file,const int line) {g_mem.Insert(p,sz,msg,file,line);}
void    DebugMem_Remove(const void* p) {g_mem.Remove(p);}
size_t  DebugMem_Size  ()              {return g_mem.size();}
void    DebugMem_Dump  (const int v)   {g_mem.Dump(v);}

class SortParticles
{
private:
	typedef vector<int>   t_arr;
	typedef vector<t_arr> t_sort_arr;
	typedef struct particle_data_gx     t_data;
	typedef const struct particle_data* t_pdata;

	t_pdata    m_p;
	t_sort_arr m_s;
	pair<t_data,t_data> m_pminmax;
	int m_res,m_N,m_Ti_Current,m_np;
	bool m_sphmode;

	size_t Getindex(const int i,const t_data& scale,const t_data& pmin) const
	{
		ASSERT_GX(i<m_N);
		t_pdata p=&m_p[i];

		int idx0 = static_cast<int>((p->Pos[0]-pmin.Pos[0])*scale.Pos[0]);
		int idx1 = static_cast<int>((p->Pos[1]-pmin.Pos[1])*scale.Pos[1]);
		int idx2 = static_cast<int>((p->Pos[2]-pmin.Pos[2])*scale.Pos[2]);

		if (idx0>=m_res) idx0=m_res-1;
		if (idx1>=m_res) idx1=m_res-1;
		if (idx2>=m_res) idx2=m_res-1;

		const int idx=idx0+idx1*m_res+idx2*m_res*m_res;

		ASSERT_GX(idx>=0 && idx<m_res*m_res*m_res);
		return idx;
	}

public:
	void Init(const int res,const int N,const t_pdata particles,const int Ti_Current,int const sphmode,const bool resetall=false)
	{
		m_p=particles;
		m_s.resize(res*res*res);

		m_res=res;
		m_N=N;
		m_Ti_Current=Ti_Current;
		m_sphmode=sphmode;

		ASSERT_GX(m_res>0 && N>0 && particles!=NULL);
		ASSERT_GX(m_N>1);

		m_np=-1;
		for(t_sort_arr::iterator itt=m_s.begin();itt!=m_s.end();++itt){
			itt->resize(0);
			if (resetall) itt->reserve(0);
		}
	}

	int Findminmax()
	{
		ASSERT_GX(m_N>0 && m_np==-1);
		t_data pmin,pmax;
		bool init=false;

		m_np=0;
		for(int i=0;i<m_N;++i) {
			if (m_sphmode) ASSERT_GX( P[i].Type==0 );
			if(P[i].Ti_endstep == m_Ti_Current) {
				m_np++;

				if (!init){
					for(int j=0;j<3;++j) pmin.Pos[j]=pmax.Pos[j]=m_p[i].Pos[j];
					init=true;
				}

				for(int j=0;j<3;++j) {
					if (m_p[i].Pos[j]<pmin.Pos[j]) pmin.Pos[j]=m_p[i].Pos[j];
					if (m_p[i].Pos[j]>pmax.Pos[j]) pmax.Pos[j]=m_p[i].Pos[j];
				}
			}
		}

		m_pminmax=make_pair<t_data,t_data>(pmin,pmax);
		if (m_np<=1) WARNING("to few particles to fullfull timestep criterion");

		#if CUDA_DEBUG_GX>0
			MESSAGE("INFO: timestep'ed particles=%d=%.2f %c",m_np,100.0*m_np/m_N,'%');
		#endif

		MESSAGE("m_Ti_Current=%d, m_np=%d, m_N=%d",All.Ti_Current,m_np,m_N);

		return m_np;
	}

	void Sort(const int Np,int*const sorted)
	{
		if (Np!=m_np) ERROR("mismatch in Np parameters, Np=%d, internal Np=%d",Np,m_np);
		if (m_np<=1) {
			WARNING("np not initialized or bad number of participating particles for this timestep");
			return;
		}
		if (m_N<=1)  ERROR("cannot sort N<=1 particles");
		//if (m_np==m_N) return m_np; // no need to sort

		t_data scale;
		for(int j=0;j<3;++j) {
			scale.Pos[j]=fabs(m_pminmax.second.Pos[j]-m_pminmax.first.Pos[j]);
			if (scale.Pos[j]<=0) {
				MESSAGE("INFO: N=%d Np=%d pmin={%.1f,%.1f,%.1f}, pmax={%.1f,%.1f,%.1f}, scale={%.1f,%.1f,%.1f}",m_N,m_np,m_pminmax.first.Pos[0],m_pminmax.first.Pos[1],m_pminmax.first.Pos[2],m_pminmax.second.Pos[0],m_pminmax.second.Pos[1],m_pminmax.second.Pos[2],1/scale.Pos[0],1/scale.Pos[1],1/scale.Pos[2]);
				ERROR("bad dist in particle data, pmax<=pmin");
			}
			scale.Pos[j]/=m_res;
			scale.Pos[j] =1.0/scale.Pos[j]; // inv scale
		}

		ASSERT_GX(m_s.size()==static_cast<size_t>(m_res*m_res*m_res));

		for(int i=0;i<m_N;++i){
			if(P[i].Ti_endstep == m_Ti_Current) {
				const size_t idx=Getindex(i,scale,m_pminmax.first);
				m_s[idx].push_back(i);
			}
		}

		#if CUDA_DEBUG_GX>0
			//MESSAGE("INFO: pmin={%.1f,%.1f,%.1f}, pmax={%.1f,%.1f,%.1f}, scale={%.1f,%.1f,%.1f}",m_pminmax.first.Pos[0],m_pminmax.first.Pos[1],m_pminmax.first.Pos[2],m_pminmax.second.Pos[0],m_pminmax.second.Pos[1],m_pminmax.second.Pos[2],1/scale.Pos[0],1/scale.Pos[1],1/scale.Pos[2]);
			double sum2=0;
		#endif

		int i=0;
		for(t_sort_arr::const_iterator itt1=m_s.begin();itt1!=m_s.end();++itt1){
			for(t_arr::const_iterator itt2=itt1->begin();itt2!=itt1->end();++itt2){
				const int id=*itt2;

				ASSERT_GX(id>=0 && id<m_N && i<m_N && i<m_np);
				if (sorted!=NULL) sorted[i++]=id;

				#if CUDA_DEBUG_GX>0
					if (itt2+1!=itt1->end())
					for(int j=0;j<3;++j) {
						const double t=m_p[id].Pos[j]-m_p[id+1].Pos[j];
						sum2 += t*t;
					}
				#endif
			}
		}

		ASSERT_GX(m_np>0 && m_np<=m_N);
		m_np=-1;

		#if CUDA_DEBUG_GX>0
			MESSAGE("INFO: sort DistRMS=%f",sqrt(sum2/(m_N-1)));
		#endif
	}
};

static SortParticles g_sort;

int SortParticles_Init(const size_t N,const struct particle_data*const particles,const int Ti_Current,const int sphmode)
{
	// MESSAGE("SortParticlesFun_GetNumberofParticipatingPartiles, Ti_Current=%d",Ti_Current);
	if (N<=1) {
		WARNING("trying to sort N<=1 particles");
		return 0;
	}
	g_sort.Init(10,N,particles,Ti_Current,sphmode);
	const int M=g_sort.Findminmax();
	return M;
}

void SortParticles_Sort(const int Np,int*const sorted)
{
	if (Np>1) g_sort.Sort(Np,sorted);
}

#endif
