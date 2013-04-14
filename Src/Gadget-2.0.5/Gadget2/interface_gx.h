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

#ifndef __INTERFACE_GX_H__
#define __INTERFACE_GX_H__

#include "assert_gx.h"
#include "defines_gx.h"

int stringhash(unsigned char *str);

#define FLOAT_GX          double // GX float mode, float: for cuda capability 1.1, float or double: for cuda capability >= 1.2
#define FLOAT_INTERNAL_GX double // internal float mode, float or double

// Check macro definitions, do not change
	#ifdef DOUBLEPRECISION
		#define FLOAT double
	#else
		#define FLOAT float
	#endif

#ifdef GADGETVERSION
	#ifndef PMGRID
		// WARNING: !PMGRID is_not_supported_yet!!
	#endif
	#ifdef ADAPTIVE_GRAVSOFT_FORGAS
		WARNING: ADAPTIVE_GRAVSOFT_FORGAS maybe_working_not_tested_yet!!
	#endif
	#ifdef TWODIMS
		WARNING: TWODIMS not tested yet!!
	#endif
#endif

#ifdef __CUDACC__
	#ifdef GADGETVERSION
		ERROR: __CUDACC__ and GADGETVERSION should not be defined at the same time
	#endif
#endif

// Macros, do not change
#ifdef PERIODIC
	#ifndef NEAREST
		#define NEAREST(x) (((x)>boxhalf)?((x)-boxsize):(((x)<-boxhalf)?((x)+boxsize):(x)))
	#endif
#endif

// Alignment macros
#ifndef __CUDACC__
	#define ALIGN_16_GX
#else
	#define ALIGN_16_GX __align__(16)
#endif

struct ALIGN_16_GX particle_data_gx
{
	// NOTE: Type, OldAcc,Ti_endstep only needed once for every particle, and are stored temporary in result_gx structure
	FLOAT_GX Pos[3]; // particle position at its current time
	FLOAT_GX Mass;   // particle mass
};

#ifdef SPH
struct ALIGN_16_GX sph_particle_data_gx
{
	FLOAT_GX Entropy;
	FLOAT_GX Density;
	FLOAT_GX Hsml;
	//FLOAT_GX Left;
	//FLOAT_GX Right;
	//FLOAT_GX NumNgb;
	FLOAT_GX Pressure;
	//FLOAT_GX DtEntropy;
	//FLOAT_GX HydroAccel[3];
	FLOAT_GX VelPred[3];
	FLOAT_GX DivVel;
	FLOAT_GX CurlVel;
	//FLOAT_GX Rot[3];
	FLOAT_GX DhsmlDensityFactor;
	//FLOAT_GX MaxSignalVel;

	int pad1,pad2;
};
#endif

struct ALIGN_16_GX NODE_gx
{
	FLOAT len;                        // sidelength of treenode
	FLOAT center[3];                  // geometrical center of node
	#ifdef ADAPTIVE_GRAVSOFT_FORGAS
		ERROR: cannot handle ADAPTIVE_GRAVSOFT_FORGAS yet, needs to move maxsoft to extNODE to keep NODE structure wihtin bounds
		// FLOAT maxsoft;           // hold the maximum gravitational softening of particles in the
		                            // node if the ADAPTIVE_GRAVSOFT_FORGAS option is selected
	#endif
	union
	{
		int suns[8];                    // temporary pointers to daughter nodes
		struct
		{
			FLOAT s[3];               // center of mass of node
			FLOAT mass;               // mass of node
			int bitflags;             // a bit-field with various information on the node
			int sibling;              // this gives the next node in the walk in case the current node can be used
			int nextnode;             // this gives the next node in case the current node needs to be opened
			int father;               // this gives the parent node of each node (or -1 if we have the root node)
		}
		d;
	}
	u;
};

struct ALIGN_16_GX extNODE_gx // this structure holds additional tree-node information which is not needed in the actual gravity computation
{
	FLOAT hmax;   // maximum SPH smoothing length in node. Only used for gas particles
	FLOAT vs[3];  // center-of-mass velocity
};

#ifdef SPH
struct ALIGN_16_GX hydrodata_in_gx
{
  FLOAT_GX Pos[3];
  FLOAT_GX Vel[3];
  FLOAT_GX Hsml;
  FLOAT_GX Mass;
  FLOAT_GX Density;
  FLOAT_GX Pressure;
  FLOAT_GX F1;
  FLOAT_GX DhsmlDensityFactor;
  int      Timestep;
  int      Task;
  int      Index;
  int      pad;
};
#endif

struct ALIGN_16_GX result_gx
{
	// NOTE gx result struct is independent of gadget result struct
	FLOAT_GX acc_x,acc_y,acc_z;
	int ninteractions;
	#ifdef CUDA_GX_DEBUG_MEMORY_ERROR
		FLOAT_GX oldacc;
		int temp2;
		int realtarget;
		int type;
	#endif
	#ifdef DOUBLEPRECISION
	int test1;	 // alignment will make this redundant
	#endif
};

#ifdef SPH
struct ALIGN_16_GX result_hydro_gx
{
	FLOAT_GX Acc[3];
	FLOAT_GX DtEntropy;
	FLOAT_GX MaxSignalVel;
	int realtarget;
	int pad0,pad1;
};
#endif

struct ALIGN_16_GX etc_gx
{
	int Type;
	int Ti_begstep;
	int Ti_endstep;
	FLOAT_GX Hsml;
	#ifdef DOUBLEPRECISION   // fixes alignment issues
	int temp2;
	#endif
};

/*
struct ALIGN_16_GX aux_gx
{
	// NOTE for read-once extra/auxilary particle data
	FLOAT_GX acc_x;    //=OldAcc;
	FLOAT_GX acc_y;    //not used
	FLOAT_GX acc_z;    //not used
	int ninteractions; //=type (3bits) | realtarget (32-3bits)
};

struct ALIGN_16_GX aux_export_gx
{
	// NOTE for read-once extra/auxilary particle data
	FLOAT acc_x; //=GravDataGet[i].u.Pos[0];
	FLOAT acc_y; //=GravDataGet[i].u.Pos[1];
	FLOAT acc_z; //=GravDataGet[i].u.Pos[2];

	int ninteractions; //=type (3bits) | realtarget (32-3bits)
	type, if UNEQUALSOFTENINGS => type=GravDataGet[target].Type, else type=typeP0

	etc.temp = GravDataGet[i].w.OldAcc;
};

// Combo structure for results and aux data
union aux_result_data_gx {
 	struct result_gx     r;
	struct aux_gx        a;
	struct aux_export_gx e;
};
*/

struct parameters_gx
{
	int MaxPart;
	int MaxNodes;

	int Ti_Current;
	#ifdef DOUBLEPRECISION
	int pad1; // padding for double alignment
	#endif

	FLOAT_GX Asmth[2];
	FLOAT_GX Rcut[2];
	FLOAT_GX BoxSize;
	FLOAT_GX ErrTolTheta;
	FLOAT_GX ErrTolForceAcc;
	FLOAT_GX ForceSoftening[6];
	#ifdef ADAPTIVE_GRAVSOFT_FORGAS
		FLOAT_GX soft;
	#endif
	FLOAT_GX Masses[6];
	int typesizes[6]; // number of particles for each particle species
	int nonsortedtypes;  // are particles sorted into types
	int ComovingIntegrationOn;
	
	FLOAT_GX Timebase_interval;
	int export_P_type;
};

#ifdef SPH
struct parameters_hydro_gx
{
	// NOTE: GCC or NCC align doubles to some boundaries...
	int N_gas;
	int szngb;

	FLOAT_GX hubble_a2;
	FLOAT_GX fac_mu;
	FLOAT_GX fac_vsic_fix;
	FLOAT_GX ArtBulkViscConst;

	#ifdef PERIODIC
		FLOAT_GX boxSize,   boxHalf;
		FLOAT_GX boxSize_X, boxHalf_X;
		FLOAT_GX boxSize_Y, boxHalf_Y;
		FLOAT_GX boxSize_Z, boxHalf_Z;
	#endif
};
#endif

struct state_gx
{
	struct particle_data_gx* P;                // holds copy of particle data structure
	struct NODE_gx*          Nodes_base;       // holds copy of NODE data structure, base pointer
	struct NODE_gx*          Nodes;            // holds copy of NODE data structure
	int*                     Nextnode;         // holds copy of nextnode data structure
	int*                     DomainTask;       // holds copy of DomainTask structure
	float*                   shortrange_table; // holds copy of shortrange_table data
	char*                    Exportflag;       // holds outgoing copy of Exportflags
	struct result_gx*        result;           // first temporary for aux particle data, finally particle result forces
	struct result_gx*        result_buffer;    // holds copy of result in case of comm Buffer reiteration, only to be used by CPU, not by GPU
	FLOAT_GX*                scratch;          // scratch area

	// SPH support
	struct sph_particle_data_gx* SphP;         // holds copy of SPH particle data structure
	struct etc_gx*           etc;              // holds various data for particles, sph particles etc.
	struct extNODE_gx*       extNodes_base;    // holds copy of extNODE data structure for SPH calc, base pointer
	struct extNODE_gx*       extNodes;         // holds copy of extNODE data structure for SPH calc
	int*                     Ngblist;          // Neighbour list
	struct result_hydro_gx*  result_hydro;     // first temporary for aux particle data, finally particle sph result
	struct hydrodata_in_gx*  hydrodata_in;     // exported hydro data, same size as result_hydro

	int*                     Psorted;          // sorted particles, same size as result, only to be used by CPU, not by GPU

	//void* pad;

	int ThisTask;                      // copy of All.ThisTask
	int NTask;                         // copy of All.NTask
	unsigned int NumPart;              // physical number of local particles, alias for sz_P
	unsigned int N_gas;                // physical number of local gas particles, alias for sz_Sph
	unsigned int Np;                   // number of particles participating in this timestep

	unsigned int segment;              // if calculation is split into smaller segments
	unsigned int sz_segments;          // size of total segments

	unsigned int MaxPart;              // copy of All.MaxPart
	unsigned int sz_memory_limit;      // mem on gx card

	// sizes are in elements, not bytes
	unsigned int sz_P;                 // size of particles, alias for NumPart
	unsigned int sz_SphP;              // size of SphP particles
	unsigned int sz_Nodes_base;        // size of nodes base structure
	unsigned int sz_Nextnode;          // size of nextnode structure
	unsigned int sz_DomainTask;        // size of DomainTask structure
	unsigned int sz_shortrange_table;  // size of shortrange_table
	unsigned int sz_Exportflag;        // size of Exportflag structure
	unsigned int sz_result;            // size of result structure
	unsigned int sz_result_buffer;     // size of result buffer structure
	unsigned int sz_Psorted;           // size of Psize structure
	unsigned int sz_scratch;           // size of scratch area

	// SPH support
	unsigned int sz_etc;               // size of etc list
	unsigned int sz_extNodes_base;     // size of extNodes base list
	unsigned int sz_Ngblist;           // size of Neighbour list
	unsigned int sz_result_hydro;      // size of hydro result list
	unsigned int sz_hydrodata_in;      // size of the export hydro list

	unsigned int sz_max_P;             // physical allocated size, leave some headroom for faster reallocation
	unsigned int sz_max_SphP;          // -
	unsigned int sz_max_Exportflag;    // -
	unsigned int sz_max_Nodes_base;    // -
	unsigned int sz_max_extNodes_base; // -
	unsigned int sz_max_result;        // -
	unsigned int sz_max_result_buffer; // -
	unsigned int sz_max_scratch;       // -
	unsigned int sz_max_etc;           // -
	unsigned int sz_max_Ngblist;       // -
	unsigned int sz_max_result_hydro;  // -
	unsigned int sz_max_hydrodata_in;  // -
	unsigned int sz_max_Psorted;       // -

	int mode;                          // forcetree/hydro evaluate mode
	int cudamode;                      // signal use of GX board
	int debugval;                      // debug signaling flag for various purp.
	int iteration;                     // debug val for counting kernel iterations

	int external_node;                 // chunck manager var: node that send this segment
	int blocks;                        // number of blocks to use in kernel
	int grids;                         // number of grids to use in kernel
	int sphmode;                       // signals SPH calculation

	// debug variables
	struct KernelSignals* kernelsignals;
	char*                 debug_msg;
	unsigned int          debug_sz_msg;

	int pad;
};

#ifndef __CUDACC__

// forward defines
struct NODE;
struct particle_data;
struct sph_particle_data;
struct extNode;
struct hydrodata_in;

// global variables
extern struct parameters_gx       p_gx;
extern struct parameters_hydro_gx h_gx;
extern struct state_gx            s_gx;

// Prototypes
void Initialize_gx						   (const int argc,char*const*const argv,const int thistask,const int ntask,const int localrank);
int  InitializeProlog_gx                   (const int N);
int  InitializeCalculation_gx              (const int N,const struct particle_data*const p,const int sphmode);
void InitializeExportCalculation_gx        (const int N,const int typeP0);
void InitializeHydraExportCalculation_gx   (const int N,const struct hydrodata_in*const s);
int  InitializeHydraCalculation_gx         (const int N,const struct particle_data*const p,const struct sph_particle_data*const sph,const int N_gas,const FLOAT_GX hubble_a2,const FLOAT_GX fac_mu,const FLOAT_GX fac_vsic_fix
								#ifdef PERIODIC
									,const FLOAT_GX boxSize,const FLOAT_GX boxHalf
								#endif
							 );

void   Finalize_gx                         (void);
double FinalizeExportCalculation_gx        (const int N);
void   FinalizeHydraExportCalculation_gx   (const int N);

struct result_gx GetTarget                 (const int target,const int n);
void WriteResultBufferData_gx              (void);
void UpdateShortrangeTableData_gx          (const int N,const float *const s);
int  CountParticlesInTimeStep_gx           (const size_t N,const struct particle_data*const p,const int Ti_Current,const int sphmode);

void force_treeevaluate_shortrange_range_gx(const int mode,const int Np);
int  GetID                                 (const int target);

#if CUDA_DEBUG_GX > 0
	struct parameters_gx FillParameters_gx(void);
#endif

#endif // __CUDACC__

#endif // __INTERFACE_GX_H__

