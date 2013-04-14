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

#ifdef CUDA_GX_TEST_WALK

#ifndef __DEVICE_EMULATION__
	ERROR: CUDA_GX_TEST_WALK only works in emu mode
#else

struct Char99
{
	char s[99];
};

int g_walknomax=-1;
int g_walknomin=-1;

Char99 PrintN(const int no,const struct parameters_gx& p)
{
	Char99 c;
	c.s[0]=0;

	if (no<0){
		sprintf(c.s,"z%d",no);
		return c;
	}
	else if(no < p.MaxPart)	{
		sprintf(c.s,"p%d",no);
		return c;
	}
	else if(no >= p.MaxPart + p.MaxNodes)	{ // pseudo particle
		sprintf(c.s,"s%d",no-p.MaxPart);
		return c;
	} else {
		sprintf(c.s,"n%d",no-p.MaxPart);
		return c;
	}
}

bool WalkSiblingPrint(const NODE_gx& node,const struct state_gx s)
{
	return (node.u.d.sibling>=0 && s.Nodes[node.u.d.sibling].len<=node.len);
}

void Printnode(const int father,const int no,const NODE_gx& node,const double boxsize,const struct parameters_gx p,const struct state_gx s,const int level)
{
	static int n=0;
	MESSAGE("@@ WALK: node=%s  father=%s  truefather=%d  level=%d  n=%d",PrintN(no,p).s,PrintN(father,p).s,node.u.d.father,level,n++);
	MESSAGE("@@         len=%1.4f  center=(%5.1f,%5.1f,%5.1f)",boxsize,node.center[0],node.center[1],node.center[2]);
	MESSAGE("@@         sibling=%s  nextnode=%s",PrintN(node.u.d.sibling,p).s,PrintN(node.u.d.nextnode,p).s);
	MESSAGE("@@         next->%s",WalkSibling(node,s) ? "sibling" : "nextnode");
}

void Walklog(FILE* fp,const char c,const int n,const int MaxPart,const int level=-1)
{
	ASSERT_GX( c=='p' || c=='n' || c=='s');
	ASSERT_GX( c=='p' || n>=MaxPart);

	const int no=c=='p' ? n : n-MaxPart;

	if (no<10)            fprintf(fp,"%c 0000000%d",c,no);
	else if (no<100)      fprintf(fp,"%c 000000%d",c,no);
	else if (no<1000)     fprintf(fp,"%c 00000%d",c,no);
	else if (no<10000)    fprintf(fp,"%c 0000%d",c,no);
	else if (no<100000)   fprintf(fp,"%c 000%d",c,no);
	else if (no<1000000)  fprintf(fp,"%c 00%d",c,no);
	else if (no<10000000) fprintf(fp,"%c 0%d",c,no);
	else if (no<100000000)fprintf(fp,"%c %d",c,no);
	else fprintf(fp,"%c XXX 0%d",c,no);

	if (level>-1) fprintf(fp,"\t L=%d",level);
	fprintf(fp,"\n");

// 	if (n==20 && c=='p') {
// 		MESSAGE("@@ noden==20\n");
// 		fprintf(fp,"\n@@  noden==20\n");
// 	}
}

int WalkParticlesPrint(FILE* fp,const int father,const int no,const struct parameters_gx p,const struct state_gx s,const int count)
{
	if(no<0) return -1;
	if(no>=p.MaxPart) return -1;

	ASSERT_GX(no>=0 && no<s.sz_Nextnode);
	const int next=s.Nextnode[no];

	ASSERT_GX(no>=0 && no<s.sz_P);
	char c[256];
	c[0]=0;
	if (next<0 || next>=p.MaxPart) sprintf(c,"  count=%d",count);

	MESSAGE("@@ WALK: p=%s  father=%s  next=%s%s",PrintN(no,p).s,PrintN(father,p).s,PrintN(next,p).s,c);
	Walklog(fp,'p',no,p.MaxPart);

	return WalkParticlesPrint(fp,no,next,p,s,count+1);
}

int WalkerPrint(FILE* fp,const int father,const int no,const struct parameters_gx p,const struct state_gx s)
{
	if (no<0) return -1;
	ASSERT_GX( s.MaxPart==p.MaxPart );
	static double rootlen=s.Nodes[p.MaxPart].len;

	if(no < p.MaxPart)	{
		return WalkParticlesPrint(fp,father,no,p,s,0);
	} else if(no >= p.MaxPart + p.MaxNodes)	{ // pseudo particle
		MESSAGE("@@ WALK: pseudo=%d",PrintN(no,p).s);
		Walklog(fp,'s',no,p.MaxPart);

		ASSERT_GX(no>=0 && no<s.sz_Nextnode);
		return -1; // Walker(s.Nextnode[no],p,s);
	} else {
		ASSERT_GX(no>=s.MaxPart);
		const NODE_gx node=s.Nodes[no];
		const int level=rootlen/node.len;

		Printnode(father,no,node,p.BoxSize,p,s,level);
		Walklog(fp,'n',no,p.MaxPart,level);

		if (g_walknomin<0 || g_walknomin>no) g_walknomin=no;
		if (g_walknomax<0 || g_walknomax<no) g_walknomax=no;

		if (WalkSibling(node,s)) Walker(fp,no,node.u.d.sibling ,p,s);
		//else if (!(node.u.d.nextnode<p.MaxPart))
		return Walker(fp,no,node.u.d.nextnode,p,s);
		//else return -1;
	}
}

void WalkNodesPrint(const struct parameters_gx p,const struct state_gx s)
{
	MESSAGE("@@ Walking...(NumPart=%d,MaxPart=%d,MaxNodes=%d)",s.NumPart,s.MaxPart,p.MaxNodes+1);
	FILE* fp=fopen("out.walk.txt","w");

	int no = p.MaxPart; // root node
	no=Walker(fp,-no,no,p,s);

	while(no >= 0) {
		const NODE_gx node=s.Nodes[no];
		no=Walker(fp,no,node.u.d.nextnode,p,s);
	}

	MESSAGE("@@ walk min/max=(%d,%d) sz=%d  All_MaxPart=%d  sz_nodebase=%d",g_walknomin,g_walknomax,g_walknomax-g_walknomin,s.MaxPart,s.sz_Nodes_base);

	fclose(fp);
}
#endif
#endif

#if 0
bool WalkSibling(const NODE_gx& node,const struct NODE_gx* Nodes)
{
	return (node.u.d.sibling>=0 && Nodes[node.u.d.sibling].len<=node.len);
}

int Walker(const int father,const int no,const int MaxPart,const int MaxNodes,const struct NODE_gx* Nodes,int* walknomin,int* walknomax)
{
	if (no<0) return -1;

	if(no < MaxPart)	{
		return -1; // WalkParticles(fp,father,no,p,s,0);
	} else if(no >=MaxPart + MaxNodes) { // pseudo particle
		//ASSERT_GX(no>=0 && no<s.sz_Nextnode);
		ASSERT_GX(0);
		return -1; // Walker(s.Nextnode[no],p,s);
	} else {
		if (!(no>=MaxPart && no<MaxPart+MaxNodes)) MESSAGE("no=%d  MaxPart=%d MaxNodes=%d",no,MaxPart,MaxNodes);

		ASSERT_GX(no>=MaxPart && no<MaxPart+MaxNodes);
		const NODE_gx node=Nodes[no];

		if (*walknomin<0 || *walknomin>no) *walknomin=no;
		if (*walknomax<0 || *walknomax<no) *walknomax=no;

		if (WalkSibling(node,Nodes)) Walker(no,node.u.d.sibling ,MaxPart,MaxNodes,Nodes,walknomin,walknomax);
		return                       Walker(no,node.u.d.nextnode,MaxPart,MaxNodes,Nodes,walknomin,walknomax);
	}
}

void WalkNodes(const int MaxPart,const int MaxNodes,const struct NODE_gx* Nodes,int* walknomin,int* walknomax)
{
	*walknomin=-1,*walknomax=-1;
	int no = MaxPart; // root node
	no=Walker(-no,no,MaxPart,MaxNodes,Nodes,walknomin,walknomax);

	while(no >= 0) {
		const NODE_gx node=Nodes[no];
		no=Walker(no,node.u.d.nextnode,MaxPart,MaxNodes,Nodes,walknomin,walknomax);
		ASSERT_GX(no<MaxNodes);
	}
}
#endif
