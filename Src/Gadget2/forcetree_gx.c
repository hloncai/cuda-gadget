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
#include "interface_gx.h"
#include "../gadget_cuda_gx.h"

void force_treeevaluate_shortrange_range_gx(const int mode,const int Np)
{
	ASSERT_GX( mode==0 || mode==1);
	ASSERT_GX( Np>0 );
	ASSERT_GX( s_gx.cudamode>=1 && s_gx.cudamode<=3 );

	force_treeevaluate_shortrange_range_cuda_gx(mode,Np,s_gx,p_gx);
	if (mode==0) WriteResultBufferData_gx();
}

