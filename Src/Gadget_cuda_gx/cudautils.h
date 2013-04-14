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

#define CUDA_DEBUG_GX 0 // 0=no check, 1=asserts, 2=verbose asserts+dbg printsystem, 3>=print calls, verbose level defined by 3,4,5...

#ifdef __CUDACC__
	#include "cudaprintf.h"
	#include "cudaassert.h"
	#include "cudamalloc.h"
	#include "cudakernelsignals.h"
#endif
