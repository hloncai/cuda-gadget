# Copyright 2009 Carsten Eie Frigaard.
#
# License GPLv3+: GNU GPL version 3 or later <http:#gnu.org/licenses/gpl.html>.
# This is free software: you are free to change and redistribute it.
# There is NO WARRANTY, to the extent permitted by law.
#
# This software is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software.  If not, see <http:#www.gnu.org/licenses/>.

#CUDACC=nvcc -ccbin /usr/local/intel12/bin/icc
#CUDACC=nvcc -ccbin tau_cxx.sh
CUDACC=nvcc
CFLAGS=-m64 -arch sm_20 --ptxas-options=-v -ftz=false -prec-div=true -prec-sqrt=true -fmad=false -Xptxas -dlcm=ca
# change above to -arch sm_13

CUFILES =$(shell ls *.cu)
CU_DEPS =$(shell ls *.h Gadget2/*_gx.h) options.h
OBJS    =$(CUFILES:.cu=.co$(DBGSFX)$(EMUSFX))
XDEPEND =Makefile

# Warning flags
CXXWARN_FLAGS = 

OLD_CXXWARN_FLAGS := \
	-W -Wall -Wextra \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wpacked \
	-Wno-unused-function \
	-Wno-unused-parameter \
	-Wno-missing-field-initializers

CWARN_FLAGS = 

OLD_CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain

CFLAGS+=$(addprefix --compiler-options ,$(CXXWARN_FLAGS))
CFLAGS+=--compiler-options -fno-strict-aliasing

ifeq ($(emu), 1)
	EMUSFX = _emu
	CFLAGS   += -deviceemu
	CXXFLAGS += -D__DEVICE_EMULATION__
	CFLAGS   += -D__DEVICE_EMULATION__
endif

ifeq ($(dbg),1)
	DBGSFX = _dbg
	CFLAGS += -g -G -D_DEBUG
else
	CFLAGS += --compiler-options -fno-strict-aliasing -O2 # -O3
endif

ifeq ($(verbose), 1)
	VERBOSE :=
else
	VERBOSE := @
endif

.PHONY:all
all: $(OBJS) $(CU_DEPS)

%.co$(DBGSFX)$(EMUSFX): %.cu $(CU_DEPS) $(XDEPEND)
	@ echo $(CUDACC) $<  [-o $@]
	$(VERBOSE) $(CUDACC) $(CFLAGS) -c -o $@ $<

options.h: Gadget2/gadget.options $(XDEPEND)
	@ cat  Gadget2/gadget.options | sed s/\#/'\/''\/'/g | sed s/OPT/\#define/g | sed s/+=//g | sed s/-D//g | sed s/'='/' '/g > options.h

# DEPEND  =$(addsuffix .dep$(DBGSFX)$(EMUSFX),$(CUFILES))
# ifneq ($(MAKECMDGOALS),clean)
# ifneq ($(MAKECMDGOALS),tar)
# %.cu.dep$(DBGSFX)$(EMUSFX): %.cu
# 	@-echo 'CDEP ' $<  [$@]
# 	@set -e; rm -f $@; \
# 	gcc -D__CUDACC__ -MM -x c $< > $@.$$$$; \
# 	sed 's,\($*\)\.o[ :]*,\1.d.o $@ : ,g' < $@.$$$$ > $@; \
# 	rm -f $@.$$$$
# include $(DEPEND)
# endif
# endif

.PHONY:clean
clean:
	@ rm -rf $(OBJS) $(DEPEND) *.linkinfo

.PHONY:clean_keep
clean_keep:
	@ rm -rf *.i *.ii *.cudafe?.* *.fatbin.c *.hash *.linkinfo *.ptx *.cubin *.cu.cpp *.cu.c

