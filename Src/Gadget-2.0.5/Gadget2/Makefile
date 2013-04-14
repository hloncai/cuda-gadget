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

#CC  = mpicc
#CPP = mpicxx

CC  = tau_cc.sh
CPP = tau_cxx.sh
LD  = tau_cxx.sh

CUDASRC = ../../Gadget_cuda_gx
ifeq (x86_64,$(shell uname -m))
	SYSTEMARCHITECTURE=64
endif

GCC_CWARN_FLAGS = 

OLD_GCC_CWARN_FLAGS = \
	-Wall \
	-Wextra \
	-Wimplicit \
	-Wchar-subscripts \
	-Wmultichar \
	-Wpointer-arith \
	-Wcast-align \
	-Wpointer-arith \
	-Wcast-align \
	-Wpacked \
	-Wstack-protector \
	-Wno-strict-aliasing\
	-Wno-unused-function\
	-Wno-unused-variable\
	-Wno-unused-parameter\
	-Wno-sign-compare
	# -Wshadow

GCC_CPPWARN_FLAGS = $(GCC_CWARN_FLAGS)

ICC_CWARN_FLAGS   =
ICC_CPPWARN_FLAGS =

# Linux/GCC/i685 like config
FLAGSGCC=-mmmx -msse -msse2 -msse3 -mfpmath=sse
ifeq (64,$(SYSTEMARCHITECTURE))
	FLAGSGCC+=-m64
	GCC_CWARN_FLAGS +=
else
	FLAGSGCC+=-m32 -march=i686
	GCC_CWARN_FLAGS += -Wpadded -Wlogical-op -Woverlength-strings
endif
# Linux/ICC/i685 like config
FLAGSICC=-fast -ipo -xSSE4.2 -fp-model precise -fp-model source -unroll-aggressive -m64 -w1 -vec-report=0 -diag-disable 188
#FLAGSICC=-g -fno-omit-frame-pointer -inline-level=0 -ip-no-inlining -no-prec-div -unroll-aggressive -m64 -w1 -vec-report=0


# architecture config
MPICCVENDOR=$(shell $(CC) --version | head -c 3)

ifeq (gcc,$(MPICCVENDOR))
	# Linux/GCC/i686 like config:
	OPTIONS += $(FLAGSGCC)
	CWARN_FLAGS  =$(GCC_CWARN_FLAGS)
	CPPWARN_FLAGS=$(GCC_CPPWARN_FLAGS)
else
ifeq (icc,$(MPICCVENDOR))
 	# Linux/ICC/i685 like config
 	OPTIONS += $(FLAGSICC)
	CWARN_FLAGS  =$(ICC_CWARN_FLAGS)
	CPPWARN_FLAGS=$(ICC_CPPWARN_FLAGS)
else
	# assume Linux/GCC
	OPTIONS += $(FLAGSGCC)
endif
endif

include gadget.options
OPTIONS += $(OPT)

# debug/emulator config
ifeq ($(dbg),1)
	DBGSFX = _dbg
	BINSUFFIX = .d
	LIBSUFFIX = D
	TARGET = debug
	# Generate debug info in objectfiles, for debuggers, callgrind, et. al.
	OPTIONS += -g -O0
else
	TARGET = release
	OPTIONS += -O3
endif

ifeq ($(emu),1)
	EMUSFX = _emu
	EMUSUFFIX = emu
	TARGET = emu$(TARGET)
endif

# Include paths, you system might be different,
#  this setup assumes local gsl and fftw directry in homepath (~/gsl and ~/ffwt)
# BOOST_INCL=-I/usr/local/boost # currently unused
CUDA_INCL =

ifeq ($(ocelot),1)
#	CUDA_LIB  =-L$(CUDA_INSTALL_PATH)/lib -L/sw/keeneland/cuda/4.2/linux_binary/lib64 -L/nics/c/home/bwibking/opt/ocelot/lib `OcelotConfig -l`
	CUDA_LIB  =-L$(CUDA_INSTALL_PATH)/lib -L/usr/local/cuda5.0/lib64 -L/home/bwibking/opt/ocelot/lib `OcelotConfig -l`
else
#	CUDA_LIB  =-L$(CUDA_INSTALL_PATH)/lib -L/sw/keeneland/cuda/4.2/linux_binary/lib64 -lcudart$(EMUSUFFIX)
	CUDA_LIB  =-L$(CUDA_INSTALL_PATH)/lib -L/usr/local/cuda5.0/lib64 -lcudart$(EMUSUFFIX)
endif

GSL_INCL = -I/usr/local/gsl/latest/x86_64/gcc46/nonet/include
GSL_LIB = -L/usr/local/gsl/latest/x86_64/gcc46/nonet/lib -lgsl -lgslcblas -lm
FFTW_INCL = -I/usr/local/fftw/latest-2/x86_64/gcc46/ether/include
FFTW_LIBS = -L/usr/local/fftw/latest-2/x86_64/gcc46/ether/lib
# GSL_INCL  =
#GSL_LIB   =-lgsl -lgslcblas -lm
#FFTW_INCL =-isystem /sw/keeneland/fftw/2.1.5/centos5.5_gnu4.1.2_openmpi1.4.3/include
#FFTW_LIBS =-L/sw/keeneland/fftw/2.1.5/centos5.5_gnu4.1.2_openmpi1.4.3/lib
#FFTW3_INCL=-isystem /home/wibkinbd/fftw3/include
#FFTW3_LIBS=-L/home/wibkinbd/fftw3/lib
#MPICH_LIB  = -lmpich
#MPICH_LIB = -llam  # -llammpio
#HDF5_INCL  =-I/usr/local/include/hdf5
#HDF5_LIB   =-L/usr/local/lib -lhdf5 -lz

ifeq (FFTW3,$(findstring FFTW3,$(OPT))) # are we using fftw3 (single precision)?
	FFTW_LIB = $(FFTW3_LIBS) -lfftw3f_mpi -lfftw3f -lm
#	FFTW_LIB = $(FFTW3_LIBS) -lfftw3_mpi -lfftw3 -lm
else
ifeq (NOTYPEPREFIX_FFTW,$(findstring NOTYPEPREFIX_FFTW,$(OPT)))  # fftw installed with type prefix?
	FFTW_LIB = $(FFTW_LIBS) -lrfftw_mpi -lfftw_mpi -lrfftw -lfftw
#	FFTW_LIB = $(FFTW_LIBS) -lfftw2x_cdft_SINGLE -Wl,--start-group /usr/local/intel12/mkl/lib/intel64/libmkl_cdft_core.a /usr/local/intel12/mkl/lib/intel64/libmkl_intel_lp64.a /usr/local/intel12/mkl/lib/intel64/libmkl_intel_thread.a /usr/local/intel12/mkl/lib/intel64/libmkl_core.a -lmkl_blacs_intelmpi_lp64 -Wl,--end-group -L/usr/local/intel12/lib/intel64/ -liomp5
else
ifeq (DOUBLEPRECISION_FFTW,$(findstring DOUBLEPRECISION_FFTW,$(OPT)))
	FFTW_LIB = $(FFTW_LIBS) -ldrfftw_mpi -ldfftw_mpi -ldrfftw -ldfftw
else
	FFTW_LIB = $(FFTW_LIBS) -lsrfftw_mpi -lsfftw_mpi -lsrfftw -lsfftw
endif
endif
endif

CUFILES = $(shell ls $(CUDASRC)/*.cu)
CUDAOBJ = $(CUFILES:.cu=.co$(DBGSFX)$(EMUSFX))
INCL     = $(GSL_INCL) $(FFTW_INCL) $(FFTW3_INCL) $(HDF5_INCL) $(BOOST_INCL) $(CUDA_INCL)
LIBS     = $(HDF5_LIB) $(MPICH_LIB) $(GSL_LIB) $(FFTW_LIB) $(CUDA_LIB)
CFLAGS   = $(INCL) $(CWARN_FLAGS)
CPPFLAGS = $(INCL) $(CPPWARN_FLAGS)
CFILES   = $(shell ls *.c   2>/dev/null)
CPPFILES = $(shell ls *.cpp 2>/dev/null)
OBJS     = $(CFILES:.c=$(BINSUFFIX).o) $(CPPFILES:.cpp=$(BINSUFFIX).oo)
DEPEND   = $(addsuffix $(BINSUFFIX).dep,$(CFILES) $(CPPFILES))
XDEPEND  = gadget.options Makefile

BIN      = $(EMUSUFFIX)Gadget2$(BINSUFFIX)

$(BIN): $(OBJS) $(CUDAOBJ)
	@ #  $(CC) $(OBJS) $(LIBS) -Wl,-shared-libgcc -o $(BIN)
	$(CC) $(OBJS) $(CUDAOBJ) $(LIBS) -o $(BIN)

%.o: %.c $(XDEPEND)
	@ echo $(CC) $<
	$(CC)  -DNDEBUG $(OPTIONS) $(CFLAGS)   -c $< -o $@

%.oo: %.cpp $(XDEPEND)
	@ echo $(CPP) $<
	$(CPP) -DNDEBUG $(OPTIONS) $(CPPFLAGS) -c $< -o $@

%.d.o: %.c $(XDEPEND)
	@ echo $(CC) $<
	$(CC)  -D_DEBUG $(OPTIONS) $(CFLAGS)   -c $< -o $@

%.d.oo: %.cpp $(XDEPEND)
	@ echo $(CPP) $<
	$(CPP) -D_DEBUG $(OPTIONS) $(CPPFLAGS) -c $< -o $@

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),tar)
%.c.dep: %.c
	@-echo 'CDEP ' $<  $@
	@set -e; rm -f $@; \
	gcc -MM $(CFLAGS) $< > $@  # gcc and g++ does not include standard libs in -MM
%.c.d.dep: %.c
	@-echo 'CDEP ' $<  $@
	@set -e; rm -f $@; \
	gcc -MM  $(CFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.d.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$
%.cpp.dep: %.cpp
	@-echo CCDEP $<  $@
	@set -e; rm -f $@; \
	g++ -MM $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.oo $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$
%.cpp.d.dep: %.cpp
	@-echo CCDEP $<  $@
	@set -e; rm -f $@; \
	g++ -MM  $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.d.oo $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$
include $(DEPEND)
endif
endif

.PHONY:clean
clean:
	@ rm -rf $(DEPEND) $(OBJS) $(BIN) core
