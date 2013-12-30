# Copyright 2009 Carsten Eie Frigaard.
#
# License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>.
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
# along with this software.  If not, see <http://www.gnu.org/licenses/>.

# ic data root file

DAT=lcdm_gas
PARAM=Dat/ana_$(DAT).param

# number of cpu's to use
NCPUS=2

# hostfile, used only if you add -machinefile $(HOSTFILE) to mpirun below
HOSTFILE=hosts.txt

# number of concurrent makejobs
MAKEJOBS=8

# Commandline argument parsing
#   DO not change these...begin

ifeq ($(ocelot),1)
	OCELOTSUFFIX := .ocelot
endif
ifeq ($(dbg),1)
	BINSUFFIX := .d
	TARGET := debug
else
	TARGET := release
endif
ifeq ($(emu),1)
	EMUSUFFIX := emu
	TARGET := emu$(TARGET)
endif
ifeq ($(cuda),0)
	CUDA := -cuda $(cuda)
endif
ifeq ($(cuda),1)
	CUDA := -cuda $(cuda)
endif
ifeq ($(cuda),2)
	CUDA := -cuda $(cuda)
endif
ifeq ($(cudadebug),0)
	CUDA := $(CUDA)-cudadebug $(cudadebug)
endif
ifeq ($(cudadebug),1)
# 	CUDA := $(CUDA)-cudadebug $(cudadebug)
endif
ifeq ($(cudadebug),2)
	CUDA := $(CUDA)-cudadebug $(cudadebug)
endif
ifeq ($(cudadebug),3)
	CUDA := $(CUDA)-cudadebug $(cudadebug)
endif
ifeq ($(cudadebug),4)
	CUDA := $(CUDA)-cudadebug $(cudadebug)
endif

#   DO not change these...end

DIR=$(shell pwd)
DATBASE=$(firstword $(subst ., ,$(subst _, ,$(DAT))))

# Directory and file dependency setup
CUDADIR     = Src
GADGETDIR   = Src/Gadget2
OUTDIR      = Out/Out_$(DAT)

CUDAFILES   = $(shell ls $(CUDADIR)/*.cu $(CUDADIR)/*.cpp $(CUDADIR)/*.h $(CUDADIR)/Makefile 2>/dev/null)
GADGETFILES = $(shell ls $(GADGETDIR)/*.c $(GADGETDIR)/*.cpp $(GADGETDIR)/*.h 2>/dev/null)
XDEPEND     = $(CUDADIR)/Makefile $(GADGETDIR)/Makefile $(GADGETDIR)/gadget.options Makefile
BIN         = $(EMUSUFFIX)Gadget2$(OCELOTSUFFIX)$(BINSUFFIX)
MPIRUN_BASE = mpiexec -machinefile $(HOSTFILE) -n $(NCPUS)
MPIRUN_BIN  = $(DIR)/Bin/$(BIN) $(PARAM) $(CUDA)
MPIRUN      = $(MPIRUN_BASE) $(MPIRUN_BIN)

# Building the executables
Bin/$(BIN): $(CUDAFILES) $(GADGETFILES) $(XDEPEND) $(GADGETDIR)/gadget.options
	@ make --no-print-directory -j$(MAKEJOBS) -C $(CUDADIR)
	@ make --no-print-directory -j$(MAKEJOBS) -C $(GADGETDIR)
	@ cp $(GADGETDIR)/$(EMUSUFFIX)Gadget2$(BINSUFFIX) $@

# Running under mpd
.NOTPARALLEL: run
.PHONY: run
run: Bin/$(BIN)
	@ - mkdir -p Out/Out_$(DAT)
	@ - rm -f Out/Out_$(DAT)/snapshot_* 2>/dev/null
	$(MPIRUN) | tee out.$(DAT).txt | grep -e "##" -e "\[" -e "\*\*"
	@ - rm Out/Out_$(DAT)/restart.* -f 2>/dev/null
	@ cp out.$(DAT).txt $(OUTDIR)/out.txt

# Testing the gadget output,
#   needs a .g suffix on all snapshotfiles (patched into the gadget code)
#   also needs gdiff, included in the Gadget_addon_tools ()
#   Ref='cuda=0', Cuda1='cuda=1' Cuda2='cuda=2', Cuda2Emu='cuda=2 emu=1'

REFDIR=cuda2-ncpu$(NCPUS)
TARDIR=$(OUTDIR)

OUTFILES=$(shell ls $(TARDIR)/*_???.g 2>/dev/null)
ASCFILES=$(OUTFILES:.g=.asc)
DIFFILES=$(OUTFILES:.g=.diff)
REFFILES=$(OUTFILES:.g=.ref)

DISTFILES=$(OUTFILES:.g=.dist)
REFFILES=$(OUTDIR)/$(REFDIR)/$(notdir $(OUTFILES:.g=.ref))

.NOTPARALLEL: %.dist
%.dist: %.g Makefile
	@ echo -n gdiff $< $(OUTDIR)/$(REFDIR)/$(notdir $<) "  "
	@ gdiff -r  $< $(OUTDIR)/$(REFDIR)/$(notdir $<) > $@
	@ tail -n 1 $@

.NOTPARALLEL: %.ref
%.ref: %.dist Makefile
	@ echo -n make ref $< "  " $@
	@ gdiff -r $@ $(OUTDIR)/$(REFDIR)/$(notdir $@)

.NOTPARALLEL: dist
dist: $(DISTFILES)

.NOTPARALLEL: fdist
fdist:
	@ rm -f $(DISTFILES)
	@ make --no-print-directory dist

.NOTPARALLEL: ref
ref: $(REFFILES)

.NOTPARALLEL: mkref
mkref:
	mkdir -p                $(OUTDIR)/$(REFDIR)
	cp -f $(OUTDIR)/out.txt $(OUTDIR)/$(REFDIR)
	cp -f $(OUTFILES)       $(OUTDIR)/$(REFDIR)
	- cp -f $(DISTFILES)    $(OUTDIR)/$(REFDIR)

# cleaning
.PHONY:clean
clean:
	@ make --no-print-directory -C $(CUDADIR)   clean
	@ make --no-print-directory -C $(GADGETDIR) clean
	@ make --no-print-directory -C $(CUDADIR)   clean dbg=1
	@ make --no-print-directory -C $(GADGETDIR) clean dbg=1
	@ make --no-print-directory -C $(CUDADIR)   clean emu=1
	@ make --no-print-directory -C $(GADGETDIR) clean emu=1
	@ make --no-print-directory -C $(CUDADIR)   clean emu=1 dbg=1
	@ make --no-print-directory -C $(GADGETDIR) clean emu=1 dbg=1
	@ rm Bin/$(BIN) Bin/emuGadget2 Bin/Gadget2.d Bin/emuGadget2.d testsuite.txt gindex.txt out.txt -f 2>/dev/null
