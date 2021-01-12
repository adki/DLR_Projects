#-------------------------------------------------------------------------------
SHELL	=/bin/bash

#------------------------------------------------------------------
ifeq ($(MAKECMDGOALS),)
ifndef XILINX_VIVADO_HLS
       $(warning XILINX_VIVADO_HLS environment variable not defined)
endif
endif
ifeq ($(MAKECMDGOALS),all)
ifndef XILINX_VIVADO_HLS
       $(warning XILINX_VIVADO_HLS environment variable not defined)
endif
endif

#------------------------------------------------------------------
export TOP        ?= lenet5
#------------------------------------------------------------------
# Max-Pool embeds ReLU at the end of its operation.
# When it is 0, mis-match may rise.
export EMBED_RELU ?= 1

#------------------------------------------------------------------
export VIVADO_HLS  = vivado_hls
export PROJECT    ?= proj_$(TOP)
export SOLUTION   ?= zed
export PART       ?= xc7z020clg484-1
export SYN        ?= 1
export CSIM       ?= 0
export COSIM      ?= 0
export GUI        ?= 1

#-------------------------------------------------------------------------------
all: syn

syn:
	$(VIVADO_HLS) -f script.tcl

open:
	$(VIVADO_HLS) -p proj_$(TOP)


#-------------------------------------------------------------------------------
clean:
	/bin/rm -f  vivado_hls.log
	/bin/rm -f  resized.png

cleanup clobber: clean
	@if [ -d proj_$(TOP) ]; then /bin/rm -fr proj_$(TOP); fi

cleanupall distclean: cleanup

#-------------------------------------------------------------------------------
.PHONY: all syn open clean cleanup cleanupall distclean
#-------------------------------------------------------------------------------
