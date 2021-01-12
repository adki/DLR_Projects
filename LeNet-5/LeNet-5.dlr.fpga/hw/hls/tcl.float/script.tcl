if {[info exists env(PROJECT)]   ==0} {set PROJECT  proj_lenet5    } else {set PROJECT    $::env(PROJECT)   }
if {[info exists env(SOLUTION)]  ==0} {set SOLUTION zed            } else {set SOLUTION   $::env(SOLUTION)  }
if {[info exists env(PART)]      ==0} {set PART     xc7z020clg484-1} else {set PART       $::env(PART)      }
if {[info exists env(FREQ)]      ==0} {set FREQ     100            } else {set FREQ       $::env(FREQ)      }
if {[info exists env(TOP)]       ==0} {set TOP      lenet5         } else {set TOP        $::env(TOP)       }
if {[info exists env(EMBED_RELU)]==0} {set EMBED_RELU 0            } else {set EMBED_RELU $::env(EMBED_RELU)}
if {[info exists env(SYN)]       ==0} {set SYN        1            } else {set SYN        $::env(SYN)       }
if {[info exists env(CSIM)]      ==0} {set CSIM       0            } else {set CSIM       $::env(CSIM)      }
if {[info exists env(COSIM)]     ==0} {set COSIM      0            } else {set COSIM      $::env(COSIM)     }
if {[info exists env(GUI)]       ==0} {set GUI        0            } else {set GUI        $::env(GUI)       }

set DIR_SRC    ../../../../LeNet-5.dlr/native.cpp/src
set DIR_TB     ../../../../LeNet-5.dlr/native.cpp/src
set DIR_LIB    ../../../../../../Deep_Learning_Routines.master/v1.3/src
set DIR_PARAMS ../../../../LeNet-5.pytorch
set PNG [file normalize ../../../../LeNet-5.pytorch/samples/t_00_c4.png]
set PERIOD [expr 1000.0/${FREQ}]

############################################################
# Create project
# if by any chance your file list changes use
# open_project -reset "VivadoTutorial"
# to make sure you delete the existing project and create a new one with the updated file list
open_project ${PROJECT}

# Adding HLS files
set CPPFLAGS    "-std=c++11 -c -O3 -m64 -mcmodel=large\
                 -D__SYNTHESIS__ -DDTYPE=float -DEMBED_RELU=${EMBED_RELU} \
                 -I${DIR_SRC} -I${DIR_LIB} -I${DIR_PARAMS} -I${DIR_TB}"
set CPPFLAGS_TB "-std=c++11    -O3 -m64 -mcmodel=large\
                 -D__SYNTHESIS__ -DDTYPE=float -DEMBED_RELU=${EMBED_RELU} \
                 -I${DIR_SRC} -I${DIR_LIB} -I${DIR_PARAMS} -I${DIR_TB}"

add_files -cflags ${CPPFLAGS}        "${DIR_SRC}/lenet5.cpp"
add_files -cflags ${CPPFLAGS_TB} -tb "${DIR_TB}/main.cpp"

# Setting the top-level function
set_top ${TOP}

################### SOLUTION SETUP ###################
# creates or, if already existing, opens a new solution
# if by any chance your clock or target device changes use
# open_solution -reset "ZED", ${SOLUTION}
# to make sure you delete the existing project and create a new one with the updated file list
open_solution ${SOLUTION}

# Sets the target device
set_part ${PART}

# Sets the clock
create_clock -period ${PERIOD} -name default

################### SOLUTION SETUP ###################
if {${CSIM}==1} {
    csim_design -clean -profile -argv "--rigor --verbose ${PNG}"
    # look "proj_lenet/solution1/csim/build"
    # $ ./proj_lenet/solution1/csim/build/csim.exe ../../../../LeNet-5.pytorch/samples/t_00_c4.png
}

if {(${SYN}==1) || (${COSIM}==1)} {
    csynth_design
    export_design -format ip_catalog
}

if {${COSIM}==1} {
    if {${GUI}==1} {
        cosim_design -disable_deadlock_detection -trace_level port -wave_debug -rtl verilog -tool xsim -argv ${PNG}
        # add '-debug off' of xelab at project_lenet/solution1/sim/verilog/run_xsim.sh
    } else {
        cosim_design -disable_deadlock_detection -rtl verilog -tool xsim -argv ${PNG}
    }
}

exit
################### HLS ###################
# refer to "lenet5/zed/impl/verilog/*.v'
# refer to "lenet5/zed/impl/ip/hdl/ip/*.vhd" if any
# refer to "lenet5/zed/impl/misc/drivers/lenet5_v1_0/src/xlenet5_hw.h"
###########################################
