#!/bin/bash

/bin/rm -f  vivado.jou
/bin/rm -f  vivado.log
/bin/rm -f  vivado_*.backup.jou
/bin/rm -f  vivado_*.backup.log
/bin/rm -f  hs_err_*.log
/bin/rm -f  vivado_pid*.str
/bin/rm -f  vivado_pid*.zip
/bin/rm -fr NA
/bin/rm -fr .Xil
/bin/rm -fr project_1
/bin/rm -fr hd_visual
/bin/rm -fr zed_example

for F in *; do
    if [[ -d "${F}" && ! -L "${F}" ]]; then
    if [ -f ${F}/Clean.sh ]; then
       ( cd ${F}; ./Clean.sh )
    fi
    fi
done
