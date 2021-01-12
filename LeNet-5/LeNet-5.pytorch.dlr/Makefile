SHELL=/bin/sh

ifndef DLR_HOME
    export DLR_HOME := ../../../Deep_Learning_Routines.master/v1.3
endif
ifdef PYTHONPATH
  export PYTHONPATH=$(DLR_HOME):$(shell printenv PYTHONPATH)
else
  export PYTHONPATH=$(DLR_HOME)
endif

PYTHON        ?= python3
DIR_SRC        = src
DIR_MODELS     = ../LeNet-5.pytorch/checkpoints
DIR_SAMPLES    = ../LeNet-5.pytorch/samples
INPUT_CHANNELS = 1

all:
	@echo "run \"$ make run\""

run:
	@if [ -f $(DIR_MODELS)/mnist_params_final.pth ] &&\
		[ -f $(DIR_SAMPLES)/t_00_c4.png ]; then\
		${PYTHON} $(DIR_SRC)/lenet5_infer.py\
			--checkpoint=$(DIR_MODELS)/mnist_params_final.pth\
			--input_channels=${INPUT_CHANNELS}\
			$(DIR_SAMPLES)/t_00_c4.png;\
	else\
		echo \"$(DIR_MODELS)/mnist_params_final.pth\" or\
			\"$(DIR_SAMPLES)/t_00_c4.png\" not found;\
	fi;

help:
	${PYTHON} $(DIR_SRC)/lenet5_infer.py --help

requirements:
	${PIP} install -r requirements.txt

clean:
	/bin/rm -rf src/__pycache__

cleanup clobber: clean
	/bin/rm -rf train test

cleanupall distclean: cleanup
	/bin/rm -rf checkpoints

.PHONY: all clean cleanup clobber cleanupall distclean
.PHONY: run_train run.train train_run train.run
.PHONY: run_infer run.infer infer_run infer.run predict depoly infer
.PHONY: run_params run.params params
.PHONY: requirements
