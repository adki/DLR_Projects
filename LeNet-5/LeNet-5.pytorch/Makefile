SHELL=/bin/sh

PYTHON         ?= python3
DIR_SRC         = src
DIR_WEIGHTS     = checkpoints
INPUT_CHANNELS ?= 1

all:

run_train run.train train_run train.run:
	${PYTHON} ${DIR_SRC}/lenet5_train.py --rigor\
			--accuracy=0.99 --epochs=100\
			--input_channels=${INPUT_CHANNELS}

run_train_more run.train.more train_run_more train.run.more:
	${PYTHON} ${DIR_SRC}/lenet5_train.py --rigor\
			--input_channels=${INPUT_CHANNELS}\
			--accuracy=0.995 --epochs=100\
			--pre_trained_type=model\
			--pre_trained_weights=checkpoints/mnist_model_0.990.pth

run_infer run.infer infer_run infer.run predict depoly infer:
	${PYTHON} ${DIR_SRC}/lenet5_infer.py\
		samples/t_00_c4.png\
		--type=model\
		--checkpoint=${DIR_WEIGHTS}/mnist_model_final.pth\
		--input_channels=${INPUT_CHANNELS}\
		--rigor #--debug --verbose

#${PYTHON} ${DIR_SRC}/lenet5_infer.py --type=weights --weights=../LeNet-5.darknet/backup/mnist_lenet.weights samples/t_00_c4.png --input_channels=${INPUT_CHANNELS} --softmax --rigor --debug --verbose
#${PYTHON} ${DIR_SRC}/lenet5_infer.py --type=weights --weights=${DIR_WEIGHTS}/mnist_final.weights    samples/t_00_c4.png --input_channels=${INPUT_CHANNELS} --rigor #--debug --verbose

#${PYTHON} ${DIR_SRC}/lenet5_infer.py --weights=lenet5.weights samples/t_00_c4.png --input_channels=${INPUT_CHANNELS} --rigor #--verbose
#${PYTHON} ${DIR_SRC}/lenet5_infer.py --type=weights --weights=../LeNet-5.darknet/backup/mnist_lenet.weights samples/t_00_c4.png --input_channels=${INPUT_CHANNELS} --rigor #--verbose

#${PYTHON} ${DIR_SRC}/lenet5_infer.py --type=params --weights=${DIR_WEIGHTS}/mnist_params_final.pth samples/t_00_c4.png --input_channels=${INPUT_CHANNELS} --verbose
#${PYTHON} ${DIR_SRC}/lenet5_infer.py --type=params --weights=${DIR_WEIGHTS}/mnist_params_final.pth samples/t_00_c4.png --input_channels=${INPUT_CHANNELS} --verbose
#${PYTHON} ${DIR_SRC}/lenet5_infer.py --type model  --weights=${DIR_WEIGHTS}/mnist_model_final.pth samples/t_00_c4.png --input_channels=${INPUT_CHANNELS} --verbose
#${PYTHON} ${DIR_SRC}/lenet5_infer.py --type model  --weights=${DIR_WEIGHTS}/mnist_model_final.pth samples/2.png --input_channels=${INPUT_CHANNELS} --verbose
#${PYTHON} ${DIR_SRC}/lenet5_infer.py --type params --weights=${DIR_WEIGHTS}/mnist_params_final.pth samples/3.jpg --input_channels=${INPUT_CHANNELS} --verbose
#${PYTHON} ${DIR_SRC}/lenet5_infer.py --weights=${DIR_WEIGHTS}/mnist_params_final.pth samples/3.png --input_channels=${INPUT_CHANNELS}
#${PYTHON} ${DIR_SRC}/lenet5_infer.py --weights=${DIR_WEIGHTS}/mnist_params_final.pth samples/4.png --input_channels=${INPUT_CHANNELS}
#${PYTHON} ${DIR_SRC}/lenet5_infer.py --weights=${DIR_WEIGHTS}/mnist_params_final.pth samples/3.png samples/4.png --input_channels=${INPUT_CHANNELS}

run_params run.params params:
	if [ -f ${DIR_WEIGHTS}/mnist_params_final.pth ]; then\
		${PYTHON} ${DIR_SRC}/lenet5_params.py\
			--input_channels=${INPUT_CHANNELS}\
			--checkpoint ${DIR_WEIGHTS}/mnist_params_final.pth\
			--txt x.txt\
			--bin y.bin\
			--header lenet5_params.h\
			--darknet lenet5.weights\
			--verbose;\
	else\
		echo \"${DIR_WEIGHTS}/mnist_params_final.pth\" not found;\
	fi

run_histogram run.histogram histogram:
	@if [ -f ${DIR_WEIGHTS}/mnist_params_final.pth ]; then\
		${PYTHON} ${DIR_SRC}/lenet5_histogram.py\
			--input_channels=${INPUT_CHANNELS}\
			--checkpoint ${DIR_WEIGHTS}/mnist_params_final.pth;\
	else\
		echo \"${DIR_WEIGHTS}/mnist_params_final.pth\" not found;\
	fi

run_viz run.viz:
	${PYTHON} ${DIR_SRC}/lenet5_viz.py\
		--input_channels=${INPUT_CHANNELS}\
		--checkpoint ${DIR_WEIGHTS}/mnist_params_final.pth\
		--result results\
		--verbose

tensorboard.voc:
	echo "You may need to run this on Andaconda, where 'tensorboard' is installed."
	tensorboard --logdir=tensorboard serve &
	echo "Now run web-browser with http://localhost:6006"
	google-chrome --no-sandbox --disable-setuid-sandbox http://localhost:6006

kill.tensorboard:
	$(eval pnum := $(shell pgrep -u ${USER} -f tensorboard | wc -l))
	if [ $(pnum) -gt 2 ]; then\
		pkill -9 -o tensorboard;\
	fi
#		pgrep -u ${USER} -f tensorboard | xargs kill -9

requirements:
	${PIP} install -r requirements.txt

clean:
	/bin/rm -rf src/__pycache__
	/bin/rm -f  lenet5_params_1ch.h
	/bin/rm -f  lenet5_params_3ch.h
	/bin/rm -f  lenet5_params.h
	/bin/rm -f  lenet5.weights
	/bin/rm -f  x.txt
	/bin/rm -f  y.bin

cleanup clobber: clean
	/bin/rm -rf dataset.train dataset.test
	/bin/rm -rf checkpoints/mnist_0.??.pth
	/bin/rm -rf checkpoints/mnist_model_0.??.pth
	/bin/rm -rf checkpoints/mnist_params_0.??.pth
	/bin/rm -rf checkpoints/mnist_model_0.??.onnx
	/bin/rm -rf tensorboard
	/bin/rm -f  x.txt y.bin

cleanupall distclean: cleanup
	/bin/rm -f  lenet5_params.h
	/bin/rm -rf checkpoints

.PHONY: all clean cleanup clobber cleanupall distclean
.PHONY: run_train run.train train_run train.run
.PHONY: run_infer run.infer infer_run infer.run predict depoly infer
.PHONY: run_params run.params params
.PHONY: requirements

# https://github.com/ChawDoe/LeNet5-MNIST-PyTorch
