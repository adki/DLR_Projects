SHELL=/bin/bash

ifndef DLR_HOME
    export DLR_HOME := ../../../../Deep_Learning_Routines.master/v1.3
endif
ifndef LENET5_HOME
    export LENET5_HOME := ../..
endif

LIB_DLR    := dlr
LIB_A      := lib$(LIB_DLR).a
DIR_LIB    := $(DLR_HOME)/lib

DIR_OBJ    := obj
DUMMY      := $(shell [ -d $(DIR_OBJ) ] || mkdir $(DIR_OBJ) )

PROG       := lenet5
DIR_SRC    := src
C_SRCS     :=
CPP_SRCS   := lenet5.cpp main.cpp
C_HDRS     := lenet5_params.h stb_image.h stb_image_resize.h stb_image_write.h
CPP_HDRS   :=
DIR_INC    := $(DIR_SRC) $(DLR_HOME)/include $(LENET5_HOME)/LeNet-5.pytorch

OBJS       := $(addprefix $(DIR_OBJ)/,$(patsubst %.c,%.o,$(notdir $(C_SRCS))))
OBJS       += $(addprefix $(DIR_OBJ)/,$(patsubst %.cpp,%.o,$(notdir $(CPP_SRCS))))
INCS       := $(addprefix -I,$(DIR_INC))

# define 'EMBED_ReLU' 1 in order to embed ReLU in Pooling2dMax
# define 'EMBED_ReLU' 0 in order to use ReLU separatly
# ReLU is not embedded in Pooling2dMax when 'EMBED_ReLU' is not defined.
DEFINES    := -DEMBED_ReLU=1 #-DDEBUG
CPPFLAGS   := -c $(INCS) $(DEFINES)
LDFLAGS    := -Wl,-Bstatic -L$(DIR_LIB) -l$(LIB_DLR) -Wl,-Bdynamic -lm

ifdef PROFILE
CPPFLAGS += -pg
LDFLAGS  += -pg
endif

vpath %.h   $(DIR_SRC) $(DIR_INC)
vpath %.c   $(DIR_SRC)
vpath %.hpp $(DIR_SRC) $(DIR_INC)
vpath %.cpp $(DIR_SRC)

$(DIR_OBJ)/%.o: %.c
	g++ $(CPPFLAGS) -o $@ $<

$(DIR_OBJ)/%.o: %.cpp
	g++ $(CPPFLAGS) -o $@ $<

all: $(OBJS)
	g++ -o $(PROG) $(OBJS) $(LDFLAGS)

run:
	./$(PROG) --rigor $(LENET5_HOME)/LeNet-5.pytorch/samples/t_00_c4.png

run.all:
	./$(PROG) --rigor\
		$(LENET5_HOME)/LeNet-5.pytorch/samples/t_00_c4.png\
		$(LENET5_HOME)/LeNet-5.pytorch/samples/t_01_c7.png\
		$(LENET5_HOME)/LeNet-5.pytorch/samples/t_02_c7.png\
		$(LENET5_HOME)/LeNet-5.pytorch/samples/t_03_c2.png\
		$(LENET5_HOME)/LeNet-5.pytorch/samples/t_04_c3.png\
		$(LENET5_HOME)/LeNet-5.pytorch/samples/t_06_c5.png\
		$(LENET5_HOME)/LeNet-5.pytorch/samples/t_07_c6.png

run.profile:
	g++ -c -pg -Isrc -I$(DLR_HOME)/include -I$(LENET5_HOME)/LeNet-5.pytorch -DEMBED_ReLU=1  -o obj/lenet5.o src/lenet5.cpp
	g++ -c -Isrc -I$(DLR_HOME)/include -I$(LENET5_HOME)/LeNet-5.pytorch -DEMBED_ReLU=1  -o obj/main.o src/main.cpp
	g++ -pg -o lenet5  obj/lenet5.o obj/main.o -Wl,-Bstatic -L$(DLR_HOME)/lib -ldlr -Wl,-Bdynamic -lm
	./$(PROG) $(LENET5_HOME)/LeNet-5.pytorch/samples/t_00_c4.png
	gprof lenet5 gmon.out > profile.txt

clean:
	/bin/rm -f   *.o
	/bin/rm -fr  $(DIR_OBJ)
	/bin/rm -f   resized.png  reverted.png
	/bin/rm -f   gmon.out profile.txt

cleanup clobber: clean
	/bin/rm -f $(PROG)

cleanupall distclean: cleanup

