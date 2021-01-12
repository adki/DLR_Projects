SHELL=/bin/bash
#
# usage: $ make                 for floating-point (32-bit float)
# usage: $ make HALF=1          for half-precision (16-bit float)
# usage: $ make FIXED_POINT=1   for fixed-point
#
ARCH=$(shell uname -s)
ifeq ($(findstring MINGW,$(ARCH)),MINGW)
      PLATFORM=mingw
else ifeq ($(findstring CYGWIN,$(ARCH)),CYGWIN)
      PLATFORM=cygwin
else ifeq ($(findstring Linux,$(ARCH)),Linux)
      PLATFORM=linux
endif
KERN:=$(PLATFORM)
MACHINE:=$(shell uname -m)
MACH:=$(MACHINE)
PROCESSOR:=$(uname -p)

ifeq ($(MAKECMDGOALS),)
ifndef CONFMC_HOME
       $(error CONFMC_HOME environment variable not defined)
endif
endif
ifeq ($(MAKECMDGOALS),all)
ifndef CONFMC_HOME
       $(error CONFMC_HOME environment variable not defined)
endif
endif

ifndef OBJECTDIR
  OBJECTDIR = obj
endif
ifeq (${wildcard $(OBJECTDIR)},)
  DUMMY:= $(shell mkdir $(OBJECTDIR))
endif
ifeq (${wildcard compile.log},)
  DUMMY:= $(shell /bin/rm -f compile.log)
endif

PROG   = lenet
SRCCXX = main.cpp
SRCC   = trx_axi_api.c
OBJS   = $(SRCCXX:.cpp=.o) $(SRCC:.c=.o)

CC          = g++
DEFINES     = -DRIGOR -DSILENCE
INCDIRS     = -Isrc -I$(CONFMC_HOME)/include
CFLAGS     += -O3
LDFLAGS    += 
CONFMC      = -L$(CONFMC_HOME)/lib/$(PLATFORM)_$(MACHINE) -lconapi -lusb-1.0
LIBS       += $(CONFMC)
ifdef HALF
    DEFINES    += -DHALF -DDTYPE="half" -DNBYTES_OF_DTYPE=2 -DHLS_NO_XIL_FPO_LIB
    INCDIRS    += -I/opt/XilinxWebpack/Vivado/2018.3/include
else
    ifdef FIXED_POINT
    #DEFINES    += -DFIXED_POINT -DDTYPE="ap_fixed<32,8>" -DNBYTES_OF_DTYPE=4
    #DEFINES    += -DFIXED_POINT -DDTYPE="ap_fixed<32,4>" -DNBYTES_OF_DTYPE=4
    #DEFINES    += -DFIXED_POINT -DDTYPE="ap_fixed<16,8>" -DNBYTES_OF_DTYPE=2
     DEFINES    += -DFIXED_POINT -DDTYPE="ap_fixed<16,6>" -DNBYTES_OF_DTYPE=2
    INCDIRS    += -I/opt/XilinxWebpack/Vivado/2018.3/include
    else
    DEFINES    += -DDTYPE=float -DNBYTES_OF_DTYPE=4
    endif
endif
ifdef DEBUG
     DEFINES    += -DDEBUG
endif

vpath %.cpp src
vpath %.c   src
ifdef FIXED_POINT
vpath %.h   src /opt/XilinxWebpack/Vivado/2018.3/include
else
vpath %.h   src
endif

$(OBJECTDIR)/%.o: %.c
	$(CC) -c $(DEFINES) $(CFLAGS) $(INCDIRS) -o $@ $<

$(OBJECTDIR)/%.o: %.cpp
	$(CC) -c $(DEFINES) $(CFLAGS) $(INCDIRS) -o $@ $<

all: $(addprefix $(OBJECTDIR)/,$(OBJS))
	$(CC) -o $(PROG) $^ $(LDFLAGS) $(LIBS)

run:
	./$(PROG) ./images/0.png
	./$(PROG) ./images/1.png

run.all:
	for N in 0 1 2 3 4 5 6 7 8 9; do\
		echo "./$(PROG) ./images/$$N.png";\
		./$(PROG) ./images/$$N.png;\
	done
	echo "./$(PROG) ./images/3b.png";\
	./$(PROG) ./images/3b.png;\
	echo "./$(PROG) ./images/0b.png";\
	./$(PROG) ./images/0b.png;\
	echo "./$(PROG) ./images/0c.png";\
	./$(PROG) ./images/0c.png;\

clean:
	/bin/rm -rf $(OBJECTDIR)
	/bin/rm -f  $(PROG)
	/bin/rm -f  resized.png reverted.png

cleanup: clean

cleanupall clobber distclean: cleanup

.PHONY: all run clean cleanup clobber cleanupall distclean
