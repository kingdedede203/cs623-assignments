NVCC=nvcc -ccbin=$(CXX) -D_FORCE_INLINES
PTXAS=ptxas
BIN2C=bin2c

DEF_OPTS=
SUFFIX=

ifeq ($(USE_ASYNC_STREAM),1)
DEF_OPTS=-DUSE_ASYNC_STREAM
SUFFIX=_async_stream
endif

ifeq ($(DEBUG),1)
NVCC_OPT=-O0 -g
else
NVCC_OPT=-O3
endif

NVCC_VER_REQ=10.1
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK=$(shell echo "${NVCC_VER} >= $(NVCC_VER_REQ)" | bc)

ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile an nvbit tool! Instrumented applications can still use lower versions of nvcc.)
endif

PTXAS_VER_ADD_FLAG=12.3
PTXAS_VER=$(shell $(PTXAS) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
PTXAS_VER_CHECK=$(shell echo "${PTXAS_VER} >= $(PTXAS_VER_ADD_FLAG)" | bc)

ifeq ($(PTXAS_VER_CHECK), 0)
MAXRREGCOUNT_FLAG=-maxrregcount=24
else
MAXRREGCOUNT_FLAG=
endif

NVBIT_PATH=../core
INCLUDES=-I$(NVBIT_PATH)

LIBS=-L$(NVBIT_PATH) -lnvbit
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

SOURCES=$(wildcard *.cu)
TOOL_FUNC_SOURCES=$(wildcard tool_func/*.cu)

OBJECTS=$(subst $(current_dir),$(current_dir)$(SUFFIX),$(SOURCES:.cu=.o))
TOOL_FUNC_FATBINS=$(TOOL_FUNC_SOURCES:.cu=.fatbin)
TOOL_FUNC_BIN2CS=$(TOOL_FUNC_FATBINS:.fatbin=.c)

ARCH?=all

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

NVBIT_TOOL=$(current_dir)$(SUFFIX).so

all: $(NVBIT_TOOL)

$(NVBIT_TOOL): $(OBJECTS) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) $(NVCC_OPT) $(OBJECTS) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

tool_func/%.fatbin: tool_func/%.cu
	$(NVCC) -arch=$(ARCH) $(INCLUDES) $(NVCC_OPT) -fatbin $< -o $@

tool_func/%.c: tool_func/%.fatbin
	$(BIN2C) -c --name $(basename $(notdir $<))_bin $< > $@

%$(SUFFIX).o: %.cu $(TOOL_FUNC_BIN2CS)
	$(NVCC) -dc -c -std=c++11 $(DEF_OPTS) $(INCLUDES) -Xptxas -cloning=no -Xcompiler -Wall -arch=$(ARCH) $(NVCC_OPT) -Xcompiler -fPIC $< -o $@

inject_funcs.o: inject_funcs.cu
	$(NVCC) $(INCLUDES) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch --keep-device-functions -arch=$(ARCH) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

clean:
	rm -f *.so *.o
