# Copyright 2023 The FLash-LLM Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# host compiler
HOST_COMPILER ?= g++
CUDA_PATH ?= /usr/local/cuda/
#below is the path for Narval
#CUDA_PATH ?= /cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.7.0/
# CUDA_PATH ?= /cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.2.2/
NVCC          := /usr/local/cuda/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m$(shell getconf LONG_BIT)
CCFLAGS     := -fPIC
LDFLAGS     :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I/usr/local/cuda/include/ -I /home/ferrar/vllm/.deps/cutlass-src/include
LIBRARIES := -lcublas -lcusparse

################################################################################

# Gencode arguments
SMS ?= 90
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

ALL_CCFLAGS += --threads 0 --std=c++11 -lineinfo -O3

FLASHLLM_CCFLAGS := -maxrregcount=255
ALL_CCFLAGS += --use_fast_math
ALL_CCFLAGS += --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills
################################################################################

HEAD_FILES = ./util/command_line.h \
			 ./util/distribution.h \
			 ./util/host_tensor.h \
			 ./util/packed_stride.hpp \
			 ./util/tensor_view_io.h \
			 ./util/reference/device/gemm.h \
			 ./util/reference/device/tensor_compare.h \
			 ./util/reference/device/tensor_fill.h


# Target rules
all: example

example: 62_hopper_sparse_gemm.cu $(HEAD_FILES)
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(OUR_CCFLAGS) $(GENCODE_FLAGS) $< -o $@

clean:
	rm -f example