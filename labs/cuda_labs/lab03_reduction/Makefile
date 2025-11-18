# Lab 03: Reduction - Makefile

NVCC = nvcc
CUDA_ARCH = sm_75
NVCC_FLAGS = -arch=$(CUDA_ARCH) -O3 --use_fast_math -lineinfo

all: starter solution test

starter: starter.cu
	$(NVCC) $(NVCC_FLAGS) starter.cu -o reduce_starter

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) solution.cu -o reduce_solution

test: test.cu
	$(NVCC) $(NVCC_FLAGS) test.cu -o reduce_test
	./reduce_test

profile: solution
	nsys profile --stats=true ./reduce_solution

clean:
	rm -f reduce_starter reduce_solution reduce_test *.o *.nsys-rep

.PHONY: all starter solution test profile clean
