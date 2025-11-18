# Lab 07: Warp Shuffle - Makefile

NVCC = nvcc
CUDA_ARCH = sm_75
NVCC_FLAGS = -arch=$(CUDA_ARCH) -O3 --use_fast_math -lineinfo

all: starter solution test

starter: starter.cu
	$(NVCC) $(NVCC_FLAGS) starter.cu -o shuffle_starter

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) solution.cu -o shuffle_solution

test: test.cu
	$(NVCC) $(NVCC_FLAGS) test.cu -o shuffle_test
	./shuffle_test

profile: solution
	ncu --metrics warp_execution_efficiency ./shuffle_solution

clean:
	rm -f shuffle_starter shuffle_solution shuffle_test *.o *.nsys-rep *.ncu-rep

.PHONY: all starter solution test profile clean
