# Lab 05: Shared Memory - Makefile

NVCC = nvcc
CUDA_ARCH = sm_75
NVCC_FLAGS = -arch=$(CUDA_ARCH) -O3 --use_fast_math -lineinfo

all: starter solution test

starter: starter.cu
	$(NVCC) $(NVCC_FLAGS) starter.cu -o stencil_starter

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) solution.cu -o stencil_solution

test: test.cu
	$(NVCC) $(NVCC_FLAGS) test.cu -o stencil_test
	./stencil_test

profile: solution
	ncu --metrics shared_efficiency,shared_load_throughput ./stencil_solution

clean:
	rm -f stencil_starter stencil_solution stencil_test *.o *.nsys-rep *.ncu-rep

.PHONY: all starter solution test profile clean
