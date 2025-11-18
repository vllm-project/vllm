# Lab 06: Atomic Operations - Makefile

NVCC = nvcc
CUDA_ARCH = sm_75
NVCC_FLAGS = -arch=$(CUDA_ARCH) -O3 --use_fast_math -lineinfo

all: starter solution test

starter: starter.cu
	$(NVCC) $(NVCC_FLAGS) starter.cu -o atomic_starter

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) solution.cu -o atomic_solution

test: test.cu
	$(NVCC) $(NVCC_FLAGS) test.cu -o atomic_test
	./atomic_test

profile: solution
	ncu --metrics atomic_transactions ./atomic_solution

clean:
	rm -f atomic_starter atomic_solution atomic_test *.o *.nsys-rep *.ncu-rep

.PHONY: all starter solution test profile clean
