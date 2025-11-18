# Lab 04: Memory Coalescing - Makefile

NVCC = nvcc
CUDA_ARCH = sm_75
NVCC_FLAGS = -arch=$(CUDA_ARCH) -O3 --use_fast_math -lineinfo

all: starter solution test

starter: starter.cu
	$(NVCC) $(NVCC_FLAGS) starter.cu -o transpose_starter

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) solution.cu -o transpose_solution

test: test.cu
	$(NVCC) $(NVCC_FLAGS) test.cu -o transpose_test
	./transpose_test

profile: solution
	ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./transpose_solution

clean:
	rm -f transpose_starter transpose_solution transpose_test *.o *.nsys-rep *.ncu-rep

.PHONY: all starter solution test profile clean
