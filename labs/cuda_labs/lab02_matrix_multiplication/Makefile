# Lab 02: Matrix Multiplication - Makefile

NVCC = nvcc
CUDA_ARCH = sm_75
NVCC_FLAGS = -arch=$(CUDA_ARCH) -O3 --use_fast_math -lineinfo

TARGETS = matmul_starter matmul_solution matmul_test

all: starter solution test

starter: matmul_starter
matmul_starter: starter.cu
	$(NVCC) $(NVCC_FLAGS) starter.cu -o matmul_starter

solution: matmul_solution
matmul_solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) solution.cu -o matmul_solution

test: matmul_test
matmul_test: test.cu
	$(NVCC) $(NVCC_FLAGS) test.cu -o matmul_test
	./matmul_test

profile: solution
	nsys profile --stats=true ./matmul_solution 2048 2048 2048

benchmark: solution
	@echo "=== Square matrices ==="
	./matmul_solution 512 512 512
	./matmul_solution 1024 1024 1024
	./matmul_solution 2048 2048 2048

clean:
	rm -f $(TARGETS) *.o *.nsys-rep *.ncu-rep

.PHONY: all starter solution test profile benchmark clean
