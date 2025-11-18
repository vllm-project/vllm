# Lab 08: Occupancy Optimization - Makefile

NVCC = nvcc
CUDA_ARCH = sm_75
NVCC_FLAGS = -arch=$(CUDA_ARCH) -O3 --use_fast_math -lineinfo

all: starter solution test

starter: starter.cu
	$(NVCC) $(NVCC_FLAGS) starter.cu -o occupancy_starter

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) solution.cu -o occupancy_solution

test: test.cu
	$(NVCC) $(NVCC_FLAGS) test.cu -o occupancy_test
	./occupancy_test

profile: solution
	ncu --set full --section SpeedOfLight --section Occupancy ./occupancy_solution

ptx: solution.cu
	$(NVCC) -arch=$(CUDA_ARCH) --ptx solution.cu -o occupancy.ptx
	@echo "PTX generated: occupancy.ptx"
	@echo "Check register usage with: grep '.reg' occupancy.ptx"

clean:
	rm -f occupancy_starter occupancy_solution occupancy_test *.o *.nsys-rep *.ncu-rep *.ptx

.PHONY: all starter solution test profile ptx clean
