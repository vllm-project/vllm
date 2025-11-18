# Lab 01: Vector Addition - Makefile
#
# Build configuration for CUDA programs
#
# Usage:
#   make              - Build starter code
#   make solution     - Build solution
#   make test         - Build and run test suite
#   make all          - Build everything
#   make clean        - Remove all build artifacts
#   make profile      - Run with profiling
#   make benchmark    - Run performance benchmark

# Compiler and flags
NVCC = nvcc
CUDA_ARCH = sm_75  # Change based on your GPU (sm_75=Turing, sm_80=Ampere, sm_86=RTX30xx, sm_89=RTX40xx)

# Compute capability reference:
# sm_75: RTX 20xx, Titan RTX, Tesla T4
# sm_80: A100
# sm_86: RTX 30xx
# sm_89: RTX 40xx
# sm_90: H100
# Use 'nvidia-smi --query-gpu=compute_cap --format=csv' to find yours

# Optimization flags
NVCC_FLAGS = -arch=$(CUDA_ARCH) \
             -O3 \
             --use_fast_math \
             -Xcompiler -Wall \
             -Xcompiler -Wextra \
             -lineinfo

# Debug flags (use with: make DEBUG=1)
DEBUG_FLAGS = -g -G -O0 -DDEBUG

# Profiling flags
PROFILE_FLAGS = -lineinfo

# Targets
STARTER_TARGET = vector_add_starter
SOLUTION_TARGET = vector_add_solution
TEST_TARGET = vector_add_test

# Source files
STARTER_SRC = starter.cu
SOLUTION_SRC = solution.cu
TEST_SRC = test.cu

# Default target
.PHONY: all
all: starter solution test

# Build starter code
.PHONY: starter
starter: $(STARTER_TARGET)

$(STARTER_TARGET): $(STARTER_SRC)
	@echo "Building starter code..."
	$(NVCC) $(NVCC_FLAGS) $(STARTER_SRC) -o $(STARTER_TARGET)
	@echo "Build complete: $(STARTER_TARGET)"
	@echo "Run with: ./$(STARTER_TARGET) [vector_size]"

# Build solution
.PHONY: solution
solution: $(SOLUTION_TARGET)

$(SOLUTION_TARGET): $(SOLUTION_SRC)
	@echo "Building solution..."
	$(NVCC) $(NVCC_FLAGS) $(SOLUTION_SRC) -o $(SOLUTION_TARGET)
	@echo "Build complete: $(SOLUTION_TARGET)"

# Build test suite
.PHONY: test
test: $(TEST_TARGET)

$(TEST_TARGET): $(TEST_SRC)
	@echo "Building test suite..."
	$(NVCC) $(NVCC_FLAGS) $(TEST_SRC) -o $(TEST_TARGET)
	@echo "Build complete: $(TEST_TARGET)"
	@echo ""
	@echo "Running tests..."
	@./$(TEST_TARGET)

# Debug build
.PHONY: debug
debug: NVCC_FLAGS = -arch=$(CUDA_ARCH) $(DEBUG_FLAGS)
debug: clean starter
	@echo "Debug build complete"
	@echo "Run with: cuda-gdb ./$(STARTER_TARGET)"

# Profile with Nsight Systems
.PHONY: profile
profile: solution
	@echo "Profiling with Nsight Systems..."
	nsys profile --stats=true -o profile_report ./$(SOLUTION_TARGET) 10000000
	@echo ""
	@echo "Profile complete. View with:"
	@echo "  nsys-ui profile_report.nsys-rep"

# Profile with Nsight Compute
.PHONY: profile-compute
profile-compute: solution
	@echo "Profiling with Nsight Compute..."
	ncu --set full -o ncu_report ./$(SOLUTION_TARGET) 10000000
	@echo ""
	@echo "Profile complete. View with:"
	@echo "  ncu-ui ncu_report.ncu-rep"

# Benchmark different configurations
.PHONY: benchmark
benchmark: solution test
	@echo "Running comprehensive benchmark..."
	@echo ""
	@echo "=== Small vector (overhead analysis) ==="
	./$(SOLUTION_TARGET) 1000
	@echo ""
	@echo "=== Medium vector (typical workload) ==="
	./$(SOLUTION_TARGET) 1000000
	@echo ""
	@echo "=== Large vector (bandwidth saturation) ==="
	./$(SOLUTION_TARGET) 100000000
	@echo ""
	@echo "=== Test suite (correctness + performance) ==="
	./$(TEST_TARGET)

# Check for CUDA errors after run
.PHONY: check
check: solution
	cuda-memcheck ./$(SOLUTION_TARGET) 1000000

# Check compute capability
.PHONY: device-info
device-info:
	@echo "Querying CUDA device information..."
	@echo ""
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
	@echo ""
	@echo "Recommended CUDA_ARCH settings:"
	@echo "  Compute 7.5 (Turing):  sm_75"
	@echo "  Compute 8.0 (Ampere):  sm_80"
	@echo "  Compute 8.6 (RTX 30):  sm_86"
	@echo "  Compute 8.9 (RTX 40):  sm_89"
	@echo "  Compute 9.0 (Hopper):  sm_90"

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(STARTER_TARGET) $(SOLUTION_TARGET) $(TEST_TARGET)
	rm -f *.o *.nsys-rep *.ncu-rep *.sqlite
	rm -rf profile_report* ncu_report*
	@echo "Clean complete"

# Help target
.PHONY: help
help:
	@echo "Vector Addition Lab - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make              - Build starter code"
	@echo "  make starter      - Build starter code"
	@echo "  make solution     - Build solution"
	@echo "  make test         - Build and run test suite"
	@echo "  make all          - Build everything"
	@echo "  make debug        - Build with debug symbols"
	@echo "  make profile      - Profile with Nsight Systems"
	@echo "  make profile-compute - Profile with Nsight Compute"
	@echo "  make benchmark    - Run comprehensive benchmark"
	@echo "  make check        - Run with cuda-memcheck"
	@echo "  make device-info  - Show GPU information"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make help         - Show this message"
	@echo ""
	@echo "Options:"
	@echo "  CUDA_ARCH=sm_XX   - Set compute capability (default: sm_75)"
	@echo ""
	@echo "Examples:"
	@echo "  make CUDA_ARCH=sm_86              # Build for RTX 30xx"
	@echo "  ./vector_add_starter 10000000     # Run with 10M elements"
	@echo "  make profile                      # Profile solution"
	@echo ""

# PTX and SASS generation (for learning)
.PHONY: ptx
ptx: $(SOLUTION_SRC)
	@echo "Generating PTX assembly..."
	$(NVCC) -arch=$(CUDA_ARCH) --ptx $(SOLUTION_SRC) -o solution.ptx
	@echo "PTX generated: solution.ptx"

.PHONY: sass
sass: $(SOLUTION_SRC)
	@echo "Generating SASS assembly..."
	$(NVCC) -arch=$(CUDA_ARCH) -cubin $(SOLUTION_SRC) -o solution.cubin
	nvdisasm solution.cubin > solution.sass
	@echo "SASS generated: solution.sass"

# Default if no target specified
.DEFAULT_GOAL := starter
