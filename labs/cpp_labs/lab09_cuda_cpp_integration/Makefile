# Makefile for Lab 09: CUDA C++ Integration

# Use nvcc for CUDA code, g++ for regular C++
NVCC = nvcc
CXX = g++
NVCCFLAGS = -std=c++17 -O2
CXXFLAGS = -std=c++17 -Wall -Wextra -g
LDFLAGS =

STARTER = starter
SOLUTION = solution
TEST = test

.PHONY: all clean run-starter run-solution run-test

# Default target - requires CUDA
all: starter test solution-if-cuda

# Always build starter and test (don't require CUDA)
starter: starter.cpp
	$(CXX) $(CXXFLAGS) -o $(STARTER) starter.cpp $(LDFLAGS)

test: test.cpp
	$(CXX) $(CXXFLAGS) -o $(TEST) test.cpp $(LDFLAGS)

# Only build solution if nvcc is available
solution-if-cuda:
	@which nvcc > /dev/null 2>&1 && $(MAKE) solution || echo "CUDA not found. Skipping solution build."

solution: solution.cu
	$(NVCC) $(NVCCFLAGS) -o $(SOLUTION) solution.cu

run-starter: starter
	./$(STARTER)

run-solution: solution
	./$(SOLUTION)

run-test: test
	./$(TEST)

clean:
	rm -f $(STARTER) $(SOLUTION) $(TEST) *.o

help:
	@echo "Available targets:"
	@echo "  make starter      - Build starter (no CUDA required)"
	@echo "  make test         - Build tests (no CUDA required)"
	@echo "  make solution     - Build solution (requires CUDA toolkit)"
	@echo "  make all          - Build all (solution only if CUDA available)"
	@echo "  make run-starter  - Run starter"
	@echo "  make run-solution - Run solution (requires GPU)"
	@echo "  make run-test     - Run tests"
	@echo "  make clean        - Remove build artifacts"
