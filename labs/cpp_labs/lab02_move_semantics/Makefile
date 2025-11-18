# Makefile for Lab 02: Move Semantics

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g -O2
LDFLAGS =

# Targets
STARTER = starter
SOLUTION = solution
TEST = test

.PHONY: all clean starter solution test run-starter run-solution run-test benchmark

all: starter solution test

# Build starter code
starter: starter.cpp
	$(CXX) $(CXXFLAGS) -o $(STARTER) starter.cpp $(LDFLAGS)

# Build solution
solution: solution.cpp
	$(CXX) $(CXXFLAGS) -o $(SOLUTION) solution.cpp $(LDFLAGS)

# Build tests
test: test.cpp
	$(CXX) $(CXXFLAGS) -o $(TEST) test.cpp $(LDFLAGS)

# Run targets
run-starter: starter
	./$(STARTER)

run-solution: solution
	./$(SOLUTION)

run-test: test
	./$(TEST)

# Benchmark copy vs move performance
benchmark: solution
	@echo "=== Running performance benchmark ==="
	./$(SOLUTION)

# Build with optimization for benchmarking
solution-opt: solution.cpp
	$(CXX) -std=c++17 -Wall -Wextra -O3 -DNDEBUG -o $(SOLUTION)-opt solution.cpp $(LDFLAGS)
	@echo "Running optimized benchmark..."
	./$(SOLUTION)-opt

# Check assembly output (see if moves are optimized out)
assembly: solution.cpp
	$(CXX) $(CXXFLAGS) -S -o solution.s solution.cpp

# Address sanitizer builds
asan-starter: starter.cpp
	$(CXX) $(CXXFLAGS) -fsanitize=address -o $(STARTER)-asan starter.cpp $(LDFLAGS)

asan-solution: solution.cpp
	$(CXX) $(CXXFLAGS) -fsanitize=address -o $(SOLUTION)-asan solution.cpp $(LDFLAGS)

# Clean build artifacts
clean:
	rm -f $(STARTER) $(SOLUTION) $(TEST) *-asan *-opt *.o *.s

# Help target
help:
	@echo "Available targets:"
	@echo "  make starter          - Build starter code"
	@echo "  make solution         - Build solution code"
	@echo "  make test             - Build test suite"
	@echo "  make all              - Build everything"
	@echo "  make run-starter      - Build and run starter"
	@echo "  make run-solution     - Build and run solution"
	@echo "  make run-test         - Build and run tests"
	@echo "  make benchmark        - Run performance benchmark"
	@echo "  make solution-opt     - Build optimized version and benchmark"
	@echo "  make assembly         - Generate assembly to see optimizations"
	@echo "  make asan-starter     - Build starter with AddressSanitizer"
	@echo "  make asan-solution    - Build solution with AddressSanitizer"
	@echo "  make clean            - Remove build artifacts"
