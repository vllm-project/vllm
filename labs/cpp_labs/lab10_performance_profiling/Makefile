# Makefile for Lab 10: Performance Profiling

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g
OPTFLAGS = -O2
PROFFLAGS = -pg

STARTER = starter
SOLUTION = solution
TEST = test

.PHONY: all clean run-starter run-solution run-test profile-gprof profile-perf

all: starter solution test

# Debug builds (no optimization)
starter: starter.cpp
	$(CXX) $(CXXFLAGS) -o $(STARTER) starter.cpp

solution: solution.cpp
	$(CXX) $(CXXFLAGS) -o $(SOLUTION) solution.cpp

test: test.cpp
	$(CXX) $(CXXFLAGS) -o $(TEST) test.cpp

# Optimized builds
starter-opt: starter.cpp
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $(STARTER)-opt starter.cpp

solution-opt: solution.cpp
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $(SOLUTION)-opt solution.cpp

# Profiling builds
solution-prof: solution.cpp
	$(CXX) $(CXXFLAGS) $(PROFFLAGS) -o $(SOLUTION)-prof solution.cpp

# Run targets
run-starter: starter
	./$(STARTER)

run-solution: solution
	./$(SOLUTION)

run-test: test
	./$(TEST)

# Run optimized version
run-opt: solution-opt
	./$(SOLUTION)-opt

# Profile with gprof
profile-gprof: solution-prof
	./$(SOLUTION)-prof
	gprof $(SOLUTION)-prof gmon.out > gprof-analysis.txt
	@echo "Profiling data written to gprof-analysis.txt"
	@head -n 50 gprof-analysis.txt

# Profile with perf (Linux only)
profile-perf: solution-opt
	@which perf > /dev/null 2>&1 && \
		(perf record -g ./$(SOLUTION)-opt && \
		 perf report) || \
		echo "perf not found. Install with: sudo apt-get install linux-tools-generic"

# Cache analysis with valgrind
profile-cache: solution
	@which valgrind > /dev/null 2>&1 && \
		valgrind --tool=cachegrind ./$(SOLUTION) || \
		echo "valgrind not found. Install with: sudo apt-get install valgrind"

# Compare optimizations
compare: solution solution-opt
	@echo "=== Debug build (no optimization) ==="
	./$(SOLUTION)
	@echo "\n=== Optimized build (-O2) ==="
	./$(SOLUTION)-opt

clean:
	rm -f $(STARTER) $(SOLUTION) $(TEST) *-opt *-prof *.o
	rm -f gmon.out perf.data perf.data.old gprof-analysis.txt
	rm -f cachegrind.out.*

help:
	@echo "Available targets:"
	@echo "  make starter          - Build starter (debug)"
	@echo "  make solution         - Build solution (debug)"
	@echo "  make solution-opt     - Build solution (optimized)"
	@echo "  make solution-prof    - Build solution (with gprof)"
	@echo "  make test             - Build tests"
	@echo "  make run-starter      - Run starter"
	@echo "  make run-solution     - Run solution"
	@echo "  make run-opt          - Run optimized solution"
	@echo "  make run-test         - Run tests"
	@echo "  make profile-gprof    - Profile with gprof"
	@echo "  make profile-perf     - Profile with perf (Linux)"
	@echo "  make profile-cache    - Analyze cache with valgrind"
	@echo "  make compare          - Compare debug vs optimized"
	@echo "  make clean            - Remove build artifacts"
