# Makefile for Lab 01: Memory Management

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g
LDFLAGS =

# Targets
STARTER = starter
SOLUTION = solution
TEST = test

.PHONY: all clean starter solution test run-starter run-solution run-test

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

# Memory leak detection (requires valgrind)
valgrind-starter: starter
	valgrind --leak-check=full --show-leak-kinds=all ./$(STARTER)

valgrind-solution: solution
	valgrind --leak-check=full --show-leak-kinds=all ./$(SOLUTION)

# Address sanitizer builds (detects memory errors at runtime)
asan-starter: starter.cpp
	$(CXX) $(CXXFLAGS) -fsanitize=address -o $(STARTER)-asan starter.cpp $(LDFLAGS)
	./$(STARTER)-asan

asan-solution: solution.cpp
	$(CXX) $(CXXFLAGS) -fsanitize=address -o $(SOLUTION)-asan solution.cpp $(LDFLAGS)
	./$(SOLUTION)-asan

# Clean build artifacts
clean:
	rm -f $(STARTER) $(SOLUTION) $(TEST) *-asan *.o

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
	@echo "  make valgrind-starter - Run starter with valgrind"
	@echo "  make valgrind-solution - Run solution with valgrind"
	@echo "  make asan-starter     - Build and run with AddressSanitizer"
	@echo "  make asan-solution    - Build and run with AddressSanitizer"
	@echo "  make clean            - Remove build artifacts"
