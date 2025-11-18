# Makefile for Lab 03: Templates

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g
LDFLAGS =

STARTER = starter
SOLUTION = solution
TEST = test

.PHONY: all clean run-starter run-solution run-test

all: starter solution test

starter: starter.cpp
	$(CXX) $(CXXFLAGS) -o $(STARTER) starter.cpp $(LDFLAGS)

solution: solution.cpp
	$(CXX) $(CXXFLAGS) -o $(SOLUTION) solution.cpp $(LDFLAGS)

test: test.cpp
	$(CXX) $(CXXFLAGS) -o $(TEST) test.cpp $(LDFLAGS)

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
	@echo "  make starter      - Build starter code"
	@echo "  make solution     - Build solution code"
	@echo "  make test         - Build test suite"
	@echo "  make run-starter  - Build and run starter"
	@echo "  make run-solution - Build and run solution"
	@echo "  make run-test     - Build and run tests"
	@echo "  make clean        - Remove build artifacts"
