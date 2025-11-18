# Makefile for Lab 05: Threading

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g -pthread
LDFLAGS = -pthread

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
