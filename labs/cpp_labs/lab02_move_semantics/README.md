# Lab 02: Move Semantics - Efficient Resource Transfer

## Overview
Learn how to use C++ move semantics to transfer ownership of resources efficiently without copying. This is crucial for high-performance GPU/ML applications where copying large tensors would be prohibitively expensive.

## Learning Objectives
1. Understand the difference between lvalues and rvalues
2. Implement move constructors and move assignment operators
3. Use `std::move` to transfer ownership explicitly
4. Recognize when the compiler uses move semantics automatically
5. Apply move semantics to large data structures (tensors, buffers)

## Estimated Time
1.5-2 hours

## Prerequisites
- Lab 01: Memory Management completed
- Understanding of copy constructors and assignment operators
- C++17 compiler with move semantics support

## Problem Statement
You'll implement a `Matrix` class that represents a large 2D array (similar to PyTorch tensors). The starter code uses expensive deep copies. Your task is to:

1. Implement move constructor for efficient transfer
2. Implement move assignment operator
3. Understand when moves happen automatically (RVO, NRVO)
4. Use perfect forwarding for generic code
5. Benchmark copy vs move performance

## Key Concepts

### Python vs C++ Object Transfer
```python
# Python - always uses references
a = [1, 2, 3, 4, 5]  # Create list
b = a                 # b references same list (no copy!)
c = a.copy()         # Explicit copy if needed
```

```cpp
// C++ - explicit control over copying vs moving
std::vector<int> a = {1, 2, 3, 4, 5};
auto b = a;                    // COPY (deep copy)
auto c = std::move(a);         // MOVE (transfer ownership)
// a is now in valid but unspecified state
```

### Lvalues vs Rvalues
- **lvalue**: Has a name, has an address (e.g., variables)
- **rvalue**: Temporary, about to be destroyed (e.g., function return values)
- **rvalue reference**: `T&&` - can bind to rvalues

### The Rule of Five
If you implement any of these, implement all five:
1. Destructor
2. Copy constructor
3. Copy assignment operator
4. Move constructor
5. Move assignment operator

## Build and Run Instructions

```bash
# Build and run starter
make run-starter

# Build and run solution
make run-solution

# Build and run tests
make run-test

# Run performance benchmark
make benchmark

# Clean
make clean
```

## Expected Output

### Starter (many copies)
```
Creating Matrix 1000x1000 (1000000 elements)
Copying Matrix 1000x1000 (expensive!)
Copying Matrix 1000x1000 (expensive!)
Time for copy: 15.2 ms
```

### Solution (with moves)
```
Creating Matrix 1000x1000 (1000000 elements)
Moving Matrix 1000x1000 (cheap!)
Moving Matrix 1000x1000 (cheap!)
Time for move: 0.1 ms
Speedup: 152x faster!
```

## Common Mistakes

1. **Forgetting to clear source after move**: Move assignment must leave source in valid state
2. **Using object after move**: Moved-from objects are valid but unspecified
3. **Moving when copy is intended**: Be careful with `std::move`
4. **Not implementing move for large objects**: Missing huge optimization opportunity
5. **Returning `std::move(local)`**: Defeats RVO, just return the local variable
6. **Self-move assignment**: Must handle `m = std::move(m)` correctly

## Stretch Goals

1. Implement perfect forwarding in a factory function
2. Add benchmark comparing copy vs move for different sizes
3. Implement move-only type (deleted copy operations)
4. Create a move-aware container class
5. Profile with `perf` to see the difference
6. Implement universal references and forwarding references

## Additional Resources

- [C++ Move Semantics Guide](https://www.cprogramming.com/c++11/rvalue-references-and-move-semantics-in-c++11.html)
- [Scott Meyers: Universal References](https://isocpp.org/blog/2012/11/universal-references-in-c11-scott-meyers)
- [CppCon: Move Semantics](https://www.youtube.com/watch?v=St0MNEU5b0o)

## Connection to vLLM

Move semantics are essential in vLLM for:
- Transferring large tensors without copying (KV cache, activations)
- Efficient batch management and request handling
- Zero-copy tensor views and slicing
- Returning large objects from functions efficiently
- Resource management in CUDA streams

Understanding move semantics will help you write efficient C++/CUDA code that doesn't waste time copying gigabytes of GPU memory.
