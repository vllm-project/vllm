# Lab 01: Memory Management - Smart Pointers, RAII, and Memory Leaks

## Overview
This lab introduces modern C++ memory management techniques crucial for GPU/ML applications. You'll learn how to manage memory safely and efficiently using smart pointers and RAII (Resource Acquisition Is Initialization) principles.

## Learning Objectives
1. Understand the difference between raw pointers and smart pointers
2. Implement RAII pattern for automatic resource management
3. Identify and fix memory leaks using modern C++ techniques
4. Use `std::unique_ptr`, `std::shared_ptr`, and `std::weak_ptr` appropriately
5. Apply these concepts to managing GPU-like resources

## Estimated Time
1-2 hours

## Prerequisites
- Basic understanding of pointers in C/C++
- Familiarity with Python's memory management (for comparison)
- C++17 compiler installed (gcc 7+, clang 5+, or MSVC 2017+)

## Problem Statement
You'll implement a `ResourceManager` class that manages simulated GPU memory buffers. The starter code has memory leaks and unsafe raw pointer usage. Your job is to:

1. Replace raw pointers with appropriate smart pointers
2. Implement RAII for automatic cleanup
3. Handle shared ownership scenarios
4. Prevent dangling pointer issues

## Key Concepts

### Python vs C++ Memory Management
```python
# Python - automatic garbage collection
data = [1, 2, 3]  # Allocated
data = None       # Eventually cleaned up by GC
```

```cpp
// C++ - manual memory management (old way)
int* data = new int[3];  // Allocated
delete[] data;           // MUST manually free

// C++ - smart pointers (modern way)
auto data = std::make_unique<int[]>(3);  // Allocated
// Automatically freed when out of scope!
```

### Smart Pointer Types
- **`std::unique_ptr`**: Exclusive ownership, non-copyable
- **`std::shared_ptr`**: Shared ownership with reference counting
- **`std::weak_ptr`**: Non-owning reference to `shared_ptr`

## Build and Run Instructions

```bash
# Build the starter code
make starter

# Run the starter (will have memory leaks!)
./starter

# Build the solution
make solution

# Run the solution (no leaks!)
./solution

# Build and run tests
make test
./test

# Clean build artifacts
make clean
```

## Expected Output

### Starter (with leaks)
```
Creating GPU buffer of size 1024
Creating GPU buffer of size 2048
Buffer data: [some values]
// Memory leaks on exit!
```

### Solution (no leaks)
```
Creating GPU buffer of size 1024
Creating GPU buffer of size 2048
Buffer data: [some values]
Destroying GPU buffer of size 2048
Destroying GPU buffer of size 1024
All tests passed!
```

## Common Mistakes

1. **Mixing raw and smart pointers**: Don't call `delete` on memory managed by smart pointers
2. **Circular references with `shared_ptr`**: Use `weak_ptr` to break cycles
3. **Using `shared_ptr` when `unique_ptr` suffices**: Prefer unique ownership for better performance
4. **Forgetting to use `make_unique`/`make_shared`**: Direct `new` defeats the purpose
5. **Storing `shared_ptr` in containers when not needed**: Can lead to unexpected lifetime extension

## Stretch Goals

1. Implement a custom deleter for `unique_ptr` that logs cleanup
2. Create a `ResourcePool` that reuses buffers using `shared_ptr`
3. Add move semantics to transfer buffer ownership efficiently
4. Implement a reference-counting mechanism from scratch to understand `shared_ptr`
5. Profile memory usage with valgrind or AddressSanitizer

## Additional Resources

- [C++ Core Guidelines on Memory](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#r-resource-management)
- [Smart Pointers in Modern C++](https://docs.microsoft.com/en-us/cpp/cpp/smart-pointers-modern-cpp)
- [RAII Explained](https://en.cppreference.com/w/cpp/language/raii)

## Connection to vLLM

In vLLM, memory management is critical for:
- Managing GPU tensors and buffers
- Handling KV cache allocation/deallocation
- Resource pooling for multiple inference requests
- Preventing memory leaks in long-running services

Understanding these patterns will help you read and contribute to vLLM's C++/CUDA codebase.
