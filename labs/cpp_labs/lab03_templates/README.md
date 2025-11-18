# Lab 03: Templates - Generic Programming

## Overview
Master C++ templates to write generic, reusable code. Templates are the foundation of ML libraries like PyTorch's C++ backend (ATen) and are essential for writing type-safe, performant code that works with different data types.

## Learning Objectives
1. Write function templates for generic algorithms
2. Create class templates for generic containers
3. Understand template specialization and SFINAE
4. Use variadic templates for flexible interfaces
5. Apply concepts (C++20) for better template constraints

## Estimated Time
1.5-2 hours

## Prerequisites
- Labs 01-02 completed
- Understanding of C++ classes and functions
- Basic knowledge of generic programming

## Problem Statement
Implement a generic `Tensor` class template that can work with different numeric types (float, double, int). Add template functions for common operations like dot product, element-wise operations, and reductions.

## Key Concepts

### Python vs C++ Generics
```python
# Python - duck typing, works with any type
def add(a, b):
    return a + b  # Works if a and b support +

# Type hints are optional
def add_typed(a: float, b: float) -> float:
    return a + b
```

```cpp
// C++ - templates provide compile-time polymorphism
template<typename T>
T add(T a, T b) {
    return a + b;  // Type-checked at compile time
}

// Can be called with any type that supports +
auto x = add(1, 2);        // T = int
auto y = add(1.5, 2.5);    // T = double
```

### Template Types
- **Function templates**: Generic functions
- **Class templates**: Generic classes (like std::vector<T>)
- **Variadic templates**: Variable number of template parameters
- **Template specialization**: Customize for specific types

## Build and Run Instructions

```bash
make run-starter
make run-solution
make run-test
make clean
```

## Common Mistakes

1. **Template code must be in headers**: Templates are instantiated at compile time
2. **Forgetting typename keyword**: Use `typename` for dependent types
3. **Template bloat**: Each instantiation creates new code
4. **Not constraining templates**: C++20 concepts help
5. **Circular dependencies**: Template headers can cause issues

## Stretch Goals

1. Implement SFINAE to enable/disable functions based on type traits
2. Add C++20 concepts for better error messages
3. Create a template meta-program to compute at compile time
4. Implement expression templates for lazy evaluation
5. Add template template parameters

## Connection to vLLM

Templates are used extensively in vLLM for:
- Generic tensor operations supporting multiple dtypes
- Kernel launchers that work with different types
- Type-safe wrappers around CUDA kernels
- Compile-time optimizations based on type properties
