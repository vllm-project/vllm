# Lab 04: Async Operations - Futures, Promises, and Async

## Overview
Learn asynchronous programming in C++ using `std::async`, `std::future`, and `std::promise`. Essential for non-blocking operations in GPU/ML inference pipelines.

## Learning Objectives
1. Use `std::async` for asynchronous function execution
2. Understand `std::future` and `std::promise` for result passing
3. Handle exceptions in async contexts
4. Implement async patterns for ML inference
5. Compare async vs threading approaches

## Estimated Time
1.5 hours

## Prerequisites
- Basic understanding of concurrency
- C++17 compiler with thread support

## Problem Statement
Implement an async inference system that processes multiple requests concurrently using futures and promises.

## Key Concepts

### Python Async vs C++ Async
```python
# Python asyncio
import asyncio

async def process():
    result = await some_async_operation()
    return result
```

```cpp
// C++ async
#include <future>

auto future = std::async(std::launch::async, []() {
    return some_operation();
});
auto result = future.get();  // Blocks until ready
```

## Build Instructions
```bash
make run-solution
make run-test
```

## Connection to vLLM
- Async request handling in inference servers
- Non-blocking GPU operations
- Parallel batch processing
