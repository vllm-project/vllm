# Lab 10: Performance Profiling - Measuring and Optimizing C++ Code

## Overview
Learn to profile C++ code using built-in timers, perf, gprof, and other tools. Understand bottlenecks and optimize hot paths.

## Learning Objectives
1. Use `std::chrono` for micro-benchmarking
2. Profile with perf and gprof
3. Identify performance bottlenecks
4. Apply compiler optimizations
5. Measure cache effects and memory bandwidth

## Estimated Time
2 hours

## Prerequisites
- All previous labs
- Linux system with perf installed (optional)

## Problem Statement
Profile and optimize a matrix multiplication implementation. Identify bottlenecks and apply optimizations.

## Key Concepts

### Profiling Tools
- **std::chrono**: High-resolution timing in C++
- **perf**: Linux performance analysis tool
- **gprof**: GNU profiler
- **valgrind --tool=callgrind**: Call graph profiler

### Common Optimizations
- Loop unrolling
- Cache-friendly access patterns
- Vectorization (SIMD)
- Compiler flags (-O2, -O3, -march=native)

## Build Instructions
```bash
make run-solution      # Run benchmarks
make profile-perf      # Profile with perf (Linux)
make profile-gprof     # Profile with gprof
```

## Connection to vLLM
- Profiling inference latency
- Identifying kernel bottlenecks
- Optimizing hot paths in attention
- Memory bandwidth analysis for KV cache
