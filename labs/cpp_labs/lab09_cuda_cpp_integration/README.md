# Lab 09: CUDA C++ Integration - Calling CUDA from C++

## Overview
Learn to integrate CUDA kernels with C++ code. Understand memory transfer between host and device, kernel launching, and error handling.

## Learning Objectives
1. Write simple CUDA kernels
2. Manage host-device memory transfers
3. Launch kernels from C++ code
4. Handle CUDA errors properly
5. Integrate CUDA with C++ classes

## Estimated Time
2-3 hours

## Prerequisites
- All previous labs
- CUDA Toolkit installed (optional for compilation)
- Basic understanding of GPU programming

## Problem Statement
Implement a C++ wrapper class that manages GPU tensors and launches CUDA kernels for element-wise operations.

## Key Concepts

### Host vs Device Memory
```cpp
// CPU (host) memory
float* h_data = new float[size];

// GPU (device) memory
float* d_data;
cudaMalloc(&d_data, size * sizeof(float));

// Copy host to device
cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

// Launch kernel
kernel<<<blocks, threads>>>(d_data, size);

// Copy device to host
cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);

// Free GPU memory
cudaFree(d_data);
```

## Build Instructions
```bash
# Requires CUDA toolkit
make solution  # Uses nvcc compiler
make run-solution
```

## Connection to vLLM
- Understanding vLLM's CUDA kernels
- Custom operator development
- GPU memory management in production
- Kernel optimization for inference
