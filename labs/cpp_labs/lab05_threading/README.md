# Lab 05: Threading - Mutexes, Condition Variables, and Thread Safety

## Overview
Master C++ threading primitives: `std::thread`, `std::mutex`, `std::condition_variable`, and synchronization. Critical for parallel GPU operations and batch processing.

## Learning Objectives
1. Create and manage threads with `std::thread`
2. Protect shared data with mutexes and locks
3. Use condition variables for thread coordination
4. Avoid race conditions and deadlocks
5. Implement thread-safe data structures

## Estimated Time
2 hours

## Prerequisites
- Basic concurrency concepts
- Understanding of race conditions

## Problem Statement
Implement a thread-safe queue for batching inference requests, with proper synchronization.

## Connection to vLLM
- Request batching across multiple threads
- Thread-safe KV cache management
- Parallel tokenization and preprocessing
