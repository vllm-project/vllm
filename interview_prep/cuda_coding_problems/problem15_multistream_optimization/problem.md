# Problem 15: Multi-Stream Overlap & Optimization

**Difficulty:** Hard
**Estimated Time:** 60-75 minutes
**Tags:** Streams, Asynchronous Execution, Overlapping, Pipelining

## Problem Statement

Implement a multi-stream pipeline that overlaps data transfers (H2D, D2H) with kernel execution. Demonstrate understanding of asynchronous CUDA operations and how to maximize GPU utilization.

## Requirements

- Use multiple CUDA streams
- Overlap H2D copy, kernel execution, and D2H copy
- Process data in chunks using a pipeline
- Use pinned (page-locked) memory for faster transfers
- Demonstrate performance improvement vs. single stream
- Proper synchronization between streams

## Pipeline Stages

```
Stream 0: H2D[0] -> Kernel[0] -> D2H[0]
Stream 1:           H2D[1] -> Kernel[1] -> D2H[1]
Stream 2:                     H2D[2] -> Kernel[2] -> D2H[2]
```

All stages can run concurrently on different chunks!

## Function Signature

```cuda
void processWithStreams(float* h_input, float* h_output, int n,
                        int num_streams, int chunk_size);
```

## Example Use Case

Process large array by:
1. Divide into chunks
2. Each chunk: H2D → Kernel → D2H
3. Overlap operations across chunks
4. Synchronize at end

## Success Criteria

- ✅ Multiple streams correctly created
- ✅ Operations overlap (H2D, kernel, D2H)
- ✅ Uses pinned memory
- ✅ Proper synchronization
- ✅ Performance improvement vs. single stream
- ✅ No race conditions
