# Hints: Multi-Stream Optimization

## Hint 1: Why Streams?
Overlap independent operations:
- Stream 0: H2D[0] | Kernel[0] | D2H[0]
- Stream 1:        | H2D[1]    | Kernel[1] | D2H[1]
- Concurrent execution!

## Hint 2: Stream Creation
```cuda
cudaStream_t stream;
cudaStreamCreate(&stream);
// Use stream...
cudaStreamDestroy(&stream);
```

## Hint 3: Async Operations
```cuda
cudaMemcpyAsync(..., cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(...);
cudaMemcpyAsync(..., cudaMemcpyDeviceToHost, stream);
```

## Hint 4: Pinned Memory
```cuda
cudaMallocHost(&pinned_ptr, size);  // Faster transfers
cudaFreeHost(pinned_ptr);
```

## Hint 5: Synchronization
```cuda
cudaStreamSynchronize(stream);  // Wait for stream
cudaDeviceSynchronize();        // Wait for all streams
```

## Hint 6: Common Mistake
Don't use regular malloc for async transfers - use pinned memory!
