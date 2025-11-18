#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define CUDA_CHECK(call) \
    do { cudaError_t err = call; \
         if (err != cudaSuccess) { \
             fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
             exit(1); \
         } \
    } while(0)

/**
 * Simple kernel for demonstration
 * In practice, this should be compute-intensive enough to benefit from overlap
 */
__global__ void processKernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Dummy computation
        float val = input[idx];
        for (int i = 0; i < 100; i++) {
            val = sqrtf(val * val + 1.0f);
        }
        output[idx] = val;
    }
}

/**
 * Single-stream baseline (no overlap)
 */
void processSingleStream(float* h_input, float* h_output, int n) {
    float *d_input, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));

    // All operations serialized
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float),
                          cudaMemcpyHostToDevice));

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    processKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, n);

    CUDA_CHECK(cudaMemcpy(h_output, d_output, n * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

/**
 * Multi-stream with overlapping (optimized)
 */
void processWithStreams(float* h_input, float* h_output, int n,
                        int num_streams) {
    cudaStream_t* streams = new cudaStream_t[num_streams];
    float *d_input, *d_output;

    // Create streams
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));

    // Use pinned memory for faster transfers
    float *h_input_pinned, *h_output_pinned;
    CUDA_CHECK(cudaMallocHost(&h_input_pinned, n * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_output_pinned, n * sizeof(float)));

    // Copy to pinned memory
    memcpy(h_input_pinned, h_input, n * sizeof(float));

    // Process in chunks with overlapping
    int chunk_size = (n + num_streams - 1) / num_streams;

    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        int size = min(chunk_size, n - offset);

        if (size <= 0) continue;

        // H2D async
        CUDA_CHECK(cudaMemcpyAsync(
            d_input + offset,
            h_input_pinned + offset,
            size * sizeof(float),
            cudaMemcpyHostToDevice,
            streams[i]
        ));

        // Kernel launch async
        int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        processKernel<<<numBlocks, BLOCK_SIZE, 0, streams[i]>>>(
            d_input + offset,
            d_output + offset,
            size
        );

        // D2H async
        CUDA_CHECK(cudaMemcpyAsync(
            h_output_pinned + offset,
            d_output + offset,
            size * sizeof(float),
            cudaMemcpyDeviceToHost,
            streams[i]
        ));
    }

    // Synchronize all streams
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    // Copy from pinned memory to output
    memcpy(h_output, h_output_pinned, n * sizeof(float));

    // Cleanup
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    CUDA_CHECK(cudaFreeHost(h_input_pinned));
    CUDA_CHECK(cudaFreeHost(h_output_pinned));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    delete[] streams;
}

/**
 * Benchmark comparison
 */
void benchmark() {
    int n = 10000000;
    float *h_input = new float[n];
    float *h_output = new float[n];

    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Single stream
    cudaEventRecord(start);
    processSingleStream(h_input, h_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_single;
    cudaEventElapsedTime(&ms_single, start, stop);

    // Multi-stream
    int num_streams = 4;
    cudaEventRecord(start);
    processWithStreams(h_input, h_output, n, num_streams);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_multi;
    cudaEventElapsedTime(&ms_multi, start, stop);

    printf("Single stream: %.2f ms\n", ms_single);
    printf("Multi-stream (%d): %.2f ms\n", num_streams, ms_multi);
    printf("Speedup: %.2fx\n", ms_single / ms_multi);

    delete[] h_input;
    delete[] h_output;
}

int main() {
    benchmark();
    return 0;
}
