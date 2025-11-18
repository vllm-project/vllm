#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { cudaError_t err = call; \
         if (err != cudaSuccess) { \
             fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
             exit(1); \
         } \
    } while(0)

/**
 * Dynamic shared memory transpose
 * Tile size determined at runtime
 */
__global__ void dynamicTransposeKernel(float* input, float* output,
                                       int rows, int cols, int tileSize) {
    // Dynamically allocated shared memory (padded to avoid bank conflicts)
    extern __shared__ float tile[];

    int x = blockIdx.x * tileSize + threadIdx.x;
    int y = blockIdx.y * tileSize + threadIdx.y;

    // Load to shared memory with padding
    // tile is 1D array, index as 2D: [row * (tileSize+1) + col]
    if (x < cols && y < rows) {
        tile[threadIdx.y * (tileSize + 1) + threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    // Transpose coordinates
    x = blockIdx.y * tileSize + threadIdx.x;
    y = blockIdx.x * tileSize + threadIdx.y;

    // Write transposed
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x * (tileSize + 1) + threadIdx.y];
    }
}

void dynamicTranspose(float* h_input, float* h_output, int rows, int cols,
                      int tileSize) {
    float *d_input, *d_output;
    size_t size = rows * cols * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    dim3 block(tileSize, tileSize);
    dim3 grid((cols + tileSize - 1) / tileSize,
              (rows + tileSize - 1) / tileSize);

    // Dynamic shared memory size: tile with padding
    size_t sharedMemSize = tileSize * (tileSize + 1) * sizeof(float);

    dynamicTransposeKernel<<<grid, block, sharedMemSize>>>(
        d_input, d_output, rows, cols, tileSize);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

int main() {
    int rows = 64, cols = 64;
    float *input = new float[rows * cols];
    float *output = new float[rows * cols];

    for (int i = 0; i < rows * cols; i++) input[i] = i;

    // Test with different tile sizes
    for (int tileSize : {16, 32}) {
        printf("Testing with tile size %d... ", tileSize);
        dynamicTranspose(input, output, rows, cols, tileSize);

        bool correct = true;
        for (int i = 0; i < rows && correct; i++) {
            for (int j = 0; j < cols && correct; j++) {
                if (output[j * rows + i] != input[i * cols + j]) {
                    correct = false;
                }
            }
        }
        printf("%s\n", correct ? "PASSED" : "FAILED");
    }

    delete[] input;
    delete[] output;
    return 0;
}
