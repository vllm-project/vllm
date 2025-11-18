#include <cuda_runtime.h>
#include <stdio.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define CUDA_CHECK(call) \
    do { cudaError_t err = call; \
         if (err != cudaSuccess) { \
             fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
             exit(1); \
         } \
    } while(0)

/**
 * Warp-level reduction using shuffle instructions
 * Much faster than shared memory for small reductions
 */
__device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float warpReduceMin(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

/**
 * Block-level reduction using warp primitives
 * Combines multiple warp results
 */
__device__ float blockReduceSum(float val) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    // Reduce within warp
    val = warpReduceSum(val);

    // Write warp result to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // First warp reduces the warp results
    if (wid == 0) {
        val = (threadIdx.x < (BLOCK_SIZE / WARP_SIZE)) ? shared[lane] : 0.0f;
        val = warpReduceSum(val);
    }

    return val;
}

/**
 * Array reduction kernel using warp primitives
 * op: 0=sum, 1=max, 2=min
 */
__global__ void warpReduceKernel(float* input, float* output, int n, int op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float val = (op == 1) ? -INFINITY : ((op == 2) ? INFINITY : 0.0f);

    // Grid-stride loop
    for (int i = idx; i < n; i += stride) {
        if (op == 0) val += input[i];
        else if (op == 1) val = fmaxf(val, input[i]);
        else if (op == 2) val = fminf(val, input[i]);
    }

    // Block-level reduction
    if (op == 0) {
        val = blockReduceSum(val);
    } else {
        // Similar block reduce for max/min
        __shared__ float shared[BLOCK_SIZE / WARP_SIZE];
        int lane = threadIdx.x % WARP_SIZE;
        int wid = threadIdx.x / WARP_SIZE;

        val = (op == 1) ? warpReduceMax(val) : warpReduceMin(val);

        if (lane == 0) shared[wid] = val;
        __syncthreads();

        if (wid == 0) {
            val = (threadIdx.x < (BLOCK_SIZE / WARP_SIZE)) ? shared[lane] :
                  ((op == 1) ? -INFINITY : INFINITY);
            val = (op == 1) ? warpReduceMax(val) : warpReduceMin(val);
        }
    }

    // Write block result
    if (threadIdx.x == 0) {
        if (op == 0) atomicAdd(&output[0], val);
        else if (op == 1) atomicMax((int*)&output[0], __float_as_int(val));
        else atomicMin((int*)&output[0], __float_as_int(val));
    }
}

void warpReduce(float* h_input, float* h_output, int n, int op) {
    float *d_input, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    numBlocks = min(numBlocks, 512);

    warpReduceKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, n, op);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

int main() {
    int n = 1000;
    float *input = new float[n];
    for (int i = 0; i < n; i++) input[i] = 1.0f;

    float result;
    warpReduce(input, &result, n, 0);  // Sum
    printf("Sum: %.0f (expected 1000)\n", result);

    delete[] input;
    return 0;
}
