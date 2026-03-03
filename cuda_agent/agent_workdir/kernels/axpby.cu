/**
 * axpby.cu
 *
 * CUDA kernel for the axpby operation: out[i] = alpha * a[i] + b[i]
 *
 * Reference implementation supplied with the agent_workdir as a
 * worked example demonstrating the expected kernel + binding pattern.
 */

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <typename T>
__global__ void axpby_kernel(const T* __restrict__ a,
                              const T* __restrict__ b,
                              T*       __restrict__ out,
                              T alpha,
                              int n) {
    // Grid-stride loop — handles arrays larger than the launch grid.
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += gridDim.x * blockDim.x) {
        out[idx] = alpha * a[idx] + b[idx];
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

/**
 * @param a       Input tensor A (device pointer, float32)
 * @param b       Input tensor B (device pointer, float32)
 * @param out     Output tensor (device pointer, float32, pre-allocated)
 * @param alpha   Scalar multiplier
 * @param n       Total number of elements
 * @param config  Thread-count selector: 1→128, 2→512, else→256
 * @param stream  CUDA stream for asynchronous execution
 */
void axpby_launcher(const float* a,
                    const float* b,
                    float*       out,
                    float        alpha,
                    int          n,
                    int          config,
                    cudaStream_t stream) {
    if (n <= 0) return;

    int threads;
    if      (config == 1) threads = 128;
    else if (config == 2) threads = 512;
    else                  threads = 256;

    int blocks = (n + threads - 1) / threads;
    axpby_kernel<float><<<blocks, threads, 0, stream>>>(a, b, out, alpha, n);
}
