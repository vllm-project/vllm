#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUDA kernel for vector addition
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CUDA kernel for element-wise multiplication
__global__ void vectorMul(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// CUDA kernel for scalar multiplication
__global__ void scalarMul(const float* a, float scalar, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * scalar;
    }
}

// C++ wrapper class for GPU tensors
class GPUTensor {
private:
    float* d_data_;
    size_t size_;

public:
    explicit GPUTensor(size_t size) : size_(size) {
        CUDA_CHECK(cudaMalloc(&d_data_, size * sizeof(float)));
        std::cout << "Allocated GPU memory: " << size * sizeof(float) << " bytes" << std::endl;
    }

    ~GPUTensor() {
        CUDA_CHECK(cudaFree(d_data_));
        std::cout << "Freed GPU memory" << std::endl;
    }

    // Delete copy constructor and assignment
    GPUTensor(const GPUTensor&) = delete;
    GPUTensor& operator=(const GPUTensor&) = delete;

    // Enable move semantics
    GPUTensor(GPUTensor&& other) noexcept : d_data_(other.d_data_), size_(other.size_) {
        other.d_data_ = nullptr;
        other.size_ = 0;
    }

    GPUTensor& operator=(GPUTensor&& other) noexcept {
        if (this != &other) {
            if (d_data_) cudaFree(d_data_);
            d_data_ = other.d_data_;
            size_ = other.size_;
            other.d_data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void copyFromHost(const float* h_data) {
        CUDA_CHECK(cudaMemcpy(d_data_, h_data, size_ * sizeof(float), cudaMemcpyHostToDevice));
    }

    void copyToHost(float* h_data) const {
        CUDA_CHECK(cudaMemcpy(h_data, d_data_, size_ * sizeof(float), cudaMemcpyDeviceToHost));
    }

    float* data() { return d_data_; }
    const float* data() const { return d_data_; }
    size_t size() const { return size_; }

    // Launch vector addition kernel
    static void add(const GPUTensor& a, const GPUTensor& b, GPUTensor& c) {
        if (a.size() != b.size() || a.size() != c.size()) {
            throw std::runtime_error("Size mismatch");
        }

        int n = a.size();
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;

        vectorAdd<<<numBlocks, blockSize>>>(a.data(), b.data(), c.data(), n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Launch vector multiplication kernel
    static void mul(const GPUTensor& a, const GPUTensor& b, GPUTensor& c) {
        if (a.size() != b.size() || a.size() != c.size()) {
            throw std::runtime_error("Size mismatch");
        }

        int n = a.size();
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;

        vectorMul<<<numBlocks, blockSize>>>(a.data(), b.data(), c.data(), n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};

int main() {
    std::cout << "=== Lab 09: CUDA C++ Integration (Solution) ===" << std::endl;

    // Check CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found. This lab requires a GPU." << std::endl;
        return 1;
    }

    // Test 1: Vector addition on GPU
    {
        std::cout << "\nTest 1: Vector Addition on GPU" << std::endl;

        const size_t N = 1000000;
        std::vector<float> h_a(N, 1.0f);
        std::vector<float> h_b(N, 2.0f);
        std::vector<float> h_c(N);

        // Create GPU tensors
        GPUTensor d_a(N), d_b(N), d_c(N);

        // Copy to device
        d_a.copyFromHost(h_a.data());
        d_b.copyFromHost(h_b.data());

        // Launch kernel
        GPUTensor::add(d_a, d_b, d_c);

        // Copy result back
        d_c.copyToHost(h_c.data());

        // Verify
        bool correct = true;
        for (size_t i = 0; i < N; ++i) {
            if (h_c[i] != 3.0f) {
                correct = false;
                break;
            }
        }

        std::cout << "Result: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;
        std::cout << "Sample: " << h_c[0] << ", " << h_c[N/2] << ", " << h_c[N-1] << std::endl;
    }

    // Test 2: Vector multiplication
    {
        std::cout << "\nTest 2: Vector Multiplication on GPU" << std::endl;

        const size_t N = 1000;
        std::vector<float> h_a(N, 2.0f);
        std::vector<float> h_b(N, 3.0f);
        std::vector<float> h_c(N);

        GPUTensor d_a(N), d_b(N), d_c(N);

        d_a.copyFromHost(h_a.data());
        d_b.copyFromHost(h_b.data());

        GPUTensor::mul(d_a, d_b, d_c);

        d_c.copyToHost(h_c.data());

        std::cout << "Result: " << h_c[0] << " (expected 6.0)" << std::endl;
    }

    std::cout << "\n=== All tests completed! ===" << std::endl;
    return 0;
}
