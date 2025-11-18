#include <iostream>
#include <vector>

// This is a simulation - actual CUDA code would use .cu files
// TODO: Implement CUDA memory management wrapper

class GPUTensor {
private:
    float* d_data_;  // Device pointer
    size_t size_;

public:
    GPUTensor(size_t size) : size_(size) {
        // TODO: Allocate GPU memory with cudaMalloc
        // cudaMalloc(&d_data_, size * sizeof(float));
        std::cout << "TODO: Allocate GPU memory" << std::endl;
    }

    ~GPUTensor() {
        // TODO: Free GPU memory with cudaFree
        // cudaFree(d_data_);
        std::cout << "TODO: Free GPU memory" << std::endl;
    }

    void copyFromHost(const float* h_data) {
        // TODO: Copy data from CPU to GPU
        // cudaMemcpy(d_data_, h_data, size_ * sizeof(float), cudaMemcpyHostToDevice);
        std::cout << "TODO: Copy from host to device" << std::endl;
    }

    void copyToHost(float* h_data) {
        // TODO: Copy data from GPU to CPU
        // cudaMemcpy(h_data, d_data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "TODO: Copy from device to host" << std::endl;
    }

    // TODO: Add method to launch CUDA kernel
};

// TODO: Define CUDA kernel (in .cu file)
// __global__ void vectorAdd(float* a, float* b, float* c, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n) {
//         c[idx] = a[idx] + b[idx];
//     }
// }

int main() {
    std::cout << "=== Lab 09: CUDA C++ Integration (Starter) ===" << std::endl;
    std::cout << "Note: This is a simulation. See solution.cu for actual CUDA code." << std::endl;

    const size_t N = 1000;
    std::vector<float> h_a(N, 1.0f);
    std::vector<float> h_b(N, 2.0f);
    std::vector<float> h_c(N);

    std::cout << "\nTODO: Implement CUDA memory management and kernel launch" << std::endl;
    std::cout << "1. Allocate device memory" << std::endl;
    std::cout << "2. Copy host to device" << std::endl;
    std::cout << "3. Launch kernel" << std::endl;
    std::cout << "4. Copy device to host" << std::endl;
    std::cout << "5. Free device memory" << std::endl;

    return 0;
}
