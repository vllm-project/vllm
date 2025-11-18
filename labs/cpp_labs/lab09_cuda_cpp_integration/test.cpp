#include <iostream>
#include <cassert>

// Mock CUDA functions for testing without GPU
namespace MockCUDA {
    void* allocate(size_t size) {
        return new char[size];
    }

    void free(void* ptr) {
        delete[] static_cast<char*>(ptr);
    }

    void copyHostToDevice(void* dst, const void* src, size_t size) {
        std::memcpy(dst, src, size);
    }

    void copyDeviceToHost(void* dst, const void* src, size_t size) {
        std::memcpy(dst, src, size);
    }

    void vectorAdd(float* a, float* b, float* c, int n) {
        for (int i = 0; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
    }
}

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { std::cout << "Running " #name "..."; test_##name(); std::cout << " PASSED" << std::endl; } while(0)

TEST(gpu_allocation) {
    void* ptr = MockCUDA::allocate(1024);
    assert(ptr != nullptr);
    MockCUDA::free(ptr);
}

TEST(vector_add_cpu) {
    const int N = 100;
    float a[N], b[N], c[N];

    for (int i = 0; i < N; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    MockCUDA::vectorAdd(a, b, c, N);

    for (int i = 0; i < N; ++i) {
        assert(c[i] == 3.0f);
    }
}

int main() {
    std::cout << "=== Running CUDA Integration Tests (CPU Mock) ===" << std::endl;
    std::cout << "Note: These are CPU-based tests. Run solution.cu for actual GPU tests." << std::endl;

    RUN_TEST(gpu_allocation);
    RUN_TEST(vector_add_cpu);

    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
