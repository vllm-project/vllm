#include <iostream>
#include <vector>
#include <chrono>
#include <random>

// Naive matrix multiplication (slow!)
void matmul_naive(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// TODO: Implement optimized version
// Hint: Transpose B for better cache locality
// void matmul_optimized(const float* A, const float* B, float* C, int N) {
//     // TODO: Implement cache-friendly version
// }

// Simple timer class
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_;

public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
};

int main() {
    std::cout << "=== Lab 10: Performance Profiling (Starter) ===" << std::endl;

    const int N = 512;
    std::vector<float> A(N * N), B(N * N), C(N * N);

    // Initialize with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < N * N; ++i) {
        A[i] = dis(gen);
        B[i] = dis(gen);
    }

    // Benchmark naive version
    {
        std::cout << "\nNaive matmul " << N << "x" << N << ":" << std::endl;
        Timer timer;
        matmul_naive(A.data(), B.data(), C.data(), N);
        std::cout << "Time: " << timer.elapsed_ms() << " ms" << std::endl;
    }

    // TODO: Benchmark optimized version
    // {
    //     std::cout << "\nOptimized matmul " << N << "x" << N << ":" << std::endl;
    //     Timer timer;
    //     matmul_optimized(A.data(), B.data(), C.data(), N);
    //     std::cout << "Time: " << timer.elapsed_ms() << " ms" << std::endl;
    // }

    return 0;
}
