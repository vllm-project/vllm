#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include <algorithm>

// Timer utility
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

// Naive matrix multiplication (poor cache locality)
void matmul_naive(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];  // B access is not cache-friendly!
            }
            C[i * N + j] = sum;
        }
    }
}

// Optimized version 1: Transpose B for better cache locality
void matmul_transposed(const float* A, const float* B, float* C, int N) {
    // Transpose B
    std::vector<float> BT(N * N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            BT[j * N + i] = B[i * N + j];
        }
    }

    // Now multiply with better cache locality
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * BT[j * N + k];  // Sequential access!
            }
            C[i * N + j] = sum;
        }
    }
}

// Optimized version 2: Blocked (tiled) multiplication for cache
void matmul_blocked(const float* A, const float* B, float* C, int N) {
    const int BLOCK_SIZE = 64;

    // Initialize C to zero
    std::fill(C, C + N * N, 0.0f);

    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                // Multiply block
                for (int i = ii; i < std::min(ii + BLOCK_SIZE, N); ++i) {
                    for (int j = jj; j < std::min(jj + BLOCK_SIZE, N); ++j) {
                        float sum = C[i * N + j];
                        for (int k = kk; k < std::min(kk + BLOCK_SIZE, N); ++k) {
                            sum += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

// Benchmark function
template<typename Func>
double benchmark(Func func, const std::vector<float>& A,
                const std::vector<float>& B, std::vector<float>& C, int N,
                const std::string& name) {
    std::cout << "\n" << name << " (" << N << "x" << N << "):" << std::endl;

    // Warmup
    func(A.data(), B.data(), C.data(), N);

    // Timed run
    Timer timer;
    func(A.data(), B.data(), C.data(), N);
    double elapsed = timer.elapsed_ms();

    // Calculate FLOPS (2*N^3 operations for matmul)
    double flops = 2.0 * N * N * N;
    double gflops = (flops / elapsed) / 1e6;

    std::cout << "  Time: " << elapsed << " ms" << std::endl;
    std::cout << "  GFLOPS: " << gflops << std::endl;

    return elapsed;
}

// Simple checksum for verification
float checksum(const float* data, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += data[i];
    }
    return sum;
}

int main() {
    std::cout << "=== Lab 10: Performance Profiling (Solution) ===" << std::endl;

    const int N = 512;
    std::vector<float> A(N * N), B(N * N);
    std::vector<float> C_naive(N * N), C_trans(N * N), C_blocked(N * N);

    // Initialize with random values
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < N * N; ++i) {
        A[i] = dis(gen);
        B[i] = dis(gen);
    }

    std::cout << "\nMatrix size: " << N << "x" << N << std::endl;
    std::cout << "Memory per matrix: " << (N * N * sizeof(float)) / 1024.0 / 1024.0 << " MB" << std::endl;

    // Benchmark naive version
    double time_naive = benchmark(matmul_naive, A, B, C_naive, N, "Naive MatMul");

    // Benchmark transposed version
    double time_trans = benchmark(matmul_transposed, A, B, C_trans, N, "Transposed MatMul");
    std::cout << "  Speedup vs naive: " << time_naive / time_trans << "x" << std::endl;

    // Benchmark blocked version
    double time_blocked = benchmark(matmul_blocked, A, B, C_blocked, N, "Blocked MatMul");
    std::cout << "  Speedup vs naive: " << time_naive / time_blocked << "x" << std::endl;

    // Verify correctness
    std::cout << "\nVerification (checksums):" << std::endl;
    float sum_naive = checksum(C_naive.data(), N * N);
    float sum_trans = checksum(C_trans.data(), N * N);
    float sum_blocked = checksum(C_blocked.data(), N * N);

    std::cout << "  Naive: " << sum_naive << std::endl;
    std::cout << "  Transposed: " << sum_trans << std::endl;
    std::cout << "  Blocked: " << sum_blocked << std::endl;

    float tolerance = 1.0f;  // Allow some floating-point error
    bool correct = (std::abs(sum_naive - sum_trans) < tolerance) &&
                   (std::abs(sum_naive - sum_blocked) < tolerance);
    std::cout << "  Results match: " << (correct ? "YES" : "NO") << std::endl;

    // Profiling tips
    std::cout << "\n=== Profiling Tips ===" << std::endl;
    std::cout << "1. Profile with perf:" << std::endl;
    std::cout << "   perf record ./solution" << std::endl;
    std::cout << "   perf report" << std::endl;
    std::cout << "\n2. Profile with gprof:" << std::endl;
    std::cout << "   g++ -pg solution.cpp -o solution" << std::endl;
    std::cout << "   ./solution" << std::endl;
    std::cout << "   gprof solution gmon.out > analysis.txt" << std::endl;
    std::cout << "\n3. Cache analysis with valgrind:" << std::endl;
    std::cout << "   valgrind --tool=cachegrind ./solution" << std::endl;

    std::cout << "\n=== All tests completed! ===" << std::endl;
    return 0;
}
