#include <iostream>
#include <cassert>
#include <vector>
#include <chrono>

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { std::cout << "Running " #name "..."; test_##name(); std::cout << " PASSED" << std::endl; } while(0)

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_;
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};

void matmul_simple(const float* A, const float* B, float* C, int N) {
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

TEST(timer_basic) {
    Timer timer;
    // Do some work
    volatile int sum = 0;
    for (int i = 0; i < 1000000; ++i) {
        sum += i;
    }
    double elapsed = timer.elapsed_ms();
    assert(elapsed > 0);
    assert(elapsed < 1000);  // Should take less than 1 second
}

TEST(matmul_correctness) {
    const int N = 4;
    std::vector<float> A(N * N), B(N * N), C(N * N);

    // Identity matrix
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = (i == j) ? 1.0f : 0.0f;
            B[i * N + j] = static_cast<float>(i * N + j);
        }
    }

    matmul_simple(A.data(), B.data(), C.data(), N);

    // C should equal B
    for (int i = 0; i < N * N; ++i) {
        assert(std::abs(C[i] - B[i]) < 1e-5);
    }
}

TEST(performance_measurement) {
    const int N = 128;
    std::vector<float> A(N * N, 1.0f), B(N * N, 1.0f), C(N * N);

    Timer timer;
    matmul_simple(A.data(), B.data(), C.data(), N);
    double elapsed = timer.elapsed_ms();

    std::cout << " (" << elapsed << " ms) ";

    // Basic sanity check on result
    assert(std::abs(C[0] - static_cast<float>(N)) < 1e-3);
}

int main() {
    std::cout << "=== Running Performance Profiling Tests ===" << std::endl;
    RUN_TEST(timer_basic);
    RUN_TEST(matmul_correctness);
    RUN_TEST(performance_measurement);
    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
