#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Include the solution (in practice, link against compiled solution)
extern float parallelSum(float* h_input, int n);

#define EPSILON 1e-3
#define ASSERT_FLOAT_EQ(expected, actual, msg) \
    do { \
        if (fabs((expected) - (actual)) > EPSILON) { \
            fprintf(stderr, "FAILED: %s\n", msg); \
            fprintf(stderr, "  Expected: %f, Got: %f, Diff: %f\n", \
                    (float)(expected), (float)(actual), \
                    fabs((expected) - (actual))); \
            return 0; \
        } \
    } while(0)

// Test 1: Small array
int test_small_array() {
    printf("Test 1: Small array... ");
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    int n = 5;
    float result = parallelSum(input, n);
    ASSERT_FLOAT_EQ(15.0f, result, "Small array sum");
    printf("PASSED\n");
    return 1;
}

// Test 2: Single element
int test_single_element() {
    printf("Test 2: Single element... ");
    float input[] = {42.0f};
    int n = 1;
    float result = parallelSum(input, n);
    ASSERT_FLOAT_EQ(42.0f, result, "Single element");
    printf("PASSED\n");
    return 1;
}

// Test 3: Empty array
int test_empty_array() {
    printf("Test 3: Empty array... ");
    float* input = nullptr;
    int n = 0;
    float result = parallelSum(input, n);
    ASSERT_FLOAT_EQ(0.0f, result, "Empty array");
    printf("PASSED\n");
    return 1;
}

// Test 4: Negative numbers
int test_negative_numbers() {
    printf("Test 4: Negative numbers... ");
    float input[] = {1.5f, -2.3f, 4.7f, 0.0f, -1.5f, 3.6f};
    int n = 6;
    float expected = 1.5f - 2.3f + 4.7f + 0.0f - 1.5f + 3.6f;
    float result = parallelSum(input, n);
    ASSERT_FLOAT_EQ(expected, result, "Negative numbers");
    printf("PASSED\n");
    return 1;
}

// Test 5: Power of 2 size
int test_power_of_2() {
    printf("Test 5: Power of 2 size (1024)... ");
    int n = 1024;
    float* input = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        input[i] = 1.0f;
    }
    float result = parallelSum(input, n);
    ASSERT_FLOAT_EQ(1024.0f, result, "Power of 2 array");
    free(input);
    printf("PASSED\n");
    return 1;
}

// Test 6: Non-power of 2 size
int test_non_power_of_2() {
    printf("Test 6: Non-power of 2 size (1000)... ");
    int n = 1000;
    float* input = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        input[i] = 1.0f;
    }
    float result = parallelSum(input, n);
    ASSERT_FLOAT_EQ(1000.0f, result, "Non-power of 2 array");
    free(input);
    printf("PASSED\n");
    return 1;
}

// Test 7: Large array
int test_large_array() {
    printf("Test 7: Large array (1M elements)... ");
    int n = 1000000;
    float* input = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        input[i] = 1.0f;
    }
    float result = parallelSum(input, n);
    // Allow larger epsilon for floating point accumulation errors
    if (fabs(result - 1000000.0f) > 1.0f) {
        fprintf(stderr, "FAILED: Large array sum\n");
        fprintf(stderr, "  Expected: %f, Got: %f\n", 1000000.0f, result);
        free(input);
        return 0;
    }
    free(input);
    printf("PASSED\n");
    return 1;
}

// Test 8: All zeros
int test_all_zeros() {
    printf("Test 8: All zeros... ");
    int n = 1000;
    float* input = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        input[i] = 0.0f;
    }
    float result = parallelSum(input, n);
    ASSERT_FLOAT_EQ(0.0f, result, "All zeros");
    free(input);
    printf("PASSED\n");
    return 1;
}

// Test 9: Sequential numbers
int test_sequential_numbers() {
    printf("Test 9: Sequential numbers (1 to 100)... ");
    int n = 100;
    float* input = (float*)malloc(n * sizeof(float));
    float expected = 0.0f;
    for (int i = 0; i < n; i++) {
        input[i] = (float)(i + 1);
        expected += input[i];
    }
    float result = parallelSum(input, n);
    ASSERT_FLOAT_EQ(expected, result, "Sequential numbers");
    free(input);
    printf("PASSED\n");
    return 1;
}

// Test 10: Very large array (10M elements)
int test_very_large_array() {
    printf("Test 10: Very large array (10M elements)... ");
    int n = 10000000;
    float* input = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        input[i] = 0.5f;
    }
    float result = parallelSum(input, n);
    float expected = n * 0.5f;
    // Allow larger epsilon for very large accumulations
    if (fabs(result - expected) > 10.0f) {
        fprintf(stderr, "FAILED: Very large array sum\n");
        fprintf(stderr, "  Expected: %f, Got: %f\n", expected, result);
        free(input);
        return 0;
    }
    free(input);
    printf("PASSED\n");
    return 1;
}

// Performance test
void performance_test() {
    printf("\n=== Performance Test ===\n");
    int sizes[] = {1000, 10000, 100000, 1000000, 10000000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        int n = sizes[i];
        float* input = (float*)malloc(n * sizeof(float));
        for (int j = 0; j < n; j++) {
            input[j] = 1.0f;
        }

        // Warmup
        parallelSum(input, n);

        // Timing
        clock_t start = clock();
        int iterations = (n < 100000) ? 100 : 10;
        for (int j = 0; j < iterations; j++) {
            parallelSum(input, n);
        }
        clock_t end = clock();

        double time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0 / iterations;
        double bandwidth = (n * sizeof(float)) / (time_ms / 1000.0) / 1e9; // GB/s

        printf("N=%d: %.3f ms, %.2f GB/s\n", n, time_ms, bandwidth);
        free(input);
    }
}

int main() {
    printf("=== CUDA Parallel Reduction Tests ===\n\n");

    int passed = 0;
    int total = 10;

    passed += test_small_array();
    passed += test_single_element();
    passed += test_empty_array();
    passed += test_negative_numbers();
    passed += test_power_of_2();
    passed += test_non_power_of_2();
    passed += test_large_array();
    passed += test_all_zeros();
    passed += test_sequential_numbers();
    passed += test_very_large_array();

    printf("\n=== Results ===\n");
    printf("Passed: %d/%d\n", passed, total);

    if (passed == total) {
        printf("All tests passed!\n");
        performance_test();
        return 0;
    } else {
        printf("Some tests failed.\n");
        return 1;
    }
}
