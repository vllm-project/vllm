#include <cuda_runtime.h>
<stdio.h>
#include <stdlib.h>
#include <math.h>

extern void matrixTranspose(float* h_input, float* h_output, int rows, int cols);

#define EPSILON 1e-5

int test_small_matrix() {
    printf("Test 1: Small 3x4 matrix... ");
    float input[] = {1,2,3,4, 5,6,7,8, 9,10,11,12};
    float expected[] = {1,5,9, 2,6,10, 3,7,11, 4,8,12};
    float output[12];

    matrixTranspose(input, output, 3, 4);

    for (int i = 0; i < 12; i++) {
        if (fabs(output[i] - expected[i]) > EPSILON) {
            printf("FAILED at index %d: expected %f, got %f\n", i, expected[i], output[i]);
            return 0;
        }
    }
    printf("PASSED\n");
    return 1;
}

int test_square_matrix() {
    printf("Test 2: Square 2x2 matrix... ");
    float input[] = {1,2, 3,4};
    float expected[] = {1,3, 2,4};
    float output[4];

    matrixTranspose(input, output, 2, 2);

    for (int i = 0; i < 4; i++) {
        if (fabs(output[i] - expected[i]) > EPSILON) {
            printf("FAILED\n");
            return 0;
        }
    }
    printf("PASSED\n");
    return 1;
}

int test_single_row() {
    printf("Test 3: Single row 1x5... ");
    float input[] = {1,2,3,4,5};
    float expected[] = {1,2,3,4,5};
    float output[5];

    matrixTranspose(input, output, 1, 5);

    for (int i = 0; i < 5; i++) {
        if (fabs(output[i] - expected[i]) > EPSILON) {
            printf("FAILED\n");
            return 0;
        }
    }
    printf("PASSED\n");
    return 1;
}

int test_single_column() {
    printf("Test 4: Single column 5x1... ");
    float input[] = {1,2,3,4,5};
    float expected[] = {1,2,3,4,5};
    float output[5];

    matrixTranspose(input, output, 5, 1);

    for (int i = 0; i < 5; i++) {
        if (fabs(output[i] - expected[i]) > EPSILON) {
            printf("FAILED\n");
            return 0;
        }
    }
    printf("PASSED\n");
    return 1;
}

int test_tile_aligned() {
    printf("Test 5: Tile-aligned 32x32 matrix... ");
    int n = 32;
    float *input = (float*)malloc(n * n * sizeof(float));
    float *output = (float*)malloc(n * n * sizeof(float));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            input[i * n + j] = i * n + j;
        }
    }

    matrixTranspose(input, output, n, n);

    bool passed = true;
    for (int i = 0; i < n && passed; i++) {
        for (int j = 0; j < n && passed; j++) {
            float expected = input[i * n + j];
            float actual = output[j * n + i];
            if (fabs(actual - expected) > EPSILON) {
                passed = false;
            }
        }
    }

    free(input);
    free(output);
    printf(passed ? "PASSED\n" : "FAILED\n");
    return passed ? 1 : 0;
}

int test_non_aligned() {
    printf("Test 6: Non-aligned 100x200 matrix... ");
    int rows = 100, cols = 200;
    float *input = (float*)malloc(rows * cols * sizeof(float));
    float *output = (float*)malloc(rows * cols * sizeof(float));

    for (int i = 0; i < rows * cols; i++) {
        input[i] = (float)i;
    }

    matrixTranspose(input, output, rows, cols);

    bool passed = true;
    for (int i = 0; i < rows && passed; i++) {
        for (int j = 0; j < cols && passed; j++) {
            float expected = input[i * cols + j];
            float actual = output[j * rows + i];
            if (fabs(actual - expected) > EPSILON) {
                passed = false;
            }
        }
    }

    free(input);
    free(output);
    printf(passed ? "PASSED\n" : "FAILED\n");
    return passed ? 1 : 0;
}

int test_large_matrix() {
    printf("Test 7: Large 4096x4096 matrix... ");
    int n = 4096;
    size_t size = n * n;
    float *input = (float*)malloc(size * sizeof(float));
    float *output = (float*)malloc(size * sizeof(float));

    for (size_t i = 0; i < size; i++) {
        input[i] = (float)(i % 1000);
    }

    matrixTranspose(input, output, n, n);

    // Spot check
    bool passed = true;
    for (int i = 0; i < 100 && passed; i++) {
        for (int j = 0; j < 100 && passed; j++) {
            float expected = input[i * n + j];
            float actual = output[j * n + i];
            if (fabs(actual - expected) > EPSILON) {
                passed = false;
            }
        }
    }

    free(input);
    free(output);
    printf(passed ? "PASSED\n" : "FAILED\n");
    return passed ? 1 : 0;
}

int main() {
    printf("=== Matrix Transpose Tests ===\n\n");

    int passed = 0;
    passed += test_small_matrix();
    passed += test_square_matrix();
    passed += test_single_row();
    passed += test_single_column();
    passed += test_tile_aligned();
    passed += test_non_aligned();
    passed += test_large_matrix();

    printf("\n=== Results: %d/7 passed ===\n", passed);
    return (passed == 7) ? 0 : 1;
}
