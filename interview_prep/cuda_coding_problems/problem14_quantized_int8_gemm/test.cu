#include <stdio.h>
extern void quantizedMatMul(float*, float*, float*, int, int, int, float, float);

int main() {
    printf("=== Quantized INT8 GEMM Test ===\n");

    int M = 2, N = 2, K = 2;
    float A[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float C[4];

    quantizedMatMul(A, B, C, M, N, K, 0.01f, 0.01f);

    printf("Results: %.2f %.2f %.2f %.2f\n", C[0], C[1], C[2], C[3]);
    printf("Test complete\n");

    return 0;
}
