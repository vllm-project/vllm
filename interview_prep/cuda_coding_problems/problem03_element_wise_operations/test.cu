#include <stdio.h>
#include <math.h>

extern void runFusedElementWise(float*, float*, float*, float*, float, float, float, int);

#define EPSILON 1e-5

int test1() {
    printf("Test 1: Small array... ");
    float A[] = {1,2,3}, B[] = {2,3,4}, C[] = {1,1,1};
    float D[3], expected[] = {9,14,19};
    runFusedElementWise(A, B, C, D, 2.0f, 3.0f, 1.0f, 3);
    for (int i = 0; i < 3; i++)
        if (fabs(D[i] - expected[i]) > EPSILON) { printf("FAILED\n"); return 0; }
    printf("PASSED\n");
    return 1;
}

int test2() {
    printf("Test 2: Large array... ");
    int n = 100000;
    float *A = new float[n], *B = new float[n], *C = new float[n], *D = new float[n];
    for (int i = 0; i < n; i++) { A[i] = 1; B[i] = 2; C[i] = 1; }
    runFusedElementWise(A, B, C, D, 2.0f, 3.0f, 1.0f, n);
    bool pass = true;
    for (int i = 0; i < n; i++) if (fabs(D[i] - 9.0f) > EPSILON) { pass = false; break; }
    delete[] A; delete[] B; delete[] C; delete[] D;
    printf(pass ? "PASSED\n" : "FAILED\n");
    return pass;
}

int main() {
    printf("=== Fused Element-wise Tests ===\n");
    int passed = test1() + test2();
    printf("Passed: %d/2\n", passed);
    return (passed == 2) ? 0 : 1;
}
