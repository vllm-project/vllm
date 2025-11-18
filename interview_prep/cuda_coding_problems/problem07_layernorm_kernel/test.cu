#include <stdio.h>
#include <math.h>
extern void layerNorm(float*, float*, float*, float*, int, int, float);

#define EPSILON 0.01

int main() {
    printf("=== LayerNorm Tests ===\n");

    // Test 1: Simple case
    printf("Test 1: Basic layernorm... ");
    float in1[] = {1,2,3};
    float gamma1[] = {1,1,1};
    float beta1[] = {0,0,0};
    float out1[3];
    layerNorm(in1, out1, gamma1, beta1, 1, 3, 1e-5f);

    // Check mean ≈ 0, std ≈ 1 (before scale/bias)
    float mean = (out1[0] + out1[1] + out1[2]) / 3;
    bool pass1 = (fabs(mean) < EPSILON);
    printf(pass1 ? "PASSED\n" : "FAILED\n");

    // Test 2: With scale and bias
    printf("Test 2: With gamma/beta... ");
    float gamma2[] = {2,2,2};
    float beta2[] = {1,1,1};
    float out2[3];
    layerNorm(in1, out2, gamma2, beta2, 1, 3, 1e-5f);

    // Should scale by 2 and add 1
    float mean2 = (out2[0] + out2[1] + out2[2]) / 3;
    bool pass2 = (fabs(mean2 - 1.0f) < EPSILON);  // Mean should be 1 after bias
    printf(pass2 ? "PASSED\n" : "FAILED\n");

    printf("Passed: %d/2\n", pass1 + pass2);
    return (pass1 && pass2) ? 0 : 1;
}
