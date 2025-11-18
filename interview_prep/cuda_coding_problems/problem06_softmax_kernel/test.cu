#include <stdio.h>
#include <math.h>
extern void softmax(float*, float*, int, int);

#define EPSILON 1e-4

int main() {
    printf("=== Softmax Tests ===\n");

    // Test 1: Small matrix
    printf("Test 1: Small matrix... ");
    float in1[] = {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f};
    float out1[6];
    softmax(in1, out1, 2, 3);

    bool pass1 = true;
    // Check row sums = 1
    for (int i = 0; i < 2; i++) {
        float sum = 0;
        for (int j = 0; j < 3; j++) sum += out1[i*3+j];
        if (fabs(sum - 1.0f) > EPSILON) pass1 = false;
    }
    // Check second row is uniform
    for (int j = 0; j < 3; j++)
        if (fabs(out1[3+j] - 0.3333f) > 0.01f) pass1 = false;
    printf(pass1 ? "PASSED\n" : "FAILED\n");

    // Test 2: Numerical stability
    printf("Test 2: Large values... ");
    float in2[] = {1000.0f, 1001.0f, 999.0f};
    float out2[3];
    softmax(in2, out2, 1, 3);

    float sum2 = 0;
    for (int i = 0; i < 3; i++) sum2 += out2[i];
    bool pass2 = (fabs(sum2 - 1.0f) < EPSILON && !isinf(out2[0]) && !isnan(out2[0]));
    printf(pass2 ? "PASSED\n" : "FAILED\n");

    printf("Passed: %d/2\n", pass1 + pass2);
    return (pass1 && pass2) ? 0 : 1;
}
