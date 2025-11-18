#include <stdio.h>
#include <math.h>

extern void prefixSum(float*, float*, int, bool);

#define EPSILON 1e-5

int main() {
    printf("=== Prefix Sum Tests ===\n");

    // Test 1: Inclusive scan
    printf("Test 1: Inclusive scan... ");
    float in1[] = {3,1,7,0,4,1,6,3};
    float exp1[] = {3,4,11,11,15,16,22,25};
    float out1[8];
    prefixSum(in1, out1, 8, true);
    bool pass1 = true;
    for (int i = 0; i < 8; i++)
        if (fabs(out1[i] - exp1[i]) > EPSILON) pass1 = false;
    printf(pass1 ? "PASSED\n" : "FAILED\n");

    // Test 2: Exclusive scan
    printf("Test 2: Exclusive scan... ");
    float exp2[] = {0,3,4,11,11,15,16,22};
    float out2[8];
    prefixSum(in1, out2, 8, false);
    bool pass2 = true;
    for (int i = 0; i < 8; i++)
        if (fabs(out2[i] - exp2[i]) > EPSILON) pass2 = false;
    printf(pass2 ? "PASSED\n" : "FAILED\n");

    // Test 3: Large array
    printf("Test 3: Large array (10000)... ");
    int n = 10000;
    float *in3 = new float[n];
    float *out3 = new float[n];
    for (int i = 0; i < n; i++) in3[i] = 1.0f;
    prefixSum(in3, out3, n, true);
    bool pass3 = true;
    for (int i = 0; i < n; i++)
        if (fabs(out3[i] - (i+1)) > 1.0f) { pass3 = false; break; }
    printf(pass3 ? "PASSED\n" : "FAILED\n");
    delete[] in3; delete[] out3;

    int total = pass1 + pass2 + pass3;
    printf("\nPassed: %d/3\n", total);
    return (total == 3) ? 0 : 1;
}
