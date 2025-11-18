#include <stdio.h>
extern void computeHistogram(int*, int*, int, int);

int main() {
    printf("=== Histogram Tests ===\n");

    // Test 1
    printf("Test 1: Small array... ");
    int in1[] = {0,1,2,1,0,3,2,1,0,2};
    int exp1[] = {3,3,3,1};
    int out1[4] = {0};
    computeHistogram(in1, out1, 10, 4);
    bool pass1 = true;
    for (int i = 0; i < 4; i++) if (out1[i] != exp1[i]) pass1 = false;
    printf(pass1 ? "PASSED\n" : "FAILED\n");

    // Test 2: Large array
    printf("Test 2: Large array (100K)... ");
    int n = 100000;
    int *in2 = new int[n];
    int out2[256] = {0};
    for (int i = 0; i < n; i++) in2[i] = i % 256;
    computeHistogram(in2, out2, n, 256);
    bool pass2 = true;
    for (int i = 0; i < 256; i++) {
        int expected = n / 256 + (i < n % 256 ? 1 : 0);
        if (out2[i] != expected) { pass2 = false; break; }
    }
    delete[] in2;
    printf(pass2 ? "PASSED\n" : "FAILED\n");

    printf("Passed: %d/2\n", pass1 + pass2);
    return (pass1 && pass2) ? 0 : 1;
}
