#include <stdio.h>
extern void processWithStreams(float*, float*, int, int);

int main() {
    printf("=== Multi-Stream Test ===\n");

    int n = 1000;
    float *input = new float[n];
    float *output = new float[n];

    for (int i = 0; i < n; i++) input[i] = 1.0f;

    processWithStreams(input, output, n, 4);

    printf("Output[0]: %.4f\n", output[0]);
    printf("Test complete\n");

    delete[] input;
    delete[] output;
    return 0;
}
