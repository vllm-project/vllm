#include <stdio.h>
extern void dynamicTranspose(float*, float*, int, int, int);

int main() {
    printf("=== Dynamic Shared Memory Test ===\n");

    float input[] = {1,2,3,4,5,6};
    float expected[] = {1,4,2,5,3,6};
    float output[6];

    dynamicTranspose(input, output, 2, 3, 16);

    bool pass = true;
    for (int i = 0; i < 6; i++)
        if (output[i] != expected[i]) pass = false;

    printf(pass ? "PASSED\n" : "FAILED\n");
    return pass ? 0 : 1;
}
