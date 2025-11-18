#include <stdio.h>
#include <math.h>
extern void warpReduce(float*, float*, int, int);

int main() {
    printf("=== Warp Reduction Test ===\n");

    float input[] = {1,2,3,4,5};
    float result;

    warpReduce(input, &result, 5, 0);  // Sum
    printf("Sum: %.0f (expected 15) - %s\n", result,
           (fabs(result - 15.0f) < 0.1f) ? "PASS" : "FAIL");

    return (fabs(result - 15.0f) < 0.1f) ? 0 : 1;
}
