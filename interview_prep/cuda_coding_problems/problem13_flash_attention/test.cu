#include <stdio.h>
extern void flashAttention(float*, float*, float*, float*, int, int);

int main() {
    printf("=== Flash Attention Test ===\n");

    int seq_len = 4, d = 2;
    float Q[] = {1,0, 0,1, 1,1, 0,0};
    float K[] = {1,0, 0,1, 1,1, 0,0};
    float V[] = {1,1, 2,2, 3,3, 4,4};
    float O[8];

    flashAttention(Q, K, V, O, seq_len, d);

    printf("Output[0]: %.4f\n", O[0]);
    printf("Test complete (check output manually)\n");

    return 0;
}
