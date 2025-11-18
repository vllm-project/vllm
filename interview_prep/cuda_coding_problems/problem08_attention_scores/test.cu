#include <stdio.h>
#include <math.h>
extern void computeAttentionScores(float*, float*, float*, int, int, int);

int main() {
    printf("=== Attention Scores Tests ===\n");

    printf("Test 1: Small matrices... ");
    float Q[] = {1,0, 0,1, 1,1};
    float K[] = {1,0, 0,1};
    float scores[6];
    computeAttentionScores(Q, K, scores, 3, 2, 2);

    float scale = 1.0f / sqrtf(2.0f);
    bool pass = (fabs(scores[0] - scale) < 0.01f &&
                 fabs(scores[1] - 0) < 0.01f &&
                 fabs(scores[4] - scale) < 0.01f);
    printf(pass ? "PASSED\n" : "FAILED\n");

    return pass ? 0 : 1;
}
