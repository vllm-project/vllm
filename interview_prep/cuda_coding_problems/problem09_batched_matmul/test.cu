#include <stdio.h>
#include <math.h>
extern void batchedMatMul(float*, float*, float*, int, int, int, int);

int main() {
    printf("=== Batched MatMul Test ===\n");

    float A[] = {1,2,3,4, 5,6,7,8};
    float B[] = {1,0,0,1, 1,1,1,1};
    float C[8];

    batchedMatMul(A, B, C, 2, 2, 2, 2);

    // Batch 0: [[1,2],[3,4]] @ [[1,0],[0,1]] = [[1,2],[3,4]]
    // Batch 1: [[5,6],[7,8]] @ [[1,1],[1,1]] = [[11,11],[15,15]]
    bool pass = (C[0]==1 && C[1]==2 && C[2]==3 && C[3]==4 &&
                 C[4]==11 && C[5]==11 && C[6]==15 && C[7]==15);

    printf(pass ? "PASSED\n" : "FAILED\n");
    return pass ? 0 : 1;
}
