#define LEGACY_HIPBLAS_DIRECT
#include <hipblaslt/hipblaslt.h>
int main() {
    hipblasLtMatmulMatrixScale_t attr = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    return 0;
}
