#define LEGACY_HIPBLAS_DIRECT
#include <hipblaslt/hipblaslt.h>
int main() {
    hipblasLtMatmulDescAttributes_t attr = HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER_VEC_EXT;
    return 0;
}
