// Instantiates the int4 skinny GEMM kernels for N_in == 1 only; see
// WVSPLITK_INT4_LANE in skinny_gemms_int4.cu.
#define WVSPLITK_INT4_LANE 1
#define wvSplitK_int4_g wvSplitK_int4_g_n1
#include "skinny_gemms_int4.cu"
