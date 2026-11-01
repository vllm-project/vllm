// Instantiates the int4 skinny GEMM kernels for N_in == 2 only; see
// WVSPLITK_INT4_LANE in skinny_gemms_int4.cu.
#define WVSPLITK_INT4_LANE 2
#define wvSplitK_int4_g wvSplitK_int4_g_n2
#include "skinny_gemms_int4.cu"
