// Copyright (c) 2023, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

// template<>
// void run_mha_fwd_<cutlass::bfloat16_t, 64>(Flash_fwd_params &params, cudaStream_t stream) {
//     using elem_type = cutlass::bfloat16_t;
//     if (params.p_dropout == 1.f) {
//         run_flash_fwd<Flash_fwd_kernel_traits<64, 128, 64, 4, true, false, elem_type>,
//         false>(params, stream);
//     } else {
//         run_flash_fwd<Flash_fwd_kernel_traits<64, 128, 64, 4, false, false, elem_type>,
//         true>(params, stream);
//     }
// }
template <>
void run_mha_fwd_<cutlass::bfloat16_t, 64>(Flash_fwd_params& params, cudaStream_t stream)
{
    run_mha_fwd_hdim64<cutlass::bfloat16_t>(params, stream);
}
