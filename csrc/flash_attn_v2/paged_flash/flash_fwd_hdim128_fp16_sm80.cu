// Copyright (c) 2023, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

// template<>
// void run_mha_fwd_<cutlass::half_t, 128>(Flash_fwd_params &params, cudaStream_t stream) {
//     using elem_type = cutlass::half_t;
//     if (params.p_dropout == 1.f) {
//         // Using 8 warps (128 x 128 and 256 x 64) is 28% slower for seqlen=2k
//         run_flash_fwd<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, elem_type>,
//         false>(params, stream);
//         // run_flash_fwd<Flash_fwd_kernel_traits<128, 128, 64, 4, true, false, elem_type>,
//         false>(params, stream);
//         // run_flash_fwd<Flash_fwd_kernel_traits<128, 128, 64, 4, false, true, elem_type>,
//         false>(params, stream);
//         // run_flash_fwd<Flash_fwd_kernel_traits<128, 128, 64, 4, true, true, elem_type>,
//         false>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<128, 128, 32, 4, false,
//         false, elem_type>, false>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<128, 64,
//         64, 4, false, false, elem_type>, false>(params, stream);
//         run_flash_fwd<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, elem_type>,
//         false>(params, stream);
//         // 1st ones are good for H100, A100
//         // 2nd one is good for A6000 bc we get slightly better occupancy
//     } else {
//         run_flash_fwd<Flash_fwd_kernel_traits<128, 128, 32, 4, false, false, elem_type>,
//         true>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<128, 128, 32, 4, true,
//         false, elem_type>, true>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<128, 128,
//         32, 4, true, true, elem_type>, true>(params, stream);
//         // 1st one is good for H100, A100, A6000
//     }
// }

template <>
void run_mha_fwd_<cutlass::half_t, 128>(Flash_fwd_params& params, cudaStream_t stream)
{
    run_mha_fwd_hdim128<cutlass::half_t>(params, stream);
}
