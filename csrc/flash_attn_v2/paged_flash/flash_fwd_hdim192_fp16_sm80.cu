// Copyright (c) 2023, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

// template<>
// void run_mha_fwd_<cutlass::half_t, 192>(Flash_fwd_params &params, cudaStream_t stream) {
//     using elem_type = cutlass::half_t;
//     BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
//         run_flash_fwd<Flash_fwd_kernel_traits<192, 64, 64, 4, false, false, elem_type>,
//         Is_dropout>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<192, 128, 32, 4,
//         false, false, elem_type>, Is_dropout>(params, stream);
//         run_flash_fwd<Flash_fwd_kernel_traits<192, 64, 32, 4, false, false, elem_type>,
//         Is_dropout>(params, stream);
//         // This one is slightly faster for causal?
//         // run_flash_fwd<Flash_fwd_kernel_traits<192, 128, 64, 8, false, elem_type>>(params,
//         stream);
//         // run_flash_fwd<Flash_fwd_kernel_traits<192, 128, 32, 4, false, elem_type>>(params,
//         stream);
//         // run_flash_fwd<Flash_fwd_kernel_traits<192, 128, 64, 4, false, elem_type>>(params,
//         stream);
//         // run_flash_fwd<Flash_fwd_kernel_traits<192, 64, 128, 4, false, elem_type>>(params,
//         stream);
//         // run_flash_fwd<Flash_fwd_kernel_traits<192, 128, 128, 8, false, elem_type>>(params,
//         stream);
//     });
//     // For A100 H100, 1st is faster with dropout, 3rd is faster without dropout
//     // For A6000, 1st is faster when causal, 3rd is faster when not causal
// }
template <>
void run_mha_fwd_<cutlass::half_t, 192>(Flash_fwd_params& params, cudaStream_t stream)
{
    run_mha_fwd_hdim192<cutlass::half_t>(params, stream);
}
