// Copyright (c) 2023, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

// template<>
// void run_mha_fwd_<cutlass::half_t, 32>(Flash_fwd_params &params, cudaStream_t stream) {
//     using elem_type = cutlass::half_t;
//     BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
//         run_flash_fwd<Flash_fwd_kernel_traits<32, 128, 128, 4, false, false, elem_type>,
//         Is_dropout>(params, stream);
//         // For dropout there might be a lot of register spilling?
//         // These two are very slow due to register spilling
//         // run_flash_fwd<Flash_fwd_kernel_traits<32, 256, 128, 4, false, elem_type>>(params,
//         stream);
//         // run_flash_fwd<Flash_fwd_kernel_traits<32, 128, 256, 4, false, elem_type>>(params,
//         stream);
//         // This one is slightly slower
//         // run_flash_fwd<Flash_fwd_kernel_traits<32, 256, 64, 4, false, elem_type>>(params,
//         stream);
//     });
// }
template <>
void run_mha_fwd_<cutlass::half_t, 32>(Flash_fwd_params& params, cudaStream_t stream)
{
    run_mha_fwd_hdim32<cutlass::half_t>(params, stream);
}
