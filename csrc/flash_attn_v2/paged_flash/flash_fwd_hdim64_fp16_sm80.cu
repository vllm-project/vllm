// Copyright (c) 2023, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

// template<>
// void run_mha_fwd_<cutlass::half_t, 64>(Flash_fwd_params &params, cudaStream_t stream) {
//     using elem_type = cutlass::half_t;
//     if (params.p_dropout == 1.f) {
//         // Using 8 warps is 18% slower for seqlen=2k, 2 warps is 5% slower
//         // Using block size (64 x 256) is 27% slower for seqlen=2k
//         // Using block size (256 x 64) is 85% slower for seqlen=2k, because of register spilling
//         run_flash_fwd<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, elem_type>,
//         false>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<64, 128, 64, 4, true,
//         false, elem_type>, false>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<64, 128,
//         64, 4, true, true, elem_type>, false>(params, stream);
//     } else {
//         run_flash_fwd<Flash_fwd_kernel_traits<64, 128, 64, 4, false, false, elem_type>,
//         true>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<64, 128, 64, 4, true, true,
//         elem_type>, true>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<64, 128, 64, 4,
//         true, false, elem_type>, true>(params, stream);
//     }
// }
template <>
void run_mha_fwd_<cutlass::half_t, 64>(Flash_fwd_params& params, cudaStream_t stream)
{
    run_mha_fwd_hdim64<cutlass::half_t>(params, stream);
}
