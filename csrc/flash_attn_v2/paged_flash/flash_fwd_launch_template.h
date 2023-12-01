/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include "flash.h"
#include "flash_fwd_kernel.h"
#include "static_switch.h"

template <typename Kernel_traits, bool Is_causal, bool Is_even_K>
__global__ void flash_fwd_kernel(Flash_fwd_params params)
{
    flash::compute_attn<Kernel_traits, Is_causal, Is_even_K>(params);
}

template <typename Kernel_traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params& params, cudaStream_t stream)
{
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    // printf("smem_size = %d\n", smem_size);

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21

    const int num_block_atoms = params.num_atoms;
    const int num_block_heads = params.h;

    dim3 grid(num_block_heads, num_block_atoms);

    const bool is_even_K = params.d == Kernel_traits::kHeadDim;

    BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
        auto kernel = &flash_fwd_kernel<Kernel_traits, Is_causal, IsEvenKConst>;

        if (smem_size >= 48 * 1024) {
            cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }
        int ctas_per_sm;
        cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
        // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
        kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
    });
}

template <typename T>
void run_mha_fwd_hdim32(Flash_fwd_params& params, cudaStream_t stream)
{
    constexpr int Headdim = 32;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_causal>(
            params, stream);
    });
}

template <typename T>
void run_mha_fwd_hdim64(Flash_fwd_params& params, cudaStream_t stream)
{
    constexpr int Headdim = 64;

    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_causal>(
            params, stream);
    });
}

template <typename T>
void run_mha_fwd_hdim96(Flash_fwd_params& params, cudaStream_t stream)
{
    constexpr int Headdim = 96;
    int cc_major, cc_minor;
    cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, 0);
    bool is_sm8x = cc_major == 8 && cc_minor > 0;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
        if (is_sm8x) {
            if constexpr (!Is_causal) {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>,
                              Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>,
                              Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_causal>(
                params, stream);
        }
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout,
        // Is_causal>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4,
        // true, true, T>, Is_dropout, Is_causal>(params, stream); These two are always slower
        // run_flash_fwd<Flash_fwd_kernel_traits<96, 128, 128, 4, true, T>>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<96, 64, 128, 4, true, T>>(params, stream);
    });
}

template <typename T>
void run_mha_fwd_hdim128(Flash_fwd_params& params, cudaStream_t stream)
{
    constexpr int Headdim = 128;
    int cc_major, cc_minor;
    cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, 0);
    bool is_sm8x = cc_major == 8 && cc_minor > 0;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
        // and 128 x 32 (48 KB smem) is the fastest for non-causal since we get 2 CTAs per SM.
        if (is_sm8x) {
            if constexpr (!Is_causal) {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>,
                              Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>,
                              Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_causal>(
                params, stream);
        }
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout,
        // Is_causal>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4,
        // true, true, T>, Is_dropout, Is_causal>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, false, T>, Is_dropout,
        // Is_causal>(params, stream); Using 8 warps (128 x 128 and 256 x 64) is 28% slower for
        // seqlen=2k run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, false, T>,
        // Is_dropout, Is_causal>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<Headdim,
        // 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream); 1st ones are good
        // for H100, A100 2nd one is good for A6000 bc we get slightly better occupancy
    });
}

template <typename T>
void run_mha_fwd_hdim160(Flash_fwd_params& params, cudaStream_t stream)
{
    constexpr int Headdim = 160;
    // auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = true; // dprops->major == 8 && dprops->minor > 0;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // For A100, H100, 128 x 32 is the fastest.
        // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
        // and 128 x 64 with 8 warps is the fastest for non-causal.
        if (is_sm8x) {
            if constexpr (!Is_causal) {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>,
                              Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>,
                              Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_causal>(
                params, stream);
        }
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, true, T>, Is_dropout,
        // Is_causal>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4,
        // false, false, T>, Is_dropout, Is_causal>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, T>>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, T>>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, T>>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, T>>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, T>>(params, stream);
    });
}

template <typename T>
void run_mha_fwd_hdim192(Flash_fwd_params& params, cudaStream_t stream)
{
    constexpr int Headdim = 192;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_causal>(
            params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout,
        // Is_causal>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 8,
        // false, false, T>, Is_dropout, Is_causal>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, T>>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, T>>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, T>>(params, stream);
    });
}

template <typename T>
void run_mha_fwd_hdim224(Flash_fwd_params& params, cudaStream_t stream)
{
    constexpr int Headdim = 224;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // printf("max_smem_per_block = %d\n", max_smem_per_block);
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        if (max_smem_per_block >= 2 * Headdim * (128 + 2 * 64)) {  // 112 KB
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_causal>(
                params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal>(
                params, stream);
        }
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout,
        // Is_causal>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4,
        // false, false, T>, Is_dropout, Is_causal>(params, stream); We can't do 128 x 32 with 8
        // warps because with headdim 224, kBlockKSmem = 32. If we have N = 32, there are only 1024
        // elements to load at once, where each load is 8 elements. This means we can only use 128
        // threads and not 256 threads. run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 8,
        // false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template <typename T>
void run_mha_fwd_hdim256(Flash_fwd_params& params, cudaStream_t stream)
{
    constexpr int Headdim = 256;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_sm, max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // printf("max_smem_per_sm = %d, max_smem_per_block = %d\n", max_smem_per_sm,
    // max_smem_per_block);
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // For A100, we want to run with 128 x 64 (128KB smem).
        // For H100 we want to run with 64 x 64 (96KB smem) since then we can get 2 CTAs per SM.
        if (max_smem_per_block >= 2 * Headdim * (128 + 2 * 64) &&
            max_smem_per_sm < 4 * Headdim * (64 + 2 * 64)) {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_causal>(
                params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal>(
                params, stream);
        }
        // 64 KB
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout,
        // Is_causal>(params, stream); 96 KB run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32,
        // 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}
