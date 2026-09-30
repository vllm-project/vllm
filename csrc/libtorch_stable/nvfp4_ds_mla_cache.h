// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

// Launchers for the nvfp4_ds_mla KV cache kernels. Those kernels live in
// their own TU (nvfp4_ds_mla_cache_kernels.cu) so that only that TU is built
// for the arch-conditional Blackwell targets their e2m1 conversions require;
// these declarations are what keeps cache_kernels.cu generic.
//
// Only defined when the build has an SM100 target (-DENABLE_NVFP4_SM100=1);
// callers must guard on the same macro.
namespace vllm {

// concat_and_cache_mla for the nvfp4_ds_mla layout.
//   kv_c:       bf16   [num_tokens, 512]
//   k_pe:       bf16   [num_tokens, 64]
//   kv_cache:   uint8  [num_blocks, block_size, 352]
void launch_concat_and_cache_nvfp4_ds_mla(const void* kv_c, const void* k_pe,
                                          void* kv_cache,
                                          const int64_t* slot_mapping,
                                          int block_stride, int entry_stride,
                                          int kv_c_stride, int k_pe_stride,
                                          int block_size, int num_tokens,
                                          cudaStream_t stream);

// Gather an nvfp4_ds_mla cache into a bf16 [total_tokens, 576] workspace.
void launch_cp_gather_and_upconvert_nvfp4_kv_cache(
    const uint8_t* src_cache, void* dst, const int32_t* block_table,
    const int32_t* workspace_starts, int32_t num_reqs, int32_t block_size,
    int32_t total_tokens, int64_t block_table_stride,
    int64_t cache_block_stride, int64_t cache_entry_stride,
    int64_t dst_entry_stride, cudaStream_t stream);

}  // namespace vllm
