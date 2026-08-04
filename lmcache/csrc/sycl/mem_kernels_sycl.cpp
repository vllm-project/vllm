// SPDX-License-Identifier: Apache-2.0
//
// SYCL implementation of LMCache memory kernels for Intel XPU
// (PVC / Arc / Battlemage).
//
// Performance-critical design choices:
//
// 1. Work-group size 256 -- keeps Intel EU ALUs fed and hides
//    global-memory latency.
// 2. [[sycl::reqd_sub_group_size(16)]] -- native SIMD width on all
//    Intel discrete GPUs; prevents IGC from falling back to width 32
//    on PVC (which would halve occupancy).
// 3. Compile-time template parameters (DIRECTION, USE_MLA) eliminate
//    run-time branches in the innermost loop.
// 4. Hoisted loop-invariant base offsets -- integer division/modulo
//    (flash_infer block indexing) is computed once per token+layer
//    rather than per inner-loop iteration.
// 5. 64-bit (int64_t) bulk transfers pack two fp32 / four fp16 /
//    eight int8 values per move.
// 6. Fused K+V work-groups (non-MLA multi-layer kernels) -- one
//    work-group handles both K and V, halving dispatch count and
//    avoiding redundant slot/pointer/division work.

// sycl/accessor.hpp references the deprecated 'host_buffer' internally
// even when user code only uses USM pointers; suppress the noise.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <sycl/sycl.hpp>
#pragma GCC diagnostic pop

#include <torch/all.h>
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>

#include "mem_kernels_sycl.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------
// Native SIMD width on PVC, DG2, and BMG.
constexpr int INTEL_SUB_GROUP_SIZE = 16;

// Max work-group size; must be a multiple of INTEL_SUB_GROUP_SIZE.
constexpr int MAX_WG_SIZE = 256;

// ---------------------------------------------------------------------------
// Round up so the work-group divides evenly into sub-groups.
// ---------------------------------------------------------------------------
inline int round_up_to_sg(int n) {
  return ((n + INTEL_SUB_GROUP_SIZE - 1) / INTEL_SUB_GROUP_SIZE) *
         INTEL_SUB_GROUP_SIZE;
}

// ---------------------------------------------------------------------------
// Namespace lmc -- device-side helper functions
// ---------------------------------------------------------------------------
namespace lmc {

template <EngineKVFormat format>
inline int64_t page_buffer_offset(const int k_or_v, const int token_idx,
                                  const int scalar_offset,
                                  const int scalars_per_token,
                                  const int page_buffer_size,
                                  const int block_size) {
  // vLLM cross layer
  if constexpr (format == EngineKVFormat::NB_NL_TWO_BS_NH_HS) {
    return k_or_v * page_buffer_size * scalars_per_token +
           token_idx * scalars_per_token + scalar_offset;
  }
  // vLLM flash attention
  else if constexpr (format == EngineKVFormat::NL_X_TWO_NB_BS_NH_HS) {
    return k_or_v * page_buffer_size * scalars_per_token +
           token_idx * scalars_per_token + scalar_offset;
  }
  // vLLM flash infer
  else if constexpr (format == EngineKVFormat::NL_X_NB_TWO_BS_NH_HS) {
    const int block_idx = token_idx / block_size;
    const int block_offset = token_idx % block_size;
    return block_idx * 2 * block_size * scalars_per_token +
           k_or_v * block_size * scalars_per_token +
           block_offset * scalars_per_token + scalar_offset;
  }
  // MLA formats: vLLM (NL_X_NB_BS_HS) and SGLang (NL_X_NBBS_ONE_HS)
  else if constexpr (format == EngineKVFormat::NL_X_NB_BS_HS ||
                     format == EngineKVFormat::NL_X_NBBS_ONE_HS) {
    return token_idx * scalars_per_token + scalar_offset;
  }
}

/// Loop-invariant base offset for the paged buffer.
/// page_buffer_offset(k_or_v, slot, i, ...) == base_offset(...) + i.
/// Hoisting this out of the inner loop avoids re-computing the
/// integer division / modulo (flash_infer) on every iteration.
template <EngineKVFormat format>
inline int64_t page_buffer_base_offset(const int k_or_v, const int token_idx,
                                       const int scalars_per_token,
                                       const int page_buffer_size,
                                       const int block_size) {
  if constexpr (format == EngineKVFormat::NB_NL_TWO_BS_NH_HS) {
    return k_or_v * page_buffer_size * scalars_per_token +
           token_idx * scalars_per_token;
  } else if constexpr (format == EngineKVFormat::NL_X_TWO_NB_BS_NH_HS) {
    return k_or_v * page_buffer_size * scalars_per_token +
           token_idx * scalars_per_token;
  } else if constexpr (format == EngineKVFormat::NL_X_NB_TWO_BS_NH_HS) {
    const int block_idx = token_idx / block_size;
    const int block_offset = token_idx % block_size;
    return block_idx * 2 * block_size * scalars_per_token +
           k_or_v * block_size * scalars_per_token +
           block_offset * scalars_per_token;
  } else if constexpr (format == EngineKVFormat::NL_X_NB_BS_HS ||
                       format == EngineKVFormat::NL_X_NBBS_ONE_HS) {
    return token_idx * scalars_per_token;
  }
}

inline int64_t page_buffer_offset_unilateral(const int token_idx,
                                             const int scalar_offset,
                                             const int scalars_per_token) {
  return token_idx * scalars_per_token + scalar_offset;
}

inline int64_t key_value_offset(const int k_or_v, const int layer_idx,
                                const int token_idx, const int scalar_offset,
                                const int scalars_per_token,
                                const int num_tokens, const int num_layers) {
  return k_or_v * num_layers * num_tokens * scalars_per_token +
         layer_idx * num_tokens * scalars_per_token +
         token_idx * scalars_per_token + scalar_offset;
}

/// Loop-invariant base offset for the LMCache key_value buffer.
/// key_value_offset(k_or_v, layer, token, i, ...) == base(...) + i.
inline int64_t key_value_base_offset(const int k_or_v, const int layer_idx,
                                     const int token_idx,
                                     const int scalars_per_token,
                                     const int num_tokens,
                                     const int num_layers) {
  return k_or_v * num_layers * num_tokens * scalars_per_token +
         layer_idx * num_tokens * scalars_per_token +
         token_idx * scalars_per_token;
}

}  // namespace lmc

// ---------------------------------------------------------------------------
// Pointer helper -- returns a kernel-accessible pointer of the given type.
// XPU tensors expose their USM device pointer; CPU tensors must be backed
// by USM host memory (e.g. sycl::malloc_host) to be device-accessible.
// ---------------------------------------------------------------------------
template <typename T, typename TENSOR_TYPE>
T* get_kernel_ptr(TENSOR_TYPE& tensor) {
  torch::Device device = tensor.device();
  if (device.is_xpu()) {
    return static_cast<T*>(tensor.data_ptr());
  } else if (device.is_cpu()) {
    // USM host pointers are device-accessible.
    return static_cast<T*>(tensor.data_ptr());
  } else {
    TORCH_CHECK(false,
                "Invalid device. Device must be xpu or cpu (USM pinned).");
  }
}

// ---------------------------------------------------------------------------
// Kernel-launch helpers -- multi-layer kernels
// ---------------------------------------------------------------------------

/**
 * Submit the multi-layer KV transfer kernel for MLA formats
 * (k_or_v_size == 1).
 *
 * nd_range layout: group(0)=k_or_v, group(1)=layer, group(2)=token;
 * local_id(2)=tid, local_range(2)=num_threads.
 *
 * Optimizations:
 *   - DIRECTION is a compile-time bool (no branch in hot loop)
 *   - Loop-invariant base offsets (including flash_infer's integer
 *     division) are computed once before the inner loop
 *   - Work-group size rounded to a sub-group multiple for full SIMD
 *     utilisation
 */
template <typename scalar_t, bool DIRECTION, EngineKVFormat format>
void submit_multi_layer_kernel(sycl::queue& queue, scalar_t* key_value_ptr,
                               scalar_t** page_buffer_ptrs,
                               const int64_t* slot_mapping_ptr,
                               int scalars_per_token, int num_tokens,
                               int num_layers, int page_buffer_size,
                               int block_size, int skip_prefix_n_tokens,
                               int k_or_v_size, int wg_size) {
  int num_transfer_tokens = num_tokens - skip_prefix_n_tokens;
  if (num_transfer_tokens <= 0 || num_layers <= 0) return;

  sycl::range<3> global_range(
      static_cast<size_t>(k_or_v_size), static_cast<size_t>(num_layers),
      static_cast<size_t>(num_transfer_tokens) * wg_size);
  sycl::range<3> local_range(1, 1, static_cast<size_t>(wg_size));

  queue.parallel_for(
      sycl::nd_range<3>(global_range, local_range),
      [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(16)]] {
        const int token_id = static_cast<int>(item.get_group(2));
        const int layer_id = static_cast<int>(item.get_group(1));
        const int k_or_v = static_cast<int>(item.get_group(0));
        const int tid = static_cast<int>(item.get_local_id(2));
        const int num_threads = static_cast<int>(item.get_local_range(2));

        const int kv_token_id = token_id + skip_prefix_n_tokens;
        const int64_t slot_idx = slot_mapping_ptr[kv_token_id];
        scalar_t* paged_buffer_ptr = page_buffer_ptrs[layer_id];

        if (slot_idx < 0) return;

        // Hoist loop-invariant base offsets (flash_infer's integer
        // division happens here, once, not per loop iteration).
        const int64_t lmc_base = lmc::key_value_base_offset(
            k_or_v, layer_id, kv_token_id, scalars_per_token, num_tokens,
            num_layers);
        const int64_t vllm_base = lmc::page_buffer_base_offset<format>(
            k_or_v, slot_idx, scalars_per_token, page_buffer_size, block_size);

        for (int i = tid; i < scalars_per_token; i += num_threads) {
          if constexpr (DIRECTION) {
            key_value_ptr[lmc_base + i] = paged_buffer_ptr[vllm_base + i];
          } else {
            paged_buffer_ptr[vllm_base + i] = key_value_ptr[lmc_base + i];
          }
        }
      });
}

/**
 * Submit a fused K+V multi-layer kernel for non-MLA formats.
 *
 * Processes both key (k_or_v=0) and value (k_or_v=1) within the same
 * work-group, halving work-group count compared to dispatching K and
 * V separately. Slot mapping, pointer-array lookup, and flash_infer's
 * block-index division are each performed once and reused for both.
 */
template <typename scalar_t, bool DIRECTION, EngineKVFormat format>
void submit_multi_layer_kernel_fused_kv(
    sycl::queue& queue, scalar_t* key_value_ptr, scalar_t** page_buffer_ptrs,
    const int64_t* slot_mapping_ptr, int scalars_per_token, int num_tokens,
    int num_layers, int page_buffer_size, int block_size,
    int skip_prefix_n_tokens, int wg_size) {
  int num_transfer_tokens = num_tokens - skip_prefix_n_tokens;
  if (num_transfer_tokens <= 0 || num_layers <= 0) return;

  // Grid: (1, num_layers, num_transfer_tokens * wg_size);
  // no k_or_v dimension — K and V share one work-group.
  sycl::range<3> global_range(
      1, static_cast<size_t>(num_layers),
      static_cast<size_t>(num_transfer_tokens) * wg_size);
  sycl::range<3> local_range(1, 1, static_cast<size_t>(wg_size));

  queue.parallel_for(
      sycl::nd_range<3>(global_range, local_range),
      [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(16)]] {
        const int token_id = static_cast<int>(item.get_group(2));
        const int layer_id = static_cast<int>(item.get_group(1));
        const int tid = static_cast<int>(item.get_local_id(2));
        const int num_threads = static_cast<int>(item.get_local_range(2));

        const int kv_token_id = token_id + skip_prefix_n_tokens;
        const int64_t slot_idx = slot_mapping_ptr[kv_token_id];
        scalar_t* paged_buffer_ptr = page_buffer_ptrs[layer_id];

        if (slot_idx < 0) return;

        // Base offsets for K (k_or_v=0) and V (k_or_v=1); the
        // flash_infer division/modulo runs once per token+layer.
        const int64_t lmc_base_k = lmc::key_value_base_offset(
            0, layer_id, kv_token_id, scalars_per_token, num_tokens,
            num_layers);
        const int64_t lmc_base_v = lmc::key_value_base_offset(
            1, layer_id, kv_token_id, scalars_per_token, num_tokens,
            num_layers);
        const int64_t vllm_base_k = lmc::page_buffer_base_offset<format>(
            0, slot_idx, scalars_per_token, page_buffer_size, block_size);
        const int64_t vllm_base_v = lmc::page_buffer_base_offset<format>(
            1, slot_idx, scalars_per_token, page_buffer_size, block_size);

        for (int i = tid; i < scalars_per_token; i += num_threads) {
          if constexpr (DIRECTION) {
            // paged buffer → LMCache
            key_value_ptr[lmc_base_k + i] = paged_buffer_ptr[vllm_base_k + i];
            key_value_ptr[lmc_base_v + i] = paged_buffer_ptr[vllm_base_v + i];
          } else {
            // LMCache → paged buffer
            paged_buffer_ptr[vllm_base_k + i] = key_value_ptr[lmc_base_k + i];
            paged_buffer_ptr[vllm_base_v + i] = key_value_ptr[lmc_base_v + i];
          }
        }
      });
}

/**
 * Submit the multi-layer unilateral kernel (SGLang MHA).
 *
 * DIRECTION = true  → paged buffer → LMCache (D2H)
 * DIRECTION = false → LMCache → paged buffer (H2D)
 *
 * Uses `if constexpr (DIRECTION)` so the compiler can
 * dead-strip the unused branch entirely.
 */
template <typename scalar_t, bool DIRECTION>
void submit_multi_layer_unilateral_kernel(
    sycl::queue& queue, scalar_t* key_value_ptr, scalar_t** page_buffer_ptrs,
    const int64_t* slot_mapping_ptr, int scalars_per_token, int num_tokens,
    int num_layers, int page_buffer_size, int k_or_v_size, int kv_num_tokens,
    int wg_size) {
  if (kv_num_tokens <= 0 || num_layers <= 0) return;

  sycl::range<3> global_range(static_cast<size_t>(k_or_v_size),
                              static_cast<size_t>(num_layers),
                              static_cast<size_t>(kv_num_tokens) * wg_size);
  sycl::range<3> local_range(1, 1, static_cast<size_t>(wg_size));

  queue.parallel_for(
      sycl::nd_range<3>(global_range, local_range),
      [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(16)]] {
        const int token_id = static_cast<int>(item.get_group(2));
        const int layer_id = static_cast<int>(item.get_group(1));
        const int k_or_v = static_cast<int>(item.get_group(0));
        const int tid = static_cast<int>(item.get_local_id(2));
        const int num_threads = static_cast<int>(item.get_local_range(2));

        const int64_t slot_idx = slot_mapping_ptr[token_id];
        scalar_t* key_ptr = page_buffer_ptrs[layer_id];
        scalar_t* value_ptr = page_buffer_ptrs[layer_id + num_layers];

        if (slot_idx < 0) return;

        const int64_t lmc_base = lmc::key_value_base_offset(
            k_or_v, layer_id, token_id, scalars_per_token, num_tokens,
            num_layers);
        const int64_t sgl_base = slot_idx * scalars_per_token;

        for (int i = tid; i < scalars_per_token; i += num_threads) {
          if (k_or_v == 0) {
            if constexpr (DIRECTION)
              key_value_ptr[lmc_base + i] = key_ptr[sgl_base + i];
            else
              key_ptr[sgl_base + i] = key_value_ptr[lmc_base + i];
          } else {
            if constexpr (DIRECTION)
              key_value_ptr[lmc_base + i] = value_ptr[sgl_base + i];
            else
              value_ptr[sgl_base + i] = key_value_ptr[lmc_base + i];
          }
        }
      });
}

// ---------------------------------------------------------------------------
// Macros to dispatch multi-layer kernels with a specific EngineKVFormat.
// MLA formats (k_or_v_size==1) use the per-component kernel.
// Non-MLA formats (k_or_v_size==2) use the fused K+V kernel.
// ---------------------------------------------------------------------------
#define LAUNCH_KERNEL_WITH_FORMAT(T, DIRECTION, FORMAT)                     \
  submit_multi_layer_kernel<T, DIRECTION, FORMAT>(                          \
      queue, key_value_ptr, page_buffer_ptrs, slot_mapping_ptr, num_xwords, \
      num_tokens, num_layers, page_buffer_size, block_size,                 \
      skip_prefix_n_tokens, k_or_v_size, wg_size);

#define LAUNCH_FUSED_KV_KERNEL_WITH_FORMAT(T, DIRECTION, FORMAT)            \
  submit_multi_layer_kernel_fused_kv<T, DIRECTION, FORMAT>(                 \
      queue, key_value_ptr, page_buffer_ptrs, slot_mapping_ptr, num_xwords, \
      num_tokens, num_layers, page_buffer_size, block_size,                 \
      skip_prefix_n_tokens, wg_size);

// ---------------------------------------------------------------------------
// multi_layer_kv_transfer -- templated implementation
// ---------------------------------------------------------------------------
template <typename T>
void multi_layer_kv_transfer_templated(
    torch::Tensor& key_value, const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping, const torch::Device& paged_memory_device,
    const int page_buffer_size, const TransferDirection direction,
    const EngineKVFormat engine_kv_format, const int block_size,
    const int skip_prefix_n_tokens) {
  T* key_value_ptr = get_kernel_ptr<T, torch::Tensor>(key_value);
  T** page_buffer_ptrs =
      get_kernel_ptr<T*, const torch::Tensor>(key_value_ptrs);
  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int num_layers = key_value.size(1);
  int num_tokens = key_value.size(2);
  int num_origin_elements = key_value.size(3);
  int elements_per_xword = sizeof(T) / key_value.element_size();
  int num_xwords = num_origin_elements / elements_per_xword;

  int k_or_v_size = ::is_mla(engine_kv_format) ? 1 : 2;

  // Round up to a sub-group multiple so every sub-group is full.
  int wg_size = round_up_to_sg(std::min(num_xwords, MAX_WG_SIZE));

  const c10::OptionalDeviceGuard device_guard(paged_memory_device);
  sycl::queue& queue =
      c10::xpu::getCurrentXPUStream(paged_memory_device.index()).queue();

  // Non-MLA formats use the fused K+V kernel; MLA formats
  // (k_or_v_size==1) use the per-component kernel.
  if (k_or_v_size == 2) {
    if (direction == TransferDirection::H2D) {
      switch (engine_kv_format) {
        case EngineKVFormat::NB_NL_TWO_BS_NH_HS:
          LAUNCH_FUSED_KV_KERNEL_WITH_FORMAT(
              T, false, EngineKVFormat::NB_NL_TWO_BS_NH_HS);
          break;
        case EngineKVFormat::NL_X_TWO_NB_BS_NH_HS:
          LAUNCH_FUSED_KV_KERNEL_WITH_FORMAT(
              T, false, EngineKVFormat::NL_X_TWO_NB_BS_NH_HS);
          break;
        case EngineKVFormat::NL_X_NB_TWO_BS_NH_HS:
          LAUNCH_FUSED_KV_KERNEL_WITH_FORMAT(
              T, false, EngineKVFormat::NL_X_NB_TWO_BS_NH_HS);
          break;
        default:
          throw std::runtime_error("Unsupported non-MLA EngineKVFormat");
      }
    } else {
      switch (engine_kv_format) {
        case EngineKVFormat::NB_NL_TWO_BS_NH_HS:
          LAUNCH_FUSED_KV_KERNEL_WITH_FORMAT(
              T, true, EngineKVFormat::NB_NL_TWO_BS_NH_HS);
          break;
        case EngineKVFormat::NL_X_TWO_NB_BS_NH_HS:
          LAUNCH_FUSED_KV_KERNEL_WITH_FORMAT(
              T, true, EngineKVFormat::NL_X_TWO_NB_BS_NH_HS);
          break;
        case EngineKVFormat::NL_X_NB_TWO_BS_NH_HS:
          LAUNCH_FUSED_KV_KERNEL_WITH_FORMAT(
              T, true, EngineKVFormat::NL_X_NB_TWO_BS_NH_HS);
          break;
        default:
          throw std::runtime_error("Unsupported non-MLA EngineKVFormat");
      }
    }
  } else {
    // MLA path (k_or_v_size == 1)
    if (direction == TransferDirection::H2D) {
      switch (engine_kv_format) {
        case EngineKVFormat::NL_X_NB_BS_HS:
          LAUNCH_KERNEL_WITH_FORMAT(T, false, EngineKVFormat::NL_X_NB_BS_HS);
          break;
        case EngineKVFormat::NL_X_NBBS_ONE_HS:
          LAUNCH_KERNEL_WITH_FORMAT(T, false, EngineKVFormat::NL_X_NBBS_ONE_HS);
          break;
        default:
          throw std::runtime_error("Unsupported MLA EngineKVFormat");
      }
    } else {
      switch (engine_kv_format) {
        case EngineKVFormat::NL_X_NB_BS_HS:
          LAUNCH_KERNEL_WITH_FORMAT(T, true, EngineKVFormat::NL_X_NB_BS_HS);
          break;
        case EngineKVFormat::NL_X_NBBS_ONE_HS:
          LAUNCH_KERNEL_WITH_FORMAT(T, true, EngineKVFormat::NL_X_NBBS_ONE_HS);
          break;
        default:
          throw std::runtime_error("Unsupported MLA EngineKVFormat");
      }
    }
  }
}

#undef LAUNCH_KERNEL_WITH_FORMAT
#undef LAUNCH_FUSED_KV_KERNEL_WITH_FORMAT

// ---------------------------------------------------------------------------
// Public API: multi_layer_kv_transfer
// ---------------------------------------------------------------------------
void multi_layer_kv_transfer(
    torch::Tensor& key_value, const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping, const torch::Device& paged_memory_device,
    const int page_buffer_size, const TransferDirection direction,
    const EngineKVFormat engine_kv_format, const int block_size,
    const int head_size, const int skip_prefix_n_tokens) {
  // head_size is currently unused in the SYCL implementation; accepted to
  // keep ABI parity with the CUDA c_ops binding so callers can pass the
  // same kwargs to either backend.
  (void)head_size;
  int num_origin_elements = key_value.size(3);
  int copy_size = num_origin_elements * key_value.element_size();

#define LAUNCH_MULTI_LAYER_KV_TRANSFER(type)                          \
  do {                                                                \
    multi_layer_kv_transfer_templated<type>(                          \
        key_value, key_value_ptrs, slot_mapping, paged_memory_device, \
        page_buffer_size, direction, engine_kv_format, block_size,    \
        skip_prefix_n_tokens);                                        \
  } while (0)
  if (copy_size % 8 == 0) {
    LAUNCH_MULTI_LAYER_KV_TRANSFER(int64_t);
  } else if (copy_size % 4 == 0) {
    LAUNCH_MULTI_LAYER_KV_TRANSFER(int32_t);
  } else if (copy_size % 2 == 0) {
    LAUNCH_MULTI_LAYER_KV_TRANSFER(int16_t);
  } else {
    LAUNCH_MULTI_LAYER_KV_TRANSFER(int8_t);
  }
#undef LAUNCH_MULTI_LAYER_KV_TRANSFER
}

// ---------------------------------------------------------------------------
// Public API: multi_layer_kv_transfer_unilateral
// ---------------------------------------------------------------------------
void multi_layer_kv_transfer_unilateral(
    torch::Tensor& key_value, const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping, const torch::Device& paged_memory_device,
    const int page_buffer_size, const TransferDirection direction,
    const EngineKVFormat engine_kv_format) {
  const bool use_mla = ::is_mla(engine_kv_format);
  // MLA case collapses back to multi_layer_kv_transfer
  if (use_mla) {
    return multi_layer_kv_transfer(key_value, key_value_ptrs, slot_mapping,
                                   paged_memory_device, page_buffer_size,
                                   direction, engine_kv_format);
  }

  int64_t* key_value_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_value);
  int64_t** page_buffer_ptrs =
      get_kernel_ptr<int64_t*, const torch::Tensor>(key_value_ptrs);
  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int num_layers = key_value.size(1);
  int num_tokens = slot_mapping.size(0);
  int num_origin_elements = key_value.size(3);
  int elements_per_qword = 8 / key_value.element_size();
  int num_qwords = num_origin_elements / elements_per_qword;

  int k_or_v_size = 2;
  int kv_num_tokens = key_value.size(2);
  int wg_size = round_up_to_sg(std::min(num_qwords, MAX_WG_SIZE));

  const c10::OptionalDeviceGuard device_guard(paged_memory_device);
  sycl::queue& queue =
      c10::xpu::getCurrentXPUStream(paged_memory_device.index()).queue();

  if (direction == TransferDirection::H2D) {
    submit_multi_layer_unilateral_kernel<int64_t, false>(
        queue, key_value_ptr, page_buffer_ptrs, slot_mapping_ptr, num_qwords,
        num_tokens, num_layers, page_buffer_size, k_or_v_size, kv_num_tokens,
        wg_size);
  } else {
    submit_multi_layer_unilateral_kernel<int64_t, true>(
        queue, key_value_ptr, page_buffer_ptrs, slot_mapping_ptr, num_qwords,
        num_tokens, num_layers, page_buffer_size, k_or_v_size, kv_num_tokens,
        wg_size);
  }
}

// ---------------------------------------------------------------------------
// single_layer_kv_transfer — helper template
// ---------------------------------------------------------------------------
// USE_MLA and IS_D2H are template parameters so the compiler can
// dead-strip the unused branch.
template <bool USE_MLA, bool IS_D2H>
void single_layer_kv_transfer_impl(sycl::queue& queue, int64_t* lmc_ptr,
                                   int64_t* vllm_ptr, const int64_t* slot_ptr,
                                   int num_tokens, int n, int lmc_stride,
                                   int lmc_value_offset, int block_size,
                                   int vllm_block_key_stride_in_64bit,
                                   int vllm_value_offset, int num_heads,
                                   int head_size_in_64bit, int wg_size) {
  if (num_tokens <= 0) return;

  sycl::range<1> global_range(static_cast<size_t>(num_tokens) * wg_size);
  sycl::range<1> local_range(static_cast<size_t>(wg_size));

  queue.parallel_for(
      sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t token_idx = static_cast<int64_t>(item.get_group(0));
        const int64_t slot_idx = slot_ptr[token_idx];
        if (slot_idx < 0) return;

        const int64_t block_idx = slot_idx / block_size;
        const int64_t block_offset = slot_idx % block_size;

        const int tid = static_cast<int>(item.get_local_id(0));
        const int nthreads = static_cast<int>(item.get_local_range(0));

        for (int i = tid; i < n; i += nthreads) {
          const int64_t lmc_key_idx = token_idx * lmc_stride + i;
          const int head_idx = i / head_size_in_64bit;
          const int head_offset = i % head_size_in_64bit;
          const int64_t vllm_key_idx =
              block_idx * vllm_block_key_stride_in_64bit +
              block_offset * num_heads * head_size_in_64bit +
              head_idx * head_size_in_64bit + head_offset;

          if constexpr (IS_D2H) {
            lmc_ptr[lmc_key_idx] = vllm_ptr[vllm_key_idx];
            if constexpr (!USE_MLA) {
              const int64_t lmc_value_idx = lmc_key_idx + lmc_value_offset;
              const int64_t vllm_value_idx = vllm_key_idx + vllm_value_offset;
              lmc_ptr[lmc_value_idx] = vllm_ptr[vllm_value_idx];
            }
          } else {
            vllm_ptr[vllm_key_idx] = lmc_ptr[lmc_key_idx];
            if constexpr (!USE_MLA) {
              const int64_t lmc_value_idx = lmc_key_idx + lmc_value_offset;
              const int64_t vllm_value_idx = vllm_key_idx + vllm_value_offset;
              vllm_ptr[vllm_value_idx] = lmc_ptr[lmc_value_idx];
            }
          }
        }
      });
}

// ---------------------------------------------------------------------------
// Public API: single_layer_kv_transfer
// ---------------------------------------------------------------------------
void single_layer_kv_transfer(torch::Tensor& lmc_key_value_cache,
                              torch::Tensor& vllm_key_value_cache,
                              torch::Tensor& slot_mapping,
                              const TransferDirection direction,
                              const EngineKVFormat engine_kv_format,
                              const bool token_major) {
  int64_t* lmc_key_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(lmc_key_value_cache);
  int64_t* vllm_key_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(vllm_key_value_cache);
  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / vllm_key_value_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads;
  int head_size_in_64bit;
  int block_size;

  const bool use_mla = ::is_mla(engine_kv_format);

  if (use_mla) {
    num_heads = 1;
    block_size = vllm_key_value_cache.size(1);
    head_size_in_64bit = vllm_key_value_cache.size(2) / elements_per_entry;
  } else {
    num_heads = vllm_key_value_cache.size(3);
    head_size_in_64bit = vllm_key_value_cache.size(4) / elements_per_entry;
    block_size = vllm_key_value_cache.size(2);
  }

  int lmc_stride;
  int lmc_value_offset;
  if (use_mla) {
    lmc_stride = lmc_key_value_cache.stride(0) / elements_per_entry;
    lmc_value_offset = 0;
  } else if (token_major) {
    lmc_stride = lmc_key_value_cache.stride(0) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(1) / elements_per_entry;
  } else {
    lmc_stride = lmc_key_value_cache.stride(1) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(0) / elements_per_entry;
  }

  int vllm_block_key_stride_in_64bit;
  int vllm_value_offset;
  if (use_mla) {
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(0) / elements_per_entry;
    vllm_value_offset = 0;
  } else if (engine_kv_format == EngineKVFormat::NL_X_TWO_NB_BS_NH_HS) {
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(1) / elements_per_entry;
    vllm_value_offset = vllm_key_value_cache.stride(0) / elements_per_entry;
  } else if (engine_kv_format == EngineKVFormat::NL_X_NB_TWO_BS_NH_HS) {
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(0) / elements_per_entry;
    vllm_value_offset = vllm_key_value_cache.stride(1) / elements_per_entry;
  } else {
    throw std::runtime_error(
        "Unsupported non-MLA EngineKVFormat in single_layer_kv_transfer: " +
        std::to_string(static_cast<int>(engine_kv_format)));
  }

  int n = num_heads * head_size_in_64bit;
  int wg_size = round_up_to_sg(std::min(n, MAX_WG_SIZE));
  if (num_tokens <= 0) return;

  const c10::OptionalDeviceGuard device_guard(device_of(vllm_key_value_cache));
  sycl::queue& queue =
      c10::xpu::getCurrentXPUStream(vllm_key_value_cache.device().index())
          .queue();

  auto lmc_ptr = lmc_key_value_cache_ptr;
  auto vllm_ptr = vllm_key_value_cache_ptr;
  auto slot_ptr = slot_mapping_ptr;

  // Dispatch to 4 compile-time specialisations (USE_MLA × IS_D2H)
  // so the inner loop is branch-free.
  if (use_mla) {
    if (direction == TransferDirection::D2H)
      single_layer_kv_transfer_impl<true, true>(
          queue, lmc_ptr, vllm_ptr, slot_ptr, num_tokens, n, lmc_stride,
          lmc_value_offset, block_size, vllm_block_key_stride_in_64bit,
          vllm_value_offset, num_heads, head_size_in_64bit, wg_size);
    else
      single_layer_kv_transfer_impl<true, false>(
          queue, lmc_ptr, vllm_ptr, slot_ptr, num_tokens, n, lmc_stride,
          lmc_value_offset, block_size, vllm_block_key_stride_in_64bit,
          vllm_value_offset, num_heads, head_size_in_64bit, wg_size);
  } else {
    if (direction == TransferDirection::D2H)
      single_layer_kv_transfer_impl<false, true>(
          queue, lmc_ptr, vllm_ptr, slot_ptr, num_tokens, n, lmc_stride,
          lmc_value_offset, block_size, vllm_block_key_stride_in_64bit,
          vllm_value_offset, num_heads, head_size_in_64bit, wg_size);
    else
      single_layer_kv_transfer_impl<false, false>(
          queue, lmc_ptr, vllm_ptr, slot_ptr, num_tokens, n, lmc_stride,
          lmc_value_offset, block_size, vllm_block_key_stride_in_64bit,
          vllm_value_offset, num_heads, head_size_in_64bit, wg_size);
  }
}

// ---------------------------------------------------------------------------
// single_layer_kv_transfer_sgl — helper template
// ---------------------------------------------------------------------------
template <bool IS_D2H>
void single_layer_kv_transfer_sgl_impl(
    sycl::queue& queue, int64_t* lmc_ptr, int64_t* sgl_k_ptr,
    int64_t* sgl_v_ptr, const int64_t* slot_ptr, int num_tokens, int n,
    int lmc_stride, int lmc_value_offset, int block_stride_in_64bit,
    int block_size, int num_heads, int head_size_in_64bit, int wg_size) {
  if (num_tokens <= 0) return;

  sycl::range<1> global_range(static_cast<size_t>(num_tokens) * wg_size);
  sycl::range<1> local_range(static_cast<size_t>(wg_size));

  queue.parallel_for(
      sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t token_idx = static_cast<int64_t>(item.get_group(0));
        const int64_t slot_idx = slot_ptr[token_idx];
        if (slot_idx < 0) return;

        const int64_t block_idx = slot_idx / block_size;
        const int64_t block_offset = slot_idx % block_size;

        const int tid = static_cast<int>(item.get_local_id(0));
        const int nthreads = static_cast<int>(item.get_local_range(0));

        for (int i = tid; i < n; i += nthreads) {
          const int64_t lmc_key_idx = token_idx * lmc_stride + i;
          const int64_t lmc_value_idx = lmc_key_idx + lmc_value_offset;

          const int head_idx = i / head_size_in_64bit;
          const int head_offset = i % head_size_in_64bit;
          const int64_t sgl_kv_idx =
              block_idx * block_stride_in_64bit +
              block_offset * num_heads * head_size_in_64bit +
              head_idx * head_size_in_64bit + head_offset;

          if constexpr (IS_D2H) {
            lmc_ptr[lmc_key_idx] = sgl_k_ptr[sgl_kv_idx];
            lmc_ptr[lmc_value_idx] = sgl_v_ptr[sgl_kv_idx];
          } else {
            sgl_k_ptr[sgl_kv_idx] = lmc_ptr[lmc_key_idx];
            sgl_v_ptr[sgl_kv_idx] = lmc_ptr[lmc_value_idx];
          }
        }
      });
}

// ---------------------------------------------------------------------------
// Public API: single_layer_kv_transfer_sgl (SGLang)
// ---------------------------------------------------------------------------
void single_layer_kv_transfer_sgl(torch::Tensor& lmc_key_value_cache,
                                  torch::Tensor& sgl_key_cache,
                                  torch::Tensor& sgl_value_cache,
                                  torch::Tensor& slot_mapping,
                                  const TransferDirection direction,
                                  const bool token_major) {
  int64_t* lmc_key_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(lmc_key_value_cache);
  int64_t* sgl_key_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(sgl_key_cache);
  int64_t* sgl_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(sgl_value_cache);
  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / sgl_key_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads = sgl_key_cache.size(2);
  int head_size_in_64bit = sgl_key_cache.size(3) / elements_per_entry;
  int block_size = sgl_key_cache.size(1);

  int lmc_stride;
  int lmc_value_offset;
  if (token_major) {
    lmc_stride = lmc_key_value_cache.stride(0) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(1) / elements_per_entry;
  } else {
    lmc_stride = lmc_key_value_cache.stride(1) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(0) / elements_per_entry;
  }

  int block_stride_in_64bit = sgl_key_cache.stride(0) / elements_per_entry;
  TORCH_CHECK(sgl_key_cache.stride(0) == sgl_value_cache.stride(0));

  int n = num_heads * head_size_in_64bit;
  int wg_size = round_up_to_sg(std::min(n, MAX_WG_SIZE));
  if (num_tokens <= 0) return;

  const c10::OptionalDeviceGuard device_guard(device_of(sgl_key_cache));
  sycl::queue& queue =
      c10::xpu::getCurrentXPUStream(sgl_key_cache.device().index()).queue();

  if (direction == TransferDirection::D2H)
    single_layer_kv_transfer_sgl_impl<true>(
        queue, lmc_key_value_cache_ptr, sgl_key_cache_ptr, sgl_value_cache_ptr,
        slot_mapping_ptr, num_tokens, n, lmc_stride, lmc_value_offset,
        block_stride_in_64bit, block_size, num_heads, head_size_in_64bit,
        wg_size);
  else
    single_layer_kv_transfer_sgl_impl<false>(
        queue, lmc_key_value_cache_ptr, sgl_key_cache_ptr, sgl_value_cache_ptr,
        slot_mapping_ptr, num_tokens, n, lmc_stride, lmc_value_offset,
        block_stride_in_64bit, block_size, num_heads, head_size_in_64bit,
        wg_size);
}

// ---------------------------------------------------------------------------
// Public API: load_and_reshape_flash (deprecated -- unit tests only)
// ---------------------------------------------------------------------------
void load_and_reshape_flash(torch::Tensor& key_value, torch::Tensor& key_cache,
                            torch::Tensor& value_cache,
                            torch::Tensor& slot_mapping, const int layer_idx) {
  int64_t* key_value_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_value);
  int64_t* key_cache_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_cache);
  int64_t* value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(value_cache);
  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / key_cache.element_size();
  int num_tokens = slot_mapping.size(0);
  int num_heads = key_cache.size(2);
  int head_size_in_64bit = key_cache.size(3) / elements_per_entry;
  int block_size = key_cache.size(1);
  int key_value_stride = key_value.stride(2) / elements_per_entry;
  int num_layers = key_value.size(1);
  int key_layer_offset = layer_idx * key_value.stride(1) / elements_per_entry;
  int value_layer_offset =
      (layer_idx + num_layers) * key_value.stride(1) / elements_per_entry;
  int block_stride_in_64bit = key_cache.stride(0) / elements_per_entry;
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  int n = num_heads * head_size_in_64bit;
  int wg_size = round_up_to_sg(std::min(n, MAX_WG_SIZE));
  if (num_tokens <= 0) return;

  sycl::range<1> global_range(static_cast<size_t>(num_tokens) * wg_size);
  sycl::range<1> local_range(static_cast<size_t>(wg_size));

  const c10::OptionalDeviceGuard device_guard(device_of(key_cache));
  sycl::queue& queue =
      c10::xpu::getCurrentXPUStream(key_cache.device().index()).queue();

  auto kv_ptr = key_value_ptr;
  auto k_ptr = key_cache_ptr;
  auto v_ptr = value_cache_ptr;
  auto slot_ptr = slot_mapping_ptr;

  queue.parallel_for(
      sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t token_idx = static_cast<int64_t>(item.get_group(0));
        const int64_t slot_idx = slot_ptr[token_idx];
        if (slot_idx < 0) return;

        const int64_t blk_idx = slot_idx / block_size;
        const int64_t blk_off = slot_idx % block_size;

        const int tid = static_cast<int>(item.get_local_id(0));
        const int nthreads = static_cast<int>(item.get_local_range(0));

        for (int i = tid; i < n; i += nthreads) {
          const int64_t tgt_key_idx =
              key_layer_offset + token_idx * key_value_stride + i;
          const int64_t tgt_value_idx =
              value_layer_offset + token_idx * key_value_stride + i;

          const int head_idx = i / head_size_in_64bit;
          const int head_offset = i % head_size_in_64bit;
          const int64_t src_kv_idx = blk_idx * block_stride_in_64bit +
                                     blk_off * num_heads * head_size_in_64bit +
                                     head_idx * head_size_in_64bit +
                                     head_offset;

          kv_ptr[tgt_key_idx] = k_ptr[src_kv_idx];
          kv_ptr[tgt_value_idx] = v_ptr[src_kv_idx];
        }
      });
}

// ---------------------------------------------------------------------------
// Public API: reshape_and_cache_back_flash (deprecated -- unit
// tests only)
// ---------------------------------------------------------------------------
void reshape_and_cache_back_flash(torch::Tensor& key_value,
                                  torch::Tensor& key_cache,
                                  torch::Tensor& value_cache,
                                  torch::Tensor& slot_mapping,
                                  const int layer_idx) {
  int64_t* key_cache_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_cache);
  int64_t* value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(value_cache);
  int64_t* key_value_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_value);
  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / key_cache.element_size();
  int num_tokens = slot_mapping.size(0);
  int num_heads = key_cache.size(2);
  int head_size_in_64bit = key_cache.size(3) / elements_per_entry;
  int block_size = key_cache.size(1);
  int key_value_stride = key_value.stride(2) / elements_per_entry;
  int num_layers = key_value.size(1);
  int key_layer_offset = layer_idx * key_value.stride(1) / elements_per_entry;
  int value_layer_offset =
      (layer_idx + num_layers) * key_value.stride(1) / elements_per_entry;
  int block_stride_in_64bit = key_cache.stride(0) / elements_per_entry;
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  int n = num_heads * head_size_in_64bit;
  int wg_size = round_up_to_sg(std::min(n, MAX_WG_SIZE));
  if (num_tokens <= 0) return;

  sycl::range<1> global_range(static_cast<size_t>(num_tokens) * wg_size);
  sycl::range<1> local_range(static_cast<size_t>(wg_size));

  const c10::OptionalDeviceGuard device_guard(device_of(key_cache));
  sycl::queue& queue =
      c10::xpu::getCurrentXPUStream(key_cache.device().index()).queue();

  auto kv_ptr = key_value_ptr;
  auto k_ptr = key_cache_ptr;
  auto v_ptr = value_cache_ptr;
  auto slot_ptr = slot_mapping_ptr;

  queue.parallel_for(
      sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t token_idx = static_cast<int64_t>(item.get_group(0));
        const int64_t slot_idx = slot_ptr[token_idx];
        if (slot_idx < 0) return;

        const int64_t blk_idx = slot_idx / block_size;
        const int64_t blk_off = slot_idx % block_size;

        const int tid = static_cast<int>(item.get_local_id(0));
        const int nthreads = static_cast<int>(item.get_local_range(0));

        for (int i = tid; i < n; i += nthreads) {
          const int64_t tgt_key_idx =
              key_layer_offset + token_idx * key_value_stride + i;
          const int64_t tgt_value_idx =
              value_layer_offset + token_idx * key_value_stride + i;

          const int head_idx = i / head_size_in_64bit;
          const int head_offset = i % head_size_in_64bit;
          const int64_t src_kv_idx = blk_idx * block_stride_in_64bit +
                                     blk_off * num_heads * head_size_in_64bit +
                                     head_idx * head_size_in_64bit +
                                     head_offset;

          k_ptr[src_kv_idx] = kv_ptr[tgt_key_idx];
          v_ptr[src_kv_idx] = kv_ptr[tgt_value_idx];
        }
      });
}

// ---------------------------------------------------------------------------
// Public API: lmcache_memcpy_async
// ---------------------------------------------------------------------------
void lmcache_memcpy_async(uintptr_t dest, uintptr_t src, size_t nbytes,
                          TransferDirection direction,
                          size_t host_buffer_offset,
                          size_t host_buffer_alignments) {
  TORCH_CHECK((host_buffer_alignments & (host_buffer_alignments - 1)) == 0,
              "host_buffer_alignments must be power of two");

  // SYCL USM memcpy infers direction from pointer allocation types;
  // the `direction` parameter is retained only for API compatibility.
  (void)direction;

  sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();

  size_t offset = 0;
  const size_t mask = host_buffer_alignments - 1;

  while (offset < nbytes) {
    size_t current_src = src + offset;
    size_t current_dest = dest + offset;

    size_t aligned_area_end =
        ((offset + host_buffer_offset) & ~mask) + host_buffer_alignments;
    size_t real_end = std::min(host_buffer_offset + nbytes, aligned_area_end);
    size_t max_nbytes = real_end - offset - host_buffer_offset;

    // USM memcpy is direction-agnostic.
    queue.memcpy(reinterpret_cast<void*>(current_dest),
                 reinterpret_cast<const void*>(current_src), max_nbytes);

    offset += max_nbytes;
  }
}

// ---------------------------------------------------------------------------
// Pinned host allocation (SYCL/XPU analog of the CUDA cudaHostAlloc path in
// csrc/mem_alloc.cpp). LMCache local_cpu backend expects a device-accessible
// (USM host) buffer; without this the XPU build silently fell back to pageable
// host memory, dropping D2H store throughput ~20x. sycl::malloc_host bound to
// the current XPU device context gives true USM-pinned host memory.
//
// IMPORTANT: We use device 0's context for both alloc and free to ensure
// context consistency. USM host memory is accessible from all devices, but
// sycl::free() requires the same context used during allocation. Using device
// 0 consistently avoids context mismatch when the current device changes
// between allocation and deallocation.
// ---------------------------------------------------------------------------
uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags) {
  // Use device 0 context consistently for host memory allocation.
  // DeviceGuard ensures we restore the original device after getting context.
  const c10::DeviceGuard device_guard(c10::Device(c10::kXPU, 0));
  void* ptr = sycl::malloc_host(size, c10::xpu::get_device_context());
  TORCH_CHECK(ptr != nullptr, "sycl::malloc_host failed for ", size,
              " bytes (XPU pinned host)");
  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_ptr(uintptr_t ptr) {
  // Use device 0 context consistently (must match alloc_pinned_ptr).
  const c10::DeviceGuard device_guard(c10::Device(c10::kXPU, 0));
  sycl::free(reinterpret_cast<void*>(ptr), c10::xpu::get_device_context());
}
