// SPDX-License-Identifier: Apache-2.0

#include "mp_mem_kernels.cuh"

namespace {

/**
 * Key logic in the kernel implementation:
 * 1. Each thread block is for (BS, NH, HS) part (i.e., a single block in the
 * paged buffer)
 * 2. Within a thread block, each warp is for a single head. Number of warps
 * in a thread block is equal to the number of heads (NH).
 * 3. Within a thread block, we do the loop over the BS (i.e., number of tokens
 * in the block) dimension.
 * 4. The grid will take over (2, NB, NL) dimensions. No matter what the actual
 * layout in memory is, we will calculate the global offset for the start of the
 * block
 * 5. For LMCache, we assume it is always using 2LTD layout, e.g.,
 * [2, L, 256, NH * HS], where 256 means that 256 tokens
 */

/**
 * Calculate the offset for the current block in the paged buffer
 */
template <typename ScalarType, EngineKVFormat format>
__device__ inline size_t calculate_engine_global_offset(
    const int k_or_v, const int engine_block_idx, const int layer_idx,
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_block = shape_desc.scalars_per_block<ScalarType>();
  if constexpr (format == EngineKVFormat::NB_NL_TWO_BS_NH_HS) {
    // Cross-layer: single tensor [NB, NL, 2, BS, NH, HS]
    return k_or_v * scalars_per_block +
           layer_idx * shape_desc.kv_size * scalars_per_block +
           engine_block_idx * shape_desc.kv_size * scalars_per_block *
               shape_desc.nl;
  } else if constexpr (format == EngineKVFormat::NL_X_TWO_NB_BS_NH_HS) {
    // Normal: L tensors [2, NB, BS, NH, HS]
    return engine_block_idx * scalars_per_block +
           k_or_v * shape_desc.nb * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::NL_X_TWO_NB_NH_BS_HS) {
    // Normal HND: L tensors [2, NB, NH, BS, HS]
    return engine_block_idx * scalars_per_block +
           k_or_v * shape_desc.nb * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::NL_X_NB_TWO_BS_NH_HS) {
    // Flash Infer: L tensors [NB, 2, BS, NH, HS]
    return engine_block_idx * shape_desc.kv_size * scalars_per_block +
           k_or_v * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::NL_X_NB_TWO_NH_BS_HS) {
    // Flash Infer HND: L tensors [NB, 2, NH, BS, HS]
    return engine_block_idx * shape_desc.kv_size * scalars_per_block +
           k_or_v * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::NL_X_NB_BS_HS ||
                       format == EngineKVFormat::NL_X_NB_BSV_BSS) {
    // MLA: L tensors [NB, BS, HS]; blocked-scale shares the block base,
    // only its within-block layout differs (handled in the transfer body).
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::TWO_X_NL_X_NBBS_NH_HS) {
    // SGLang MHA (in-process): 2L tensors [NBBS, NH, HS]
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::TWO_X_NL_X_NB_BS_NH_HS) {
    // SGLang MHA (MP daemon): 2L tensors [NB, BS, NH, HS]
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::NL_X_NBBS_ONE_HS) {
    // SGLang MLA: L tensors [NBBS, 1, HS]
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::NL_X_NB_NH_BS_TWO_HS) {
    // Fused-K/V HND: L tensors [NB, NH, BS, 2*HS], handled like
    // NL_X_TWO_NB_NH_BS_HS but with an empty K/V axis: the desc carries
    // kv_size == 1 and hs == 2 * head_size, so k_or_v is always 0 and each
    // head copy moves the packed K+V pair.
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::NL_X_NB_BS_NH_TWO_HS) {
    // Fused-K/V NHD: L tensors [NB, BS, NH, 2*HS]; same empty K/V axis as
    // NL_X_NB_NH_BS_TWO_HS (kv_size == 1, hs == 2 * head_size), tokens
    // before heads within a block.
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::NL_X_NB_NH_BS_CS) {
    // Content-size HND: L tensors [NB, NH, BS, CS]; same empty K/V axis as
    // NL_X_NB_NH_BS_TWO_HS (kv_size == 1, hs == content_size).
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::NL_X_NB_BS_NH_CS) {
    // Content-size NHD: L tensors [NB, BS, NH, CS]; same empty K/V axis as
    // NL_X_NB_NH_BS_CS, tokens before heads within a block.
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::NB_NL_TWO_NH_BS_HS) {
    // TRT-LLM cross-layer HND: single tensor [NB, NL, 2, NH, BS, HS]
    // same block-level strides as NB_NL_TWO_BS_NH_HS
    return k_or_v * scalars_per_block +
           layer_idx * shape_desc.kv_size * scalars_per_block +
           engine_block_idx * shape_desc.kv_size * scalars_per_block *
               shape_desc.nl;
  }
}

/**
 * Calculate the offset for the current token against the start
 * of the block in the paged buffer.
 */
template <typename ScalarType, EngineKVFormat format>
__device__ inline size_t calculate_engine_local_offset(
    const int token_offset, const int head_idx,
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_head = shape_desc.scalars_per_head<ScalarType>();
  size_t scalars_per_token = shape_desc.scalars_per_token<ScalarType>();
  if constexpr (format == EngineKVFormat::NB_NL_TWO_NH_BS_HS ||
                format == EngineKVFormat::NL_X_TWO_NB_NH_BS_HS ||
                format == EngineKVFormat::NL_X_NB_TWO_NH_BS_HS ||
                format == EngineKVFormat::NL_X_NB_NH_BS_TWO_HS ||
                format == EngineKVFormat::NL_X_NB_NH_BS_CS) {
    // HND: [NH, BS, HS] — heads are outermost within a block
    size_t scalars_per_head_block =
        shape_desc.bs * scalars_per_head;  // BS * HS
    return head_idx * scalars_per_head_block + token_offset * scalars_per_head;
  } else {
    // NHD: [BS, NH, HS] — tokens are outermost within a block
    return head_idx * scalars_per_head + token_offset * scalars_per_token;
  }
}

/**
 * Calculate the global offset for the current `block` in the LMCache object.
 * The `block` here is the memory region corresponding to a thread-block.
 */
template <typename ScalarType, EngineKVFormat format>
__device__ inline size_t calculate_lmcache_global_offset(
    const int k_or_v,
    const int
        token_offset_in_lmcache_object,  // 0~255 if LMCache chunk size is 256
    const int layer_idx,
    const int lmcache_chunk_size,  // e.g., 256
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_token = shape_desc.scalars_per_token<ScalarType>();
  // LMCache is using 2LTD all the times
  return token_offset_in_lmcache_object * scalars_per_token +
         layer_idx * lmcache_chunk_size * scalars_per_token +
         k_or_v * shape_desc.nl * lmcache_chunk_size * scalars_per_token;
}

/**
 * Calculate the local offset for the current token against the start of the
 * block in the LMCache object.
 */
template <typename ScalarType, EngineKVFormat format>
__device__ inline size_t calculate_lmcache_local_offset(
    const int token_offset, const int head_idx,
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_head = shape_desc.scalars_per_head<ScalarType>();
  size_t scalars_per_token = shape_desc.scalars_per_token<ScalarType>();
  return head_idx * scalars_per_head + token_offset * scalars_per_token;
}

__device__ inline uint4 ld_cs(const uint4* addr) {
#ifdef __CUDA_ARCH__
  uint4 val;
  asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr));
  return val;
#else
  return *addr;
#endif
}

__device__ inline void st_cs(uint4* addr, uint4 val) {
#ifdef __CUDA_ARCH__
  asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(addr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#else
  *addr = val;
#endif
}

template <typename ScalarType>
__device__ inline void warp_copy(ScalarType* __restrict__ dst,
                                 const ScalarType* __restrict__ src,
                                 size_t num_elements) {
  int idx = threadIdx.x;
  int stride = blockDim.x;
  if constexpr (std::is_same_v<ScalarType, uint4>) {
    for (size_t i = idx; i < num_elements; i += stride) {
      st_cs(dst + i, ld_cs(src + i));
    }
  } else {
    for (size_t i = idx; i < num_elements; i += stride) {
      dst[i] = src[i];
    }
  }
}

template <typename ScalarType, bool lmcache_to_engine, EngineKVFormat format>
__device__ void multi_layer_block_transfer_single_block(
    ScalarType* __restrict__ lmcache_object,
    ScalarType** __restrict__ paged_buffer_ptrs, const int engine_block_idx,
    const int offset_in_lmcache_block, const PageBufferShapeDesc shape_desc,
    const int lmcache_chunk_size  // e.g., 256, used to calculate global offset
                                  // in LMCache object
) {
  const int head_idx = threadIdx.y;
  const int k_or_v = blockIdx.x;
  const int layer_idx = blockIdx.z;

  const size_t engine_global_offset =
      calculate_engine_global_offset<ScalarType, format>(
          k_or_v, engine_block_idx, layer_idx, shape_desc);
  const size_t lmcache_global_offset =
      calculate_lmcache_global_offset<ScalarType, format>(
          k_or_v, offset_in_lmcache_block, layer_idx, lmcache_chunk_size,
          shape_desc);
  ScalarType* paged_buffer_layer_ptr;
  if constexpr (format == EngineKVFormat::NB_NL_TWO_BS_NH_HS ||
                format == EngineKVFormat::NB_NL_TWO_NH_BS_HS) {
    paged_buffer_layer_ptr = (ScalarType*)paged_buffer_ptrs[0];
  } else if constexpr (format == EngineKVFormat::TWO_X_NL_X_NBBS_NH_HS) {
    // SGLang MHA (in-process): ptrs[0..NL-1] = K per layer, ptrs[NL..2NL-1] = V
    // per layer
    paged_buffer_layer_ptr =
        (ScalarType*)paged_buffer_ptrs[k_or_v * shape_desc.nl + layer_idx];
  } else if constexpr (format == EngineKVFormat::TWO_X_NL_X_NB_BS_NH_HS) {
    // SGLang MHA (MP daemon): ptrs[0..NL-1] = K per layer, ptrs[NL..2NL-1] = V
    // per layer
    paged_buffer_layer_ptr =
        (ScalarType*)paged_buffer_ptrs[k_or_v * shape_desc.nl + layer_idx];
  } else {
    paged_buffer_layer_ptr = (ScalarType*)paged_buffer_ptrs[layer_idx];
  }

  if constexpr (format == EngineKVFormat::NL_X_NB_BSV_BSS) {
    // Blocked page [BSxvals][BSxscales] vs token-major chunk row: two
    // copies per token (host pins <=4B units, so scale is whole units).
    const size_t spt = shape_desc.scalars_per_token<ScalarType>();
    const size_t scale_units = 4 / sizeof(ScalarType);
    const size_t val_units = spt - scale_units;
    for (int t = 0; t < shape_desc.bs; ++t) {
      ScalarType* eng_vals =
          paged_buffer_layer_ptr + engine_global_offset + t * val_units;
      ScalarType* eng_scale = paged_buffer_layer_ptr + engine_global_offset +
                              shape_desc.bs * val_units + t * scale_units;
      ScalarType* lmc_row = lmcache_object + lmcache_global_offset + t * spt;
      if constexpr (lmcache_to_engine) {
        warp_copy<ScalarType>(eng_vals, lmc_row, val_units);
        warp_copy<ScalarType>(eng_scale, lmc_row + val_units, scale_units);
      } else {
        warp_copy<ScalarType>(lmc_row, eng_vals, val_units);
        warp_copy<ScalarType>(lmc_row + val_units, eng_scale, scale_units);
      }
    }
    return;
  }

  for (int token_offset = 0; token_offset < shape_desc.bs; ++token_offset) {
    const size_t engine_local_offset =
        calculate_engine_local_offset<ScalarType, format>(token_offset,
                                                          head_idx, shape_desc);
    const size_t lmcache_local_offset =
        calculate_lmcache_local_offset<ScalarType, format>(
            token_offset, head_idx, shape_desc);
    ScalarType* engine_ptr =
        paged_buffer_layer_ptr + engine_global_offset + engine_local_offset;
    ScalarType* lmcache_ptr =
        lmcache_object + lmcache_global_offset + lmcache_local_offset;
    if constexpr (lmcache_to_engine) {
      warp_copy<ScalarType>(engine_ptr, lmcache_ptr,
                            shape_desc.scalars_per_head<ScalarType>());
    } else {
      warp_copy<ScalarType>(lmcache_ptr, engine_ptr,
                            shape_desc.scalars_per_head<ScalarType>());
    }
  }
}

template <typename ScalarType, bool lmcache_to_engine, EngineKVFormat format>
__global__ void multi_layer_block_transfer_kernel(
    MemoryObj4<ScalarType> lmcache_objects,
    ScalarType** __restrict__ paged_buffer_ptrs,
    const int64_t* engine_block_ids,
    const int num_blocks_per_object,  // e.g. 16 for lmcache chunk size =
                                      // 256 and block size = 16
    const PageBufferShapeDesc shape_desc,
    const int lmcache_chunk_size,  // e.g., 256, used to calculate global offset
                                   // in LMCache object
    const int skip_prefix_n_blocks) {
  // blockIdx.y spans all blocks across all objects (total_blocks).
  // Derive which object and local block index from the flat index.
  const int flat_block_idx = blockIdx.y;
  if (flat_block_idx < skip_prefix_n_blocks) {
    return;
  }
  const int obj_idx = flat_block_idx / num_blocks_per_object;
  const int block_idx_in_object = flat_block_idx % num_blocks_per_object;

  const int engine_block_idx = engine_block_ids[flat_block_idx];
  multi_layer_block_transfer_single_block<ScalarType, lmcache_to_engine,
                                          format>(
      lmcache_objects.objects[obj_idx], paged_buffer_ptrs, engine_block_idx,
      block_idx_in_object * shape_desc.bs,  // offset in LMCache object
      shape_desc, lmcache_chunk_size);
}

#define LAUNCH_KERNEL(DIRECTION, FORMAT)                                 \
  multi_layer_block_transfer_kernel<ScalarType, DIRECTION, FORMAT>       \
      <<<grid, block, 0, stream>>>(lmcache_obj4, paged_buffer_ptrs,      \
                                   block_ids_ptr, num_blocks_per_object, \
                                   shape_desc, lmcache_chunk_size,       \
                                   skip_prefix_n_blocks);                \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define DISPATCH_FORMAT(DIRECTION)                                      \
  switch (engine_kv_format) {                                           \
    case EngineKVFormat::NB_NL_TWO_BS_NH_HS:                            \
      LAUNCH_KERNEL(DIRECTION, EngineKVFormat::NB_NL_TWO_BS_NH_HS);     \
      break;                                                            \
    case EngineKVFormat::NL_X_TWO_NB_BS_NH_HS:                          \
      LAUNCH_KERNEL(DIRECTION, EngineKVFormat::NL_X_TWO_NB_BS_NH_HS);   \
      break;                                                            \
    case EngineKVFormat::NL_X_TWO_NB_NH_BS_HS:                          \
      LAUNCH_KERNEL(DIRECTION, EngineKVFormat::NL_X_TWO_NB_NH_BS_HS);   \
      break;                                                            \
    case EngineKVFormat::NL_X_NB_TWO_BS_NH_HS:                          \
      LAUNCH_KERNEL(DIRECTION, EngineKVFormat::NL_X_NB_TWO_BS_NH_HS);   \
      break;                                                            \
    case EngineKVFormat::NL_X_NB_TWO_NH_BS_HS:                          \
      LAUNCH_KERNEL(DIRECTION, EngineKVFormat::NL_X_NB_TWO_NH_BS_HS);   \
      break;                                                            \
    case EngineKVFormat::NL_X_NB_BS_HS:                                 \
      LAUNCH_KERNEL(DIRECTION, EngineKVFormat::NL_X_NB_BS_HS);          \
      break;                                                            \
    case EngineKVFormat::TWO_X_NL_X_NBBS_NH_HS:                         \
      LAUNCH_KERNEL(DIRECTION, EngineKVFormat::TWO_X_NL_X_NBBS_NH_HS);  \
      break;                                                            \
    case EngineKVFormat::TWO_X_NL_X_NB_BS_NH_HS:                        \
      LAUNCH_KERNEL(DIRECTION, EngineKVFormat::TWO_X_NL_X_NB_BS_NH_HS); \
      break;                                                            \
    case EngineKVFormat::NL_X_NBBS_ONE_HS:                              \
      LAUNCH_KERNEL(DIRECTION, EngineKVFormat::NL_X_NBBS_ONE_HS);       \
      break;                                                            \
    case EngineKVFormat::NB_NL_TWO_NH_BS_HS:                            \
      LAUNCH_KERNEL(DIRECTION, EngineKVFormat::NB_NL_TWO_NH_BS_HS);     \
      break;                                                            \
    case EngineKVFormat::NL_X_NB_NH_BS_TWO_HS:                          \
      LAUNCH_KERNEL(DIRECTION, EngineKVFormat::NL_X_NB_NH_BS_TWO_HS);   \
      break;                                                            \
    case EngineKVFormat::NL_X_NB_BS_NH_TWO_HS:                          \
      LAUNCH_KERNEL(DIRECTION, EngineKVFormat::NL_X_NB_BS_NH_TWO_HS);   \
      break;                                                            \
    case EngineKVFormat::NL_X_NB_NH_BS_CS:                              \
      LAUNCH_KERNEL(DIRECTION, EngineKVFormat::NL_X_NB_NH_BS_CS);       \
      break;                                                            \
    case EngineKVFormat::NL_X_NB_BS_NH_CS:                              \
      LAUNCH_KERNEL(DIRECTION, EngineKVFormat::NL_X_NB_BS_NH_CS);       \
      break;                                                            \
    case EngineKVFormat::NL_X_NB_BSV_BSS:                               \
      LAUNCH_KERNEL(DIRECTION, EngineKVFormat::NL_X_NB_BSV_BSS);        \
      break;                                                            \
    default:                                                            \
      TORCH_CHECK(false, "Unsupported EngineKVFormat: ",                \
                  static_cast<int>(engine_kv_format));                  \
  }

template <typename ScalarType>
void multi_layer_block_kv_transfer_templated(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    EngineKVFormat engine_kv_format, int skip_prefix_n_blocks) {
  // --- Validation ---
  int num_objects = static_cast<int>(lmcache_objects_ptrs.size());
  TORCH_CHECK(num_objects >= 1 && num_objects <= 4,
              "Expected 1-4 LMCache objects, got ", num_objects);

  int total_blocks = block_ids.size(0);
  TORCH_CHECK(total_blocks % num_objects == 0, "block_ids length (",
              total_blocks, ") must be divisible by num_objects (", num_objects,
              ")");
  int num_blocks_per_object = total_blocks / num_objects;

  TORCH_CHECK(num_blocks_per_object * shape_desc.bs == lmcache_chunk_size,
              "blocks_per_object * block_size (",
              num_blocks_per_object * shape_desc.bs,
              ") must equal lmcache_chunk_size (", lmcache_chunk_size, ")");

  // --- Build MemoryObj4 ---
  MemoryObj4<ScalarType> lmcache_obj4;
  lmcache_obj4.num_objects = num_objects;
  for (int i = 0; i < 4; ++i) {
    lmcache_obj4.objects[i] =
        (i < num_objects)
            ? reinterpret_cast<ScalarType*>(lmcache_objects_ptrs[i])
            : nullptr;
  }

  // --- Build paged buffer pointer array ---
  ScalarType** paged_buffer_ptrs =
      reinterpret_cast<ScalarType**>(paged_buffer_ptrs_tensor.data_ptr());

  const at::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // --- block_ids is a GPU int64 tensor, read directly ---
  const int64_t* block_ids_ptr = block_ids.data_ptr<int64_t>();

  // --- Grid and block dimensions ---
  int elements_per_head = shape_desc.hs * shape_desc.element_size /
                          static_cast<int>(sizeof(ScalarType));
  int thread_dim_x = std::min(elements_per_head, 32);
  int thread_dim_y = shape_desc.nh;

  dim3 block(thread_dim_x, thread_dim_y);
  dim3 grid(shape_desc.kv_size, total_blocks, shape_desc.nl);

  if (direction == TransferDirection::H2D) {
    DISPATCH_FORMAT(true);
  } else {
    DISPATCH_FORMAT(false);
  }
}

#undef DISPATCH_FORMAT
#undef LAUNCH_KERNEL

}  // namespace

#define LAUNCH_TEMPLATED(type)                                             \
  do {                                                                     \
    multi_layer_block_kv_transfer_templated<type>(                         \
        paged_buffer_ptrs_tensor, lmcache_objects_ptrs, block_ids, device, \
        direction, shape_desc, lmcache_chunk_size, engine_kv_format,       \
        skip_prefix_n_blocks);                                             \
  } while (0)

void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    EngineKVFormat engine_kv_format, int skip_prefix_n_blocks) {
  int head_bytes = shape_desc.hs * shape_desc.element_size;
  TORCH_CHECK(head_bytes % sizeof(uint16_t) == 0, "head_size * element_size (",
              head_bytes, ") must be divisible by 2 for vectorized access");

  if (engine_kv_format == EngineKVFormat::NL_X_NB_BSV_BSS) {
    // Blocked-scale indexer cache: the per-token fp32 scale must be a whole
    // number of transfer units, so pin 4-byte units regardless of row width.
    TORCH_CHECK(head_bytes % sizeof(uint32_t) == 0,
                "NL_X_NB_BSV_BSS row bytes (", head_bytes,
                ") must be divisible by 4");
    LAUNCH_TEMPLATED(uint32_t);
    return;
  }

  if (head_bytes % sizeof(uint4) == 0) {
    LAUNCH_TEMPLATED(uint4);  // 16 bytes per copy
  } else if (head_bytes % sizeof(uint32_t) == 0) {
    LAUNCH_TEMPLATED(uint32_t);  // 4 bytes per copy
  } else {
    LAUNCH_TEMPLATED(uint16_t);  // 2 bytes per copy (minimum granularity)
  }
}

#undef LAUNCH_TEMPLATED

void execute_object_group_transfer(
    TransferDirection direction, const torch::Device& device,
    size_t host_buffer_alignment,
    const std::vector<KernelGroupSpec>& kernel_group_specs,
    const std::vector<BatchStep>& batch_steps) {
  // Set the device guard once for the whole plan so every staging copy and
  // kernel launch below is enqueued on this device's current stream, in order.
  const at::cuda::OptionalCUDAGuard device_guard(device);
  const bool is_h2d = (direction == TransferDirection::H2D);
  const auto int64_opts = at::TensorOptions().dtype(at::kLong).device(device);

  const auto do_staging = [&](const std::vector<StagingCopy>& staging) {
    for (const auto& copy : staging) {
      lmcache_memcpy_async(copy.dest, copy.src, copy.nbytes, direction,
                           copy.host_offset, host_buffer_alignment);
    }
  };

  for (const auto& step : batch_steps) {
    // H2D stages CPU->GPU temp buffers before the kernel reads them; D2H stages
    // GPU->CPU after the kernel writes them. The per-step ordering must be
    // preserved because temp buffers are reused across steps.
    if (is_h2d) {
      do_staging(step.staging);
    }
    for (const auto& launch : step.launches) {
      TORCH_CHECK(
          launch.group_idx >= 0 &&
              launch.group_idx < static_cast<int>(kernel_group_specs.size()),
          "LaunchVar.group_idx out of range: ", launch.group_idx);
      const KernelGroupSpec& group = kernel_group_specs[launch.group_idx];
      TORCH_CHECK(launch.num_objects >= 1 &&
                      launch.num_objects <=
                          static_cast<int>(group.lmcache_objects_ptrs.size()),
                  "LaunchVar.num_objects (", launch.num_objects,
                  ") exceeds available temp buffers (",
                  group.lmcache_objects_ptrs.size(), ")");
      // Bounds-check the block_ids slice before the kernel dereferences it on
      // device: an out-of-range offset/length would otherwise be a silent
      // out-of-bounds device read (CUDA fault or garbage), not a clean error.
      TORCH_CHECK(launch.block_ids_offset >= 0,
                  "LaunchVar.block_ids_offset must be non-negative, got ",
                  launch.block_ids_offset);
      TORCH_CHECK(launch.total_blocks >= 0,
                  "LaunchVar.total_blocks must be non-negative, got ",
                  launch.total_blocks);
      TORCH_CHECK(launch.block_ids_offset + launch.total_blocks <=
                      group.block_ids_capacity,
                  "LaunchVar block_ids slice [", launch.block_ids_offset, ", ",
                  launch.block_ids_offset + launch.total_blocks,
                  ") exceeds block_ids capacity ", group.block_ids_capacity);

      // Wrap the plan's pre-resolved raw device addresses as non-owning tensor
      // views so we can reuse the existing multi_layer_block_kv_transfer entry
      // point without touching any of its code. The backing storage is owned by
      // the caller's tensors (kept alive for the duration of this call); these
      // views only carry the pointer/shape each launch needs. Downstream only
      // reads paged_buffer_ptrs_tensor.data_ptr() and block_ids.{data_ptr,
      // size(0)}.
      const uintptr_t block_ids_addr =
          group.block_ids_base +
          static_cast<uintptr_t>(launch.block_ids_offset) * sizeof(int64_t);
      const at::Tensor paged_buffer_ptrs_tensor = at::from_blob(
          reinterpret_cast<void*>(group.paged_buffer_ptrs), {1}, int64_opts);
      const at::Tensor block_ids = at::from_blob(
          reinterpret_cast<void*>(block_ids_addr),
          {static_cast<int64_t>(launch.total_blocks)}, int64_opts);
      std::vector<int64_t> lmcache_objects_ptrs(
          group.lmcache_objects_ptrs.begin(),
          group.lmcache_objects_ptrs.begin() + launch.num_objects);

      multi_layer_block_kv_transfer(
          paged_buffer_ptrs_tensor, std::move(lmcache_objects_ptrs), block_ids,
          device, direction, group.shape_desc, group.lmcache_chunk_size,
          group.engine_kv_format, launch.skip_prefix_n_blocks);
    }
    if (!is_h2d) {
      do_staging(step.staging);
    }
  }
}
