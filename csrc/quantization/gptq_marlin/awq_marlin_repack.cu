#include "marlin.cuh"

#include "core/registration.h"

namespace marlin {

template <int const num_threads, int const num_bits>
__global__ void awq_marlin_repack_kernel(
    uint32_t const* __restrict__ b_q_weight_ptr, uint32_t* __restrict__ out_ptr,
    int size_k, int size_n) {
  constexpr int pack_factor = 32 / num_bits;

  int k_tiles = size_k / tile_k_size;
  int n_tiles = size_n / tile_n_size;
  int block_k_tiles = div_ceil(k_tiles, gridDim.x);

  int start_k_tile = blockIdx.x * block_k_tiles;
  if (start_k_tile >= k_tiles) {
    return;
  }

  int finish_k_tile = min(start_k_tile + block_k_tiles, k_tiles);

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering
    // and can only issue the next fetch when it is guaranteed that the previous
    // shared memory load is fully complete (as it may otherwise be
    // overwritten).
    cp_async_wait<repack_stages - 2>();
    __syncthreads();
  };

  extern __shared__ int4 sh[];

  constexpr int tile_n_ints = tile_n_size / pack_factor;

  constexpr int stage_n_threads = tile_n_ints / 4;
  constexpr int stage_k_threads = tile_k_size;
  constexpr int stage_size = stage_k_threads * stage_n_threads;

  auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      cp_async_fence();
      return;
    }

    int first_n = n_tile_id * tile_n_size;
    int first_n_packed = first_n / pack_factor;

    int4* sh_ptr = sh + stage_size * pipe;

    if (threadIdx.x < stage_size) {
      int k_id = threadIdx.x / stage_n_threads;
      int n_id = threadIdx.x % stage_n_threads;

      int first_k = k_tile_id * tile_k_size;

      cp_async4(&sh_ptr[k_id * stage_n_threads + n_id],
                reinterpret_cast<int4 const*>(
                    &(b_q_weight_ptr[(first_k + k_id) * (size_n / pack_factor) +
                                     first_n_packed + (n_id * 4)])));
    }

    cp_async_fence();
  };

  auto repack_tile = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      return;
    }

    int warp_id = threadIdx.x / 32;
    int th_id = threadIdx.x % 32;

    if (warp_id >= 4) {
      return;
    }

    int tc_col = th_id / 4;
    int tc_row = (th_id % 4) * 2;

    constexpr int tc_offsets[4] = {0, 1, 8, 9};

    int cur_n = warp_id * 16 + tc_col;
    int cur_n_packed = cur_n / pack_factor;
    int cur_n_pos = cur_n % pack_factor;

    constexpr int sh_stride = tile_n_ints;
    constexpr uint32_t mask = (1 << num_bits) - 1;

    int4* sh_stage_ptr = sh + stage_size * pipe;
    uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh_stage_ptr);

    // Undo interleaving
    int cur_n_pos_unpacked;
    if constexpr (num_bits == 4) {
      constexpr int undo_pack[8] = {0, 4, 1, 5, 2, 6, 3, 7};
      cur_n_pos_unpacked = undo_pack[cur_n_pos];
    } else {
      constexpr int undo_pack[4] = {0, 2, 1, 3};
      cur_n_pos_unpacked = undo_pack[cur_n_pos];
    }

    uint32_t vals[8];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int cur_elem = tc_row + tc_offsets[i];

      int packed_src_0 = sh_stage_int_ptr[cur_n_packed + sh_stride * cur_elem];
      int packed_src_1 = sh_stage_int_ptr[cur_n_packed + (8 / pack_factor) +
                                          sh_stride * cur_elem];

      vals[i] = (packed_src_0 >> (cur_n_pos_unpacked * num_bits)) & mask;
      vals[4 + i] = (packed_src_1 >> (cur_n_pos_unpacked * num_bits)) & mask;
    }

    constexpr int tile_size = tile_k_size * tile_n_size / pack_factor;
    int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_size;

    // Result of:
    // https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
    if constexpr (num_bits == 4) {
      constexpr int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};

      uint32_t res = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        res |= vals[pack_idx[i]] << (i * 4);
      }

      out_ptr[out_offset + th_id * 4 + warp_id] = res;

    } else {
      constexpr int pack_idx[4] = {0, 2, 1, 3};

      uint32_t res1 = 0;
      uint32_t res2 = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        res1 |= vals[pack_idx[i]] << (i * 8);
        res2 |= vals[4 + pack_idx[i]] << (i * 8);
      }

      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 0] = res1;
      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 1] = res2;
    }
  };

  auto start_pipes = [&](int k_tile_id, int n_tile_id) {
#pragma unroll
    for (int pipe = 0; pipe < repack_stages - 1; pipe++) {
      fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
    }

    wait_for_stage();
  };
#pragma unroll
  for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++) {
    int n_tile_id = 0;

    start_pipes(k_tile_id, n_tile_id);

    while (n_tile_id < n_tiles) {
#pragma unroll
      for (int pipe = 0; pipe < repack_stages; pipe++) {
        fetch_to_shared((pipe + repack_stages - 1) % repack_stages, k_tile_id,
                        n_tile_id + pipe + repack_stages - 1);
        repack_tile(pipe, k_tile_id, n_tile_id + pipe);
        wait_for_stage();
      }
      n_tile_id += repack_stages;
    }
  }
}

}  // namespace marlin

#define CALL_IF(NUM_BITS)                                                   \
  else if (num_bits == NUM_BITS) {                                          \
    cudaFuncSetAttribute(                                                   \
        marlin::awq_marlin_repack_kernel<marlin::repack_threads, NUM_BITS>, \
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);       \
    marlin::awq_marlin_repack_kernel<marlin::repack_threads, NUM_BITS>      \
        <<<blocks, marlin::repack_threads, max_shared_mem, stream>>>(       \
            b_q_weight_ptr, out_ptr, size_k, size_n);                       \
  }

torch::Tensor awq_marlin_repack(torch::Tensor& b_q_weight, int64_t size_k,
                                int64_t size_n, int64_t num_bits) {
  // Verify compatibility with marlin tile of 16x64
  TORCH_CHECK(size_k % marlin::tile_k_size == 0, "size_k = ", size_k,
              " is not divisible by tile_k_size = ", marlin::tile_k_size);
  TORCH_CHECK(size_n % marlin::tile_n_size == 0, "size_n = ", size_n,
              " is not divisible by tile_n_size = ", marlin::tile_n_size);

  TORCH_CHECK(num_bits == 4 || num_bits == 8,
              "num_bits must be 4 or 8. Got = ", num_bits);
  int const pack_factor = 32 / num_bits;

  // Verify B
  TORCH_CHECK(b_q_weight.size(0) == size_k,
              "b_q_weight.size(0) = ", b_q_weight.size(0),
              " is not size_k = ", size_k);
  TORCH_CHECK((size_n / pack_factor) == b_q_weight.size(1),
              "Shape mismatch: b_q_weight.size(1) = ", b_q_weight.size(1),
              ", size_n = ", size_n, ", pack_factor = ", pack_factor);

  // Verify device and strides
  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");
  TORCH_CHECK(b_q_weight.dtype() == at::kInt, "b_q_weight type is not kInt");

  // Alloc buffers
  const at::cuda::OptionalCUDAGuard device_guard(device_of(b_q_weight));
  auto options = torch::TensorOptions()
                     .dtype(b_q_weight.dtype())
                     .device(b_q_weight.device());
  torch::Tensor out = torch::empty(
      {size_k / marlin::tile_size, size_n * marlin::tile_size / pack_factor},
      options);

  // Get ptrs
  uint32_t const* b_q_weight_ptr =
      reinterpret_cast<uint32_t const*>(b_q_weight.data_ptr());
  uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out.data_ptr());

  // Get dev info
  int dev = b_q_weight.get_device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);
  int blocks;
  cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, dev);

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(max_shared_mem > 0);

  if (false) {
  }
  CALL_IF(4)
  CALL_IF(8)
  else {
    TORCH_CHECK(false, "Unsupported repack config: num_bits = ", num_bits);
  }

  return out;
}

torch::Tensor awq_marlin_repack_meta(torch::Tensor& b_q_weight,
                                     c10::SymInt size_k, c10::SymInt size_n,
                                     int64_t num_bits) {
  int const pack_factor = 32 / num_bits;
  auto options = torch::TensorOptions()
                     .dtype(b_q_weight.dtype())
                     .device(b_q_weight.device());
  return torch::empty_symint(
      {size_k / marlin::tile_size, size_n * marlin::tile_size / pack_factor},
      options);
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("awq_marlin_repack", &awq_marlin_repack);
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Meta, m) {
  m.impl("awq_marlin_repack", &awq_marlin_repack_meta);
}