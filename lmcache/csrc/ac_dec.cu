// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include "cachegen_kernels.cuh"

#define MAX_LP 64
#define MAX_THREAD_PER_BLOCK 128
#define MAX_SHARED_MEMORY_PER_THREAD (0xc000 / MAX_THREAD_PER_BLOCK)
#if MAX_SHARED_MEMORY_PER_THREAD - MAX_LP * 2 >= 256
  #define MAX_TOKENS_PER_THREAD 256
  #define OUTPUT_BUFFER_LENGTH_PER_THREAD 256
#else
  #define OUTPUT_BUFFER_LENGTH_PER_THREAD \
    (MAX_SHARED_MEMORY_PER_THREAD - MAX_LP * 2)
  #define MAX_TOKENS_PER_THREAD (OUTPUT_BUFFER_LENGTH_PER_THREAD)
#endif
#define PRECISION 16

extern int get_block_size(int);

template <typename T>
__inline__ __device__ T big_to_small(T value) {
  return value;
}

template <>
__inline__ __device__ uint32_t big_to_small<uint32_t>(uint32_t value) {
  return ((value & 0xFF000000U) >> 24) | ((value & 0x00FF0000U) >> 8) |
         ((value & 0x0000FF00U) << 8) | ((value & 0x000000FFU) << 24);
}

template <>
__inline__ __device__ uint8_t big_to_small<uint8_t>(uint8_t value) {
  return value;
}

template <int BUFFER_BITS, typename BUFFER_TYPE>
__inline__ __device__ void read_next_bit(uint32_t& value,
                                         BUFFER_TYPE& byte_buffer,
                                         int& bit_idx) {
  value <<= 1;
  value |= (byte_buffer >> (BUFFER_BITS - bit_idx)) & 1;
  bit_idx += 1;
}

template <int BUFFER_BITS, typename BUFFER_TYPE>
__inline__ __device__ void check_and_update_byte_buffer(
    BUFFER_TYPE& byte_buffer, int& bit_idx, int& byte_buffer_offset,
    uint8_t* bytestream) {
  if (bit_idx == BUFFER_BITS + 1) {
    bit_idx = 1;
    byte_buffer_offset++;
    byte_buffer = big_to_small<BUFFER_TYPE>(
        ((BUFFER_TYPE*)bytestream)[byte_buffer_offset]);
  }
}

template <int BLOCK_SIZE>
__inline__ __device__ uint16_t binsearch(const uint16_t* cdf_shared,
                                         uint16_t target, uint8_t max_sym,
                                         const int tid) {
  uint16_t left = 0;
  uint16_t right = max_sym + 1;  // len(cdf) == max_sym + 2

  while (left + 1 < right) {  // ?
    const auto m = static_cast<uint16_t>((left + right) / 2);
    const auto offset = m * BLOCK_SIZE + tid;
    const auto v = cdf_shared[offset];
    if (v < target) {
      left = m;
    } else if (v > target) {
      right = m;
    } else {
      return m;
    }
  }
  return left;
}

// BLOCK_SIZE SHOULD ALWAYS BE THE SAME AS blockDim.x
template <int BLOCK_SIZE, typename CDF_ACC_T, typename BS_ACC_T,
          typename LEN_ACC_T, typename OUT_ACC_T>
__global__ void decode_with_accessor_kernel(CDF_ACC_T cdf, BS_ACC_T bytestreams,
                                            LEN_ACC_T lengths, OUT_ACC_T output,
                                            int32_t lp, int32_t ntokens) {
  // The shared memory will be split to 3 parts:
  // 1. The CDF tensor, with shape [MAX_LP, BLOCK_SIZE)] (only used [LP,
  // BLOCK_SIZE] part)
  // 2. The bytestream buffer, with shape [BLOCK_SIZE,
  // OUTPUT_BUFFER_LENGTH_PER_THREAD] uint8s
  // 3. The lengths buffer, with shape [BLOCK_SIZE] int32s
  __shared__ __align__(4) uint16_t cdf_shared[MAX_LP][BLOCK_SIZE];
  __shared__ __align__(4)
      uint8_t bytestreams_shared[BLOCK_SIZE][OUTPUT_BUFFER_LENGTH_PER_THREAD];
  int32_t* lengths_shared = (int32_t*)&cdf_shared[0][0];

  const int layer_id = blockIdx.x;
  const int global_channel_offset = blockIdx.y * BLOCK_SIZE;
  const int local_channel_id = threadIdx.x;
  const int global_channel_id = global_channel_offset + local_channel_id;
  const int max_symbol = lp - 2;

  // copy lengths[layer_id,
  // global_channel_offset:global_channel_offset+BLOCK_SIZE] to shared memory
  for (int i = threadIdx.x; i < BLOCK_SIZE; i += BLOCK_SIZE) {
    lengths_shared[i] = lengths[layer_id][global_channel_offset + i];
  }

  __syncthreads();

  // copy bytestreams[layer_id,
  // global_channel_offset:global_channel_offset+BLOCK_SIZE, :] to shared
  // memory, do this channel by channel
  for (int i = 0; i < BLOCK_SIZE; i++) {
    const int channel_id = global_channel_offset + i;
    const int length = lengths_shared[i];  // shared memory broadcast
    // TODO: optimized this by a packed-32bits read instead of 8bits read
    for (int j = threadIdx.x; j < OUTPUT_BUFFER_LENGTH_PER_THREAD;
         j += BLOCK_SIZE) {
      const uint8_t value =
          j < length ? bytestreams[layer_id][channel_id][j] : 0;
      bytestreams_shared[i][j] = value;
    }
  }
  __syncthreads();

  // copy CDF[layer_id, global_channel_offset:global_channel_offset+BLOCK_SIZE,
  // :] to shared memory
  const int cdf_size = lp * BLOCK_SIZE;
  for (int i = threadIdx.x; i < cdf_size; i += BLOCK_SIZE) {
    const int cid = i / lp;
    const int lid = i % lp;
    cdf_shared[lid][cid] = cdf[layer_id][global_channel_offset + cid][lid];
  }

  __syncthreads();

  // decode the bytestreams
  uint32_t low = 0;
  uint32_t high = 0xFFFFFFFFU;
  uint32_t value = 0;
  const uint32_t c_count = 0x10000U;
  const int precision = 16;

  uint8_t byte_buffer = 0;
  int bit_idx = 1;  // next bit to read: (byte_buffer >> (8 - bit_idx)) & 1
  int byte_buffer_offset =
      sizeof(value) / sizeof(byte_buffer);  // where to read the next byte

  // Get the initial value and byte buffer
  value = big_to_small<uint32_t>(
      ((uint32_t*)bytestreams_shared[local_channel_id])[0]);
  // byte_buffer = ((uint32_t
  // *)bytestreams_shared[local_channel_id])[byte_buffer_offset];
  byte_buffer = bytestreams_shared[local_channel_id][byte_buffer_offset];

  for (int i = 0; i < ntokens; ++i) {
    const uint64_t span =
        static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;
    // always < 0x10000 ???
    const uint16_t count =
        ((static_cast<uint64_t>(value) - static_cast<uint64_t>(low) + 1) *
             c_count -
         1) /
        span;

    // TODO: implement the binsearch here!
    auto sym_i = binsearch<BLOCK_SIZE>(&cdf_shared[0][0], count, max_symbol,
                                       local_channel_id);

    output[layer_id][i][global_channel_id] = sym_i;

    if (i == ntokens - 1) {
      break;
    }

    const uint32_t c_low = cdf_shared[sym_i][local_channel_id];
    const uint32_t c_high = sym_i == max_symbol
                                ? 0x10000U
                                : cdf_shared[sym_i + 1][local_channel_id];

    high = (low - 1) + ((span * static_cast<uint64_t>(c_high)) >> precision);
    low = (low) + ((span * static_cast<uint64_t>(c_low)) >> precision);

    while (true) {
      if (low >= 0x80000000U || high < 0x80000000U) {
        low <<= 1;
        high <<= 1;
        high |= 1;
        read_next_bit<8>(value, byte_buffer, bit_idx);
        check_and_update_byte_buffer<8, uint8_t>(
            byte_buffer, bit_idx, byte_buffer_offset,
            bytestreams_shared[local_channel_id]);
      } else if (low >= 0x40000000U && high < 0xC0000000U) {
        low <<= 1;
        low &= 0x7FFFFFFFU;  // make MSB 0
        high <<= 1;
        high |= 0x80000001U;  // add 1 at the end, retain MSB = 1
        value -= 0x40000000U;
        read_next_bit<8>(value, byte_buffer, bit_idx);
        check_and_update_byte_buffer<8, uint8_t>(
            byte_buffer, bit_idx, byte_buffer_offset,
            bytestreams_shared[local_channel_id]);
      } else {
        break;
      }
    }
  }
}

// BLOCK_SIZE SHOULD ALWAYS BE THE SAME AS blockDim.x
template <int BLOCK_SIZE, typename CDF_ACC_T, typename BS_ACC_T,
          typename LEN_ACC_T, typename OUT_ACC_T>
__global__ void decode_prefix_with_accessor_kernel(CDF_ACC_T cdf,
                                                   BS_ACC_T bytestreams,
                                                   LEN_ACC_T lengths_prefix,
                                                   OUT_ACC_T output, int32_t lp,
                                                   int32_t ntokens) {
  // The shared memory will be split to 3 parts:
  // 1. The CDF tensor, with shape [MAX_LP, BLOCK_SIZE)] (only used [LP,
  // BLOCK_SIZE] part)
  // 2. The bytestream buffer, with shape [BLOCK_SIZE,
  // OUTPUT_BUFFER_LENGTH_PER_THREAD] uint8s
  // 3. The lengths buffer, with shape [BLOCK_SIZE] int32s
  __shared__ __align__(4) uint16_t cdf_shared[MAX_LP][BLOCK_SIZE];
  __shared__ __align__(4)
      uint8_t bytestreams_shared[BLOCK_SIZE][OUTPUT_BUFFER_LENGTH_PER_THREAD];
  int32_t* sum_lengths_shared = (int32_t*)&cdf_shared[0][0];

  const int layer_id = blockIdx.x;
  const int global_channel_offset = blockIdx.y * BLOCK_SIZE;
  const int local_channel_id = threadIdx.x;
  const int global_channel_id = global_channel_offset + local_channel_id;
  const int max_symbol = lp - 2;
  const int nchannels = gridDim.y * BLOCK_SIZE;

  // copy lengths[layer_id,
  // global_channel_offset:global_channel_offset+BLOCK_SIZE] to shared memory
  for (int i = threadIdx.x; i < BLOCK_SIZE + 1; i += BLOCK_SIZE) {
    int gid = layer_id * nchannels + global_channel_offset + i - 1;
    sum_lengths_shared[i] =
        gid >= 0 ? lengths_prefix[gid / nchannels][gid % nchannels] : 0;
  }

  __syncthreads();

  // copy bytestreams[layer_id,
  // global_channel_offset:global_channel_offset+BLOCK_SIZE, :] to shared
  // memory, do this channel by channel
  for (int i = 0; i < BLOCK_SIZE; i++) {
    [[maybe_unused]] const int channel_id = global_channel_offset + i;
    const int start_offset = sum_lengths_shared[i];
    const int end_offset = sum_lengths_shared[i + 1];
    const int length = end_offset - start_offset;
    // TODO: optimized this by a packed-32bits read instead of 8bits read
    for (int j = threadIdx.x; j < OUTPUT_BUFFER_LENGTH_PER_THREAD;
         j += BLOCK_SIZE) {
      const uint8_t value = j < length ? bytestreams[start_offset + j] : 0;
      bytestreams_shared[i][j] = value;
    }
  }
  __syncthreads();

  // copy CDF[layer_id, global_channel_offset:global_channel_offset+BLOCK_SIZE,
  // :] to shared memory
  const int cdf_size = lp * BLOCK_SIZE;
  for (int i = threadIdx.x; i < cdf_size; i += BLOCK_SIZE) {
    const int cid = i / lp;
    const int lid = i % lp;
    cdf_shared[lid][cid] = cdf[layer_id][global_channel_offset + cid][lid];
  }

  __syncthreads();

  // decode the bytestreams
  uint32_t low = 0;
  uint32_t high = 0xFFFFFFFFU;
  uint32_t value = 0;
  const uint32_t c_count = 0x10000U;
  const int precision = 16;

  uint8_t byte_buffer = 0;
  int bit_idx = 1;  // next bit to read: (byte_buffer >> (8 - bit_idx)) & 1
  int byte_buffer_offset =
      sizeof(value) / sizeof(byte_buffer);  // where to read the next byte

  // Get the initial value and byte buffer
  value = big_to_small<uint32_t>(
      ((uint32_t*)bytestreams_shared[local_channel_id])[0]);
  // byte_buffer = ((uint32_t
  // *)bytestreams_shared[local_channel_id])[byte_buffer_offset];
  byte_buffer = bytestreams_shared[local_channel_id][byte_buffer_offset];

  for (int i = 0; i < ntokens; ++i) {
    const uint64_t span =
        static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;
    // always < 0x10000 ???
    const uint16_t count =
        ((static_cast<uint64_t>(value) - static_cast<uint64_t>(low) + 1) *
             c_count -
         1) /
        span;

    // TODO: implement the binsearch here!
    auto sym_i = binsearch<BLOCK_SIZE>(&cdf_shared[0][0], count, max_symbol,
                                       local_channel_id);

    output[layer_id][i][global_channel_id] = sym_i;

    if (i == ntokens - 1) {
      break;
    }

    const uint32_t c_low = cdf_shared[sym_i][local_channel_id];
    const uint32_t c_high = sym_i == max_symbol
                                ? 0x10000U
                                : cdf_shared[sym_i + 1][local_channel_id];

    high = (low - 1) + ((span * static_cast<uint64_t>(c_high)) >> precision);
    low = (low) + ((span * static_cast<uint64_t>(c_low)) >> precision);

    while (true) {
      if (low >= 0x80000000U || high < 0x80000000U) {
        low <<= 1;
        high <<= 1;
        high |= 1;
        read_next_bit<8>(value, byte_buffer, bit_idx);
        check_and_update_byte_buffer<8, uint8_t>(
            byte_buffer, bit_idx, byte_buffer_offset,
            bytestreams_shared[local_channel_id]);
      } else if (low >= 0x40000000U && high < 0xC0000000U) {
        low <<= 1;
        low &= 0x7FFFFFFFU;  // make MSB 0
        high <<= 1;
        high |= 0x80000001U;  // add 1 at the end, retain MSB = 1
        value -= 0x40000000U;
        read_next_bit<8>(value, byte_buffer, bit_idx);
        check_and_update_byte_buffer<8, uint8_t>(
            byte_buffer, bit_idx, byte_buffer_offset,
            bytestreams_shared[local_channel_id]);
      } else {
        break;
      }
    }
  }
}

/**
 * @brief CUDA kernel to decode a compressed bytestream using the given CDF.
 *
 * @param cdf the int16 CDF tensor, with shape [nlayers, nchannels, LP], should
 * be on GPU
 * @param bytestreams The uint8 bytestreams tensor, with shape [nlayers,
 * nchannels, OUTPUT_BUFFER_LENGTH_PER_THREAD], should be on GPU
 * @param lengths The int32 lengths tensor, with shape [nlayers, nchannels],
 * should be on GPU
 * @param output The uint8 output tensor, with shape nlayers, ntokens,
 * nchannels], should be on GPU.
 */
void decode_cuda_new(const at::Tensor& cdf, const at::Tensor& bytestreams,
                     const at::Tensor& lengths, at::Tensor& output) {
  TORCH_CHECK(cdf.is_cuda(), "CDF should be on GPU");
  TORCH_CHECK(bytestreams.is_cuda(), "Bytestreams should be on GPU");
  TORCH_CHECK(lengths.is_cuda(), "Lengths should be on GPU");
  TORCH_CHECK(output.is_cuda(), "Output should be on GPU");

  const auto cdf_shape = cdf.sizes();
  const auto bs_shape = bytestreams.sizes();
  const auto lengths_shape = lengths.sizes();
  const auto output_shape = output.sizes();
  TORCH_CHECK(cdf_shape[0] == bs_shape[0],
              "CDF and bytestreams should have the same number of layers");
  TORCH_CHECK(cdf_shape[1] == bs_shape[1],
              "CDF and bytestreams should have the same number of channels");
  TORCH_CHECK(cdf_shape[0] == lengths_shape[0],
              "CDF and lengths should have the same number of layers");
  TORCH_CHECK(cdf_shape[1] == lengths_shape[1],
              "CDF and lengths should have the same number of channels");
  TORCH_CHECK(cdf_shape[0] == output_shape[0],
              "CDF and output should have the same number of layers");
  TORCH_CHECK(cdf_shape[1] == output_shape[2],
              "CDF and output should have the same number of channels");

  const int nlayers = cdf_shape[0];
  const int nchannels = cdf_shape[1];
  const int ntokens = output_shape[1];
  const int lp = cdf_shape[2];
  const int block_size = get_block_size(nchannels);
  TORCH_CHECK(ntokens <= MAX_TOKENS_PER_THREAD,
              "Number of tokens should be less than or equal to",
              MAX_TOKENS_PER_THREAD);
  TORCH_CHECK(nchannels % block_size == 0,
              "Number of channels should be divisible by block size");
  TORCH_CHECK(lp <= MAX_LP, "CDF should have at most", MAX_LP, "Lps");

  dim3 block_dim(block_size, 1, 1);
  dim3 grid_dim(nlayers, nchannels / block_size, 1);

  auto cdf_accessor =
      cdf.packed_accessor32<int16_t, 3, torch::RestrictPtrTraits>();
  auto bytestreams_accessor =
      bytestreams.packed_accessor32<uint8_t, 3, torch::RestrictPtrTraits>();
  auto lengths_accessor =
      lengths.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>();
  auto output_accessor =
      output.packed_accessor32<uint8_t, 3, torch::RestrictPtrTraits>();

#ifndef LAUNCH_DECODE_KERNEL
  #define LAUNCH_DECODE_KERNEL(block_size)                                     \
    decode_with_accessor_kernel<block_size><<<grid_dim, block_dim>>>(          \
        cdf_accessor, bytestreams_accessor, lengths_accessor, output_accessor, \
        lp, ntokens)
#endif

  switch (block_size) {
    case 1:
      LAUNCH_DECODE_KERNEL(1);
      break;
    case 2:
      LAUNCH_DECODE_KERNEL(2);
      break;
    case 4:
      LAUNCH_DECODE_KERNEL(4);
      break;
    case 8:
      LAUNCH_DECODE_KERNEL(8);
      break;
    case 16:
      LAUNCH_DECODE_KERNEL(16);
      break;
    case 32:
      LAUNCH_DECODE_KERNEL(32);
      break;
    case 64:
      LAUNCH_DECODE_KERNEL(64);
      break;
    case 128:
      LAUNCH_DECODE_KERNEL(128);
      break;
    default:
      throw std::runtime_error("Unsupported block size");
  }
}

/**
 * @brief CUDA kernel to decode a compressed bytestream using the given CDF.
 *
 * @param cdf the int16 CDF tensor, with shape [nlayers, nchannels, LP], should
 * be on GPU
 * @param bytestreams The 1-D uint8 bytestreams tensor containing [nlayers,
 * nchannels] bytestreams, should be on GPU
 * @param lengths_prefsum The int64 tensor containing the prefix sum of the
 * lengths, with shape [nlayers, nchannels], should be on GPU
 * @param output The uint8 output tensor, with shape nlayers, ntokens,
 * nchannels], should be on GPU.
 */
void decode_cuda_prefsum(const at::Tensor& cdf, const at::Tensor& bytestreams,
                         const at::Tensor& lengths_prefsum,
                         at::Tensor& output) {
  TORCH_CHECK(cdf.is_cuda(), "CDF should be on GPU");
  TORCH_CHECK(bytestreams.is_cuda(), "Bytestreams should be on GPU");
  TORCH_CHECK(lengths_prefsum.is_cuda(), "Lengths should be on GPU");
  TORCH_CHECK(output.is_cuda(), "Output should be on GPU");

  const auto cdf_shape = cdf.sizes();
  const auto lengths_shape = lengths_prefsum.sizes();
  const auto output_shape = output.sizes();
  TORCH_CHECK(cdf_shape[0] == lengths_shape[0],
              "CDF and lengths should have the same number of layers");
  TORCH_CHECK(cdf_shape[1] == lengths_shape[1],
              "CDF and lengths should have the same number of channels");
  TORCH_CHECK(cdf_shape[0] == output_shape[0],
              "CDF and output should have the same number of layers");
  TORCH_CHECK(cdf_shape[1] == output_shape[2],
              "CDF and output should have the same number of channels");

  const int nlayers = cdf_shape[0];
  const int nchannels = cdf_shape[1];
  const int ntokens = output_shape[1];
  const int lp = cdf_shape[2];
  const int block_size = get_block_size(nchannels);
  TORCH_CHECK(ntokens <= MAX_TOKENS_PER_THREAD,
              "Number of tokens should be less than or equal to",
              MAX_TOKENS_PER_THREAD);
  TORCH_CHECK(nchannels % block_size == 0,
              "Number of channels should be divisible by block size");
  TORCH_CHECK(lp <= MAX_LP, "CDF should have at most", MAX_LP, "Lps");

  dim3 block_dim(block_size, 1, 1);
  dim3 grid_dim(nlayers, nchannels / block_size, 1);

  auto cdf_accessor =
      cdf.packed_accessor32<int16_t, 3, torch::RestrictPtrTraits>();
  auto bytestreams_accessor =
      bytestreams.packed_accessor32<uint8_t, 1, torch::RestrictPtrTraits>();
  auto lengths_accessor =
      lengths_prefsum.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>();
  auto output_accessor =
      output.packed_accessor32<uint8_t, 3, torch::RestrictPtrTraits>();

#ifndef LAUNCH_DECODE_PREFIX_KERNEL
  #define LAUNCH_DECODE_PREFIX_KERNEL(block_size)                              \
    decode_prefix_with_accessor_kernel<block_size><<<grid_dim, block_dim>>>(   \
        cdf_accessor, bytestreams_accessor, lengths_accessor, output_accessor, \
        lp, ntokens)
#endif

  switch (block_size) {
    case 1:
      LAUNCH_DECODE_PREFIX_KERNEL(1);
      break;
    case 2:
      LAUNCH_DECODE_PREFIX_KERNEL(2);
      break;
    case 4:
      LAUNCH_DECODE_PREFIX_KERNEL(4);
      break;
    case 8:
      LAUNCH_DECODE_PREFIX_KERNEL(8);
      break;
    case 16:
      LAUNCH_DECODE_PREFIX_KERNEL(16);
      break;
    case 32:
      LAUNCH_DECODE_PREFIX_KERNEL(32);
      break;
    case 64:
      LAUNCH_DECODE_PREFIX_KERNEL(64);
      break;
    case 128:
      LAUNCH_DECODE_PREFIX_KERNEL(128);
      break;
    default:
      throw std::runtime_error("Unsupported block size");
  }
}
