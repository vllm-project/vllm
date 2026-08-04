// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include "cachegen_kernels.cuh"

#define MAX_LP 48
#define MAX_THREAD_PER_BLOCK 128
#define MAX_SHARED_MEMORY_PER_THREAD (0xc000 / MAX_THREAD_PER_BLOCK)
#if MAX_SHARED_MEMORY_PER_THREAD - MAX_LP * 2 > 256
  #define MAX_TOKENS_PER_THREAD 256
  #define OUTPUT_BUFFER_LENGTH_PER_THREAD 256
#else
  #define OUTPUT_BUFFER_LENGTH_PER_THREAD \
    (MAX_SHARED_MEMORY_PER_THREAD - MAX_LP * 2)
  #define MAX_TOKENS_PER_THREAD (OUTPUT_BUFFER_LENGTH_PER_THREAD)
#endif
#define PRECISION 16

/**
 * Spill the register to the shared memory
 */
__device__ __inline__ void spill_reg_to_shared(uint32_t output_reg,
                                               int& output_reg_len,
                                               uint8_t* output_shared,
                                               int& output_shared_offset) {
  output_reg <<= 32 - output_reg_len;
  output_reg_len = 0;
  // TODO: potential optimization: since it uses little endian, we can directly
  // write the 4 bytes
  output_shared[output_shared_offset] = output_reg >> 24;
  output_shared[output_shared_offset + 1] = (output_reg >> 16) & 0xFF;
  output_shared[output_shared_offset + 2] = (output_reg >> 8) & 0xFF;
  output_shared[output_shared_offset + 3] = output_reg & 0xFF;
  output_shared_offset += 4;
  output_reg = 0;
}

/**
 * Spill the register to the shared memory, when it is not a full 32 bits
 */
__device__ __inline__ void spill_partial_reg_to_shared(
    uint32_t output_reg, int& output_reg_len, uint8_t* output_shared,
    int& output_shared_offset) {
  output_reg <<= 32 - output_reg_len;
  while (output_reg_len > 0) {
    output_reg_len -= 8;
    output_shared[output_shared_offset] = output_reg >> 24;
    output_shared_offset++;
    output_reg <<= 8;
  }
}

/**
 * Write N bits of 0/1 to the output. Save the overflow bits to the shared
 * memory
 */
__device__ __inline__ void add_bits_to_output(uint32_t bit, int num,
                                              uint32_t& output_reg,
                                              int& output_reg_len,
                                              uint8_t* output_shared,
                                              int& output_shared_offset) {
  do {
    const int remaining = min(num, 32 - output_reg_len);
    output_reg <<= remaining;
    output_reg |= (bit << remaining) - bit;
    num -= remaining;
    output_reg_len += remaining;
    if (output_reg_len == 32) {
      spill_reg_to_shared(output_reg, output_reg_len, output_shared,
                          output_shared_offset);
    }
  } while (num > 0);
}

/**
 * Append a bit to the output, and add the pending bits
 */
__device__ __inline__ void append_bit_and_pending(
    uint32_t bit, uint64_t& pending_bits, uint32_t& output_reg,
    int& output_reg_len, uint8_t* output_shared, int& output_shared_offset) {
  add_bits_to_output(bit, 1, output_reg, output_reg_len, output_shared,
                     output_shared_offset);
  add_bits_to_output(1 - bit, pending_bits, output_reg, output_reg_len,
                     output_shared, output_shared_offset);
  pending_bits = 0;
}

__inline__ __device__ void warp_scan(volatile int* temp, int tid) {
  int offset = 1;
  int n = blockDim.x;
  while (offset < n) {
    if (tid >= offset) temp[tid] += temp[tid - offset];
    offset *= 2;
    __syncthreads();
  }
}

// This is assuming each thread will process all the tokens in the same layer
// and channel
template <int BLOCK_SIZE>
__global__ void encode_kernel(
    const uint16_t* cdf,       // shape [nlayers, nchannels, Lp]
    const uint8_t* input_sym,  // shape [nlayers, ntokens, nchannels]
    uint8_t* output_buffer,    // shape [nlayers, nchannels,
                               // OUTPUT_BUFFER_LENGTH_PER_THREAD]
    int32_t* output_lengths,   // shape [nlayers, nchannels]
    int32_t lp, int32_t ntokens, int32_t output_buffer_length_per_thread) {
  // The shared memory will be split to 2 parts:
  // 1. The CDF tensor, with shape [MAX_LP, BLOCK_SIZE)] (only used [LP,
  // BLOCK_SIZE] part)
  // 2. The output buffer, with shape [BLOCK_SIZE, MAX_SHARED_PER_THREAD -
  // MAX_LP * 2] uint8s
  __shared__ uint16_t cdf_shared[MAX_LP * BLOCK_SIZE];
  __shared__ uint8_t
      output_shared[BLOCK_SIZE * OUTPUT_BUFFER_LENGTH_PER_THREAD];

  const int nchannels = gridDim.y * BLOCK_SIZE;

  const int layer_id = blockIdx.x;
  const int channel_id = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Copy the CDF[layer_id, channel_start:channel_end, :] to shared memory
  const int cdf_offset = (blockIdx.y * BLOCK_SIZE) * lp;
  const int cdf_size = BLOCK_SIZE * lp;
  for (int i = threadIdx.x; i < cdf_size; i += blockDim.x) {
    const int cid = i / lp;
    const int lid = i % lp;
    const int shared_offset = lid * BLOCK_SIZE + cid;
    cdf_shared[shared_offset] = reinterpret_cast<uint16_t>(
        cdf[layer_id * nchannels * lp + cdf_offset + i]);
  }

  __syncthreads();

  // Do the actual encoding
  uint32_t low = 0U;
  uint32_t high = 0xFFFFFFFFU;
  uint64_t pending_bits = 0;
  const int max_symbol = lp - 2;

  uint32_t output_reg = 0;
  int output_reg_len = 0;
  int output_shared_offset = threadIdx.x * OUTPUT_BUFFER_LENGTH_PER_THREAD;

  for (int i = 0; i < ntokens; i++) {
    const uint8_t sym =
        input_sym[layer_id * ntokens * nchannels + i * nchannels + channel_id];
    const uint64_t span =
        static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;

    const uint32_t c_low = cdf_shared[sym * BLOCK_SIZE + threadIdx.x];
    const uint32_t c_high =
        sym == max_symbol ? 0x10000U
                          : cdf_shared[(sym + 1) * BLOCK_SIZE + threadIdx.x];

    high = (low - 1) + ((span * static_cast<uint64_t>(c_high)) >> precision);
    low = (low) + ((span * static_cast<uint64_t>(c_low)) >> precision);

    while (true) {
      if (high < 0x80000000U) {
        append_bit_and_pending(0, pending_bits, output_reg, output_reg_len,
                               output_shared, output_shared_offset);
        low <<= 1;
        high <<= 1;
        high |= 1;
      } else if (low >= 0x80000000U) {
        append_bit_and_pending(1, pending_bits, output_reg, output_reg_len,
                               output_shared, output_shared_offset);
        low <<= 1;
        high <<= 1;
        high |= 1;
      } else if (low >= 0x40000000U && high < 0xC0000000U) {
        pending_bits++;
        low <<= 1;
        low &= 0x7FFFFFFF;
        high <<= 1;
        high |= 0x80000001;
      } else {
        break;
      }
    }
  }

  pending_bits += 1;

  if (low < 0x40000000U) {
    append_bit_and_pending(0, pending_bits, output_reg, output_reg_len,
                           output_shared, output_shared_offset);
  } else {
    append_bit_and_pending(1, pending_bits, output_reg, output_reg_len,
                           output_shared, output_shared_offset);
  }

  spill_partial_reg_to_shared(output_reg, output_reg_len, output_shared,
                              output_shared_offset);
  output_lengths[layer_id * nchannels + channel_id] =
      output_shared_offset - threadIdx.x * OUTPUT_BUFFER_LENGTH_PER_THREAD;

  __syncthreads();

  // reuse cdf for the prefix sum
  int* output_lengths_shared = reinterpret_cast<int*>(cdf_shared);
  output_lengths_shared[threadIdx.x] =
      output_shared_offset - threadIdx.x * OUTPUT_BUFFER_LENGTH_PER_THREAD;
  __syncthreads();

  // Copy the output buffer to global memory
  // then copy one "row" at a time, and make sure the write is coalesced
  for (int i = 0; i < BLOCK_SIZE; i++) {
    int length = output_lengths_shared[i];
    int current_channel = blockIdx.y * BLOCK_SIZE + i;
    for (int j = threadIdx.x; j < length; j += blockDim.x) {
      int global_offset =
          layer_id * nchannels * output_buffer_length_per_thread +
          current_channel * output_buffer_length_per_thread + j;
      int local_offset = i * OUTPUT_BUFFER_LENGTH_PER_THREAD + j;
      output_buffer[global_offset] = output_shared[local_offset];
    }
  }
}

// This is assuming each thread will process all the tokens in the same layer
// and channel
template <int BLOCK_SIZE, typename CDF_ACC_T, typename SYM_ACC_T,
          typename OUTPUT_ACC_T, typename LEN_ACC_T>
__global__ void encode_with_accessor_kernel(
    CDF_ACC_T cdf,               // shape [nlayers, nchannels, Lp]
    SYM_ACC_T input_sym,         // shape [nlayers, ntokens, nchannels]
    OUTPUT_ACC_T output_buffer,  // shape [nlayers, nchannels,
                                 // OUTPUT_BUFFER_LENGTH_PER_THREAD]
    LEN_ACC_T output_lengths,    // shape [nlayers, nchannels]
    int32_t lp, int32_t ntokens) {
  // The shared memory will be split to 2 parts:
  // 1. The CDF tensor, with shape [MAX_LP, BLOCK_SIZE)] (only used [LP,
  // BLOCK_SIZE] part)
  // 2. The output buffer, with shape [BLOCK_SIZE, MAX_SHARED_PER_THREAD -
  // MAX_LP * 2] uint8s
  __shared__ uint16_t cdf_shared[MAX_LP * BLOCK_SIZE];
  __shared__ uint8_t
      output_shared[BLOCK_SIZE * OUTPUT_BUFFER_LENGTH_PER_THREAD];

  const int layer_id = blockIdx.x;
  const int channel_id = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  // Copy the CDF[layer_id, channel_start:channel_end, :] to shared memory
  const int cdf_size = BLOCK_SIZE * lp;
  for (int i = threadIdx.x; i < cdf_size; i += blockDim.x) {
    const int cid = i / lp;
    const int lid = i % lp;
    const int shared_offset = lid * BLOCK_SIZE + cid;
    const int16_t value = cdf[layer_id][cid + blockIdx.y * BLOCK_SIZE][lid];
    cdf_shared[shared_offset] = static_cast<uint16_t>(value);
  }

  __syncthreads();

  // Do the actual encoding
  uint32_t low = 0U;
  uint32_t high = 0xFFFFFFFFU;
  uint64_t pending_bits = 0;
  const int max_symbol = lp - 2;

  uint32_t output_reg = 0;
  int output_reg_len = 0;
  int output_shared_offset = threadIdx.x * OUTPUT_BUFFER_LENGTH_PER_THREAD;

  for (int i = 0; i < ntokens; i++) {
    const uint8_t sym =
        input_sym[layer_id][i][channel_id];  //[layer_id * ntokens * nchannels +
                                             // i * nchannels + channel_id];
    const uint64_t span =
        static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;

    const uint32_t c_low = cdf_shared[sym * BLOCK_SIZE + threadIdx.x];
    const uint32_t c_high =
        sym == max_symbol ? 0x10000U
                          : cdf_shared[(sym + 1) * BLOCK_SIZE + threadIdx.x];

    high = (low - 1) + ((span * static_cast<uint64_t>(c_high)) >> precision);
    low = (low) + ((span * static_cast<uint64_t>(c_low)) >> precision);

    while (true) {
      if (high < 0x80000000U) {
        append_bit_and_pending(0, pending_bits, output_reg, output_reg_len,
                               output_shared, output_shared_offset);
        low <<= 1;
        high <<= 1;
        high |= 1;
      } else if (low >= 0x80000000U) {
        append_bit_and_pending(1, pending_bits, output_reg, output_reg_len,
                               output_shared, output_shared_offset);
        low <<= 1;
        high <<= 1;
        high |= 1;
      } else if (low >= 0x40000000U && high < 0xC0000000U) {
        pending_bits++;
        low <<= 1;
        low &= 0x7FFFFFFF;
        high <<= 1;
        high |= 0x80000001;
      } else {
        break;
      }
    }
  }

  pending_bits += 1;

  if (low < 0x40000000U) {
    append_bit_and_pending(0, pending_bits, output_reg, output_reg_len,
                           output_shared, output_shared_offset);
  } else {
    append_bit_and_pending(1, pending_bits, output_reg, output_reg_len,
                           output_shared, output_shared_offset);
  }

  spill_partial_reg_to_shared(output_reg, output_reg_len, output_shared,
                              output_shared_offset);
  output_lengths[layer_id][channel_id] =
      output_shared_offset - threadIdx.x * OUTPUT_BUFFER_LENGTH_PER_THREAD;

  __syncthreads();

  // reuse cdf for the prefix sum
  int* output_lengths_shared = reinterpret_cast<int*>(cdf_shared);
  output_lengths_shared[threadIdx.x] =
      output_shared_offset - threadIdx.x * OUTPUT_BUFFER_LENGTH_PER_THREAD;
  __syncthreads();

  // Copy the output buffer to global memory
  // then copy one "row" at a time, and make sure the write is coalesced
  for (int i = 0; i < BLOCK_SIZE; i++) {
    int length = output_lengths_shared[i];
    int current_channel = blockIdx.y * BLOCK_SIZE + i;
    for (int j = threadIdx.x; j < length; j += blockDim.x) {
      output_buffer[layer_id][current_channel][j] =
          output_shared[i * OUTPUT_BUFFER_LENGTH_PER_THREAD + j];
    }
  }
}

int get_block_size(int nchannels) {
  // find the biggest 2^n that can divide nchannels
  int factor = (nchannels ^ (nchannels - 1)) + 1;
  factor >>= 1;
  if (factor > MAX_THREAD_PER_BLOCK) {
    factor = MAX_THREAD_PER_BLOCK;
  }
  return factor;
}

/**
 * Encodes the input symbols to a list of bytestreams
 *
 * Input:
 *   cdf: the int16 CDF tensor, with shape [nlayers, nchannels, Lp], should be
 * on GPU input_sym: the int8 symbols tensor, with shape [nlayers, ntokens,
 * nchannels], should be on GPU output_buffer: the output buffer of int8, with
 * shape [nlayers, nchannels, ntokens * 2] should be on GPU output_lengths: the
 * output lengths of int32, with shape [nlayers, nchannels], should be on GPU
 *
 * Note:
 *   The block and grid mapping is as follows:
 *   - Each thread is responsible for 1 layer, 1 channel, all the tokens
 *   - Each block is responsible for 1 layer, 256 channel # TODO: maybe make
 * '256' configurable
 *   - Each grid consists of (nlayers, nchannels / 256) blocks
 *   Each block will copy all the CDFs and output buffers to shared memory
 *   On Volta, ADA and Ampere, the shared memory per thread is ~1KB. The CDF
 * will take Lp * 2 bytes (66 bytes), and the output buffer will take ntokens *
 * 2 bytes. So ideally the ntokens should be less than 512.
 */
void encode_cuda_new(const at::Tensor& cdf, const at::Tensor& input_sym,
                     at::Tensor& output_buffer, at::Tensor& output_lengths) {
  // TODO: if the input tensor have too many tokens, the shared memory will not
  // be enough to hold the output
  //       The current boundary is around 256 tokens

  /* Input validation */
  TORCH_CHECK(cdf.is_cuda(), "CDF should be on GPU");
  TORCH_CHECK(input_sym.is_cuda(), "Input symbols should be on GPU");
  TORCH_CHECK(output_buffer.is_cuda(), "Output buffer should be on GPU");
  TORCH_CHECK(output_lengths.is_cuda(), "Output lengths should be on GPU");

  const auto cdf_shape = cdf.sizes();
  const auto input_shape = input_sym.sizes();
  const auto output_shape = output_buffer.sizes();
  const auto output_lengths_shape = output_lengths.sizes();
  TORCH_CHECK(cdf_shape[0] == input_shape[0],
              "CDF and input should have the same number of layers");
  TORCH_CHECK(cdf_shape[1] == input_shape[2],
              "CDF and input should have the same number of layers");
  TORCH_CHECK(output_shape[0] == cdf_shape[0],
              "Output buffer should have the same number of layers as CDF");
  TORCH_CHECK(output_shape[1] == cdf_shape[1],
              "Output buffer should have the same number of channels as CDF");
  TORCH_CHECK(output_lengths_shape[0] == cdf_shape[0],
              "Output lengths should have the same number of layers as CDF");
  TORCH_CHECK(output_lengths_shape[1] == cdf_shape[1],
              "Output lengths should have the same number of channels as CDF");

  /* set block and grid size */
  const int nlayers = cdf_shape[0];
  const int nchannels = cdf_shape[1];
  const int ntokens = input_shape[1];
  // const int output_buffer_length_per_thread = output_shape[2];
  const int block_size = get_block_size(nchannels);
  TORCH_CHECK(ntokens <= MAX_TOKENS_PER_THREAD,
              "Number of tokens should be less than or equal to ",
              MAX_TOKENS_PER_THREAD);
  TORCH_CHECK(nchannels % block_size == 0,
              "Number of channels should be divisible by block size");
  TORCH_CHECK(cdf_shape[2] <= MAX_LP, "CDF length should be less than MAX_LP");

  dim3 block_dim(block_size, 1, 1);
  dim3 grid_dim(nlayers, nchannels / block_size, 1);

  // TODO: potential optimization: use PackedAccessor32 to access the tensors,
  // in case the tensor is not contiguous
  auto cdf_accessor =
      cdf.packed_accessor32<int16_t, 3, torch::RestrictPtrTraits>();
  auto input_sym_accessor =
      input_sym.packed_accessor32<int8_t, 3, torch::RestrictPtrTraits>();
  auto output_buffer_accessor =
      output_buffer.packed_accessor32<uint8_t, 3, torch::RestrictPtrTraits>();
  auto output_lengths_accessor =
      output_lengths.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>();

  /* Call the kernel */
#ifndef LAUNCH_ENCODE_KERNEL
  #define LAUNCH_ENCODE_KERNEL(block_size)                            \
    encode_with_accessor_kernel<block_size><<<grid_dim, block_dim>>>( \
        cdf_accessor, input_sym_accessor, output_buffer_accessor,     \
        output_lengths_accessor, cdf_shape[2], ntokens)
//#define LAUNCH_ENCODE_KERNEL(block_size) \
//    encode_kernel<block_size><<<grid_dim, block_dim>>>( \
//        (const uint16_t *)cdf.data_ptr<int16_t>(), \
//        (const uint8_t *)input_sym.data_ptr<int8_t>(), \
//        output_buffer.data_ptr<uint8_t>(), \
//        output_lengths.data_ptr<int32_t>(), \
//        cdf_shape[2], \
//        ntokens, \
//        output_buffer_length_per_thread \
//    )
#endif

  switch (block_size) {
    case 1:
      LAUNCH_ENCODE_KERNEL(1);
      break;
    case 2:
      LAUNCH_ENCODE_KERNEL(2);
      break;
    case 4:
      LAUNCH_ENCODE_KERNEL(4);
      break;
    case 8:
      LAUNCH_ENCODE_KERNEL(8);
      break;
    case 16:
      LAUNCH_ENCODE_KERNEL(16);
      break;
    case 32:
      LAUNCH_ENCODE_KERNEL(32);
      break;
    case 64:
      LAUNCH_ENCODE_KERNEL(64);
      break;
    case 128:
      LAUNCH_ENCODE_KERNEL(128);
      break;
    default:
      throw std::runtime_error("Unsupported block size");
  }
}