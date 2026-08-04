// SPDX-License-Identifier: Apache-2.0
//
// Arithmetic-coding decoder for Intel XPU.  Provides:
//   decode_fast_new_xpu      -- per-channel buffer input
//   decode_fast_prefsum_xpu  -- 1-D packed bytestream + prefix-sum offsets
//
// Algorithm
// ---------
//   One work-item per (layer, channel); each runs an independent
//   arithmetic-decoder state machine over `ntokens` symbols, with an
//   unrolled binary search over the channel's CDF.
//
// Intel XPU mapping
// -----------------
//   - 2-D nd_range (layer x channel_block * BLOCK_SIZE).
//   - SLM: CDF [MAX_LP * BLOCK_SIZE] uint16, per-channel byte buffer
//     [BLOCK_SIZE * OUTPUT_BUFFER_LENGTH_PER_THREAD] uint8.
//   - Sub-group size locked to 16 (Xe native SIMD).
//   - Binary search uses a simple loop with compile-time bound
//     max_symbol <= MAX_LP-2 < 64 so the IGC unrolls it to ~6 comparisons.
//
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <sycl/sycl.hpp>
#pragma GCC diagnostic pop

#include <torch/all.h>
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>

#include "cachegen_kernels_sycl.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace {

constexpr int MAX_LP = 64;
constexpr int OUTPUT_BUFFER_LENGTH_PER_THREAD = 256;
constexpr int MAX_TOKENS_PER_THREAD = 256;
constexpr int PRECISION = 16;
constexpr int INTEL_SUB_GROUP_SIZE = 16;

inline uint32_t big_to_small_u32(uint32_t value) {
  return ((value & 0xFF000000u) >> 24) | ((value & 0x00FF0000u) >> 8) |
         ((value & 0x0000FF00u) << 8) | ((value & 0x000000FFu) << 24);
}

template <int BLOCK_SIZE, typename SLMRef>
inline uint16_t binsearch(SLMRef cdf_shared, uint16_t target, uint8_t max_sym,
                          int tid) {
  uint16_t left = 0;
  uint16_t right = static_cast<uint16_t>(max_sym + 1);
  while (left + 1 < right) {
    const uint16_t m = static_cast<uint16_t>((left + right) / 2);
    const int offset = m * BLOCK_SIZE + tid;
    const uint16_t v = cdf_shared[offset];
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

template <int BLOCK_SIZE>
void launch_decode_per_channel(sycl::queue& queue, const int16_t* cdf,
                               const uint8_t* bytestreams,
                               const int32_t* lengths, uint8_t* output,
                               int nlayers, int nchannels, int ntokens,
                               int lp) {
  const int channel_blocks = nchannels / BLOCK_SIZE;
  sycl::range<2> global_range(static_cast<size_t>(nlayers),
                              static_cast<size_t>(channel_blocks) * BLOCK_SIZE);
  sycl::range<2> local_range(1, BLOCK_SIZE);

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<uint16_t, 1> cdf_shared(
        sycl::range<1>(MAX_LP * BLOCK_SIZE), cgh);
    sycl::local_accessor<uint8_t, 1> bs_shared(
        sycl::range<1>(BLOCK_SIZE * OUTPUT_BUFFER_LENGTH_PER_THREAD), cgh);
    sycl::local_accessor<int32_t, 1> len_shared(sycl::range<1>(BLOCK_SIZE),
                                                cgh);

    cgh.parallel_for(
        sycl::nd_range<2>(global_range, local_range),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(
            INTEL_SUB_GROUP_SIZE)]] {
          const int layer_id = static_cast<int>(item.get_group(0));
          const int block_y = static_cast<int>(item.get_group(1));
          const int tx = static_cast<int>(item.get_local_id(1));
          const int global_channel_offset = block_y * BLOCK_SIZE;
          const int global_channel_id = global_channel_offset + tx;
          const int max_symbol = lp - 2;

          // Load per-channel lengths into SLM.
          len_shared[tx] =
              lengths[layer_id * nchannels + global_channel_offset + tx];
          item.barrier(sycl::access::fence_space::local_space);

          // Load per-channel byte buffers into SLM, flat sweep over
          // BLOCK_SIZE * OUTPUT_BUFFER_LENGTH_PER_THREAD bytes -- each
          // sub-group lane handles every BLOCK_SIZE-th byte across the
          // entire tile, removing the per-channel outer loop and giving
          // the IGC more freedom to schedule the loads.
          const int total_bytes = BLOCK_SIZE * OUTPUT_BUFFER_LENGTH_PER_THREAD;
          for (int k = tx; k < total_bytes; k += BLOCK_SIZE) {
            const int i = k / OUTPUT_BUFFER_LENGTH_PER_THREAD;
            const int j = k - i * OUTPUT_BUFFER_LENGTH_PER_THREAD;
            const int channel_id = global_channel_offset + i;
            const int length = len_shared[i];
            uint8_t v = 0;
            if (j < length) {
              const int64_t off =
                  (static_cast<int64_t>(layer_id) * nchannels + channel_id) *
                      OUTPUT_BUFFER_LENGTH_PER_THREAD +
                  j;
              v = bytestreams[off];
            }
            bs_shared[k] = v;
          }
          item.barrier(sycl::access::fence_space::local_space);

          // Load CDF[layer_id, channel_block, :] into SLM (column-major).
          const int cdf_size = lp * BLOCK_SIZE;
          for (int i = tx; i < cdf_size; i += BLOCK_SIZE) {
            const int cid = i / lp;
            const int lid = i % lp;
            const int16_t v = cdf[(static_cast<int64_t>(layer_id) * nchannels +
                                   global_channel_offset + cid) *
                                      lp +
                                  lid];
            cdf_shared[lid * BLOCK_SIZE + cid] = static_cast<uint16_t>(v);
          }
          item.barrier(sycl::access::fence_space::local_space);

          // ---------- AC decode state ----------
          uint32_t low = 0u;
          uint32_t high = 0xFFFFFFFFu;
          const uint32_t c_count = 0x10000u;

          uint8_t byte_buffer = 0;
          int bit_idx = 1;  // next bit: (byte_buffer >> (8-bit_idx)) & 1
          int byte_buffer_offset = 4;

          // Initial 32-bit value from first 4 bytes (big-endian); on
          // little-endian devices this matches a packed load followed
          // by a byte-swap.
          const int row_base = tx * OUTPUT_BUFFER_LENGTH_PER_THREAD;
          uint32_t v0 = (static_cast<uint32_t>(bs_shared[row_base]) << 24) |
                        (static_cast<uint32_t>(bs_shared[row_base + 1]) << 16) |
                        (static_cast<uint32_t>(bs_shared[row_base + 2]) << 8) |
                        static_cast<uint32_t>(bs_shared[row_base + 3]);
          (void)big_to_small_u32;
          uint32_t value = v0;
          byte_buffer = bs_shared[row_base + byte_buffer_offset];

          for (int i = 0; i < ntokens; ++i) {
            const uint64_t span =
                static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;
            const uint16_t count =
                static_cast<uint16_t>(((static_cast<uint64_t>(value) -
                                        static_cast<uint64_t>(low) + 1) *
                                           c_count -
                                       1) /
                                      span);

            const uint16_t sym_i = binsearch<BLOCK_SIZE>(
                cdf_shared, count, static_cast<uint8_t>(max_symbol), tx);

            output[(static_cast<int64_t>(layer_id) * ntokens + i) * nchannels +
                   global_channel_id] = static_cast<uint8_t>(sym_i);

            if (i == ntokens - 1) break;

            const uint32_t c_low = cdf_shared[sym_i * BLOCK_SIZE + tx];
            const uint32_t c_high =
                sym_i == max_symbol ? 0x10000u
                                    : cdf_shared[(sym_i + 1) * BLOCK_SIZE + tx];

            high =
                (low - 1) + static_cast<uint32_t>((span * c_high) >> PRECISION);
            low = low + static_cast<uint32_t>((span * c_low) >> PRECISION);

            while (true) {
              if (low >= 0x80000000u || high < 0x80000000u) {
                low <<= 1;
                high <<= 1;
                high |= 1;
                value = (value << 1) | ((byte_buffer >> (8 - bit_idx)) & 1u);
                bit_idx += 1;
              } else if (low >= 0x40000000u && high < 0xC0000000u) {
                low <<= 1;
                low &= 0x7FFFFFFFu;
                high <<= 1;
                high |= 0x80000001u;
                value -= 0x40000000u;
                value = (value << 1) | ((byte_buffer >> (8 - bit_idx)) & 1u);
                bit_idx += 1;
              } else {
                break;
              }

              if (bit_idx == 9) {
                bit_idx = 1;
                byte_buffer_offset += 1;
                byte_buffer = bs_shared[row_base + byte_buffer_offset];
              }
            }
          }
        });
  });
}

template <int BLOCK_SIZE>
void launch_decode_prefsum(sycl::queue& queue, const int16_t* cdf,
                           const uint8_t* bytestreams,
                           const int64_t* lengths_prefix, uint8_t* output,
                           int nlayers, int nchannels, int ntokens, int lp) {
  const int channel_blocks = nchannels / BLOCK_SIZE;
  sycl::range<2> global_range(static_cast<size_t>(nlayers),
                              static_cast<size_t>(channel_blocks) * BLOCK_SIZE);
  sycl::range<2> local_range(1, BLOCK_SIZE);

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<uint16_t, 1> cdf_shared(
        sycl::range<1>(MAX_LP * BLOCK_SIZE), cgh);
    sycl::local_accessor<uint8_t, 1> bs_shared(
        sycl::range<1>(BLOCK_SIZE * OUTPUT_BUFFER_LENGTH_PER_THREAD), cgh);
    sycl::local_accessor<int32_t, 1> sum_shared(sycl::range<1>(BLOCK_SIZE + 1),
                                                cgh);

    cgh.parallel_for(
        sycl::nd_range<2>(global_range, local_range),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(
            INTEL_SUB_GROUP_SIZE)]] {
          const int layer_id = static_cast<int>(item.get_group(0));
          const int block_y = static_cast<int>(item.get_group(1));
          const int tx = static_cast<int>(item.get_local_id(1));
          const int global_channel_offset = block_y * BLOCK_SIZE;
          const int global_channel_id = global_channel_offset + tx;
          const int max_symbol = lp - 2;

          // Load running prefix sums for [BLOCK_SIZE + 1] positions.
          // Index is the *global* (layer, channel) flattened position.
          for (int i = tx; i < BLOCK_SIZE + 1; i += BLOCK_SIZE) {
            const int gid =
                layer_id * nchannels + global_channel_offset + i - 1;
            int32_t val = 0;
            if (gid >= 0) {
              const int row = gid / nchannels;
              const int col = gid % nchannels;
              val = static_cast<int32_t>(lengths_prefix[row * nchannels + col]);
            }
            sum_shared[i] = val;
          }
          item.barrier(sycl::access::fence_space::local_space);

          // Load per-channel byte buffers via prefix-sum windowing.
          for (int i = 0; i < BLOCK_SIZE; ++i) {
            const int start_offset = sum_shared[i];
            const int end_offset = sum_shared[i + 1];
            const int length = end_offset - start_offset;
            for (int j = tx; j < OUTPUT_BUFFER_LENGTH_PER_THREAD;
                 j += BLOCK_SIZE) {
              uint8_t v = (j < length) ? bytestreams[start_offset + j] : 0;
              bs_shared[i * OUTPUT_BUFFER_LENGTH_PER_THREAD + j] = v;
            }
          }
          item.barrier(sycl::access::fence_space::local_space);

          // Load CDF.
          const int cdf_size = lp * BLOCK_SIZE;
          for (int i = tx; i < cdf_size; i += BLOCK_SIZE) {
            const int cid = i / lp;
            const int lid = i % lp;
            const int16_t v = cdf[(static_cast<int64_t>(layer_id) * nchannels +
                                   global_channel_offset + cid) *
                                      lp +
                                  lid];
            cdf_shared[lid * BLOCK_SIZE + cid] = static_cast<uint16_t>(v);
          }
          item.barrier(sycl::access::fence_space::local_space);

          // Decode (identical body to per-channel kernel).
          uint32_t low = 0u;
          uint32_t high = 0xFFFFFFFFu;
          const uint32_t c_count = 0x10000u;

          uint8_t byte_buffer = 0;
          int bit_idx = 1;
          int byte_buffer_offset = 4;

          const int row_base = tx * OUTPUT_BUFFER_LENGTH_PER_THREAD;
          uint32_t value =
              (static_cast<uint32_t>(bs_shared[row_base]) << 24) |
              (static_cast<uint32_t>(bs_shared[row_base + 1]) << 16) |
              (static_cast<uint32_t>(bs_shared[row_base + 2]) << 8) |
              static_cast<uint32_t>(bs_shared[row_base + 3]);
          byte_buffer = bs_shared[row_base + byte_buffer_offset];

          for (int i = 0; i < ntokens; ++i) {
            const uint64_t span =
                static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;
            const uint16_t count =
                static_cast<uint16_t>(((static_cast<uint64_t>(value) -
                                        static_cast<uint64_t>(low) + 1) *
                                           c_count -
                                       1) /
                                      span);

            const uint16_t sym_i = binsearch<BLOCK_SIZE>(
                cdf_shared, count, static_cast<uint8_t>(max_symbol), tx);
            output[(static_cast<int64_t>(layer_id) * ntokens + i) * nchannels +
                   global_channel_id] = static_cast<uint8_t>(sym_i);

            if (i == ntokens - 1) break;

            const uint32_t c_low = cdf_shared[sym_i * BLOCK_SIZE + tx];
            const uint32_t c_high =
                sym_i == max_symbol ? 0x10000u
                                    : cdf_shared[(sym_i + 1) * BLOCK_SIZE + tx];

            high =
                (low - 1) + static_cast<uint32_t>((span * c_high) >> PRECISION);
            low = low + static_cast<uint32_t>((span * c_low) >> PRECISION);

            while (true) {
              if (low >= 0x80000000u || high < 0x80000000u) {
                low <<= 1;
                high <<= 1;
                high |= 1;
                value = (value << 1) | ((byte_buffer >> (8 - bit_idx)) & 1u);
                bit_idx += 1;
              } else if (low >= 0x40000000u && high < 0xC0000000u) {
                low <<= 1;
                low &= 0x7FFFFFFFu;
                high <<= 1;
                high |= 0x80000001u;
                value -= 0x40000000u;
                value = (value << 1) | ((byte_buffer >> (8 - bit_idx)) & 1u);
                bit_idx += 1;
              } else {
                break;
              }
              if (bit_idx == 9) {
                bit_idx = 1;
                byte_buffer_offset += 1;
                byte_buffer = bs_shared[row_base + byte_buffer_offset];
              }
            }
          }
        });
  });
}

int decoder_get_block_size(int nchannels) {
  int factor = (nchannels ^ (nchannels - 1)) + 1;
  factor >>= 1;
  if (factor > 128) factor = 128;
  return factor;
}

#define DISPATCH_DECODE_BLOCKSIZE(LAUNCHER, BLOCKSIZE_VAR, ...) \
  switch (BLOCKSIZE_VAR) {                                      \
    case 1:                                                     \
      LAUNCHER<1>(__VA_ARGS__);                                 \
      break;                                                    \
    case 2:                                                     \
      LAUNCHER<2>(__VA_ARGS__);                                 \
      break;                                                    \
    case 4:                                                     \
      LAUNCHER<4>(__VA_ARGS__);                                 \
      break;                                                    \
    case 8:                                                     \
      LAUNCHER<8>(__VA_ARGS__);                                 \
      break;                                                    \
    case 16:                                                    \
      LAUNCHER<16>(__VA_ARGS__);                                \
      break;                                                    \
    case 32:                                                    \
      LAUNCHER<32>(__VA_ARGS__);                                \
      break;                                                    \
    case 64:                                                    \
      LAUNCHER<64>(__VA_ARGS__);                                \
      break;                                                    \
    case 128:                                                   \
      LAUNCHER<128>(__VA_ARGS__);                               \
      break;                                                    \
    default:                                                    \
      throw std::runtime_error("unsupported block size");       \
  }

}  // namespace

void decode_fast_new_xpu(const at::Tensor& cdf, const at::Tensor& bytestreams,
                         const at::Tensor& lengths, at::Tensor& output) {
  if (!cdf.device().is_xpu() || !bytestreams.device().is_xpu() ||
      !lengths.device().is_xpu() || !output.device().is_xpu()) {
    throw std::runtime_error("decode_fast_new_xpu: tensors must be on XPU");
  }

  TORCH_CHECK(cdf.scalar_type() == at::kShort,
              "decode_fast_new_xpu: cdf must be int16");
  TORCH_CHECK(bytestreams.scalar_type() == at::kByte,
              "decode_fast_new_xpu: bytestreams must be uint8");
  TORCH_CHECK(lengths.scalar_type() == at::kInt,
              "decode_fast_new_xpu: lengths must be int32");
  TORCH_CHECK(
      output.scalar_type() == at::kByte || output.scalar_type() == at::kChar,
      "decode_fast_new_xpu: output must be uint8 or int8");

  TORCH_CHECK(cdf.dim() == 3,
              "decode_fast_new_xpu: cdf must be 3-D [nlayers, nchannels, lp]");
  TORCH_CHECK(bytestreams.dim() == 3,
              "decode_fast_new_xpu: bytestreams must be 3-D "
              "[nlayers, nchannels, OUTPUT_BUFFER_LENGTH_PER_THREAD]");
  TORCH_CHECK(lengths.dim() == 2,
              "decode_fast_new_xpu: lengths must be 2-D [nlayers, nchannels]");
  TORCH_CHECK(
      output.dim() == 3,
      "decode_fast_new_xpu: output must be 3-D [nlayers, ntokens, nchannels]");

  TORCH_CHECK(bytestreams.size(2) == OUTPUT_BUFFER_LENGTH_PER_THREAD,
              "decode_fast_new_xpu: bytestreams last dim must equal "
              "OUTPUT_BUFFER_LENGTH_PER_THREAD (256)");

  TORCH_CHECK(lengths.numel() == 0 || lengths.max().item<int32_t>() <=
                                          OUTPUT_BUFFER_LENGTH_PER_THREAD,
              "decode_fast_new_xpu: per-channel length exceeds "
              "OUTPUT_BUFFER_LENGTH_PER_THREAD (256)");

  const auto cdf_shape = cdf.sizes();
  const auto out_shape = output.sizes();
  const int nlayers = static_cast<int>(cdf_shape[0]);
  const int nchannels = static_cast<int>(cdf_shape[1]);
  const int lp = static_cast<int>(cdf_shape[2]);
  const int ntokens = static_cast<int>(out_shape[1]);

  TORCH_CHECK(lp >= 2, "decode_fast_new_xpu: cdf last dim (lp) must be >= 2");

  if (ntokens > MAX_TOKENS_PER_THREAD) {
    throw std::runtime_error(
        "decode_fast_new_xpu: ntokens > MAX_TOKENS_PER_THREAD");
  }
  if (lp > MAX_LP) {
    throw std::runtime_error("decode_fast_new_xpu: lp > MAX_LP");
  }

  const int block_size = decoder_get_block_size(nchannels);
  if (nchannels % block_size != 0) {
    throw std::runtime_error(
        "decode_fast_new_xpu: nchannels must be divisible by block size");
  }

  auto cdf_c = cdf.is_contiguous() ? cdf : cdf.contiguous();
  auto bs_c =
      bytestreams.is_contiguous() ? bytestreams : bytestreams.contiguous();
  auto len_c = lengths.is_contiguous() ? lengths : lengths.contiguous();

  const c10::DeviceGuard guard(cdf.device());
  sycl::queue& queue =
      c10::xpu::getCurrentXPUStream(cdf.device().index()).queue();

  const int16_t* cdf_ptr = cdf_c.data_ptr<int16_t>();
  const uint8_t* bs_ptr = reinterpret_cast<const uint8_t*>(bs_c.data_ptr());
  const int32_t* len_ptr = len_c.data_ptr<int32_t>();
  uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data_ptr());

  DISPATCH_DECODE_BLOCKSIZE(launch_decode_per_channel, block_size, queue,
                            cdf_ptr, bs_ptr, len_ptr, out_ptr, nlayers,
                            nchannels, ntokens, lp);
}

void decode_fast_prefsum_xpu(const at::Tensor& cdf,
                             const at::Tensor& bytestreams,
                             const at::Tensor& lengths_prefsum,
                             at::Tensor& output) {
  if (!cdf.device().is_xpu() || !bytestreams.device().is_xpu() ||
      !lengths_prefsum.device().is_xpu() || !output.device().is_xpu()) {
    throw std::runtime_error("decode_fast_prefsum_xpu: tensors must be on XPU");
  }

  TORCH_CHECK(cdf.scalar_type() == at::kShort,
              "decode_fast_prefsum_xpu: cdf must be int16");
  TORCH_CHECK(bytestreams.scalar_type() == at::kByte,
              "decode_fast_prefsum_xpu: bytestreams must be uint8");
  TORCH_CHECK(lengths_prefsum.scalar_type() == at::kLong,
              "decode_fast_prefsum_xpu: lengths_prefsum must be int64");
  TORCH_CHECK(
      output.scalar_type() == at::kByte || output.scalar_type() == at::kChar,
      "decode_fast_prefsum_xpu: output must be uint8 or int8");

  TORCH_CHECK(
      cdf.dim() == 3,
      "decode_fast_prefsum_xpu: cdf must be 3-D [nlayers, nchannels, lp]");
  TORCH_CHECK(lengths_prefsum.dim() == 2,
              "decode_fast_prefsum_xpu: lengths_prefsum must be 2-D "
              "[nlayers, nchannels]");
  TORCH_CHECK(output.dim() == 3,
              "decode_fast_prefsum_xpu: output must be 3-D "
              "[nlayers, ntokens, nchannels]");

  const auto cdf_shape = cdf.sizes();
  const auto out_shape = output.sizes();
  const int nlayers = static_cast<int>(cdf_shape[0]);
  const int nchannels = static_cast<int>(cdf_shape[1]);
  const int lp = static_cast<int>(cdf_shape[2]);
  const int ntokens = static_cast<int>(out_shape[1]);

  TORCH_CHECK(lp >= 2,
              "decode_fast_prefsum_xpu: cdf last dim (lp) must be >= 2");

  if (ntokens > MAX_TOKENS_PER_THREAD) {
    throw std::runtime_error(
        "decode_fast_prefsum_xpu: ntokens > MAX_TOKENS_PER_THREAD");
  }
  if (lp > MAX_LP) {
    throw std::runtime_error("decode_fast_prefsum_xpu: lp > MAX_LP");
  }

  const int block_size = decoder_get_block_size(nchannels);
  if (nchannels % block_size != 0) {
    throw std::runtime_error(
        "decode_fast_prefsum_xpu: nchannels must be divisible by block size");
  }

  auto cdf_c = cdf.is_contiguous() ? cdf : cdf.contiguous();
  auto bs_c =
      bytestreams.is_contiguous() ? bytestreams : bytestreams.contiguous();
  auto pref_c = lengths_prefsum.is_contiguous() ? lengths_prefsum
                                                : lengths_prefsum.contiguous();

  const c10::DeviceGuard guard(cdf.device());
  sycl::queue& queue =
      c10::xpu::getCurrentXPUStream(cdf.device().index()).queue();

  const int16_t* cdf_ptr = cdf_c.data_ptr<int16_t>();
  const uint8_t* bs_ptr = reinterpret_cast<const uint8_t*>(bs_c.data_ptr());
  const int64_t* pref_ptr = pref_c.data_ptr<int64_t>();
  uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data_ptr());

  DISPATCH_DECODE_BLOCKSIZE(launch_decode_prefsum, block_size, queue, cdf_ptr,
                            bs_ptr, pref_ptr, out_ptr, nlayers, nchannels,
                            ntokens, lp);
}
