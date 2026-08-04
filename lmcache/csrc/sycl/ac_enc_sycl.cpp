// SPDX-License-Identifier: Apache-2.0
//
// SYCL/XPU arithmetic encoder for CacheGen.
//
// Algorithm
// ---------
//   Each (layer, channel) pair runs an independent arithmetic-coder state
//   machine over `ntokens` 8-bit symbols and writes a per-channel byte
//   stream into `output_buffer` along with the stream length into
//   `output_lengths`.
//
// Intel XPU mapping
// -----------------
//   - 2-D nd_range:
//       group axis 0 = layer
//       group axis 1 = channel block (BLOCK_SIZE work-items per WG)
//   - Each work-item owns ONE (layer, channel).  The state machine is
//     intrinsically sequential, so we don't parallelise across tokens.
//   - SLM:
//       cdf_shared      [MAX_LP * BLOCK_SIZE] uint16  -- column-major
//                                                       (lid * BLOCK_SIZE + tx)
//       output_shared   [BLOCK_SIZE * OUTPUT_BUFFER_LENGTH_PER_THREAD] u8
//     Column-major CDF layout lets a sub-group of 16 lanes read 16
//     *contiguous* uint16s when looking up cdf[s], giving fully coalesced
//     SLM access without bank conflicts (Xe SLM banks are 4-byte wide, so
//     two consecutive lanes pack into the same bank line).
//   - Sub-group size locked to 16 (Intel native SIMD).
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

// Encoder caps lp at 48 to keep cdf_shared SLM at 48*BLOCK_SIZE*2B = 12KB
// for BLOCK_SIZE=128.  Combined with output_shared (32KB) this fits one WG
// per Xe sub-slice; bumping to 64 pushes total SLM past the occupancy
// sweet spot and measurably slows the kernel.
constexpr int MAX_LP = 48;
constexpr int OUTPUT_BUFFER_LENGTH_PER_THREAD = 256;
constexpr int MAX_TOKENS_PER_THREAD = 256;
constexpr int PRECISION = 16;
constexpr int INTEL_SUB_GROUP_SIZE = 16;

// ----- bit-stream helpers (functor-style; no recursion, no allocation) ---

template <typename SLMRef>
inline void spill_reg_to_shared(uint32_t& output_reg, int& output_reg_len,
                                SLMRef output_shared,
                                int& output_shared_offset) {
  // output_reg holds 32 filled bits; write them out big-endian.
  // Byte-by-byte because output_shared_offset is not guaranteed to be
  // 4-byte aligned.
  output_shared[output_shared_offset] = static_cast<uint8_t>(output_reg >> 24);
  output_shared[output_shared_offset + 1] =
      static_cast<uint8_t>((output_reg >> 16) & 0xFFu);
  output_shared[output_shared_offset + 2] =
      static_cast<uint8_t>((output_reg >> 8) & 0xFFu);
  output_shared[output_shared_offset + 3] =
      static_cast<uint8_t>(output_reg & 0xFFu);
  output_shared_offset += 4;
  output_reg = 0;
  output_reg_len = 0;
}

template <typename SLMRef>
inline void spill_partial_reg_to_shared(uint32_t output_reg,
                                        int& output_reg_len,
                                        SLMRef output_shared,
                                        int& output_shared_offset) {
  output_reg <<= 32 - output_reg_len;
  while (output_reg_len > 0) {
    output_reg_len -= 8;
    output_shared[output_shared_offset] =
        static_cast<uint8_t>(output_reg >> 24);
    output_shared_offset++;
    output_reg <<= 8;
  }
}

template <typename SLMRef>
inline void add_bits_to_output(uint32_t bit, int num, uint32_t& output_reg,
                               int& output_reg_len, SLMRef output_shared,
                               int& output_shared_offset) {
  do {
    const int remaining = sycl::min<int>(num, 32 - output_reg_len);
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

template <typename SLMRef>
inline void append_bit_and_pending(uint32_t bit, uint64_t& pending_bits,
                                   uint32_t& output_reg, int& output_reg_len,
                                   SLMRef output_shared,
                                   int& output_shared_offset) {
  add_bits_to_output(bit, 1, output_reg, output_reg_len, output_shared,
                     output_shared_offset);
  add_bits_to_output(1 - bit, static_cast<int>(pending_bits), output_reg,
                     output_reg_len, output_shared, output_shared_offset);
  pending_bits = 0;
}

// -------------------------------------------------------------------------

template <int BLOCK_SIZE>
void launch_encode(sycl::queue& queue, const int16_t* cdf,
                   const uint8_t* input_sym, uint8_t* output_buffer,
                   int32_t* output_lengths, int nlayers, int nchannels,
                   int ntokens, int lp) {
  const int channel_blocks = nchannels / BLOCK_SIZE;
  const int output_buffer_length_per_thread = OUTPUT_BUFFER_LENGTH_PER_THREAD;

  sycl::range<2> global_range(static_cast<size_t>(nlayers),
                              static_cast<size_t>(channel_blocks) * BLOCK_SIZE);
  sycl::range<2> local_range(1, BLOCK_SIZE);

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<uint16_t, 1> cdf_shared(
        sycl::range<1>(MAX_LP * BLOCK_SIZE), cgh);
    sycl::local_accessor<uint8_t, 1> output_shared(
        sycl::range<1>(BLOCK_SIZE * OUTPUT_BUFFER_LENGTH_PER_THREAD), cgh);
    // Per-channel byte-count scratch (used after the encode loop to drive
    // the coalesced write-out).
    sycl::local_accessor<int32_t, 1> lengths_shared(sycl::range<1>(BLOCK_SIZE),
                                                    cgh);

    cgh.parallel_for(
        sycl::nd_range<2>(global_range, local_range),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(
            INTEL_SUB_GROUP_SIZE)]] {
          const int layer_id = static_cast<int>(item.get_group(0));
          const int block_y = static_cast<int>(item.get_group(1));
          const int tx = static_cast<int>(item.get_local_id(1));
          const int channel_id = block_y * BLOCK_SIZE + tx;

          // Load CDF[layer_id, block_y*BLOCK_SIZE : +BLOCK_SIZE, :]
          const int cdf_size = BLOCK_SIZE * lp;
          for (int i = tx; i < cdf_size; i += BLOCK_SIZE) {
            const int cid = i / lp;
            const int lid = i % lp;
            const int shared_offset = lid * BLOCK_SIZE + cid;
            const int16_t v = cdf[(static_cast<int64_t>(layer_id) * nchannels +
                                   block_y * BLOCK_SIZE + cid) *
                                      lp +
                                  lid];
            cdf_shared[shared_offset] = static_cast<uint16_t>(v);
          }

          item.barrier(sycl::access::fence_space::local_space);

          // Per-(layer, channel) AC state machine.
          uint32_t low = 0u;
          uint32_t high = 0xFFFFFFFFu;
          uint64_t pending_bits = 0;
          const int max_symbol = lp - 2;

          uint32_t output_reg = 0;
          int output_reg_len = 0;
          int output_shared_offset = tx * OUTPUT_BUFFER_LENGTH_PER_THREAD;

          auto out_acc = output_shared;  // alias

          for (int i = 0; i < ntokens; ++i) {
            const uint8_t sym =
                input_sym[(static_cast<int64_t>(layer_id) * ntokens + i) *
                              nchannels +
                          channel_id];
            const uint64_t span =
                static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;
            const uint32_t c_low = cdf_shared[sym * BLOCK_SIZE + tx];
            const uint32_t c_high =
                sym == max_symbol ? 0x10000u
                                  : cdf_shared[(sym + 1) * BLOCK_SIZE + tx];

            high =
                (low - 1) + static_cast<uint32_t>((span * c_high) >> PRECISION);
            low = low + static_cast<uint32_t>((span * c_low) >> PRECISION);

            while (true) {
              if (high < 0x80000000u) {
                append_bit_and_pending(0, pending_bits, output_reg,
                                       output_reg_len, out_acc,
                                       output_shared_offset);
                low <<= 1;
                high <<= 1;
                high |= 1;
              } else if (low >= 0x80000000u) {
                append_bit_and_pending(1, pending_bits, output_reg,
                                       output_reg_len, out_acc,
                                       output_shared_offset);
                low <<= 1;
                high <<= 1;
                high |= 1;
              } else if (low >= 0x40000000u && high < 0xC0000000u) {
                pending_bits++;
                low <<= 1;
                low &= 0x7FFFFFFFu;
                high <<= 1;
                high |= 0x80000001u;
              } else {
                break;
              }
            }
          }

          pending_bits += 1;
          if (low < 0x40000000u) {
            append_bit_and_pending(0, pending_bits, output_reg, output_reg_len,
                                   out_acc, output_shared_offset);
          } else {
            append_bit_and_pending(1, pending_bits, output_reg, output_reg_len,
                                   out_acc, output_shared_offset);
          }
          spill_partial_reg_to_shared(output_reg, output_reg_len, out_acc,
                                      output_shared_offset);

          const int my_len =
              output_shared_offset - tx * OUTPUT_BUFFER_LENGTH_PER_THREAD;
          output_lengths[layer_id * nchannels + channel_id] = my_len;
          lengths_shared[tx] = my_len;

          item.barrier(sycl::access::fence_space::local_space);

          // Coalesced write-out by channel.
          for (int i = 0; i < BLOCK_SIZE; ++i) {
            const int peer_len = lengths_shared[i];
            const int current_channel = block_y * BLOCK_SIZE + i;
            for (int j = tx; j < peer_len; j += BLOCK_SIZE) {
              const int64_t global_offset =
                  (static_cast<int64_t>(layer_id) * nchannels +
                   current_channel) *
                      output_buffer_length_per_thread +
                  j;
              const int local_offset = i * OUTPUT_BUFFER_LENGTH_PER_THREAD + j;
              output_buffer[global_offset] = output_shared[local_offset];
            }
          }
        });
  });
}

int encoder_get_block_size(int nchannels) {
  int factor = (nchannels ^ (nchannels - 1)) + 1;
  factor >>= 1;
  if (factor > 128) factor = 128;
  return factor;
}

}  // namespace

void encode_fast_new_xpu(const at::Tensor& cdf, const at::Tensor& input_sym,
                         at::Tensor& output_buffer,
                         at::Tensor& output_lengths) {
  if (!cdf.device().is_xpu() || !input_sym.device().is_xpu() ||
      !output_buffer.device().is_xpu() || !output_lengths.device().is_xpu()) {
    throw std::runtime_error("encode_fast_new_xpu: all tensors must be on XPU");
  }

  TORCH_CHECK(cdf.scalar_type() == at::kShort,
              "encode_fast_new_xpu: cdf must be int16");
  TORCH_CHECK(input_sym.scalar_type() == at::kByte ||
                  input_sym.scalar_type() == at::kChar,
              "encode_fast_new_xpu: input_sym must be uint8 or int8");
  TORCH_CHECK(output_buffer.scalar_type() == at::kByte,
              "encode_fast_new_xpu: output_buffer must be uint8");
  TORCH_CHECK(output_lengths.scalar_type() == at::kInt,
              "encode_fast_new_xpu: output_lengths must be int32");

  TORCH_CHECK(cdf.dim() == 3,
              "encode_fast_new_xpu: cdf must be 3-D [nlayers, nchannels, lp]");
  TORCH_CHECK(input_sym.dim() == 3,
              "encode_fast_new_xpu: input_sym must be 3-D [nlayers, ntokens, "
              "nchannels]");
  TORCH_CHECK(output_buffer.dim() == 3,
              "encode_fast_new_xpu: output_buffer must be 3-D");
  TORCH_CHECK(output_lengths.dim() == 2,
              "encode_fast_new_xpu: output_lengths must be 2-D");

  const auto cdf_shape = cdf.sizes();
  const auto input_shape = input_sym.sizes();
  const auto output_shape = output_buffer.sizes();

  const int nlayers = static_cast<int>(cdf_shape[0]);
  const int nchannels = static_cast<int>(cdf_shape[1]);
  const int lp = static_cast<int>(cdf_shape[2]);
  const int ntokens = static_cast<int>(input_shape[1]);

  TORCH_CHECK(lp >= 2, "encode_fast_new_xpu: cdf last dim (lp) must be >= 2");

  if (ntokens > MAX_TOKENS_PER_THREAD) {
    throw std::runtime_error(
        "encode_fast_new_xpu: ntokens must be <= MAX_TOKENS_PER_THREAD");
  }
  if (lp > MAX_LP) {
    throw std::runtime_error(
        "encode_fast_new_xpu: cdf last dim must be <= MAX_LP");
  }
  if (output_shape[2] != OUTPUT_BUFFER_LENGTH_PER_THREAD) {
    throw std::runtime_error(
        "encode_fast_new_xpu: output buffer last dim must equal "
        "OUTPUT_BUFFER_LENGTH_PER_THREAD (256)");
  }

  const int block_size = encoder_get_block_size(nchannels);
  if (nchannels % block_size != 0) {
    throw std::runtime_error(
        "encode_fast_new_xpu: nchannels must be divisible by block size");
  }

  auto cdf_c = cdf.is_contiguous() ? cdf : cdf.contiguous();
  auto sym_c = input_sym.is_contiguous() ? input_sym : input_sym.contiguous();

  const c10::DeviceGuard guard(cdf.device());
  sycl::queue& queue =
      c10::xpu::getCurrentXPUStream(cdf.device().index()).queue();

  const int16_t* cdf_ptr = cdf_c.data_ptr<int16_t>();
  // Accept both Byte (uint8) and Char (int8) input symbols; the AC
  // state machine reads the raw bit pattern either way.
  const uint8_t* sym_ptr = reinterpret_cast<const uint8_t*>(sym_c.data_ptr());
  uint8_t* out_buf_ptr = reinterpret_cast<uint8_t*>(output_buffer.data_ptr());
  int32_t* out_len_ptr = output_lengths.data_ptr<int32_t>();

  switch (block_size) {
    case 1:
      launch_encode<1>(queue, cdf_ptr, sym_ptr, out_buf_ptr, out_len_ptr,
                       nlayers, nchannels, ntokens, lp);
      break;
    case 2:
      launch_encode<2>(queue, cdf_ptr, sym_ptr, out_buf_ptr, out_len_ptr,
                       nlayers, nchannels, ntokens, lp);
      break;
    case 4:
      launch_encode<4>(queue, cdf_ptr, sym_ptr, out_buf_ptr, out_len_ptr,
                       nlayers, nchannels, ntokens, lp);
      break;
    case 8:
      launch_encode<8>(queue, cdf_ptr, sym_ptr, out_buf_ptr, out_len_ptr,
                       nlayers, nchannels, ntokens, lp);
      break;
    case 16:
      launch_encode<16>(queue, cdf_ptr, sym_ptr, out_buf_ptr, out_len_ptr,
                        nlayers, nchannels, ntokens, lp);
      break;
    case 32:
      launch_encode<32>(queue, cdf_ptr, sym_ptr, out_buf_ptr, out_len_ptr,
                        nlayers, nchannels, ntokens, lp);
      break;
    case 64:
      launch_encode<64>(queue, cdf_ptr, sym_ptr, out_buf_ptr, out_len_ptr,
                        nlayers, nchannels, ntokens, lp);
      break;
    case 128:
      launch_encode<128>(queue, cdf_ptr, sym_ptr, out_buf_ptr, out_len_ptr,
                         nlayers, nchannels, ntokens, lp);
      break;
    default:
      throw std::runtime_error("encode_fast_new_xpu: unsupported block size");
  }
}
