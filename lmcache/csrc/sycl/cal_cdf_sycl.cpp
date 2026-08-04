// SPDX-License-Identifier: Apache-2.0
//
// SYCL/XPU implementation of calculate_cdf, hand-tuned for the Intel Xe
// architecture (PVC / DG2 / BMG / Arc).
//
// Algorithm
// ---------
//   For each (layer_id, channel_id) we build a histogram of the uint8
//   input values along the `ntokens` axis, prefix-sum it into a CDF,
//   normalise to a uint16 range, and add the bin index so values are
//   strictly monotonic (see normalize_cdf_value()).
//
// Intel XPU mapping
// -----------------
//   - 2-D nd_range: dim 0 = layer, dim 1 = channel (BLOCK_SIZE per WG).
//   - Each work-item owns one channel (no atomic contention).
//   - Histogram lives in Shared Local Memory (SLM) as a 2-D tile of
//     [MAX_BINS_SUPPORTED+1][BLOCK_SIZE] uint16s -- one column per
//     channel.  All traffic stays local; no cross work-item sync in the
//     inner increment loop.
//   - sub_group_size locked to 16 (native SIMD on Intel discrete GPUs).
//   - BLOCK_SIZE is a template parameter so IGC can fully unroll and
//     statically size the SLM allocation.
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

#include <cstdint>
#include <stdexcept>

namespace {

constexpr int MAX_BINS_SUPPORTED = 64;
constexpr int INTEL_SUB_GROUP_SIZE = 16;

// Linear remap into [0, 0xFFFF - max_bins].
inline uint16_t normalize_cdf_value(uint16_t cdf_value, uint16_t total_count,
                                    int max_bins) {
  const uint32_t MAX_UINT16_VALUE = 0xFFFFu - static_cast<uint32_t>(max_bins);
  return static_cast<uint16_t>(MAX_UINT16_VALUE *
                               static_cast<uint32_t>(cdf_value) /
                               static_cast<uint32_t>(total_count));
}

template <int BLOCK_SIZE>
void launch_calculate_cdf(sycl::queue& queue, const uint8_t* input,
                          int16_t* output, int nlayers, int ntokens,
                          int nchannels, int max_bins) {
  const int channel_blocks = (nchannels + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int lp = max_bins + 1;

  sycl::range<2> global_range(static_cast<size_t>(nlayers),
                              static_cast<size_t>(channel_blocks) * BLOCK_SIZE);
  sycl::range<2> local_range(1, BLOCK_SIZE);

  queue.submit([&](sycl::handler& cgh) {
    // SLM histogram: [MAX_BINS_SUPPORTED + 1][BLOCK_SIZE] uint16.
    sycl::local_accessor<uint16_t, 2> hist(
        sycl::range<2>(MAX_BINS_SUPPORTED + 1, BLOCK_SIZE), cgh);

    cgh.parallel_for(
        sycl::nd_range<2>(global_range, local_range),
        [=](sycl::nd_item<2> item)
            [[sycl::reqd_sub_group_size(INTEL_SUB_GROUP_SIZE)]] {
              const int layer_id = static_cast<int>(item.get_group(0));
              const int tx = static_cast<int>(item.get_local_id(1));
              const int channel_block = static_cast<int>(item.get_group(1));
              const int start_channel = channel_block * BLOCK_SIZE;
              const int channel_id = start_channel + tx;
              const bool in_range = (channel_id < nchannels);

              // Zero this column of the SLM histogram.  No barrier
              // needed: every work-item only ever touches column `tx`,
              // so there is zero cross-thread sharing in SLM.  Profiler
              // (unitrace --stall-sampling) confirmed that the previous
              // barriers caused ~67% SyncStall — removing them was the
              // single largest win on this kernel.
              for (int i = 0; i <= max_bins; ++i) {
                hist[i][tx] = 0;
              }

              if (in_range) {
                // Histogram pass: counts at hist[v+1][tx].  hist[0][tx] stays
                // 0.
                for (int i = 0; i < ntokens; ++i) {
                  const uint8_t v =
                      input[(static_cast<int64_t>(layer_id) * ntokens + i) *
                                nchannels +
                            channel_id];
                  hist[v + 1][tx] += 1;
                }

                // Inclusive prefix sum so hist[i] = sum of counts of
                // bins [0..i-1].
                uint16_t total = 0;
                for (int i = 0; i < max_bins; ++i) {
                  uint16_t value = hist[i + 1][tx];
                  hist[i + 1][tx] = total + value;
                  total += value;
                }

                // Normalise + add bin index for strict monotonicity.
                if (total > 0) {
                  for (int i = 0; i <= max_bins; ++i) {
                    hist[i][tx] =
                        normalize_cdf_value(hist[i][tx], total, max_bins) +
                        static_cast<uint16_t>(i);
                  }
                } else {
                  for (int i = 0; i <= max_bins; ++i) {
                    hist[i][tx] = static_cast<uint16_t>(i);
                  }
                }
              }

              if (in_range) {
                const int64_t base =
                    (static_cast<int64_t>(layer_id) * nchannels + channel_id) *
                    lp;
                for (int i = 0; i <= max_bins; ++i) {
                  output[base + i] = static_cast<int16_t>(hist[i][tx]);
                }
              }
            });
  });
}

// Largest power-of-two divisor of nchannels, capped at 128.
int get_block_size_xpu(int nchannels) {
  static const int kCandidates[] = {128, 64, 32, 16, 8, 4, 2, 1};
  for (int bs : kCandidates) {
    if (nchannels % bs == 0) return bs;
  }
  return 1;
}

}  // namespace

at::Tensor calculate_cdf_xpu(const at::Tensor& input, int64_t max_bins) {
  if (!input.device().is_xpu()) {
    throw std::runtime_error("calculate_cdf_xpu: input must be an XPU tensor");
  }
  if (max_bins >= MAX_BINS_SUPPORTED) {
    throw std::runtime_error("calculate_cdf_xpu: max_bins must be < 64");
  }
  if (input.dim() != 3) {
    throw std::runtime_error(
        "calculate_cdf_xpu: input must be 3-D [nlayers, ntokens, nchannels]");
  }

  const auto sizes = input.sizes();
  const int nlayers = static_cast<int>(sizes[0]);
  const int ntokens = static_cast<int>(sizes[1]);
  const int nchannels = static_cast<int>(sizes[2]);

  auto contiguous = input.is_contiguous() ? input : input.contiguous();
  auto output = torch::zeros({nlayers, nchannels, max_bins + 1},
                             input.options().dtype(at::kShort));

  sycl::queue& queue =
      c10::xpu::getCurrentXPUStream(input.device().index()).queue();
  // Accept both Byte (uint8) and Char (int8) inputs: values are bin
  // indices in [0, max_bins), so byte-pattern is identical either way.
  const uint8_t* in_ptr =
      reinterpret_cast<const uint8_t*>(contiguous.data_ptr());
  int16_t* out_ptr = output.data_ptr<int16_t>();

  const int block_size = get_block_size_xpu(nchannels);
  switch (block_size) {
    case 1:
      launch_calculate_cdf<1>(queue, in_ptr, out_ptr, nlayers, ntokens,
                              nchannels, static_cast<int>(max_bins));
      break;
    case 2:
      launch_calculate_cdf<2>(queue, in_ptr, out_ptr, nlayers, ntokens,
                              nchannels, static_cast<int>(max_bins));
      break;
    case 4:
      launch_calculate_cdf<4>(queue, in_ptr, out_ptr, nlayers, ntokens,
                              nchannels, static_cast<int>(max_bins));
      break;
    case 8:
      launch_calculate_cdf<8>(queue, in_ptr, out_ptr, nlayers, ntokens,
                              nchannels, static_cast<int>(max_bins));
      break;
    case 16:
      launch_calculate_cdf<16>(queue, in_ptr, out_ptr, nlayers, ntokens,
                               nchannels, static_cast<int>(max_bins));
      break;
    case 32:
      launch_calculate_cdf<32>(queue, in_ptr, out_ptr, nlayers, ntokens,
                               nchannels, static_cast<int>(max_bins));
      break;
    case 64:
      launch_calculate_cdf<64>(queue, in_ptr, out_ptr, nlayers, ntokens,
                               nchannels, static_cast<int>(max_bins));
      break;
    case 128:
      launch_calculate_cdf<128>(queue, in_ptr, out_ptr, nlayers, ntokens,
                                nchannels, static_cast<int>(max_bins));
      break;
    default:
      throw std::runtime_error("calculate_cdf_xpu: unsupported block size");
  }

  return output;
}
