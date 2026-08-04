// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <ATen/ATen.h>

#define MAX_BINS_SUPPORTED 64

extern int get_block_size(int);

__inline__ __device__ uint16_t normalize_cdf_value(uint16_t cdf_value,
                                                   uint16_t max_cdf_value,
                                                   const int max_bins) {
  uint32_t MAX_UINT16_VALUE = 0xFFFF - max_bins;
  return (uint16_t)(MAX_UINT16_VALUE * (uint32_t)cdf_value / max_cdf_value);
}

// BLOCK_SIZE should be equal to blockDim.x, and blockDim.y should be equal to 1
template <int BLOCK_SIZE, int MAXBINS, typename INPUT_ACC_T,
          typename OUTPUT_ACC_T>
__global__ void calculate_cdf_kernel(INPUT_ACC_T input, OUTPUT_ACC_T output,
                                     const int max_bins, const int ntokens) {
  __shared__ uint16_t hist[MAXBINS][BLOCK_SIZE];

  const int channel_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const int layer_id = blockIdx.y;

  for (int i = 0; i <= max_bins; i++) {
    hist[i][threadIdx.x] = 0;
  }

  __syncthreads();

  for (int i = 0; i < ntokens; i++) {
    uint8_t value = input[layer_id][i][channel_id];
    hist[value + 1][threadIdx.x]++;
  }

  uint16_t local_sum = 0;
  for (int i = 0; i < max_bins; i++) {
    uint16_t value = hist[i + 1][threadIdx.x];
    hist[i + 1][threadIdx.x] += local_sum;
    local_sum += value;
  }

  for (int i = 0; i <= max_bins; i++) {
    hist[i][threadIdx.x] =
        normalize_cdf_value(hist[i][threadIdx.x], local_sum, max_bins) + i;
  }

  __syncthreads();
  const int num_elements = BLOCK_SIZE * (max_bins + 1);
  const int start_channel = blockIdx.x * BLOCK_SIZE;
  for (int i = threadIdx.x; i < num_elements; i += BLOCK_SIZE) {
    int bin_id = i % (max_bins + 1);
    int cid = i / (max_bins + 1);
    output[layer_id][start_channel + cid][bin_id] = hist[bin_id][cid];
  }
}

/**
 * @brief Calculate the cdf across tokens in the same (layer, channel) pair of
 * the input tensor
 *
 * @param input The input uint8 GPU tensor with shape [nlayers, ntokens,
 * nchannels]
 * @param max_bins The maximum number of bins (i.e., Lp - 1)
 * @return at::Tensor The normalized int16t cdf that can be used for torchac
 * with shape [nlayers, nchannels, max_bins + 1]
 */
at::Tensor calculate_cdf(const at::Tensor& input, const int max_bins) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(max_bins < MAX_BINS_SUPPORTED, "Max bins must be less than ",
              MAX_BINS_SUPPORTED);

  const auto input_shape = input.sizes();
  const int nlayers = input_shape[0];
  const int ntokens = input_shape[1];
  const int nchannels = input_shape[2];

  auto output = torch::zeros({nlayers, nchannels, max_bins + 1},
                             input.options().dtype(at::kShort));

  auto input_accessor = input.packed_accessor64<int8_t, 3>();
  auto output_accessor = output.packed_accessor64<int16_t, 3>();

  int block_size = get_block_size(nchannels);
  dim3 block_dim(block_size, 1, 1);
  dim3 grid_dim(nchannels / block_size, nlayers, 1);

#ifndef LAUNCH_CDF_KERNEL
  #define LAUNCH_CDF_KERNEL(block_size)                                      \
    calculate_cdf_kernel<block_size, MAX_BINS_SUPPORTED>                     \
        <<<grid_dim, block_dim>>>(input_accessor, output_accessor, max_bins, \
                                  ntokens)
#endif
  switch (block_size) {
    case 1:
      LAUNCH_CDF_KERNEL(1);
      break;
    case 2:
      LAUNCH_CDF_KERNEL(2);
      break;
    case 4:
      LAUNCH_CDF_KERNEL(4);
      break;
    case 8:
      LAUNCH_CDF_KERNEL(8);
      break;
    case 16:
      LAUNCH_CDF_KERNEL(16);
      break;
    case 32:
      LAUNCH_CDF_KERNEL(32);
      break;
    case 64:
      LAUNCH_CDF_KERNEL(64);
      break;
    case 128:
      LAUNCH_CDF_KERNEL(128);
      break;
    default:
      throw std::runtime_error("Unsupported block size");
  }

  return output;
}
