/*
Adapted from https://github.com/mit-han-lab/llm-awq
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and
Acceleration}, author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang,
Shang and Dang, Xingyu and Han, Song}, journal={arXiv}, year={2023}
}
 */

#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#include <torch/extension.h>
//#include <c10/cuda/CUDAGuard.h>
#include "dequantize.h"
#include "utils.h"
#include "xpu_types.h"

void awq_dequantize_impl(
    int* __restrict__ input,
    sycl::half* __restrict__ scaling_factors,
    int* __restrict__ zeros,
    sycl::half* __restrict__ output,
    int G,
    sycl::nd_item<3> item_ct1) {
  int j_factors1 = 4;
  int row_stride2 = 4;
  int split_k_iters = 1;
  sycl::half2 ZERO_HALF2{0, 0};
  sycl::half input_shared[8];

  int N = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
  int col = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);
  int row = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
      item_ct1.get_local_id(1);
  int index1 = 8 * col + 8 * row * N;
  sycl::half* output_ptr2 = output + index1;

  int index2 = col + row * N;
  int* input_ptr2 = input + index2;

  int index3 = col + (int)(row / G) * N;
  int* zeros_ptr2 = zeros + index3;
  int index4 = 8 * col + (int)(row / G) * N * 8;
  sycl::half* scale_loaded = scaling_factors + index4;

  uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr2);
  sycl::uint4 zero_loaded_u4 = vllm::awq::dequantize_s4_to_fp16x2(zeros_loaded);
  // sycl::uint4 scale_loaded_u4 = *(sycl::uint4*)(scaling_factors_ptr2);
  // int j = 0;

  uint32_t input_loaded = *(uint32_t*)(input_ptr2);
  sycl::uint4 input_loaded_fp16 =
      vllm::awq::dequantize_s4_to_fp16x2(input_loaded);

  sycl::half2* input_loaded_h2 = (sycl::half2*)(&input_loaded_fp16);
  sycl::half2* zero_loaded_h2 = (sycl::half2*)(&zero_loaded_u4);
  sycl::half2* scale_loaded_h2 = (sycl::half2*)scale_loaded;
  for (int i = 0; i < 4; i++) {
    input_loaded_h2[i] = sycl_half_sub2(input_loaded_h2[i], zero_loaded_h2[i]);
    input_loaded_h2[i] =
        sycl_half_fma2(input_loaded_h2[i], scale_loaded_h2[i], ZERO_HALF2);
  }
  *(sycl::uint4*)(input_shared) = input_loaded_fp16;

  for (int i = 0; i < 8; ++i) {
    *(output_ptr2 + i) = input_shared[i];
  }
}

torch::Tensor awq_dequantize(
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int split_k_iters,
    int thx,
    int thy) {
  int in_c = _kernel.size(0);
  int qout_c = _kernel.size(1);
  int out_c = qout_c * 8;
  int G = in_c / _scaling_factors.size(0);

  int x_thread = thx;
  int y_thread = thy;

  int x_blocks = 1;
  int y_blocks = 1;
  if (thx == 0) {
    x_thread = qout_c;
  }
  if (thy == 0) {
    y_thread = in_c;
  }
  if (thx == 0 && thy == 0) {
    x_thread = 8;
    y_thread = 8;
    x_blocks = (int)(qout_c / 8);
    y_blocks = (int)(in_c / 8);
  }

  auto options = torch::TensorOptions()
                     .dtype(_scaling_factors.dtype())
                     .device(_scaling_factors.device());
  at::Tensor _de_kernel = torch::empty({in_c, out_c}, options);
  auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
  auto de_kernel =
      reinterpret_cast<sycl::half*>(_de_kernel.data_ptr<at::Half>());
  auto scaling_factors =
      reinterpret_cast<sycl::half*>(_scaling_factors.data_ptr<at::Half>());
  auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());

  sycl::range<3> num_blocks(1, y_blocks, x_blocks);
  sycl::range<3> threads_per_block(1, y_thread, x_thread);
  auto& queue = vllm::xpu::vllmGetQueue();

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(num_blocks * threads_per_block, threads_per_block),
        [=](sycl::nd_item<3> item_ct1) {
          awq_dequantize_impl(
              kernel, scaling_factors, zeros, de_kernel, G, item_ct1);
        });
  });
  return _de_kernel;
}