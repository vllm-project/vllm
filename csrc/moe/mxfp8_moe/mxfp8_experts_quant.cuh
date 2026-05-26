// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from SGLang:
// https://github.com/sgl-project/sglang/blob/ded068a76e00878881d52d5bfb791e0f60d7311b/sgl-kernel/csrc/expert_specialization/es_sm100_mxfp8_blockscaled_group_quant.cuh

#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/all.h>

#include <cuda/ptx>

#include "cute/tensor.hpp"

namespace expert_specialization {

using namespace cute;

constexpr uint32_t THREAD_BLOCK_SIZE = 128;
constexpr uint32_t WARP_SIZE = 32;
constexpr int BLOCK_M = 128;
constexpr int BLOCK_K = 128;
using ThrLayout = Layout<Shape<_16, _8>, Stride<_8, _1>>;
using ValLayout = Layout<Shape<_1, _16>>;
using SfR2SThrLayout = Layout<Shape<_16, _4>, Stride<_4, _1>>;
using SfR2SValLayout = Layout<Shape<_1, _1>>;
using ScaleFactorTileLayout =
    Layout<Shape<Shape<_32, _4>, _4>, Stride<Stride<_16, _4>, _1>>;

// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

// Some code references TRT-LLM:
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/quantization.cuh
template <typename FragmentS, typename FragmentD>
__inline__ __device__ uint8_t cvt_warp_fp16_to_mxfp8(FragmentS& fragment_s,
                                                     FragmentD& fragment_d) {
  using FragmentSLayout = typename FragmentS::layout_type;
  using FragmentDLayout = typename FragmentD::layout_type;
  FragmentSLayout fragment_s_layout;
  FragmentDLayout fragment_d_layout;
  static_assert(is_static<FragmentSLayout>::value &&
                size(fragment_s_layout) == 16);
  static_assert(is_static<FragmentDLayout>::value &&
                size(fragment_d_layout) == 16);

  constexpr int eles_per_thr = 16;
  using ValType = typename FragmentS::element_type;
  using VecType = std::conditional_t<std::is_same_v<ValType, __nv_bfloat16>,
                                     __nv_bfloat162, __half2>;
  VecType vec[8];
  // Assign vals
  vec[0].x = fragment_s(Int<0>{});
  vec[0].y = fragment_s(Int<1>{});
  vec[1].x = fragment_s(Int<2>{});
  vec[1].y = fragment_s(Int<3>{});
  vec[2].x = fragment_s(Int<4>{});
  vec[2].y = fragment_s(Int<5>{});
  vec[3].x = fragment_s(Int<6>{});
  vec[3].y = fragment_s(Int<7>{});
  vec[4].x = fragment_s(Int<8>{});
  vec[4].y = fragment_s(Int<9>{});
  vec[5].x = fragment_s(Int<10>{});
  vec[5].y = fragment_s(Int<11>{});
  vec[6].x = fragment_s(Int<12>{});
  vec[6].y = fragment_s(Int<13>{});
  vec[7].x = fragment_s(Int<14>{});
  vec[7].y = fragment_s(Int<15>{});

  auto local_max = __habs2(vec[0]);
  for (int i = 1; i < eles_per_thr / 2; i++) {
    local_max = __hmax2(__habs2(vec[i]), local_max);
  }
  local_max = __hmax2(__shfl_xor_sync(uint32_t(-1), local_max, 1), local_max);

  // Get the final absolute maximum values.
  float block_max(0.0f);
  if constexpr (std::is_same_v<ValType, __nv_bfloat16>) {
    block_max = __bfloat162float(__hmax(local_max.x, local_max.y));
  } else {
    block_max = __half2float(__hmax(local_max.x, local_max.y));
  }
  // Get the SF (max value of the vector / max value of mxfp8).
  float sf_val = block_max * reciprocal_approximate_ftz(448.0f);
  // 8 bits representation of the SF.
  uint8_t fp8_sf_val;

  __nv_fp8_e8m0 tmp_sf_val;
  tmp_sf_val.__x =
      __nv_cvt_float_to_e8m0(sf_val, __NV_SATFINITE, cudaRoundPosInf);
  sf_val = static_cast<float>(tmp_sf_val);
  fp8_sf_val = tmp_sf_val.__x;
  // Get the output scale (reciprocal of the SFValue).
  float output_scale =
      block_max != 0.f ? reciprocal_approximate_ftz(sf_val) : 0.0f;

  // Convert the input to float.
  float2 fp2_vals[eles_per_thr / 2];

#pragma unroll
  for (int i = 0; i < eles_per_thr / 2; i++) {
    if constexpr (std::is_same_v<ValType, __half>) {
      fp2_vals[i] = __half22float2(vec[i]);
    } else {
      fp2_vals[i] = __bfloat1622float2(vec[i]);
    }
    fp2_vals[i].x *= output_scale;
    fp2_vals[i].y *= output_scale;
  }
  union {
    uint8_t bytes[16];
    __nv_fp8x2_e4m3 elts[8];
  } u;
  u.elts[0] = __nv_fp8x2_e4m3(fp2_vals[0]);
  u.elts[1] = __nv_fp8x2_e4m3(fp2_vals[1]);
  u.elts[2] = __nv_fp8x2_e4m3(fp2_vals[2]);
  u.elts[3] = __nv_fp8x2_e4m3(fp2_vals[3]);
  u.elts[4] = __nv_fp8x2_e4m3(fp2_vals[4]);
  u.elts[5] = __nv_fp8x2_e4m3(fp2_vals[5]);
  u.elts[6] = __nv_fp8x2_e4m3(fp2_vals[6]);
  u.elts[7] = __nv_fp8x2_e4m3(fp2_vals[7]);
  fragment_d(Int<0>{}) = cutlass::float_e4m3_t::bitcast(u.bytes[0]);
  fragment_d(Int<1>{}) = cutlass::float_e4m3_t::bitcast(u.bytes[1]);
  fragment_d(Int<2>{}) = cutlass::float_e4m3_t::bitcast(u.bytes[2]);
  fragment_d(Int<3>{}) = cutlass::float_e4m3_t::bitcast(u.bytes[3]);
  fragment_d(Int<4>{}) = cutlass::float_e4m3_t::bitcast(u.bytes[4]);
  fragment_d(Int<5>{}) = cutlass::float_e4m3_t::bitcast(u.bytes[5]);
  fragment_d(Int<6>{}) = cutlass::float_e4m3_t::bitcast(u.bytes[6]);
  fragment_d(Int<7>{}) = cutlass::float_e4m3_t::bitcast(u.bytes[7]);
  fragment_d(Int<8>{}) = cutlass::float_e4m3_t::bitcast(u.bytes[8]);
  fragment_d(Int<9>{}) = cutlass::float_e4m3_t::bitcast(u.bytes[9]);
  fragment_d(Int<10>{}) = cutlass::float_e4m3_t::bitcast(u.bytes[10]);
  fragment_d(Int<11>{}) = cutlass::float_e4m3_t::bitcast(u.bytes[11]);
  fragment_d(Int<12>{}) = cutlass::float_e4m3_t::bitcast(u.bytes[12]);
  fragment_d(Int<13>{}) = cutlass::float_e4m3_t::bitcast(u.bytes[13]);
  fragment_d(Int<14>{}) = cutlass::float_e4m3_t::bitcast(u.bytes[14]);
  fragment_d(Int<15>{}) = cutlass::float_e4m3_t::bitcast(u.bytes[15]);
  return fp8_sf_val;
}

template <typename TensorS, typename TensorP, typename TensorD,
          typename TensorSharedSF, typename TensorSF, typename TiledCopyG2R,
          typename TiledCopyR2G, typename TiledCopyR2S>
__inline__ __device__ void mxfp8_experts_quant_tile(
    TensorS& tensor_s, TensorP& tensor_p, TensorD& tensor_d,
    TensorSharedSF& tensor_shared_sf, TensorSF& tensor_sf, int m,
    TiledCopyG2R& tiled_copy_g2r, TiledCopyR2G& tiled_copy_r2g,
    TiledCopyR2S& tiled_copy_r2s) {
  static_assert(size(get<0>(typename TensorS::layout_type{})) == 128 &&
                size(get<1>(typename TensorS::layout_type{})) == 128 &&
                stride(get<1>(typename TensorS::layout_type{})) == 1);
  static_assert(size(get<0>(typename TensorD::layout_type{})) == 128 &&
                size(get<1>(typename TensorD::layout_type{})) == 128 &&
                stride(get<1>(typename TensorD::layout_type{})) == 1);
  static_assert(size(get<0>(typename TensorP::layout_type{})) == 128 &&
                size(get<1>(typename TensorP::layout_type{})) == 128);
  static_assert(size(get<0>(typename TensorSharedSF::layout_type{})) == 128 &&
                size(get<1>(typename TensorSharedSF::layout_type{})) == 4);
  static_assert(size(get<0>(typename TensorSF::layout_type{})) == 128 &&
                size(get<1>(typename TensorSF::layout_type{})) == 4);

  using Tiler_MN = typename TiledCopyG2R::Tiler_MN;
  auto tiler_mn = Tiler_MN{};
  static_assert(size<0>(tiler_mn) == 16 && size<1>(tiler_mn) == 128);

  auto tiled_tensor_s = tiled_divide(tensor_s, tiler_mn);
  auto tiled_tensor_p = tiled_divide(tensor_p, tiler_mn);
  auto tiled_tensor_d = tiled_divide(tensor_d, tiler_mn);
  static_assert(size<2>(tiled_tensor_s) == 1);
  static_assert(size<2>(tiled_tensor_p) == 1);
  static_assert(size<2>(tiled_tensor_d) == 1);
  auto squeeze_tiled_tensor_s = take<0, 2>(tiled_tensor_s);
  auto squeeze_tiled_tensor_p = take<0, 2>(tiled_tensor_p);
  auto squeeze_tiled_tensor_d = take<0, 2>(tiled_tensor_d);

  using SF_Tiler_MN = typename TiledCopyR2S::Tiler_MN;
  auto sf_tiler_mn = SF_Tiler_MN{};
  static_assert(size<0>(sf_tiler_mn) == 16 && size<1>(sf_tiler_mn) == 4);

  auto tiled_tensor_sf = tiled_divide(tensor_sf, sf_tiler_mn);
  auto tiled_tensor_shared_sf = tiled_divide(tensor_shared_sf, sf_tiler_mn);
  auto squeeze_tiled_tensor_sf = take<0, 2>(tiled_tensor_sf);
  auto squeeze_tiled_tensor_shared_sf = take<0, 2>(tiled_tensor_shared_sf);

  constexpr int tile_loop_count = size<1>(tiled_tensor_s);
  constexpr int rows_in_tile = 16;
  // We don't need to clear shared memory
  // clear(squeeze_tiled_tensor_shared_sf);
#pragma unroll 4
  for (int t = 0; t < tile_loop_count; t++) {
    if (t * rows_in_tile >= m) {
      break;
    }
    auto current_copy_tile_s = tensor<0>(squeeze_tiled_tensor_s(_, t));
    auto current_copy_tile_p = tensor<0>(squeeze_tiled_tensor_p(_, t));
    auto current_copy_tile_d = tensor<0>(squeeze_tiled_tensor_d(_, t));
    auto current_copy_tile_sf = tensor<0>(squeeze_tiled_tensor_sf(_, t));
    auto current_copy_tile_shared_sf =
        tensor<0>(squeeze_tiled_tensor_shared_sf(_, t));

    // Global to Register copy
    auto thr_copy_g2r = tiled_copy_g2r.get_thread_slice(threadIdx.x);
    auto thr_tile_g2r_s = thr_copy_g2r.partition_S(current_copy_tile_s);
    auto thr_tile_g2r_p = thr_copy_g2r.partition_S(current_copy_tile_p);
    auto input_fragment = make_fragment_like(thr_tile_g2r_s);

    // Register to Global copy
    auto thr_copy_r2g = tiled_copy_r2g.get_thread_slice(threadIdx.x);
    auto thr_tile_r2g_d = thr_copy_r2g.partition_D(current_copy_tile_d);
    auto thr_tile_r2g_p = thr_copy_r2g.partition_D(current_copy_tile_p);
    auto output_fragment = make_fragment_like(thr_tile_r2g_d);

    // Register to Shared copy
    auto thr_copy_r2s = tiled_copy_r2s.get_thread_slice(threadIdx.x / 2);
    auto thr_tile_r2s_shared_sf =
        thr_copy_r2s.partition_D(current_copy_tile_shared_sf);
    auto shared_sf_fragment = make_fragment_like(thr_tile_r2s_shared_sf);

    // CopyG2R & convert & CopyR2G
    copy_if(tiled_copy_g2r, thr_tile_g2r_p, thr_tile_g2r_s, input_fragment);
    uint8_t fp8_sf_val =
        cvt_warp_fp16_to_mxfp8(input_fragment, output_fragment);
    copy_if(tiled_copy_r2g, thr_tile_r2g_p, output_fragment, thr_tile_r2g_d);
    shared_sf_fragment[0] = fp8_sf_val;

    // Before first copy r2s, clear shared memory and wait previous group
    if (t == 0 && threadIdx.x == 0) {
      // Wait for the group to have completed reading from shared memory.
      cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>());
    }
    __syncthreads();

    if (threadIdx.x % 2 == 0) {
      copy(tiled_copy_r2s, shared_sf_fragment, thr_tile_r2s_shared_sf);
    }
    __syncthreads();
  }

  // Wait for shared memory writes to be visible to TMA engine.
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);  // b)
  __syncthreads();

  if (threadIdx.x == 0) {
    cuda::ptx::cp_async_bulk(cuda::ptx::space_global, cuda::ptx::space_shared,
                             squeeze_tiled_tensor_sf.data().get(),
                             squeeze_tiled_tensor_shared_sf.data().get(), 512);
    // Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    cuda::ptx::cp_async_bulk_commit_group();
  }
  __syncthreads();
}

template <typename T_IN, typename TiledCopyG2R, typename TiledCopyR2G,
          typename TiledCopyR2S>
__global__ void mxfp8_experts_quant_kernel(
    const T_IN* input, const int* problem_sizes, const int* expert_offsets,
    const int* blockscale_offsets, cutlass::float_e4m3_t* quant_output,
    uint8_t* scale_factor, int groups, TiledCopyG2R tiled_copy_g2r,
    TiledCopyR2G tiled_copy_r2g, TiledCopyR2S tiled_copy_r2s) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  __shared__ __align__(512) uint8_t shared_memory[512];
  ScaleFactorTileLayout scale_factor_tile_layout{};
  auto scale_factor_shared =
      make_tensor(make_smem_ptr(shared_memory),
                  scale_factor_tile_layout);  // ((_32,_4), _4):((_16,_4), _1)
  // TODO: Transform Groupwise Schedule into a more efficient Schedule
  for (int g = 0; g < groups; g++) {
    int m = problem_sizes[g * 3 + 0];
    int k = problem_sizes[g * 3 + 2];
    int64_t expert_offset = static_cast<int64_t>(expert_offsets[g]);
    int64_t blockscale_offset = static_cast<int64_t>(blockscale_offsets[g]);

    auto input_tensor = make_tensor(
        make_gmem_ptr(input + expert_offset * k),
        make_layout(make_shape(m, k),
                    LayoutRight{}));  // (M, K):(K, 1) half_t/bfloat16_t

    auto quant_output_tensor = make_tensor(
        make_gmem_ptr(quant_output + expert_offset * k),
        make_layout(make_shape(m, k),
                    LayoutRight{}));  // (M, K):(K, 1) cutlass::float_e4m3_t

    auto scale_factor_shape = make_shape(ceil_div(m, 128) * 128, k / 32);
    auto scale_factor_layout = tile_to_shape(scale_factor_tile_layout,
                                             scale_factor_shape, LayoutRight{});
    // layout<0>(layout<0>(scale_factor_layout))  (_32,_4):(_16,_4) -- static
    // layout<1>(layout<0>(scale_factor_layout))  M_align_128 / 128 -- dynamic
    // shape dynamic stride layout<0>(layout<1>(scale_factor_layout))  _4:_1 --
    // static layout<1>(layout<1>(scale_factor_layout))  (K / 32) / 4 : _512 --
    // dynamic shape static stride

    // Reshape to zipped layout for 1D indexing
    auto zipped_scale_factor_layout = make_layout(
        make_layout(layout<0>(layout<0>(scale_factor_layout)),
                    layout<0>(layout<1>(scale_factor_layout))),
        make_layout(
            layout<1>(layout<0>(scale_factor_layout)),
            layout<1>(layout<1>(
                scale_factor_layout))));  // (((_32,_4),_4),(M_align_128 /
                                          // 128,(K / 32) /
                                          // 4)):(((_16,_4),_1),(?,_512))

    auto scale_factor_tensor =
        make_tensor(make_gmem_ptr(scale_factor + blockscale_offset * (k / 32)),
                    zipped_scale_factor_layout);

    // Used for cases where M is not divisible by 128 (most scenarios).
    auto input_shape = shape(input_tensor);  // (M, K):(K, 1)
    auto identity_tensor = make_identity_tensor(input_shape);
    auto predict_tensor = cute::lazy::transform(
        identity_tensor, [&](auto c) { return elem_less(c, input_shape); });

    // (_128, _128)
    auto tiler = make_shape(Int<BLOCK_M>{}, Int<BLOCK_K>{});

    auto tiled_input_tensor = zipped_divide(
        input_tensor, tiler);  // ((128, 128), (cdiv(M, 128), cdiv(K, 128)))
    auto tiled_quant_output_tensor =
        zipped_divide(quant_output_tensor,
                      tiler);  // ((128, 128), (cdiv(M, 128), cdiv(K, 128)))
    auto tiled_predict_tensor = zipped_divide(
        predict_tensor, tiler);  // ((128, 128), (cdiv(M, 128), cdiv(K, 128)))

    auto total_tiles =
        size<1>(tiled_input_tensor);  // cdiv(M, 128) * cdiv(K, 128)
    decltype(total_tiles) blk_offset = blockIdx.x;
    while (blk_offset < total_tiles) {
      auto current_input_tile = tensor<0>(tiled_input_tensor(_, blk_offset));
      auto current_quant_output_tile =
          tensor<0>(tiled_quant_output_tensor(_, blk_offset));
      auto current_predict_tile =
          tensor<0>(tiled_predict_tensor(_, blk_offset));
      auto current_scale_factor_tile =
          tensor<0>(scale_factor_tensor(_, blk_offset));

      mxfp8_experts_quant_tile<
          decltype(current_input_tile), decltype(current_predict_tile),
          decltype(current_quant_output_tile), decltype(scale_factor_shared),
          decltype(current_scale_factor_tile), TiledCopyG2R, TiledCopyR2G,
          TiledCopyR2S>(current_input_tile, current_predict_tile,
                        current_quant_output_tile, scale_factor_shared,
                        current_scale_factor_tile, m, tiled_copy_g2r,
                        tiled_copy_r2g, tiled_copy_r2s);
      blk_offset += gridDim.x;
    }
  }
#endif
}

template <typename T_IN>
void launch_mxfp8_experts_quant(const torch::Tensor& input,
                                const torch::Tensor& problem_sizes,
                                const torch::Tensor& expert_offsets,
                                const torch::Tensor& blockscale_offsets,
                                torch::Tensor& quant_output,
                                torch::Tensor& scale_factor) {
  ThrLayout thr_layout{};
  ValLayout val_layout{};
  SfR2SThrLayout r2s_thr_layout{};
  SfR2SValLayout r2s_val_layout{};

  using CopyOpG2R =
      UniversalCopy<cutlass::AlignedArray<T_IN, size(val_layout)>>;
  using CopyAtomG2R = cute::Copy_Atom<CopyOpG2R, T_IN>;
  auto tiled_copy_g2r = cute::make_tiled_copy(
      CopyAtomG2R{}, thr_layout, val_layout);  // Tiler_MN: (16, 128)

  using CopyOpR2G = UniversalCopy<
      cutlass::AlignedArray<cutlass::float_e4m3_t, size(val_layout)>>;
  using CopyAtomR2G = cute::Copy_Atom<CopyOpR2G, cutlass::float_e4m3_t>;
  auto tiled_copy_r2g = cute::make_tiled_copy(
      CopyAtomR2G{}, thr_layout, val_layout);  // Tiler_MN: (16, 128)

  using CopyOpR2S =
      UniversalCopy<cutlass::AlignedArray<uint8_t, size(r2s_val_layout)>>;
  using CopyAtomR2S = cute::Copy_Atom<CopyOpR2S, uint8_t>;
  auto tiled_copy_r2s = cute::make_tiled_copy(
      CopyAtomR2S{}, r2s_thr_layout, r2s_val_layout);  // Tiler_MN: (16, 4)

  int max_active_blocks_per_sm = -1;
  AT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_per_sm,
      mxfp8_experts_quant_kernel<T_IN, decltype(tiled_copy_g2r),
                                 decltype(tiled_copy_r2g),
                                 decltype(tiled_copy_r2s)>,
      THREAD_BLOCK_SIZE, 0));

  dim3 grid(at::cuda::getCurrentDeviceProperties()->multiProcessorCount *
                max_active_blocks_per_sm,
            1, 1);
  dim3 block(THREAD_BLOCK_SIZE, 1, 1);
  int num_experts = (int)problem_sizes.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  mxfp8_experts_quant_kernel<T_IN, decltype(tiled_copy_g2r),
                             decltype(tiled_copy_r2g), decltype(tiled_copy_r2s)>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<const T_IN*>(input.data_ptr()),
          reinterpret_cast<const int*>(problem_sizes.data_ptr()),
          reinterpret_cast<const int*>(expert_offsets.data_ptr()),
          reinterpret_cast<const int*>(blockscale_offsets.data_ptr()),
          reinterpret_cast<cutlass::float_e4m3_t*>(quant_output.data_ptr()),
          reinterpret_cast<uint8_t*>(scale_factor.data_ptr()), num_experts,
          tiled_copy_g2r, tiled_copy_r2g, tiled_copy_r2s);
}

}  // namespace expert_specialization