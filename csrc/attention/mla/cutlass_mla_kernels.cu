/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

#include "device/sm100_mla.hpp"
#include "kernel/sm100_mla_tile_scheduler.hpp"

#define CUTLASS_CHECK(status)                       \
  {                                                 \
    cutlass::Status error = status;                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, \
                cutlassGetStatusString(error));     \
  }

using namespace cute;
using namespace cutlass::fmha::kernel;

template<bool v>
struct IsPersistent {
  static const bool value = v;
};

template <typename T, typename PersistenceOption = IsPersistent<true>>
struct MlaSm100 {
  using Element = T;
  using ElementAcc = float;
  using ElementOut = T;

  using TileShape = Shape<_128, _128, Shape<_512, _64>>;
  using TileShapeH = cute::tuple_element_t<0, TileShape>;
  using TileShapeD = cute::tuple_element_t<2, TileShape>;

  // H K (D_latent D_rope) B
  using ProblemShape = cute::tuple<TileShapeH, int, TileShapeD, int>;
  
  using StrideQ = cute::tuple<int64_t, _1, int64_t>;  // H D B
  using StrideK = cute::tuple<int64_t, _1, int64_t>;  // K D B
  using StrideO = StrideK;                            // H D B
  using StrideLSE = cute::tuple<_1, int>;             // H B

  using TileScheduler = std::conditional_t<
      PersistenceOption::value,
      Sm100MlaPersistentTileScheduler,
      Sm100MlaIndividualTileScheduler>;

  using FmhaKernel =
      cutlass::fmha::kernel::Sm100FmhaMlaKernelTmaWarpspecialized<
          TileShape, Element, ElementAcc, ElementOut, ElementAcc,
          TileScheduler, /*kIsCpAsync=*/true>;
  using Fmha = cutlass::fmha::device::MLA<FmhaKernel>;
};


template <typename T>
typename T::Fmha::Arguments args_from_options(at::Tensor const& out,
                                              at::Tensor const& q_nope_and_q_pe,
                                              at::Tensor const& kv_c_and_k_pe_cache,
                                              at::Tensor const& seq_lens,
                                              at::Tensor const& page_table) {
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = q_nope_and_q_pe.device().index();
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  int batches = q_nope_and_q_pe.sizes()[0];
  int page_count_per_seq = page_table.sizes()[1];
  int page_count_total = kv_c_and_k_pe_cache.sizes()[0];
  int page_size = kv_c_and_k_pe_cache.sizes()[1];
  int max_seq_len = page_size * page_count_per_seq;
  using TileShapeH = typename T::TileShapeH;
  using TileShapeD = typename T::TileShapeD;
  auto problem_shape = cute::make_tuple(
      TileShapeH{}, max_seq_len, TileShapeD{}, batches);

  auto [H, K, D, B] = problem_shape;
  auto [D_latent, D_rope] = D;

  // the scale is based on the non-absorbed sizes, change as appropriate
  // we can't determine this parameter from the info we have, it's an input
  int D_non_latent = 128;
  float scale = 1.0 / sqrt(1.0 * (D_non_latent + D_rope));

  using StrideQ = typename T::StrideQ;
  using StrideK = typename T::StrideK;
  using StrideO = typename T::StrideO;
  using StrideLSE = typename T::StrideLSE;

  StrideQ stride_Q = cute::make_tuple(
      static_cast<int64_t>(0 + D_latent + D_rope),
      _1{},
      static_cast<int64_t>(H * (0 + D_latent + D_rope)));
  StrideK stride_C = cute::make_tuple(
      static_cast<int64_t>(0 + D_latent + D_rope),
      _1{},
      static_cast<int64_t>(page_size * (D_latent + D_rope)));
  StrideLSE stride_PT = cute::make_stride(_1{}, page_count_per_seq);
  StrideLSE stride_LSE = cute::make_tuple(_1{}, 0 + H);
  StrideO stride_O = cute::make_tuple(
      static_cast<int64_t>(0 + D_latent),
      _1{},
      static_cast<int64_t>(0 + H * D_latent));

  using Element = typename T::Element;
  using ElementOut = typename T::ElementOut;
  using ElementAcc = typename T::ElementAcc;
  auto Q_ptr = static_cast<Element*>(q_nope_and_q_pe.data_ptr());
  auto C_ptr = static_cast<Element*>(kv_c_and_k_pe_cache.data_ptr());
  typename T::Fmha::Arguments arguments{
    problem_shape,
    { scale,
      Q_ptr, stride_Q,
      Q_ptr + D_latent, stride_Q,
      C_ptr, stride_C,
      C_ptr + D_latent, stride_C,
      static_cast<int*>(seq_lens.data_ptr()),
      static_cast<int*>(page_table.data_ptr()), stride_PT,
      page_count_total, page_size},
    { static_cast<ElementOut*>(out.data_ptr()), stride_O,
      // static_cast<ElementAcc*>(lse.data_ptr()), stride_LSE},
      static_cast<ElementAcc*>(nullptr), stride_LSE},
    hw_info,
    -1, // split_kv
    nullptr, // is_var_split_kv
  };
  // TODO(kaixih@nvidia): When split_kv=-1 and is_var_split_kv=false, we compute
  // split_kv automatically based on batch size and sequence length to balance 
  // workload across available SMs. Consider using var_split_kv for manual
  // control if needed.
  T::Fmha::set_split_kv(arguments);
  return arguments;
}

template <typename Element>
void runMla(at::Tensor const& out,
            at::Tensor const& q_nope_and_q_pe,
            at::Tensor const& kv_c_and_k_pe_cache,
            at::Tensor const& seq_lens,
            at::Tensor const& page_table,
            cudaStream_t stream) {
  using MlaSm100Type = MlaSm100<Element>;
  typename MlaSm100Type::Fmha fmha;
  auto arguments =
      args_from_options<MlaSm100Type>(out, q_nope_and_q_pe, kv_c_and_k_pe_cache,
                                      seq_lens, page_table);
  size_t workspace_size = MlaSm100Type::Fmha::get_workspace_size(arguments);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(q_nope_and_q_pe.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  CUTLASS_CHECK(fmha.can_implement(arguments));

  CUTLASS_CHECK(fmha.initialize(arguments, workspace.data_ptr(), stream));

  CUTLASS_CHECK(fmha.run(arguments, workspace.data_ptr(), stream));
}

void cutlass_mla_decode_sm100a(torch::Tensor const& out,
                               torch::Tensor const& q_nope_and_q_pe,
                               torch::Tensor const& kv_c_and_k_pe_cache,
                               torch::Tensor const& seq_lens,
                               torch::Tensor const& page_table) {
    auto in_dtype = q_nope_and_q_pe.dtype();
    at::cuda::CUDAGuard device_guard{(char)q_nope_and_q_pe.get_device()};
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
                                    q_nope_and_q_pe.get_device());
    if (in_dtype == at::ScalarType::Half) {
        runMla<cutlass::half_t>(
            out, q_nope_and_q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, stream);
    } else if (in_dtype == at::ScalarType::BFloat16) {
        runMla<cutlass::bfloat16_t>(
            out, q_nope_and_q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, stream);
    } else if (in_dtype == at::ScalarType::Float8_e4m3fn) {
        runMla<cutlass::float_e4m3_t>(
            out, q_nope_and_q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, stream);
    } else {
        TORCH_CHECK(false, "Unsupported input data type of MLA");
    }
}
