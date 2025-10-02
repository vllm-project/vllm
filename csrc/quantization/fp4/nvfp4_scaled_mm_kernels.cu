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

#include "cutlass_extensions/common.hpp"

#include "cutlass/cutlass.h"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/packed_stride.hpp"

#include "core/math.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Configuration for M in (256, inf)
struct sm100_fp4_config_default {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<_2, _1, _1>;
  using PerSmTileShape_MNK = Shape<_128, _256, _256>;
};

// Configuration for M in (16, 256]
struct sm100_fp4_config_M256 {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_256, _128, _256>;
  using ClusterShape = Shape<_2, _1, _1>;
  using PerSmTileShape_MNK = Shape<_128, _128, _256>;
};

// Configuration for M in [1, 16]
struct sm100_fp4_config_M16 {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _128, _256>;
  using ClusterShape = Shape<_1, _1, _1>;
  using PerSmTileShape_MNK = Shape<_128, _128, _256>;
};

template <typename Config, typename OutType>
struct Fp4GemmSm100 {
  // A matrix configuration
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  // B matrix configuration
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  // C/D matrix configuration
  using ElementD = OutType;
  using ElementC = OutType;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  // Kernel functional config
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  // Use config's tile shapes
  using MmaTileShape = typename Config::TileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD,
          LayoutDTag, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using LayoutA = decltype(cute::make_layout(make_shape(0, 0, 0), StrideA{}));
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using LayoutB = decltype(cute::make_layout(make_shape(0, 0, 0), StrideB{}));
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using LayoutC = decltype(cute::make_layout(make_shape(0, 0, 0), StrideC{}));
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using LayoutD = decltype(cute::make_layout(make_shape(0, 0, 0), StrideD{}));
};

template <typename Config>
typename Config::Gemm::Arguments args_from_options(
    at::Tensor& D, at::Tensor const& A, at::Tensor const& B,
    at::Tensor const& A_sf, at::Tensor const& B_sf, at::Tensor const& alpha,
    int64_t M, int64_t N, int64_t K) {
  using ElementA = typename Config::Gemm::ElementA;
  using ElementB = typename Config::Gemm::ElementB;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementD = typename Config::Gemm::ElementD;
  using ElementCompute = float;
  using StrideA = typename Config::StrideA;
  using StrideB = typename Config::StrideB;
  using StrideD = typename Config::StrideD;
  using Sm100BlkScaledConfig = typename Config::Gemm::GemmKernel::
      CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  auto layout_SFA = Sm100BlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm100BlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(m, n, k, 1));

  typename Config::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<ElementA const*>(A.data_ptr()), stride_A,
       static_cast<ElementB const*>(B.data_ptr()), stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()), layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()), layout_SFB},
      {     // Epilogue arguments
       {},  // epilogue.thread
       static_cast<ElementD const*>(D.data_ptr()),
       stride_D,
       static_cast<ElementD*>(D.data_ptr()),
       stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(alpha.data_ptr());
  return arguments;
}

template <typename Config>
void runGemm(at::Tensor& D, at::Tensor const& A, at::Tensor const& B,
             at::Tensor const& A_sf, at::Tensor const& B_sf,
             at::Tensor const& alpha, int64_t m, int64_t n, int64_t k,
             cudaStream_t stream) {
  typename Config::Gemm gemm;

  auto arguments =
      args_from_options<Config>(D, A, B, A_sf, B_sf, alpha, m, n, k);

  size_t workspace_size = Config::Gemm::get_workspace_size(arguments);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  CUTLASS_CHECK(gemm.can_implement(arguments));

  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));

  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}

// Dispatch function to select appropriate config based on M
template <typename OutType>
void cutlass_fp4_gemm_dispatch(torch::Tensor& D, torch::Tensor const& A,
                               torch::Tensor const& B,
                               torch::Tensor const& A_sf,
                               torch::Tensor const& B_sf,
                               torch::Tensor const& alpha, int64_t m, int64_t n,
                               int64_t k, cudaStream_t stream) {
  uint32_t const mp2 = std::max(static_cast<uint32_t>(16), next_pow_2(m));

  if (mp2 <= 16) {
    // m in [1, 16]
    runGemm<Fp4GemmSm100<sm100_fp4_config_M16, OutType>>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else if (mp2 <= 256) {
    // m in (16, 256]
    runGemm<Fp4GemmSm100<sm100_fp4_config_M256, OutType>>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else {
    // m in (256, inf)
    runGemm<Fp4GemmSm100<sm100_fp4_config_default, OutType>>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  }
}

#else
template <typename OutType>
void cutlass_fp4_gemm_dispatch(torch::Tensor& D, torch::Tensor const& A,
                               torch::Tensor const& B,
                               torch::Tensor const& A_sf,
                               torch::Tensor const& B_sf,
                               torch::Tensor const& alpha, int64_t m, int64_t n,
                               int64_t k, cudaStream_t stream) {
  TORCH_CHECK(false,
              "Unsupported CUTLASS version. Set VLLM_CUTLASS_SRC_DIR to "
              "a CUTLASS 3.8 source directory to enable support.");
}
#endif  // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#define CHECK_TYPE(x, st, m) \
  TORCH_CHECK(x.scalar_type() == st, ": Inconsistency of Tensor type:", m)
#define CHECK_TH_CUDA(x, m) \
  TORCH_CHECK(x.is_cuda(), m, ": must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) \
  TORCH_CHECK(x.is_contiguous(), m, ": must be contiguous")
#define CHECK_INPUT(x, st, m) \
  CHECK_TH_CUDA(x, m);        \
  CHECK_CONTIGUOUS(x, m);     \
  CHECK_TYPE(x, st, m)

constexpr auto FLOAT4_E2M1X2 = at::ScalarType::Byte;
constexpr auto SF_DTYPE = at::ScalarType::Float8_e4m3fn;

void cutlass_scaled_fp4_mm_sm100a(torch::Tensor& D, torch::Tensor const& A,
                                  torch::Tensor const& B,
                                  torch::Tensor const& A_sf,
                                  torch::Tensor const& B_sf,
                                  torch::Tensor const& alpha) {
  CHECK_INPUT(A, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(B, FLOAT4_E2M1X2, "b");

  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");

  CHECK_INPUT(alpha, at::ScalarType::Float, "alpha");

  TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  TORCH_CHECK(B.dim() == 2, "b must be a matrix");
  TORCH_CHECK(A.sizes()[1] == B.sizes()[1],
              "a and b shapes cannot be multiplied (", A.sizes()[0], "x",
              A.sizes()[1], " and ", B.sizes()[0], "x", B.sizes()[1], ")");

  auto const m = A.sizes()[0];
  auto const n = B.sizes()[0];
  auto const k = A.sizes()[1] * 2;

  constexpr int alignment = 32;
  TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ", alignment,
              ", but got a shape: (", A.sizes()[0], "x", A.sizes()[1],
              "), k: ", k, ".");
  TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment,
              ", but got b shape: (", B.sizes()[0], "x", B.sizes()[1], ").");

  auto round_up = [](int x, int y) { return (x + y - 1) / y * y; };
  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(n, 128);
  // Since k is divisible by 32 (alignment), k / 16 is guaranteed to be an
  // integer.
  int rounded_k = round_up(k / 16, 4);

  TORCH_CHECK(A_sf.dim() == 2, "scale_a must be a matrix");
  TORCH_CHECK(B_sf.dim() == 2, "scale_b must be a matrix");
  TORCH_CHECK(A_sf.sizes()[1] == B_sf.sizes()[1],
              "scale_a and scale_b shapes cannot be multiplied (",
              A_sf.sizes()[0], "x", A_sf.sizes()[1], " and ", B_sf.sizes()[0],
              "x", B_sf.sizes()[1], ")");
  TORCH_CHECK(A_sf.sizes()[0] == rounded_m && A_sf.sizes()[1] == rounded_k,
              "scale_a must be padded and swizzled to a shape (", rounded_m,
              "x", rounded_k, "), but got a shape (", A_sf.sizes()[0], "x",
              A_sf.sizes()[1], ")");
  TORCH_CHECK(B_sf.sizes()[0] == rounded_n && B_sf.sizes()[1] == rounded_k,
              "scale_b must be padded and swizzled to a shape (", rounded_n,
              "x", rounded_k, "), but got a shape (", B_sf.sizes()[0], "x",
              B_sf.sizes()[1], ")");

  auto out_dtype = D.dtype();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());

  if (out_dtype == at::ScalarType::Half) {
    cutlass_fp4_gemm_dispatch<cutlass::half_t>(D, A, B, A_sf, B_sf, alpha, m, n,
                                               k, stream);
  } else if (out_dtype == at::ScalarType::BFloat16) {
    cutlass_fp4_gemm_dispatch<cutlass::bfloat16_t>(D, A, B, A_sf, B_sf, alpha,
                                                   m, n, k, stream);
  } else {
    TORCH_CHECK(false, "Unsupported output data type of nvfp4 mm (", out_dtype,
                ")");
  }
}
