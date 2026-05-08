// AIESW-32176: CK WMMA W4A16 b_scale GEMM wrapper.
// Tuned for gfx1151 (Strix Halo) at the Qwen3-4B gate_up_proj prefill shape
// (M=3968, N=19456, K=2560, group=128). Out of scope for any other shape;
// the Python dispatcher is responsible for restricting calls to the target.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3_b_scale.hpp"

namespace {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType = ck::half_t;
using BDataType = ck::pk_i4_t;
using BScaleDataType = ck::half_t;
using AccDataType = float;
using CShuffleDataType = ck::half_t;
using CDataType = ck::half_t;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

static constexpr auto GemmDefault =
    ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr bool PermuteA = false;
static constexpr bool PermuteB = true;
static constexpr ck::index_t Scale_Block_N = 1;
static constexpr ck::index_t Scale_Block_K = 128;

// EXP1_FINAL config from Phase 1 sweep (30.0 TFLOPS verified at the target
// shape). See AIInfo memory project_aiesw_32176_phase1_2_results for the full
// sweep table.
static constexpr ck::index_t KPerBlock = 32;

// clang-format off
using DeviceGemmInstance =
    ck::tensor_operation::device::DeviceGemm_BScale_Wmma_CShuffleV3<
        ALayout,   BLayout,  CLayout,
        ADataType, BDataType, BScaleDataType, CDataType, AccDataType, CShuffleDataType,
        PassThrough, PassThrough, PassThrough, GemmDefault,
        256, Scale_Block_N, Scale_Block_K,
        128, 128,
        KPerBlock, 8, 8,
        16,  16,
        4,    2,
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 8, 8, 1,
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 8, 8, 1,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1,
        CDataType, CDataType, PermuteA, PermuteB>;
// clang-format on

}  // namespace

// in_a:  [M, K]            fp16, contiguous (row-major)
// in_b:  [K0, N, K1/2]     int8 in CK pk_i4_v3 b_scale layout (K0 =
// K/KPerBlock, K1 = KPerBlock) in_s:  [N, K/G]          fp16, contiguous
// row-major (vLLM HybridW4A16 native layout).
//                          CK calls this `b1_k_n` shape [K/G, N] with stride
//                          (K/G, 1), which is exactly a view over [N, K/G]
//                          row-major storage — no transpose needed.
// Returns out [M, N] fp16, freshly allocated.
torch::Tensor ck_w4a16_b_scale_gemm(const at::Tensor& in_a,
                                    const at::Tensor& in_b,
                                    const at::Tensor& in_s,
                                    int64_t group_size) {
  TORCH_CHECK(in_a.is_cuda() && in_b.is_cuda() && in_s.is_cuda(),
              "All inputs must be on GPU");
  TORCH_CHECK(in_a.dtype() == at::kHalf, "in_a must be fp16");
  TORCH_CHECK(in_s.dtype() == at::kHalf, "in_s must be fp16");
  TORCH_CHECK(in_a.dim() == 2, "in_a must be 2-D [M, K]");
  TORCH_CHECK(in_b.dim() == 3,
              "in_b must be 3-D [K0, N, K1/2] (CK pk_i4 layout)");
  TORCH_CHECK(in_s.dim() == 2,
              "in_s must be 2-D [N, K/G] row-major (vLLM HybridW4A16 native "
              "scale layout)");
  TORCH_CHECK(group_size == Scale_Block_K,
              "group_size must equal CK Scale_Block_K (", Scale_Block_K, ")");

  const int64_t M = in_a.size(0);
  const int64_t K = in_a.size(1);
  // CK packs 2 nibbles per int8 in the inner K1/2 dim, plus K0 = K/KPerBlock.
  const int64_t K0 = in_b.size(0);
  const int64_t N = in_b.size(1);
  const int64_t K1_half = in_b.size(2);
  TORCH_CHECK(K0 * KPerBlock == K, "K0 * KPerBlock != K (", K0, "*", KPerBlock,
              "!=", K, ")");
  TORCH_CHECK(K1_half * 2 == KPerBlock, "in_b last dim must be KPerBlock/2 (",
              K1_half, "*2 !=", KPerBlock, ")");
  TORCH_CHECK(in_s.size(0) == N && in_s.size(1) * group_size == K,
              "in_s shape must be [N, K/G]; got [", in_s.size(0), ",",
              in_s.size(1), "] for K=", K, " N=", N, " G=", group_size);
  TORCH_CHECK(in_s.is_contiguous(),
              "in_s must be contiguous row-major [N, K/G]");

  auto out = torch::empty({M, N}, in_a.options());

  const at::cuda::OptionalCUDAGuard guard(device_of(in_a));

  // Logical strides per the CK device-op signature:
  //   ALayout=Row -> StrideA = K, BLayout=Col -> StrideB = K, CLayout=Row ->
  //   StrideC = N.
  // The b_scale device-op consumes the permuted in_b buffer directly
  // (PermuteB=true bakes the K-block tiling into how the threads read; the
  // logical stride is unchanged).
  const ck::index_t StrideA = static_cast<ck::index_t>(K);
  const ck::index_t StrideB = static_cast<ck::index_t>(K);
  const ck::index_t StrideC = static_cast<ck::index_t>(N);
  const ck::index_t Scale_Stride_BN = static_cast<ck::index_t>(K / group_size);
  const ck::index_t KBatch = 1;

  auto gemm = DeviceGemmInstance{};
  auto invoker = gemm.MakeInvoker();
  auto argument = gemm.MakeArgument(
      reinterpret_cast<const ADataType*>(in_a.data_ptr()),
      reinterpret_cast<const BDataType*>(in_b.data_ptr()),
      reinterpret_cast<CDataType*>(out.data_ptr()), static_cast<ck::index_t>(M),
      static_cast<ck::index_t>(N), static_cast<ck::index_t>(K), StrideA,
      StrideB, StrideC, Scale_Stride_BN,
      reinterpret_cast<const BScaleDataType*>(in_s.data_ptr()), KBatch,
      PassThrough{}, PassThrough{}, PassThrough{});

  TORCH_CHECK(gemm.IsSupportedArgument(argument),
              "CK W4A16 b_scale device op rejected the argument; ",
              "shape (M=", M, ", N=", N, ", K=", K, ", G=", group_size,
              ") not supported by this build");

  StreamConfig stream;
  stream.stream_id_ = at::cuda::getCurrentCUDAStream();
  invoker.Run(argument, stream);

  return out;
}
