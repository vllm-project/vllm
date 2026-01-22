#pragma once

#include "cutlass_extensions/epilogue/broadcast_load_epilogue_c2x.hpp"
#include <torch/csrc/stable/tensor.h>

/*
   This file defines custom epilogues for fusing channel scales, token scales,
   bias, and activation zero-points onto a GEMM operation using the
   CUTLASS 2.x API, for sm80 (Ampere) NVIDIA GPUs.


   Epilogues must contain a public type named EVTCompute of type Sm80EVT,
   as well as a static prepare_args function that constructs an
   EVTCompute::Arguments struct.
*/

namespace vllm::c2x {

using namespace cute;

/*
 * This class provides the common load descriptors for the
 * ScaledEpilogue[...] classes
 */
template <typename ElementD, typename OutputTileThreadMap>
struct ScaledEpilogueBase {
 protected:
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  template <typename T>
  using ColOrScalarLoad =
      cutlass::epilogue::threadblock::VisitorColOrScalarBroadcast<
          OutputTileThreadMap, T, Stride<Int<1>, Int<0>, Int<0>>>;

  template <typename T>
  using RowOrScalarLoad =
      cutlass::epilogue::threadblock::VisitorRowOrScalarBroadcast<
          OutputTileThreadMap, T, Stride<Int<0>, Int<1>, Int<0>>>;

  template <typename T>
  using ColLoad = cutlass::epilogue::threadblock::VisitorColBroadcast<
      OutputTileThreadMap, T, Stride<Int<1>, Int<0>, Int<0>>>;

  template <typename T>
  using RowLoad = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, T, Stride<Int<0>, Int<1>, Int<0>>>;

  template <typename T>
  using RowOrZeroLoad =
      cutlass::epilogue::threadblock::VisitorRowOrZeroBroadcast<
          OutputTileThreadMap, T, Stride<Int<0>, Int<1>, Int<0>>>;

  // This utility function constructs the arguments for the load descriptors
  // from a tensor. It can handle both row and column, as well as row/column or
  // scalar cases.
  template <typename Descriptor, typename T>
  static auto args_from_tensor(torch::stable::Tensor const& tensor) {
    using Arguments = typename Descriptor::Arguments;
    auto* data_ptr = static_cast<T*>(tensor.data_ptr());
    if constexpr (std::is_same_v<Descriptor, ColOrScalarLoad<T>> ||
                  std::is_same_v<Descriptor, RowOrScalarLoad<T>>) {
      return Arguments{data_ptr, tensor.numel() != 1};
    } else {
      // it would technically work but no use case as data_ptr is never nullptr
      static_assert(!std::is_same_v<Descriptor, RowOrZeroLoad<T>>);
      return Arguments{data_ptr};
    }
  }

  // This overload handles the case where there might not be a tensor, in which
  // case a nullptr is passed and a constant (0) is used.
  template <typename Descriptor, typename T>
  static auto args_from_tensor(
      std::optional<torch::stable::Tensor> const& tensor) {
    static_assert(std::is_same_v<Descriptor, RowOrZeroLoad<T>>);
    using Arguments = typename Descriptor::Arguments;
    auto* data_ptr = tensor ? static_cast<T*>(tensor->data_ptr()) : nullptr;
    return Arguments{data_ptr};
  }
};

/*
 This epilogue function defines a quantized GEMM operation similar to
 torch._scaled_mm.

 A and B may be both either int8 or fp8_e4m3. A can be quantized per-tensor or
 per-row. B can be quantized per-tensor or per-column.
 Any combination of per-tensor and per-row or column is supported.
 A and B must have symmetric quantization (zero point == 0).

 So the GEMM operation is D = (a_scales * A) (b_scales * B), where the
 scales are applied elementwise with numpy-style broadcasting.

 ScaleA and ScaleB define the epilogue functions that apply the scales for
 the A and B operands respectively. These scales may be either per-tensor or
 per row or column.
*/
template <typename ElementD, typename OutputTileThreadMap>
struct ScaledEpilogue
    : private ScaledEpilogueBase<ElementD, OutputTileThreadMap> {
 private:
  using SUPER = ScaledEpilogueBase<ElementD, OutputTileThreadMap>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;

  using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::threadblock::Sm80EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::threadblock::Sm80EVT<Compute1, ScaleA, EVTCompute0>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(torch::stable::Tensor const& a_scales,
                                   torch::stable::Tensor const& b_scales) {
    auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales);
    auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales);

    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, {}};
  }
};

/*
 * This epilogue performs the same operation as ScaledEpilogue, but adds a bias.
 * This bias can also be used in the per-tensor azp case, where the activation
 * zero point (azp) is used to compute an azp correction term,
 * which is folded into the bias.
 *
 * The bias tensor must be per-output channel.
 * ScaleA and ScaleB can be per-tensor or per-token/per-channel.
 */
template <typename ElementD, typename OutputTileThreadMap>
struct ScaledEpilogueBias
    : protected ScaledEpilogueBase<ElementD, OutputTileThreadMap> {
 protected:
  using SUPER = ScaledEpilogueBase<ElementD, OutputTileThreadMap>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template RowLoad<ElementD>;
  using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::threadblock::Sm80EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute = cutlass::epilogue::threadblock::Sm80EVT<Compute1, ScaleA,
                                                             EVTCompute0, Bias>;
  using ArgumentType = typename EVTCompute::Arguments;
  static ArgumentType prepare_args(torch::stable::Tensor const& a_scales,
                                   torch::stable::Tensor const& b_scales,
                                   torch::stable::Tensor const& bias) {
    auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales);
    auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales);
    auto bias_args = SUPER::template args_from_tensor<Bias, ElementD>(bias);

    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, bias_args, {}};
  }
};

/*
 * This epilogue directly supports per-tensor azp in int32 form.
 * As opposed to the per-token epilogue below, this epilogue only has an azp_adj
 * term, which should already be multiplied with the scalar azp.
 * The azp_adj term is a 1D tensor of shape (1,n), computed as azp * J @ B.
 *
 * This epilogue also supports bias, which remains per-channel.
 */
template <typename ElementD, typename OutputTileThreadMap>
struct ScaledEpilogueBiasAzp
    : protected ScaledEpilogueBase<ElementD, OutputTileThreadMap> {
 private:
  using SUPER = ScaledEpilogueBase<ElementD, OutputTileThreadMap>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template RowOrZeroLoad<ElementD>;

  // This is the full AZP term, azp * J @ B, shape (1,n)
  using AzpWithAdj = typename SUPER::template RowLoad<int32_t>;

  // Compute float(accum - azp_adj), both operands are int32_t
  using ComputeAzp = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::minus, float, int32_t,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeAzp =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeAzp, Accum, AzpWithAdj>;

  using ComputeScaleB = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeScaleB =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeScaleB, ScaleB,
                                              EVTComputeAzp>;

  using ComputeScaleBiasA = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeScaleBiasA, ScaleA,
                                              EVTComputeScaleB, Bias>;

  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(
      torch::stable::Tensor const& a_scales,
      torch::stable::Tensor const& b_scales,
      torch::stable::Tensor const& azp_adj,
      std::optional<torch::stable::Tensor> const& bias) {
    auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales);
    auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales);
    auto bias_args = SUPER::template args_from_tensor<Bias, ElementD>(bias);
    auto azp_adj_args =
        SUPER::template args_from_tensor<AzpWithAdj, int32_t>(azp_adj);

    typename EVTComputeAzp::Arguments evt_azp_args{{}, azp_adj_args, {}};
    typename EVTComputeScaleB::Arguments evt_scale_b_args{
        b_args, evt_azp_args, {}};
    return ArgumentType{a_args, evt_scale_b_args, bias_args, {}};
  }
};

/*
 * This epilogue supports per-token azp by computing and applying
 * the correction term using a rank-1 update. If the term were materialized,
 * it would require O(m*n) space, and this way it only requires O(m+n) space.
 * The azp term is a 1D tensor of shape (m,1), and represents the unscaled zero
 * point for each row of A.
 * The azp_adj term is a 1D tensor of shape (1,n), computed as J @ B.
 *
 * This epilogue also supports bias, which remains per-channel.
 */
template <typename ElementD, typename OutputTileThreadMap>
struct ScaledEpilogueBiasAzpToken
    : protected ScaledEpilogueBase<ElementD, OutputTileThreadMap> {
 private:
  using SUPER = ScaledEpilogueBase<ElementD, OutputTileThreadMap>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template RowOrZeroLoad<ElementD>;

  // Per-token azp term, shape (m,1)
  using Azp = typename SUPER::template ColLoad<int32_t>;

  // This is the AZP adjustment term, J @ B, shape (1,n)
  using AzpAdj = typename SUPER::template RowLoad<int32_t>;

  // Compute azp * azp_adj
  using ComputeAzp = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, int32_t, int32_t,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeAzp =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeAzp, Azp, AzpAdj>;

  // Compute float(accum - azp*azp_adj), all operands are int32_t
  using ComputeAcc = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::minus, float, int32_t,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeAcc =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeAcc, Accum, EVTComputeAzp>;

  using ComputeScaleB = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeScaleB =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeScaleB, ScaleB,
                                              EVTComputeAcc>;

  using ComputeScaleBiasA = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeScaleBiasA, ScaleA,
                                              EVTComputeScaleB, Bias>;

  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(
      torch::stable::Tensor const& a_scales,
      torch::stable::Tensor const& b_scales,
      torch::stable::Tensor const& azp_adj, torch::stable::Tensor const& azp,
      std::optional<torch::stable::Tensor> const& bias) {
    auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales);
    auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales);
    auto bias_args = SUPER::template args_from_tensor<Bias, ElementD>(bias);
    auto azp_args = SUPER::template args_from_tensor<Azp, int32_t>(azp);
    auto azp_adj_args =
        SUPER::template args_from_tensor<AzpAdj, int32_t>(azp_adj);

    typename EVTComputeAzp::Arguments evt_azp_args{azp_args, azp_adj_args, {}};
    typename EVTComputeAcc::Arguments evt_acc_args{{}, evt_azp_args, {}};
    typename EVTComputeScaleB::Arguments evt_scale_b_args{
        b_args, evt_acc_args, {}};
    return ArgumentType{a_args, evt_scale_b_args, bias_args, {}};
  }
};

};  // namespace vllm::c2x
