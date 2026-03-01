#pragma once

#include "cutlass_extensions/epilogue/broadcast_load_epilogue_c3x.hpp"
#include "cutlass_extensions/epilogue/broadcast_load_epilogue_array_c3x.hpp"

/*
   This file defines custom epilogues for fusing channel scales, token scales,
   bias, and activation zero-points onto a GEMM operation using the
   CUTLASS 3.x API, for NVIDIA GPUs with sm90a (Hopper) or later.

   Epilogues must contain a public type named EVTCompute of type Sm90EVT,
   as well as a static prepare_args function that constructs an
   EVTCompute::Arguments struct.
*/

namespace vllm::c3x {

using namespace cute;

template <typename T>
struct identity {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs) const { return lhs; }
};

template <typename ElementAcc, typename ElementD, typename TileShape>
struct TrivialEpilogue {
 private:
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using Compute = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::epilogue::thread::Identity, ElementD, ElementAcc,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute = cutlass::epilogue::fusion::Sm90EVT<Compute, Accum>;
  using ArgumentType = typename EVTCompute::Arguments;

  template <typename... Args>
  static ArgumentType prepare_args(Args... args) {
    return {};
  }
};

/*
 * This class provides the common load descriptors for the
 * ScaledEpilogue[...] classes
 */
template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogueBase {
 protected:
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  template <typename T>
  using ColOrScalarLoad = cutlass::epilogue::fusion::Sm90ColOrScalarBroadcast<
      0 /*Stages*/, TileShape, T, Stride<Int<1>, Int<0>, Int<0>>>;

  template <typename T>
  using RowOrScalarLoad = cutlass::epilogue::fusion::Sm90RowOrScalarBroadcast<
      0 /*Stages*/, TileShape, T, Stride<Int<0>, Int<1>, Int<0>>>;

  // Don't want to support nullptr by default
  template <typename T, bool EnableNullPtr = false>
  using ColLoad = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0 /*Stages*/, TileShape, T, T, Stride<Int<1>, Int<0>, Int<0>>,
      128 / sizeof_bits_v<T>, EnableNullPtr>;

  // Don't want to support nullptr by default
  template <typename T, bool EnableNullPtr = false>
  using RowLoad = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0 /*Stages*/, TileShape, T, T, Stride<Int<0>, Int<1>, Int<0>>,
      128 / sizeof_bits_v<T>, EnableNullPtr>;

  template <typename T>
  using ColOrScalarLoadArray =
      cutlass::epilogue::fusion::Sm90ColOrScalarBroadcastArray<
          0 /*Stages*/, TileShape, T, Stride<Int<1>, Int<0>, Int<0>>>;

  template <typename T>
  using RowOrScalarLoadArray =
      cutlass::epilogue::fusion::Sm90RowOrScalarBroadcastArray<
          0 /*Stages*/, TileShape, T, Stride<Int<0>, Int<1>, Int<0>>>;

  // This utility function constructs the arguments for the load descriptors
  // from a tensor. It can handle both row and column, as well as row/column or
  // scalar cases.
  template <typename Descriptor, typename T>
  static auto args_from_tensor(torch::Tensor const& tensor) {
    using Arguments = typename Descriptor::Arguments;
    auto* data_ptr = static_cast<T*>(tensor.data_ptr());
    if constexpr (std::is_same_v<Descriptor, ColOrScalarLoad<T>> ||
                  std::is_same_v<Descriptor, RowOrScalarLoad<T>>) {
      return Arguments{data_ptr, tensor.numel() != 1};
    } else {
      static_assert(!std::is_same_v<Descriptor, ColLoad<T, true>> &&
                    !std::is_same_v<Descriptor, RowLoad<T, true>>);
      return Arguments{data_ptr};
    }
  }

  // This overload handles the case where there might not be a tensor, in which
  // case a nullptr is passed and a constant (0) is used.
  template <typename Descriptor, typename T>
  static auto args_from_tensor(std::optional<torch::Tensor> const& tensor) {
    using Arguments = typename Descriptor::Arguments;
    auto* data_ptr = tensor ? static_cast<T*>(tensor->data_ptr()) : nullptr;
    static_assert(std::is_same_v<Descriptor, ColLoad<T, true>> ||
                  std::is_same_v<Descriptor, RowLoad<T, true>>);
    return Arguments{data_ptr};
  }

  template <typename Descriptor, typename T>
  static auto args_from_tensor(const T* const* data_ptr, bool do_broadcast) {
    using Arguments = typename Descriptor::Arguments;
    static_assert(std::is_same_v<Descriptor, ColOrScalarLoadArray<T>> ||
                  std::is_same_v<Descriptor, RowOrScalarLoadArray<T>>);
    return Arguments{data_ptr, do_broadcast};
  }
};

/*
   This epilogue function defines a quantized GEMM operation similar to
   torch.scaled_mm_.

   A and B may be both either int8 or fp8_e4m3. A can be
   quantized per-tensor or per-row. B can be quantized per-tensor or per-column.
   Any combination of per-tensor and per-row or column is supported.
   A and B must have symmetric quantization (zero point == 0).

   So the GEMM operation is D = (a_scales * A) (b_scales * B), where the
   scales are applied elementwise with numpy-style broadcasting.

   ScaleA and ScaleB define the epilogue functions that apply the scales for
   the A and B operands respectively. These scales may be either per-tensor or
   per row or column.
*/
template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogue
    : private ScaledEpilogueBase<ElementAcc, ElementD, TileShape> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, TileShape>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(torch::Tensor const& a_scales,
                                   torch::Tensor const& b_scales) {
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
template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogueBias
    : private ScaledEpilogueBase<ElementAcc, ElementD, TileShape> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, TileShape>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template RowLoad<ElementD>;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0, Bias>;

  using ArgumentType = typename EVTCompute::Arguments;
  static ArgumentType prepare_args(torch::Tensor const& a_scales,
                                   torch::Tensor const& b_scales,
                                   torch::Tensor const& bias) {
    auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales);
    auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales);
    auto bias_args = SUPER::template args_from_tensor<Bias, ElementD>(bias);

    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, bias_args, {}};
  }
};

/*
 * This epilogue performs the same operation as ScaledEpilogueBias, but the
 * bias is a column vector instead of a row vector. Useful e.g. if we are
 * computing a GEMM via C^T += B^T A^T. This happens in the 2:4 sparse kernels.
 */
template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogueColumnBias
    : private ScaledEpilogueBase<ElementAcc, ElementD, TileShape> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, TileShape>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template ColLoad<ElementD>;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0, Bias>;

  using ArgumentType = typename EVTCompute::Arguments;
  static ArgumentType prepare_args(torch::Tensor const& a_scales,
                                   torch::Tensor const& b_scales,
                                   torch::Tensor const& bias) {
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
template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogueBiasAzp
    : private ScaledEpilogueBase<ElementAcc, ElementD, TileShape> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, TileShape>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template RowLoad<ElementD, true>;

  // This is the full AZP term, azp * J @ B, shape (1,n)
  using AzpWithAdj = typename SUPER::template RowLoad<int32_t>;

  // Compute float(accum - azp_adj), both operands are int32_t
  using ComputeAzp = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::minus, float, int32_t,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeAzp =
      cutlass::epilogue::fusion::Sm90EVT<ComputeAzp, Accum, AzpWithAdj>;

  using ComputeScaleB = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeScaleB =
      cutlass::epilogue::fusion::Sm90EVT<ComputeScaleB, ScaleB, EVTComputeAzp>;

  using ComputeScaleBiasA = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<ComputeScaleBiasA, ScaleA,
                                         EVTComputeScaleB, Bias>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(torch::Tensor const& a_scales,
                                   torch::Tensor const& b_scales,
                                   torch::Tensor const& azp_adj,
                                   std::optional<torch::Tensor> const& bias) {
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
template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogueBiasAzpToken
    : private ScaledEpilogueBase<ElementAcc, ElementD, TileShape> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, TileShape>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template RowLoad<ElementD, true>;

  // Per-token azp term, shape (m,1)
  using Azp = typename SUPER::template ColLoad<int32_t>;

  // This is the AZP adjustment term, J @ B, shape (1,n)
  using AzpAdj = typename SUPER::template RowLoad<int32_t>;

  // Compute azp * azp_adj
  using ComputeAzp = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, int32_t, int32_t,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeAzp =
      cutlass::epilogue::fusion::Sm90EVT<ComputeAzp, Azp, AzpAdj>;

  // Compute float(accum - azp*azp_adj), all operands are int32_t
  using ComputeAcc = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::minus, float, int32_t,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeAcc =
      cutlass::epilogue::fusion::Sm90EVT<ComputeAcc, Accum, EVTComputeAzp>;

  using ComputeScaleB = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeScaleB =
      cutlass::epilogue::fusion::Sm90EVT<ComputeScaleB, ScaleB, EVTComputeAcc>;

  using ComputeScaleBiasA = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<ComputeScaleBiasA, ScaleA,
                                         EVTComputeScaleB, Bias>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(torch::Tensor const& a_scales,
                                   torch::Tensor const& b_scales,
                                   torch::Tensor const& azp_adj,
                                   torch::Tensor const& azp,
                                   std::optional<torch::Tensor> const& bias) {
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

/*
    This epilogue works like ScaledEpilogue, but ScaleA and ScaleB are pointers
    to arrays containing different scales used in group gemm. The number of
   pointers in ScaleA and the number of pointers in ScaleB are equal to the
   group size.
*/
template <typename ElementAcc, typename ElementD, typename EpilogueDescriptor>
struct ScaledEpilogueArray
    : private ScaledEpilogueBase<ElementAcc, ElementD, EpilogueDescriptor> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, EpilogueDescriptor>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoadArray<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoadArray<float>;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0>;
  using ArgumentType = typename EVTCompute::Arguments;

  using ScaleAArray = typename SUPER::template ColOrScalarLoadArray<float>;
  using ScaleBArray = typename SUPER::template RowOrScalarLoadArray<float>;

  static ArgumentType prepare_args(float const* const* a_scales_ptr,
                                   float const* const* b_scales_ptr,
                                   bool a_col_broadcast, bool b_row_broadcast) {
    auto a_args = SUPER::template args_from_tensor<ScaleAArray, float>(
        a_scales_ptr, a_col_broadcast);
    auto b_args = SUPER::template args_from_tensor<ScaleBArray, float>(
        b_scales_ptr, b_row_broadcast);

    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, {}};
  }
};

};  // namespace vllm::c3x
