#pragma once

#include <torch/csrc/stable/ops.h>

#include "machete_prepack_kernel.cuh"
#include "stable/cutlass_extensions/torch_utils.hpp"
#include "stable/torch_utils.h"
#include "stable/core/scalar_type.hpp"

namespace machete {

struct PrepackBArgs {
  torch::stable::Tensor const& B;
  torch::headeronly::ScalarType a_type;
  vllm::ScalarType b_type;
  std::optional<torch::headeronly::ScalarType> maybe_group_scales_type;
};

template <typename PrepackedLayoutB>
torch::stable::Tensor prepack_impl(torch::stable::Tensor const B) {
  using ElementB = typename PrepackedLayoutB::ElementB;
  using PPBlockShape_NK = typename PrepackedLayoutB::PPBlockShape_NK;

  auto device_index = B.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  auto stream = get_current_cuda_stream(device_index);
  auto B_ptr = static_cast<ElementB const*>(B.const_data_ptr());
  // elements per storage item for B
  auto eles_per_storage =
      (B.element_size() * 8) / cute::sizeof_bits_v<ElementB>;

  // torch B passed in is/should be (packed_K,N), the kernel expects (N,K,L) (to
  // match cutlass using (N,K,L) for B), so we transpose B to (N,packed_K,L)
  auto Bt_packed = torch::stable::transpose(B, 0, 1);

  STD_TORCH_CHECK(
      (B.size(0) * eles_per_storage) % size<1>(PPBlockShape_NK{}) == 0,
      "B.shape[0] (in terms of unpacked elements) must be a multiple of ",
      size<1>(PPBlockShape_NK{}));
  STD_TORCH_CHECK(B.size(1) % size<0>(PPBlockShape_NK{}) == 0,
                  "B.shape[1] must be a multiple of ",
                  size<0>(PPBlockShape_NK{}));

  using StrideB = cutlass::detail::TagToStrideB_t<cutlass::layout::ColumnMajor>;
  auto const l_Bt_packed = make_cute_layout<StrideB>(Bt_packed, "B");

  // convert (N,packed_K,L) layout to (N,K,L) layout
  //  in effect we want to do: blocked_product(layout_Bt_packed,
  //      make_ordered_layout(make_shape(_1{}, eles_per_storage, _1{}),
  //                          Step<_1, _0, _2>{}));
  // but blocked_product does not support dynamic strides so we implement the
  // equivalent manually,
  //   new_shape = (N, packed_K, L) * (1, eles_per_storage, 1) -> (N, K, L)
  //   new_stride = (s0, s1, s2) * (eles_per_storage, 1, eles_per_storage)
  //                 when s1 == 1
  STD_TORCH_CHECK(stride<1>(l_Bt_packed) == 1);
  // clang-format off
  auto const layout_Bt = make_layout(
      transform_with_idx(l_Bt_packed.shape(), [&](auto ele, auto idx) {
        return idx == 1 ? ele * eles_per_storage : ele;
      }),
      transform_with_idx(l_Bt_packed.stride(), [&](auto ele, auto idx) {
        return idx != 1 ? ele * eles_per_storage : ele;
      }));
  // clang-format on

  // Allocate output with same shape and dtype as B
  torch::stable::Tensor D = torch::stable::empty(
      B.sizes(), B.scalar_type(), torch::headeronly::Layout::Strided,
      B.device(), std::nullopt, torch::headeronly::MemoryFormat::Contiguous);

  prepack_B_template<PrepackedLayoutB>(
      stream, B_ptr, layout_Bt, static_cast<ElementB*>(D.mutable_data_ptr()));

  return D;
};

torch::stable::Tensor prepack_B_dispatch(PrepackBArgs args);

};  // namespace machete
