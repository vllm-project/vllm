#pragma once

#include <cute/tensor.hpp>
#include <torch/all.h>
namespace cute {

////////////////////////////////////////////////////////////////////
// layout utils
////////////////////////////////////////////////////////////////////

// Permute layout based on indices, example:
//   permute_layout<1, 0>(layout) will swap the two dimensions
//   permute_layout<0, 2, 1>(layout) will swap the last two dimensions
template <size_t... I, typename Layout>
auto permute_layout(Layout l) {
  static_assert(rank(l) == sizeof...(I), "Invalid permutation, rank mismatch");
  return cute::make_layout(cute::get<I>(l)...);
}

////////////////////////////////////////////////////////////////////
// Pointer utils
////////////////////////////////////////////////////////////////////

template <class PointerType>
static constexpr auto get_logical_ptr(PointerType* ptr) {
  if constexpr (cute::sizeof_bits_v<PointerType> < 8) {
    return cute::subbyte_iterator<PointerType>(ptr);
  } else {
    return ptr;
  }
}

};  // namespace cute
