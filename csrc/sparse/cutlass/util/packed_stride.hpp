/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Utilities for packing constructing canonical CuTe stride types for 3.x mainloop params.
*/

#pragma once

#include "cute/layout.hpp"
#include "cute/container/array.hpp"   // cute::array
#include "cutlass/conv/convolution.h" // cutlass::conv::Operator

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Strides without batch mode

template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<IntT, cute::Int<1>>
make_cute_packed_stride(cute::Stride<IntT, cute::Int<1>> s, cute::Shape<int,int,int> shape_MKL) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<0>(s_copy) = static_cast<IntT>(cute::get<1>(shape_MKL));
  return s_copy;
}

template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<cute::Int<1>, IntT>
make_cute_packed_stride(cute::Stride<cute::Int<1>, IntT> s, cute::Shape<int,int,int> shape_MKL) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<1>(s_copy) = static_cast<IntT>(cute::get<0>(shape_MKL));
  return s_copy;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// Strides with batch mode

template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<IntT, cute::Int<1>, int64_t>
make_cute_packed_stride(cute::Stride<IntT, cute::Int<1>, int64_t> s, cute::Shape<int,int,int> shape_MKL) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<0>(s_copy) = static_cast<IntT>(cute::get<1>(shape_MKL));
  int batch_count =  cute::get<2>(shape_MKL);
  if (batch_count > 1) {
    cute::get<2>(s_copy) = static_cast<IntT>(cute::get<0>(shape_MKL) * cute::get<1>(shape_MKL));
  }
  else {
    cute::get<2>(s_copy) = static_cast<IntT>(0);
  }
  return s_copy;
}

template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<cute::Int<1>, IntT, int64_t>
make_cute_packed_stride(cute::Stride<cute::Int<1>, IntT, int64_t> s, cute::Shape<int,int,int> shape_MKL) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<1>(s_copy) = static_cast<IntT>(cute::get<0>(shape_MKL));
  int batch_count =  cute::get<2>(shape_MKL);
  if (batch_count > 1) {
    cute::get<2>(s_copy) = static_cast<IntT>(cute::get<0>(shape_MKL) * cute::get<1>(shape_MKL));
  }
  else {
    cute::get<2>(s_copy) = static_cast<IntT>(0);
  }
  return s_copy;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// Strides with group mode

template <class StrideIntT>
CUTLASS_HOST_DEVICE
cute::Stride<StrideIntT, cute::Int<1>, cute::Int<0>>
make_cute_packed_stride(cute::Stride<StrideIntT, cute::Int<1>, cute::Int<0>> s, cute::Shape<int,int,int> shape_MKL) {
  static_assert(std::is_integral_v<StrideIntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<0>(s_copy) = static_cast<StrideIntT>(cute::get<1>(shape_MKL));
  return s_copy;
}

template <class StrideIntT>
CUTLASS_HOST_DEVICE
cute::Stride<cute::Int<1>, StrideIntT, cute::Int<0>>
make_cute_packed_stride(cute::Stride<cute::Int<1>, StrideIntT, cute::Int<0>> s, cute::Shape<int,int,int> shape_MKL) {
  static_assert(std::is_integral_v<StrideIntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<1>(s_copy) = static_cast<StrideIntT>(cute::get<0>(shape_MKL));
  return s_copy;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// Strides for convolutions

// Output cutlass::layout::TensorNDHWC -> rank-3 stride (InT,_1,_0)
// Note: For fprop/dgrad kernel, strides are assumed to be layout right in NZPQK/NDHWC order
// and therefore can be coalesced to just q/w. For wgrad kernel, strides are assumed to be layout
// right in KTRSC order and can be coalesced to just k.
// We enforce this condition here with asserts.
template <class IntT, size_t RankT_>
CUTLASS_HOST_DEVICE
cute::Stride<IntT, cute::Int<1>, cute::Int<0>>
make_cute_packed_stride(
    cute::Stride<IntT, cute::Int<1>, cute::Int<0>> s,
    cute::array<int32_t, RankT_> shape_output,
    cute::array<IntT, RankT_> stride_output,
    cutlass::conv::Operator conv_op) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  static_assert(RankT_ >= 3u);
  constexpr static int RankT = static_cast<int>(RankT_);

  assert(stride_output[RankT-1] == 1);
  cute::for_each(cute::make_seq<RankT-2>{}, [&](auto i) {
    assert(stride_output[i] == shape_output[i+1] * stride_output[i+1]);
  });

  auto s_copy = s;
  cute::get<0>(s_copy) = (conv_op == cutlass::conv::Operator::kWgrad) ?
      stride_output[0] :
      stride_output[RankT-2];
  return s_copy;
}

//
// Activation tensor ((w, h, d, n), _1) for fprop kernel
//

// Activation cutlass::layout::TensorNWC -> rank-2 stride ((W,N),_1)
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<cute::Stride<IntT, IntT>, cute::Int<1>>
make_cute_packed_stride(
    cute::Stride<cute::Stride<IntT, IntT>, cute::Int<1>> s,
    cute::array<IntT, 3> stride_nwc,
    conv::Operator ConvOp) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  assert(stride_nwc[2] == 1);
  auto s_copy = s;
  cute::get<0,0>(s_copy) = stride_nwc[1];
  cute::get<0,1>(s_copy) = stride_nwc[0];
  return s_copy;
}

// Activation cutlass::layout::TensorNHWC -> rank-2 stride ((W,H,N),_1)
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<cute::Stride<IntT, IntT, IntT>, cute::Int<1>>
make_cute_packed_stride(
    cute::Stride<cute::Stride<IntT, IntT, IntT>, cute::Int<1>> s,
    cute::array<IntT, 4> stride_nhwc,
    conv::Operator ConvOp) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  assert(stride_nhwc[3] == 1);
  auto s_copy = s;
  cute::for_each(cute::make_seq<3>{}, [&](auto i) {
    cute::get<0,i>(s_copy) = stride_nhwc[2-i];
  });
  return s_copy;
}

// Activation cutlass::layout::TensorNDHWC -> rank-2 stride ((W,H,D,N),_1)
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<cute::Stride<IntT, IntT, IntT, IntT>, cute::Int<1>>
make_cute_packed_stride(
    cute::Stride<cute::Stride<IntT, IntT, IntT, IntT>, cute::Int<1>> s,
    cute::array<IntT, 5> stride_ndhwc,
    conv::Operator ConvOp) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");

  assert(stride_ndhwc[4] == 1);
  auto s_copy = s;
  cute::for_each(cute::make_seq<4>{}, [&](auto i) {
    cute::get<0,i>(s_copy) = stride_ndhwc[3-i];
  });
  return s_copy;
}

//
// Filter tensor (k, (_1, s, r, t)) for fprop kernel
//

// Filter cutlass::layout::TensorNWC -> rank-2 stride (k, (_1, s))
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<IntT, cute::Stride<cute::Int<1>, IntT>>
make_cute_packed_stride(
    cute::Stride<IntT, cute::Stride<cute::Int<1>, IntT>> s,
    cute::array<IntT, 3> stride_ksc,
    conv::Operator ConvOp) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");

  assert(stride_ksc[2] == 1);
  auto s_copy = s;
  cute::get<0,0>(s_copy) = stride_ksc[0];
  cute::get<1,1>(s_copy) = stride_ksc[1];
  return s_copy;
}

// Filter cutlass::layout::TensorNHWC -> rank-2 stride (k, (_1, s, r))
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<IntT, cute::Stride<cute::Int<1>, IntT, IntT>>
make_cute_packed_stride(
    cute::Stride<IntT, cute::Stride<cute::Int<1>, IntT, IntT>> s,
    cute::array<IntT, 4> stride_krsc,
    conv::Operator ConvOp) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");

  assert(stride_krsc[3] == 1);
  auto s_copy = s;
  cute::get<0,0>(s_copy) = stride_krsc[0];
  cute::for_each(cute::make_seq<2>{}, [&](auto i) {
    cute::get<1,2-i>(s_copy) = stride_krsc[i+1];
  });
  return s_copy;
}

// Filter cutlass::layout::TensorNDHWC -> rank-2 stride (k, (_1, s, r, t))
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<IntT, cute::Stride<cute::Int<1>, IntT, IntT, IntT>>
make_cute_packed_stride(
    cute::Stride<IntT, cute::Stride<cute::Int<1>, IntT, IntT, IntT>> s,
    cute::array<IntT, 5> stride_ktrsc,
    conv::Operator ConvOp) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");

  assert(stride_ktrsc[4] == 1);
  auto s_copy = s;
  cute::get<0,0>(s_copy) = stride_ktrsc[0];
  cute::for_each(cute::make_seq<3>{}, [&](auto i) {
    cute::get<1,3-i>(s_copy) = stride_ktrsc[i+1];
  });
  return s_copy;
}

//
// Activation tensor (_1, (w, h, d, n)) for wgrad kernel
//
// It is also Filter tensor ((_1), (k, s, r, t)) for dgrad kernel
//

// Activation cutlass::layout::TensorNWC -> rank-2 stride (_1, (W,N)) in wgrad
// Filter cutlass::layout::TensorNWC -> rank-2 stride ((_1), (k, s)) in dgrad
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<cute::Int<1>, cute::Stride<IntT, IntT>>
make_cute_packed_stride(
    cute::Stride<cute::Int<1>, cute::Stride<IntT, IntT>> s,
    cute::array<IntT, 3> stride_nwc,
    conv::Operator ConvOp) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");

  assert(stride_nwc[2] == 1);
  auto s_copy = s;
  if (ConvOp == cutlass::conv::Operator::kWgrad) {
    cute::get<1,0>(s_copy) = stride_nwc[1];
    cute::get<1,1>(s_copy) = stride_nwc[0];
  }
  else if (ConvOp == cutlass::conv::Operator::kDgrad) {
    // stride_nwc in dgrad is ksc.
    cute::get<1,0>(s_copy) = stride_nwc[0];
    cute::get<1,1>(s_copy) = stride_nwc[1];
  }
  return s_copy;
}

// Activation cutlass::layout::TensorNHWC -> rank-2 stride (_1, (W,H,N)) in wgrad
// Filter cutlass::layout::TensorNHWC -> rank-2 stride ((_1), (k, s, r)) in dgrad
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<cute::Int<1>, cute::Stride<IntT, IntT, IntT>>
make_cute_packed_stride(
    cute::Stride<cute::Int<1>, cute::Stride<IntT, IntT, IntT>> s,
    cute::array<IntT, 4> stride_nhwc,
    conv::Operator ConvOp) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");

  assert(stride_nhwc[3] == 1);
  auto s_copy = s;
  if (ConvOp == cutlass::conv::Operator::kWgrad) {
    cute::for_each(cute::make_seq<3>{}, [&](auto i) {
      cute::get<1,i>(s_copy) = stride_nhwc[2-i];
    });
  }
  else if (ConvOp == cutlass::conv::Operator::kDgrad) {
    // stride_nhwc in dgrad is krsc.
    cute::get<1,0>(s_copy) = stride_nhwc[0];
    cute::for_each(cute::make_seq<2>{}, [&](auto i) {
      cute::get<1,2-i>(s_copy) = stride_nhwc[i+1];
    });
  }
  return s_copy;
}

// Activation cutlass::layout::TensorNDHWC -> rank-2 stride (_1, (W,H,D,N)) in wgrad
// Filter cutlass::layout::TensorNDHWC -> rank-2 stride ((_1), (k, s, r, t)) in dgrad
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<cute::Int<1>, cute::Stride<IntT, IntT, IntT, IntT>>
make_cute_packed_stride(
    cute::Stride<cute::Int<1>, cute::Stride<IntT, IntT, IntT, IntT>> s,
    cute::array<IntT, 5> stride_ndhwc,
    conv::Operator ConvOp) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");

  assert(stride_ndhwc[4] == 1);
  auto s_copy = s;
  if (ConvOp == cutlass::conv::Operator::kWgrad) {
    cute::for_each(cute::make_seq<4>{}, [&](auto i) {
      cute::get<1,i>(s_copy) = stride_ndhwc[3-i];
    });
  }
  else if (ConvOp == cutlass::conv::Operator::kDgrad) {
    // stride_ndhwc in dgrad is ktrsc.
    cute::get<1,0>(s_copy) = stride_ndhwc[0];
    cute::for_each(cute::make_seq<3>{}, [&](auto i) {
      cute::get<1,3-i>(s_copy) = stride_ndhwc[i+1];
    });
  }
  return s_copy;
}

//
// NZPQ tensor (_1, nzpq) for wgrad kernel
//

// cutlass::layout::TensorNWC -> rank-2 stride (_1, nzpq)
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<cute::Int<1>, IntT>
make_cute_packed_stride(
    cute::Stride<cute::Int<1>, IntT> s,
    cute::array<IntT, 3> stride_nqk,
    conv::Operator ConvOp) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");

  assert(stride_nqk[2] == 1);
  auto s_copy = s;
  cute::get<1>(s_copy) = stride_nqk[1];
  return s_copy;
}

// cutlass::layout::TensorNHWC -> rank-2 stride (_1, nzpq)
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<cute::Int<1>, IntT>
make_cute_packed_stride(
    cute::Stride<cute::Int<1>, IntT> s,
    cute::array<IntT, 4> stride_npqk,
    conv::Operator ConvOp) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");

  assert(stride_npqk[3] == 1);
  auto s_copy = s;
  cute::get<1>(s_copy) = stride_npqk[2];
  return s_copy;
}

// cutlass::layout::TensorNDHWC -> rank-2 stride (_1, nzpq)
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<cute::Int<1>, IntT>
make_cute_packed_stride(
    cute::Stride<cute::Int<1>, IntT> s,
    cute::array<IntT, 5> stride_nzpqk,
    conv::Operator ConvOp) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");

  assert(stride_nzpqk[4] == 1);
  auto s_copy = s;
  cute::get<1>(s_copy) = stride_nzpqk[3];
  return s_copy;
}



//
// Wgrad output tensor (k, (_1, s, r, t), _0)
//

// Filter cutlass::layout::TensorKCS -> rank-3 stride (k, (_1, s), _0)
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<IntT, cute::Stride<cute::Int<1>, IntT>, cute::Int<0>>
make_cute_packed_stride(
    cute::Stride<IntT, cute::Stride<cute::Int<1>, IntT>, cute::Int<0>> s,
    [[maybe_unused]] cute::array<int32_t, 3> shape_output,
    cute::array<IntT, 3> stride_ksc,
    conv::Operator ConvOp) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");

  assert(stride_ksc[2] == 1);
  auto s_copy = s;
  cute::get<0,0>(s_copy) = stride_ksc[0];
  cute::get<1,1>(s_copy) = stride_ksc[1];
  return s_copy;
}

// Filter cutlass::layout::TensorKCSR -> rank-3 stride (k, (_1, s, r), _0)
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<IntT, cute::Stride<cute::Int<1>, IntT, IntT>, cute::Int<0>>
make_cute_packed_stride(
    cute::Stride<IntT, cute::Stride<cute::Int<1>, IntT, IntT>, cute::Int<0>> s,
    [[maybe_unused]] cute::array<int32_t, 4> shape_output,
    cute::array<IntT, 4> stride_krsc,
    conv::Operator ConvOp) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");

  assert(stride_krsc[3] == 1);
  auto s_copy = s;
  cute::get<0,0>(s_copy) = stride_krsc[0];
  cute::for_each(cute::make_seq<2>{}, [&](auto i) {
    cute::get<1,2-i>(s_copy) = stride_krsc[i+1];
  });
  return s_copy;
}

// Filter cutlass::layout::TensorKCSRT -> rank-3 stride (k, (_1, s, r, t), _0)
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<IntT, cute::Stride<cute::Int<1>, IntT, IntT, IntT>, cute::Int<0>>
make_cute_packed_stride(
    cute::Stride<IntT, cute::Stride<cute::Int<1>, IntT, IntT, IntT>, cute::Int<0>> s,
    [[maybe_unused]] cute::array<int32_t, 5> shape_output,
    cute::array<IntT, 5> stride_ktrsc,
    conv::Operator ConvOp) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");

  assert(stride_ktrsc[4] == 1);
  auto s_copy = s;
  cute::get<0,0>(s_copy) = stride_ktrsc[0];
  cute::for_each(cute::make_seq<3>{}, [&](auto i) {
    cute::get<1,3-i>(s_copy) = stride_ktrsc[i+1];
  });
  return s_copy;
}


//
// Wgrad output tensor ((_1, s, r, t), k, _0)
//

// Filter cutlass::layout::TensorCSK -> rank-3 stride ((_1, s), k, _0)
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<cute::Stride<cute::Int<1>, IntT>, IntT, cute::Int<0>>
make_cute_packed_stride(
    cute::Stride<cute::Stride<cute::Int<1>, IntT>, IntT, cute::Int<0>> s,
    [[maybe_unused]] cute::array<int32_t, 3> shape_output,
    cute::array<IntT, 3> stride_ksc,
    conv::Operator ConvOp) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");

  assert(stride_ksc[2] == 1);
  auto s_copy = s;
  cute::get<1,0>(s_copy) = stride_ksc[0];
  cute::get<0,1>(s_copy) = stride_ksc[1];
  return s_copy;
}

// Filter cutlass::layout::TensorCSRK -> rank-3 stride ((_1, s, r), k, _0)
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<cute::Stride<cute::Int<1>, IntT, IntT>, IntT, cute::Int<0>>
make_cute_packed_stride(
    cute::Stride<cute::Stride<cute::Int<1>, IntT, IntT>, IntT, cute::Int<0>> s,
    [[maybe_unused]] cute::array<int32_t, 4> shape_output,
    cute::array<IntT, 4> stride_krsc,
    conv::Operator ConvOp) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");

  assert(stride_krsc[3] == 1);
  auto s_copy = s;
  cute::get<1,0>(s_copy) = stride_krsc[0];
  cute::for_each(cute::make_seq<2>{}, [&](auto i) {
    cute::get<0,2-i>(s_copy) = stride_krsc[i+1];
  });
  return s_copy;
}

// Filter cutlass::layout::TensorCSRTK -> rank-3 stride ((_1, s, r, t), k, _0)
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<cute::Stride<cute::Int<1>, IntT, IntT, IntT>, IntT, cute::Int<0>>
make_cute_packed_stride(
    cute::Stride<cute::Stride<cute::Int<1>, IntT, IntT, IntT>, IntT, cute::Int<0>> s,
    [[maybe_unused]] cute::array<int32_t, 5> shape_output,
    cute::array<IntT, 5> stride_ktrsc,
    conv::Operator ConvOp) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");

  assert(stride_ktrsc[4] == 1);
  auto s_copy = s;
  cute::get<1,0>(s_copy) = stride_ktrsc[0];
  cute::for_each(cute::make_seq<3>{}, [&](auto i) {
    cute::get<0,3-i>(s_copy) = stride_ktrsc[i+1];
  });
  return s_copy;
}
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
