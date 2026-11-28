// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Entry point of the int8 skinny GEMM. The kernels are instantiated one unit
// per N_in (see skinny_gemms_int8_n<n>.cu); this unit only dispatches, so it
// includes the declaration header rather than the kernel header.
#include "skinny_gemms_int8.h"
// ops.h carries the public declaration of wvSplitK_int8(), so a mismatch with
// VLLM_SKINNY_INT8_SIG fails to compile here instead of at link time.
#include "ops.h"

torch::Tensor wvSplitK_int8(VLLM_SKINNY_INT8_SIG) {
  switch (in_b.size(0)) {
    case 1:
      return wvSplitK_int8_impl<1>(in_a, in_b, in_scale, in_bias, CuCount);
    case 2:
      return wvSplitK_int8_impl<2>(in_a, in_b, in_scale, in_bias, CuCount);
    case 3:
      return wvSplitK_int8_impl<3>(in_a, in_b, in_scale, in_bias, CuCount);
    case 4:
      return wvSplitK_int8_impl<4>(in_a, in_b, in_scale, in_bias, CuCount);
    case 5:
      return wvSplitK_int8_impl<5>(in_a, in_b, in_scale, in_bias, CuCount);
    default:
      TORCH_CHECK(false, "wvSplitK_int8 supports N <= 5, got N=", in_b.size(0));
  }
}
