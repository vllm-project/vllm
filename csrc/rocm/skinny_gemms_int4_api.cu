// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Entry point of the int4 skinny GEMM. The kernels are instantiated one unit
// per N_in (see skinny_gemms_int4_n<n>.cu); this unit only dispatches, so it
// includes the declaration header rather than the kernel header.
#include "skinny_gemms_int4.h"
// ops.h carries the public declaration of wvSplitK_int4_g(), so the defining
// translation unit also sees it (matches attention_dispatch.cu); a mismatch
// with VLLM_SKINNY_INT4_SIG would otherwise only show up as a link error.
#include "ops.h"

#include <stdexcept>
#include <string>

torch::Tensor wvSplitK_int4_g(VLLM_SKINNY_INT4_SIG) {
  switch (in_b.size(0)) {
    case 1:
      return wvSplitK_int4_g_impl<1>(in_a, in_b, in_scale, in_zero_points,
                                     in_bias, CuCount, group_size);
    case 2:
      return wvSplitK_int4_g_impl<2>(in_a, in_b, in_scale, in_zero_points,
                                     in_bias, CuCount, group_size);
    case 3:
      return wvSplitK_int4_g_impl<3>(in_a, in_b, in_scale, in_zero_points,
                                     in_bias, CuCount, group_size);
    case 4:
      return wvSplitK_int4_g_impl<4>(in_a, in_b, in_scale, in_zero_points,
                                     in_bias, CuCount, group_size);
    case 5:
      return wvSplitK_int4_g_impl<5>(in_a, in_b, in_scale, in_zero_points,
                                     in_bias, CuCount, group_size);
    default:
      throw std::runtime_error("Unsupported N value: " +
                               std::to_string(in_b.size(0)));
  }
}
