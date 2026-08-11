// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project
//
// Separate .so from litedsa_jit_binding.cu: the DSv4 header is a second copy
// of the FlashMLA amalgamation, so the two cannot share a translation unit.

#include "litedsa_dsv4.cu"
#include "tvm_ffi_utils.h"

void litedsa_masked_mla_dsv4_bf16(TensorView q, TensorView kv, TensorView indices,
                                  TensorView pref, TensorView win_lo, TensorView win_hi,
                                  TensorView counts, double sm_scale, int64_t h_per_q,
                                  TensorView out, TensorView lse, TensorView attn_sink);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dsv4_masked_mla_bf16, litedsa_masked_mla_dsv4_bf16);
