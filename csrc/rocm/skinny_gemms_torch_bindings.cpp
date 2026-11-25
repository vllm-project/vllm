// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Registers the skinny GEMMs from _C_stable_libtorch into the _rocm_C
// namespace; the rest of _rocm_C is registered in rocm/torch_bindings.cpp.
// This unit is only built with the skinny GEMMs themselves, which are excluded
// on gfx1250.
#include <torch/csrc/stable/library.h>

#include "ops_stable.h"

STABLE_TORCH_LIBRARY_FRAGMENT(_rocm_C, rocm_ops) {
  // Custom gemm op for matrix-vector multiplication
  rocm_ops.def(
      "LLMM1(Tensor in_a, Tensor in_b, int rows_per_block) -> "
      "Tensor");

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitK(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");

  // W4A16 grouped skinny GEMM: packed int4 weights, per-group scales,
  // optional zero points [M/8, K/group_size] int32 for asymmetric
  // quantization
  rocm_ops.def(
      "wvSplitK_int4_g(Tensor in_a, Tensor in_b, Tensor in_scale, "
      "Tensor? in_zero_points, Tensor? in_bias, int CuCount, "
      "int group_size) -> Tensor");

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitKrc(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");

  // wvSplitK for fp8
  rocm_ops.def(
      "wvSplitKQ(Tensor in_a, Tensor in_b, Tensor? in_bias, Tensor! out_c, "
      "Tensor scale_a, "
      "          Tensor scale_b, int CuCount) -> ()");

  rocm_ops.def(
      "swmmacGEMM(Tensor in_a, Tensor in_b, "
      "Tensor? in_bias, int logical_M, int CuCount) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(_rocm_C, CUDA, rocm_ops) {
  rocm_ops.impl("LLMM1", TORCH_BOX(&LLMM1));
  rocm_ops.impl("wvSplitK", TORCH_BOX(&wvSplitK));
  rocm_ops.impl("wvSplitK_int4_g", TORCH_BOX(&wvSplitK_int4_g));
  rocm_ops.impl("wvSplitKrc", TORCH_BOX(&wvSplitKrc));
  rocm_ops.impl("wvSplitKQ", TORCH_BOX(&wvSplitKQ));
  rocm_ops.impl("swmmacGEMM", TORCH_BOX(&swmmacGEMM));
}
