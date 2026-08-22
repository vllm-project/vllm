// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "core/registration.h"
#include "../ops.h"

#include <torch/csrc/stable/library.h>

// CPU stable-ABI extension. Kernels move here from csrc/cpu/ over subsequent
// PRs. Register under namespace "_C" so ops stay at torch.ops._C.<name>.
// REGISTER_EXTENSION uses TORCH_EXTENSION_NAME (the Python module:
// _C_stable_libtorch / _C_AVX512_stable_libtorch / _C_AVX2_stable_libtorch).

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  // Activation function used in SwiGLU.
  ops.def("silu_and_mul(Tensor! out, Tensor input) -> ()");

  // Activation function used in GeGLU with `none` approximation.
  ops.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");

  // Activation function used in GeGLU with `tanh` approximation.
  ops.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");

  // GELU tanh implementation.
  ops.def("gelu_tanh(Tensor! out, Tensor input) -> ()");

  // GELU implementation used in GPT-2.
  ops.def("gelu_new(Tensor! out, Tensor input) -> ()");

  // Approximate GELU implementation.
  ops.def("gelu_fast(Tensor! out, Tensor input) -> ()");

  // Quick GELU implementation.
  ops.def("gelu_quick(Tensor! out, Tensor input) -> ()");

  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.def(
      "rms_norm(Tensor! out, Tensor input, Tensor? weight, float epsilon) -> "
      "()");

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor? weight, "
      "float epsilon) -> ()");

  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor!? key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox, int "
      "rope_dim_offset=0, bool inverse=False) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CPU, ops) {
  ops.impl("silu_and_mul", TORCH_BOX(&silu_and_mul));
  ops.impl("gelu_and_mul", TORCH_BOX(&gelu_and_mul));
  ops.impl("gelu_tanh_and_mul", TORCH_BOX(&gelu_tanh_and_mul));
  ops.impl("gelu_tanh", TORCH_BOX(&gelu_tanh));
  ops.impl("gelu_new", TORCH_BOX(&gelu_new));
  ops.impl("gelu_fast", TORCH_BOX(&gelu_fast));
  ops.impl("gelu_quick", TORCH_BOX(&gelu_quick));
  ops.impl("rms_norm", TORCH_BOX(&rms_norm));
  ops.impl("fused_add_rms_norm", TORCH_BOX(&fused_add_rms_norm));
  ops.impl("rotary_embedding", TORCH_BOX(&rotary_embedding));
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
