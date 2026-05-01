/* SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Vendored CUTLASS FA3 MLA attention kernel binding for vLLM.
 * Based on sgl-kernel/csrc/flash_extension.cc from SGLang.
 *
 * This registers the FA3 forward pass as a PyTorch C++ extension under
 * the _cutlass_fa3_C namespace, enabling torch.ops._cutlass_fa3_C.fwd().
 *
 * Original source:
 *   https://github.com/sgl-project/sgl-attn (commit bcf72ccc)
 *   sgl-kernel/csrc/flash_extension.cc
 */
#include <Python.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "sgl_flash_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(_cutlass_fa3_C, m) {
  /*
   * CUTLASS FA3 MLA forward pass.
   * Signature matches sgl-attn's mha_fwd() exactly.
   */
  m.def(
      "fwd(Tensor   q,"
      "    Tensor   k,"
      "    Tensor   v,"
      "    Tensor?  k_new,"
      "    Tensor?  v_new,"
      "    Tensor?  q_v,"
      "    Tensor?  out,"
      "    Tensor?  cu_seqlens_q,"
      "    Tensor?  cu_seqlens_k,"
      "    Tensor?  cu_seqlens_k_new,"
      "    Tensor?  seqused_q,"
      "    Tensor?  seqused_k,"
      "    int?     max_seqlen_q,"
      "    int?     max_seqlen_k,"
      "    Tensor?  page_table,"
      "    Tensor?  kv_batch_idx,"
      "    Tensor?  leftpad_k,"
      "    Tensor?  rotary_cos,"
      "    Tensor?  rotary_sin,"
      "    Tensor?  seqlens_rotary,"
      "    Tensor?  q_descale,"
      "    Tensor?  k_descale,"
      "    Tensor?  v_descale,"
      "    float?   softmax_scale,"
      "    bool     is_causal,"
      "    int      window_size_left,"
      "    int      window_size_right,"
      "    int      attention_chunk,"
      "    float    softcap,"
      "    bool     is_rotary_interleaved,"
      "    Tensor?  scheduler_metadata,"
      "    int      num_splits,"
      "    bool?    pack_gqa,"
      "    int      sm_margin,"
      "    Tensor?  sinks"
      ") -> (Tensor, Tensor, Tensor, Tensor)");

  m.impl("fwd", torch::kCUDA, make_pytorch_shim(&mha_fwd));
}

// Python module initialization for _cutlass_fa3_C
PyMODINIT_FUNC PyInit__cutlass_fa3_C() {
  static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_cutlass_fa3_C",
                                      nullptr, 0, nullptr};
  return PyModule_Create(&module);
}
