# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SparseAttnIndexer — re-exported from upstream.

The upstream ``SparseAttnIndexer`` (in
``vllm/model_executor/layers/sparse_attn_indexer.py``) does the
sparse-attention indexer compute (FP8 MQA logits → top-k). It has no
portable PyTorch reference — the runtime path is gated by
``current_platform.is_cuda() / is_xpu() / is_rocm()`` and dispatches
to backend-specific kernels (DeepGEMM, AITER, FlashInfer). The
hw-agnostic OOT plugin patch we applied at
``vllm/model_executor/layers/sparse_attn_indexer.py`` adds an
``is_out_of_tree()`` branch that uses the pure-PyTorch references
``fp8_mqa_logits_torch`` / ``fp8_paged_mqa_logits_torch`` from
``vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`` — that's the
agnostic forward.

We previously vendored a stub that just raised
``NotImplementedError``; the runtime hits it during the profile run
and crashes. Re-exporting the upstream class (which now has the OOT
torch fallback) is the correct shape — same justification as the
``MoEActivation`` / ``FusedMoEMethodBase`` / ``MoERunnerInterface``
re-exports.
"""

from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

__all__ = ["SparseAttnIndexer"]
