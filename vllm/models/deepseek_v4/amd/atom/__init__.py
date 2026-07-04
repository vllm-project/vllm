# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM DeepSeek V4 attention module, ported into the vLLM source tree.

This package is a faithful port of AMD ATOM's DeepSeek V4 hybrid-attention
subsystem — kept as ATOM-native code (imports repointed to this vendored root),
NOT rewritten to vLLM conventions. It bundles the four pieces the port targets:

* **KV cache layout adapter** — ``plugin.vllm.deepseek_v4_bridge``: a single
  vLLM proxy ``AttentionBackend`` / ``AttentionLayerBase`` whose
  ``FullAttentionSpec`` lets vLLM's pager own the classical KV blocks, plus
  ``slice_deepseek_v4_proxy_cache_views`` which carves ATOM's unified
  SWA-ring + CSA/HCA-compress views out of that vLLM-managed storage, and
  ``build_atom_v4_attention_metadata`` which re-derives ATOM's V4 metadata from
  vLLM's ``CommonAttentionMetadata``.
* **Ring SWA buffer** — the per-request sliding-window ring: the layout is
  produced by ``slice_deepseek_v4_proxy_cache_views`` (unified_kv / swa views)
  and the ``swa_pages + ring_idx`` addressing is written/read by the vendored
  Triton kernels (``swa_write``, ``write_v4_paged_decode_indices``,
  ``write_v4_decode_indices_fused``).
* **All related kernels** — ``model_ops.v4_kernels`` (compress plan, fused
  compress, sparse paged decode/prefill, csa translate-pack, qk-norm-rope,
  state writes, index builders) plus the bridge-local decode-index ops in
  ``plugin.vllm.deepseek_v4_ops`` and the ragged sparse-attn reference in
  ``model_ops.sparse_attn_v4``.
  The compressors run synchronously on the current stream; ATOM's
  multi-stream compressor overlap is intentionally not wired here (it is a
  separate, follow-up optimization).

Scope / single-node: this is a single-node port. Distributed features (Prefill
Context Parallel, tensor-parallel all-reduce) are stubbed to single-rank — the
PCP helpers report world size 1 and their code paths are never taken. ATOM's
engine-side pieces (ModelRunner, scheduler, config/quant parsers, weight loader)
are shimmed to the minimal surface the attention module imports; the ATOM-native
metadata builder is ModelRunner-coupled and is included as ported source, while
the vLLM-runnable adapter is the bridge.
"""
