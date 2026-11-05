# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An MLA layer's decode between the mono kernels' halves (M1a before, M1b in
K2 after):

- ``attend``: AITER's MLA decode on M1a's (gathered, padded) fp8 query
  (``forward_mqa``), at DCP8.
- ``m1c_inputs``: at DCP1 K2 runs the decode itself (M1c) on M1a's bf16 query.

Pure decode only (every row a decode row), as the mono step is.
"""

import torch

from vllm.model_executor.layers.attention.attention import get_attention_context


def context(wrapper):
    """``(metadata, MLAAttention, kv_cache, slot_mapping)`` of the step."""
    return get_attention_context(wrapper.mla_attn.layer_name)


def qkv_gate_weights(wrapper):
    """fused_qkv_a's [2112, 7168] and g_proj's [1536, 7168] weights: row
    slices of one [q_a | kv_a | gate] weight where vLLM merges the gate in
    (``KimiK3MergedQKVGateLinear``); the gate None without one."""
    w = wrapper.fused_qkv_a_proj.weight
    if wrapper.g_proj is not None:
        return w, wrapper.g_proj.weight
    if not getattr(wrapper, "_qkv_gate_fused", False):
        return w, None
    n = wrapper.q_lora_rank + wrapper.mla_attn.kv_lora_rank + wrapper.qk_rope_head_dim
    return w[:n], w[n:]


def attend(wrapper, q8, kv_cache, md):
    """The partial (this rank's KV) attention: ``o`` [S, heads * dcp, 512] and
    its LSE [S, heads * dcp] (None at DCP1), both possibly strided views.

    ``q8`` may carry more heads than the layer: AITER's cprr pad (e.g. 96 ->
    128, M1a writes it). forward_mqa gets the real heads, a view whose row
    stride spans the pad, and AITER's ``get_mla_padded_q`` widens it back to
    ``q8`` instead of copying."""
    from vllm.platforms import current_platform

    attn = wrapper.mla_attn
    kv = kv_cache.view(current_platform.fp8_dtype())
    n = attn.impl._decode_num_heads
    return attn.impl.forward_mqa(q8[:, :n], kv, md, attn)


def m1c_inputs(wrapper, q, kv_cache, md):
    """K2's ``m1c_in`` for M1a's bf16 query ``q`` [S, 12, 576] and its new
    partials, o [splits, S, 12, 512] bf16 and LSE [splits, S, 12] f32 (M1b
    merges the splits)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import mla_attn
    from vllm.platforms import current_platform

    attn = wrapper.mla_attn
    decode = md.decode
    kv = kv_cache.view(current_platform.fp8_dtype())
    key = mla_attn.MlaAttnBuild(tokens=q.size(0), qlen=int(decode.max_qo_len))
    so, sl = mla_attn.partial_shapes(key)
    m1c_in = dict(
        q=q,
        kv=kv.view(-1, kv.size(-1)),  # the flat token rows
        pages=decode.paged_kv_indices,
        indptr=decode.paged_kv_indptr,
        scale=attn.impl.scale,
        kv_scale=float(attn._k_scale_float),
    )
    po = q.new_empty(so)
    return m1c_in, po, torch.empty(sl, dtype=torch.float32, device=q.device)
