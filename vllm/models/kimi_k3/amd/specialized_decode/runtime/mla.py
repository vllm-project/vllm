# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An MLA layer's decode, split where the mono kernels split it (vLLM's ops,
the reference M1a / M1b reproduce):

- ``pre``: qkv_a, the q / kv norms, q_b, W_UK (fp4), the query's fp8 quant,
  this rank's KV-cache rows, the DCP query all-gather (DCP8), the output gate.
- ``attend``: AITER's MLA decode on the (gathered) query (``forward_mqa``),
  on M1a's padded query as is. At DCP1 with qlen 4 / 8 K2 runs the decode
  itself (M1c, ``m1c_inputs``).
- ``post``: the DCP LSE combine (DCP8), W_UV (fp4), the gate: o_proj's input.

Pure decode only (every row a decode row), as the mono step is.
"""

import torch

from vllm._aiter_ops import rocm_aiter_ops
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


def pre(wrapper, hidden, kv_cache, slots, bf16=False):
    """``hidden``: input_layernorm's output [S, 7168]. Returns the gathered fp8
    query [S, heads * dcp, 576] (``bf16``, DCP1: bf16, unquantized) and the
    gate sigmoid(g_proj) [S, 1536]; writes the rows whose slot is this
    rank's."""
    attn = wrapper.mla_attn
    nope, rope = wrapper.qk_nope_head_dim, wrapper.qk_rope_head_dim
    qkv = wrapper.fused_qkv_a_proj(hidden)[0]
    if wrapper.g_proj is None:  # [q_a | kv_a | gate] in one GEMM
        qkv, g = qkv.split([wrapper.q_lora_rank + attn.kv_lora_rank + rope, -1], -1)
    else:
        g = wrapper.g_proj(hidden)[0]
    q_c, kv_lora = qkv.split([wrapper.q_lora_rank, attn.kv_lora_rank + rope], -1)
    kv_c, k_pe = kv_lora.split([attn.kv_lora_rank, rope], -1)
    q_c, kv_c = wrapper._normalize_q_kv(q_c, kv_c)
    q = wrapper.q_b_proj(q_c)[0].view(-1, wrapper.num_heads, nope + rope)
    q_nope, q_pe = q.split([nope, rope], -1)
    ql_nope = rocm_aiter_ops.batched_gemm_a16wfp4(
        q_nope.transpose(0, 1),
        attn.W_K,
        attn.W_K_scale,
        transpose_bm=True,
        prequant=True,
        y_scale=None if bf16 else attn._q_scale,
    )
    if bf16:
        q8 = torch.cat([ql_nope, q_pe], -1)
    else:
        q8 = attn._decode_concat_quant_fp8_op(ql_nope, q_pe, attn._q_scale)
    attn.impl.do_kv_cache_update(
        kv_c, k_pe.unsqueeze(1), kv_cache, slots, attn.kv_cache_dtype, attn._k_scale
    )
    if attn.impl.dcp_world_size > 1:
        q8 = attn.dcp_manager.query_gather(q8)
    gate = g.sigmoid()
    return q8, gate


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


def post(wrapper, o, lse, md, gate):
    """o_proj's input [S, heads * 128] bf16."""
    attn = wrapper.mla_attn
    latent = o
    if attn.impl.dcp_world_size > 1:
        latent = attn.dcp_manager.combine(
            o,
            lse,
            seq_lens=md.decode.seq_lens,
            query_start_loc=md.query_start_loc[: md.num_decodes + 1],
        )
    core = gate.new_empty(gate.shape)
    attn._v_up_proj(latent, out=core)
    return core * gate
