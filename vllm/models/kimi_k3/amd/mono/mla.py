# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An MLA layer's decode, split where the mono kernels split it (vLLM's ops,
the reference M1a / M1b reproduce):

- ``pre``: qkv_a, the q / kv norms, q_b, W_UK (fp4), the query's fp8 quant,
  this rank's KV-cache rows, the DCP query all-gather (DCP8), the output gate.
- ``attend``: AITER's MLA decode on the (gathered) query (``forward_mqa``).
- ``post``: the DCP LSE combine (DCP8), W_UV (fp4), the gate: o_proj's input.

Pure decode only (every row a decode row), as the mono step is.
"""

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.attention.attention import get_attention_context


def context(wrapper):
    """``(metadata, MLAAttention, kv_cache, slot_mapping)`` of the step."""
    return get_attention_context(wrapper.mla_attn.layer_name)


def pre(wrapper, hidden, kv_cache, slots):
    """``hidden``: input_layernorm's output [S, 7168]. Returns the gathered fp8
    query [S, heads * dcp, 576] and the gate sigmoid(g_proj) [S, 1536]; writes
    the rows whose slot is this rank's."""
    attn = wrapper.mla_attn
    nope, rope = wrapper.qk_nope_head_dim, wrapper.qk_rope_head_dim
    qkv = wrapper.fused_qkv_a_proj(hidden)[0]
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
        y_scale=attn._q_scale,
    )
    q8 = attn._decode_concat_quant_fp8_op(ql_nope, q_pe, attn._q_scale)
    attn.impl.do_kv_cache_update(
        kv_c, k_pe.unsqueeze(1), kv_cache, slots, attn.kv_cache_dtype, attn._k_scale
    )
    if attn.impl.dcp_world_size > 1:
        q8 = attn.dcp_manager.query_gather(q8)
    gate = wrapper.g_proj(hidden)[0].sigmoid()
    return q8, gate


def attend(wrapper, q8, kv_cache, md):
    """The partial (this rank's KV) attention: ``o`` [S, heads * dcp, 512] and
    its LSE [S, heads * dcp] (None at DCP1), both possibly strided views."""
    from vllm.platforms import current_platform

    attn = wrapper.mla_attn
    return attn.impl.forward_mqa(
        q8, kv_cache.view(current_platform.fp8_dtype()), md, attn
    )


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
