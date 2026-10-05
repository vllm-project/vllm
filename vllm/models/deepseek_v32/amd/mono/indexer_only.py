# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run only the sparse-attention *indexer* of a ``DeepseekV32MLAAttention`` layer
(ROCm).

Mirrors ``DeepseekV32MLAAttention.forward`` / ``_fused_attention`` (amd/rocm.py) up to
and including ``_run_indexer``, but:
  * passes ``mla_kv_cache=None`` to ``fused_norm_rope`` (a zero entry stride disables
    the MLA cache write), so the MLA KV row is written exactly once -- by the
    MonoKernel;
  * skips q_b_proj / W_UK / sparse MQA / W_UV / o_proj; ``fused_q`` gets dummy MQA query
    tensors (its MQA outputs are discarded; the indexer outputs do not depend on them).
It still writes the index-K cache (at slot_mapping) and the shared top-k buffer, exactly
like the full attention does.

``trim=True`` (``LiveConfig.indexer_trim``, default) removes work whose results are
discarded, without changing any value the indexer consumes:
  * the two per-call ``torch.zeros`` MQA dummies (``q_pe``, ``ql_nope``) become
    persistent zero buffers. ``fused_q`` only reads them to produce MQA outputs that are
    discarded, and RoPE of zeros stays zero, so their contents never matter;
  * the kv rows (kv_lora + rope) of ``fused_qkv_a_proj`` are dead here
    (``mla_kv_cache=None``): only the q_lora rows feed ``wq_b``. The q rows are computed
    by the SAME linear method on a row-slice view of the weight, but only after a
    one-time bitwise self-check per layer and width (sliced q == full GEMM's q); if it
    ever differs or raises, that layer keeps the full GEMM. kv_c / k_pe then come from
    persistent dummies (program 1 of fused_norm_rope reads them but writes nothing when
    the MLA stride is 0).
"""

from __future__ import annotations

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v32.common.kernels import fused_norm_rope, fused_q

# (id(attn), name) -> persistent zero buffer (rows grow to the largest T seen)
_DUMMY: dict = {}
# (id(attn), T) -> True (sliced q GEMM verified bitwise) / False (keep the full GEMM)
_QSLICE: dict = {}


def _dummy(attn, name, T, shape_tail, dtype, device):
    key = (id(attn), name)
    buf = _DUMMY.get(key)
    if buf is None or buf.shape[0] < T or buf.dtype != dtype or buf.device != device:
        buf = torch.zeros((max(T, 16), *shape_tail), dtype=dtype, device=device)
        _DUMMY[key] = buf
    return buf[:T]


def _q_rows(attn, hidden_states):
    """q_lora rows of fused_qkv_a_proj only, via the layer's own linear method on a
    row-slice of its weight, or None when not verified bit-identical (then the caller
    runs the full GEMM)."""
    lin = attn.fused_qkv_a_proj
    T = hidden_states.shape[0]
    key = (id(attn), T)
    ok = _QSLICE.get(key)
    if (
        ok is False
        or not hasattr(lin, "weight")
        or getattr(lin, "_use_min_latency_gemm", False)
    ):
        return None
    if (
        ok is None
        and torch.cuda.is_available()
        and torch.cuda.is_current_stream_capturing()
    ):
        # the self-check syncs: never inside a capture (warm-up runs verify each width
        # first)
        return None
    from types import SimpleNamespace

    try:
        w = lin.weight[: attn.q_lora_rank]
        q = lin.quant_method.apply(SimpleNamespace(weight=w), hidden_states, None)
    except Exception:  # noqa: BLE001 -- any quant method that needs more than .weight: keep the full GEMM
        _QSLICE[key] = False
        return None
    if (
        ok is None
        # one-time bitwise self-check for this layer and width (eager warm-up / first
        # call)
    ):
        full = lin(hidden_states)[0][:, : attn.q_lora_rank]
        _QSLICE[key] = bool(torch.equal(q, full))
        if not _QSLICE[key]:
            return None
    return q


def prealloc(attn, rows: int, dtype, device) -> None:
    """Allocate the trim-mode persistent dummies before any graph capture (live
    install)."""
    _dummy(attn, "kv_c", rows, (attn.kv_lora_rank,), dtype, device)
    _dummy(attn, "k_pe", rows, (attn.qk_rope_head_dim,), dtype, device)
    _dummy(
        attn, "q_pe", rows, (attn.num_local_heads, attn.qk_rope_head_dim), dtype, device
    )
    _dummy(
        attn, "ql_nope", rows, (attn.num_local_heads, attn.kv_lora_rank), dtype, device
    )


def refresh_indexer(
    attn,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    trim: bool = False,
) -> None:
    assert attn.indexer is not None and not attn.skip_topk
    T0 = hidden_states.shape[0]
    q_c = _q_rows(attn, hidden_states) if trim else None
    if q_c is not None:
        dev, dt = hidden_states.device, hidden_states.dtype
        kv_c = _dummy(attn, "kv_c", T0, (attn.kv_lora_rank,), dt, dev)
        k_pe = _dummy(attn, "k_pe", T0, (attn.qk_rope_head_dim,), dt, dev)
    else:
        qkv_lora = attn.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_c, k_pe = qkv_lora.split(
            [attn.q_lora_rank, attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1
        )
    kw = attn.indexer.wk_weights_proj(hidden_states)[0]
    index_k = kw[:, : attn.indexer.head_dim]
    index_weights = kw[:, attn.indexer.head_dim :]

    fc = get_forward_context()
    all_md = fc.attn_metadata
    if isinstance(all_md, list):
        all_md = all_md[0]
    md = all_md.get(attn.layer_name) if isinstance(all_md, dict) else None
    assert md is not None, "indexer-only refresh needs attention metadata"
    slot_mapping = fc.slot_mapping
    assert isinstance(slot_mapping, dict)
    mla_slot = slot_mapping.get(attn.layer_name)

    q_c = fused_norm_rope(
        positions,
        q_c,
        attn.q_a_layernorm.weight,
        attn.q_a_layernorm.variance_epsilon,
        kv_c,
        attn.kv_a_layernorm.weight,
        attn.kv_a_layernorm.variance_epsilon,
        k_pe,
        attn.rotary_emb.cos_sin_cache,
        index_k,
        attn.indexer.k_norm.weight,
        attn.indexer.k_norm.bias,
        attn.indexer.k_norm.eps,
        attn.indexer_rope_emb.cos_sin_cache,
        attn.topk_indices_buffer,
        slot_mapping=mla_slot,
        indexer_k_cache=attn.indexer.k_cache.kv_cache,
        indexer_cache_shuffled=attn.indexer.k_cache.uses_shuffled_layout,
        mla_kv_cache=None,  # MLA row is the MonoKernel's to write
        mla_kv_cache_dtype=attn.kv_cache_dtype,
        mla_k_scale=None,
        has_indexer=True,
        index_rope_interleave=attn._index_rope_interleave,
    )
    T = hidden_states.shape[0]
    index_q = attn.indexer.wq_b(q_c)[0].view(
        -1, attn.indexer.n_head, attn.indexer.head_dim
    )
    if trim:
        q_pe = _dummy(
            attn,
            "q_pe",
            T,
            (attn.num_local_heads, attn.qk_rope_head_dim),
            hidden_states.dtype,
            hidden_states.device,
        )
        ql_nope = _dummy(
            attn,
            "ql_nope",
            T,
            (attn.num_local_heads, attn.kv_lora_rank),
            hidden_states.dtype,
            hidden_states.device,
        )
    else:
        q_pe = torch.zeros(
            T,
            attn.num_local_heads,
            attn.qk_rope_head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        ql_nope = torch.zeros(
            T,
            attn.num_local_heads,
            attn.kv_lora_rank,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    index_q_fp8, index_weights_out, _ = fused_q(
        positions,
        q_pe,
        attn.rotary_emb.cos_sin_cache,
        index_q,
        attn.indexer_rope_emb.cos_sin_cache,
        ql_nope,
        attn._q_scale,
        index_weights,
        attn.indexer.softmax_scale,
        attn.indexer.n_head**-0.5,
        has_indexer=True,
        index_rope_interleave=attn._index_rope_interleave,
        quantize_mqa=attn._fp8_kv,
    )
    attn._run_indexer(q_c, index_q_fp8, index_weights_out)
