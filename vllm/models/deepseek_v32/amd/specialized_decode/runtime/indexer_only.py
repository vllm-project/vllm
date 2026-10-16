# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run only the sparse-attention indexer of a ``DeepseekV32MLAAttention`` layer (ROCm).

Mirrors ``DeepseekV32MLAAttention.forward`` / ``_fused_attention`` (amd/rocm.py) up to
``_run_indexer``: it writes the index-K cache and the shared top-k buffer, but passes
``mla_kv_cache=None`` to ``fused_norm_rope`` (the MonoKernel writes the MLA row) and
skips q_b_proj / W_UK / sparse MQA / W_UV / o_proj (``fused_q`` gets zero MQA dummies
whose outputs are discarded).

``trim=True`` (``DecodeConfig.indexer_trim``) drops discarded work without changing a
value the indexer consumes: the MQA dummies become persistent zero buffers, and only the
q_lora rows of ``fused_qkv_a_proj`` are computed (same linear method on a row-slice of
the weight, kept per layer and row count only after a one-time bitwise self-check
against the full GEMM; kv_c / k_pe then come from persistent dummies)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm.models.deepseek_v32.amd.specialized_decode.runtime.mono_layers import (
    layer_metadata,
)
from vllm.models.deepseek_v32.common.kernels import fused_norm_rope, fused_q


def _dummy_shapes(attn) -> dict[str, tuple[int, ...]]:
    h, r, kv = attn.num_local_heads, attn.qk_rope_head_dim, attn.kv_lora_rank
    return dict(kv_c=(kv,), k_pe=(r,), q_pe=(h, r), ql_nope=(h, kv))


class IndexerRefresh:
    """The indexer-only refresh of one layer, with its trim-mode state: persistent zero
    dummies (allocated here, before any graph capture) and, per row count, whether the
    sliced q GEMM was verified bit-identical to the full one."""

    def __init__(self, attn, rows: int, dtype, device, trim: bool = False):
        assert attn.indexer is not None and not attn.skip_topk
        self.attn, self.trim = attn, trim
        self.shapes = _dummy_shapes(attn)
        self.dummies: dict[str, torch.Tensor] = {}
        # T -> True (sliced q GEMM verified bitwise) / False (keep the full GEMM)
        self.q_slice_ok: dict[int, bool] = {}
        if trim:
            for name in self.shapes:
                self._dummy(name, rows, dtype, device)

    def _dummy(self, name, T, dtype, device) -> torch.Tensor:
        buf = self.dummies.get(name)
        if (
            buf is None
            or buf.shape[0] < T
            or buf.dtype != dtype
            or buf.device != device
        ):
            buf = torch.zeros(
                (max(T, 16), *self.shapes[name]), dtype=dtype, device=device
            )
            self.dummies[name] = buf
        return buf[:T]

    def _q_rows(self, hidden_states):
        """q_lora rows of fused_qkv_a_proj via a row-slice of its weight, or None unless
        verified bit-identical to the full GEMM (then the caller runs the full GEMM)."""
        attn = self.attn
        lin = attn.fused_qkv_a_proj
        T = hidden_states.shape[0]
        ok = self.q_slice_ok.get(T)
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
            # the self-check syncs (warm-up runs verify each row count first)
            return None
        try:
            w = lin.weight[: attn.q_lora_rank]
            q = lin.quant_method.apply(SimpleNamespace(weight=w), hidden_states, None)
        except Exception:  # the quant method needs more than .weight
            self.q_slice_ok[T] = False
            return None
        if ok is None:
            full = lin(hidden_states)[0][:, : attn.q_lora_rank]
            self.q_slice_ok[T] = bool(torch.equal(q, full))
            if not self.q_slice_ok[T]:
                return None
        return q

    def __call__(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> None:
        attn, trim = self.attn, self.trim
        T, dev, dt = hidden_states.shape[0], hidden_states.device, hidden_states.dtype

        def zeros(name):
            if trim:
                return self._dummy(name, T, dt, dev)
            return torch.zeros(T, *self.shapes[name], dtype=dt, device=dev)

        q_c = self._q_rows(hidden_states) if trim else None
        if q_c is not None:
            kv_c, k_pe = zeros("kv_c"), zeros("k_pe")
        else:
            qkv_lora = attn.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_c, k_pe = qkv_lora.split(
                [attn.q_lora_rank, attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1
            )
        kw = attn.indexer.wk_weights_proj(hidden_states)[0]
        index_k = kw[:, : attn.indexer.head_dim]
        index_weights = kw[:, attn.indexer.head_dim :]
        md, mla_slot = layer_metadata(attn.layer_name)
        assert md is not None, "indexer-only refresh needs attention metadata"

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
        index_q = attn.indexer.wq_b(q_c)[0].view(
            -1, attn.indexer.n_head, attn.indexer.head_dim
        )
        index_q_fp8, index_weights_out, _ = fused_q(
            positions,
            zeros("q_pe"),
            attn.rotary_emb.cos_sin_cache,
            index_q,
            attn.indexer_rope_emb.cos_sin_cache,
            zeros("ql_nope"),
            attn._q_scale,
            index_weights,
            attn.indexer.softmax_scale,
            attn.indexer.n_head**-0.5,
            has_indexer=True,
            index_rope_interleave=attn._index_rope_interleave,
            quantize_mqa=attn._fp8_kv,
        )
        attn._run_indexer(q_c, index_q_fp8, index_weights_out)
