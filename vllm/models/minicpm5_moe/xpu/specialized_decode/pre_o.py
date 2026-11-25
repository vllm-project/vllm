# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Input to the output projection of the TP2 Rows8 mono split."""

from contextvars import ContextVar
from functools import partial

import torch

from vllm.model_executor.layers.linear import apply_flattened


class PreOFa2SplitOne:
    """Set one FA2 KV split only during this instance's selected pre-O call."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.force_one = False
        self.active = ContextVar("minicpm5_pre_o_fa2_split1", default=False)
        self.calls = 0

    def call_attention(self, attention, q, k, v):
        token = self.active.set(True)
        try:
            return attention(q, k, v)
        finally:
            self.active.reset(token)

    def __call__(self, *args, **kwargs):
        if self.force_one and self.active.get() and not args and self._supports(kwargs):
            kwargs["num_splits"] = 1
            self.calls += 1
        return self.kernel(*args, **kwargs)

    @staticmethod
    def _supports(kwargs):
        q, k, v = (kwargs[name] for name in ("q", "k", "v"))
        table = kwargs.get("block_table")
        return (
            kwargs.get("fa_version") == 2
            and kwargs.get("max_seqlen_q") == 1
            and q.shape == (8, 16, 128)
            and k.shape[1:] == v.shape[1:] == (64, 1, 128)
            and q.dtype == k.dtype == v.dtype == torch.float16
            and table is not None
            and table.ndim == 2
            and table.shape[0] == 8
            and kwargs.get("window_size") in (None, (-1, -1), [-1, -1])
        )


def attention_before_output_projection(attention, positions, hidden, split_one=None):
    """Use the model's projection, rotary and cached attention instances."""
    qkv, _ = attention.qkv_proj(hidden)
    q, k, v = qkv.split(
        [
            attention.q_size * (2 if attention.gated else 1),
            attention.kv_size,
            attention.kv_size,
        ],
        dim=-1,
    )
    gate = None
    if attention.gated:
        q, gate = q.reshape(-1, attention.num_heads, 2 * attention.head_dim).chunk(
            2, -1
        )
        q = q.reshape(-1, attention.q_size)
        gate = gate.reshape(-1, attention.q_size)
    q, k = attention.rotary_emb(positions, q, k)
    if split_one is None:
        output = attention.attn(q, k, v)
    else:
        output = split_one.call_attention(attention.attn, q, k, v)
    if gate is not None:
        output = output * gate.sigmoid()
    return output


def rank_local_output_projection(projection, attention):
    """Compute the model's bias-free row-parallel GEMM before its collective."""
    return apply_flattened(
        partial(projection.quant_method.apply, projection, bias=None), attention
    )
