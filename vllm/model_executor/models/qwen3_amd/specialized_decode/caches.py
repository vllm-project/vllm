# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache binding helpers for the Qwen3 RDNA dense specialized decode."""

from __future__ import annotations

from .backend import kv_planes


def gather_kv_bindings(model, num_kv_heads: int, head_dim: int):
    keys: list[int] = []
    values: list[int] = []
    block_sizes: set[int] = set()
    for layer in model.layers:
        kv = layer.self_attn.attn.kv_cache
        planes = kv_planes(kv, num_kv_heads, head_dim)
        if planes is None:
            return None
        keys.append(planes[0])
        values.append(planes[1])
        block_sizes.add(planes[2])
    block_size = block_sizes.pop() if len(block_sizes) == 1 else -1
    return keys, values, block_size
