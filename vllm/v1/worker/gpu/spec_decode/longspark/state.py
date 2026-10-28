# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions adapted from LongSpark at commit 9ab6526e481bbea67489f3a73ebc782218d56f71:
# https://github.com/Hao-Yuan-He/LongSpark/tree/9ab6526e481bbea67489f3a73ebc782218d56f71/vendor/specforge/modeling/draft
#
# MIT License
# Copyright (c) 2026 The LongSpark Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Online-softmax primitives for LongSpark global-memory request state."""

import torch


def normalized_gqa_attention_batched(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched normalized GQA output for a shared query bank.

    Shapes are ``query=[Hq, Q, D]``, ``keys/values=[B, Hkv, K, D]``, and
    ``valid_mask=[B, K]``. Every batch row must contain at least one valid key.
    """
    if query.ndim != 3 or keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError(
            "LongSpark batched Global16 expects rank-3 query and rank-4 K/V"
        )
    batch_size, kv_heads, key_rows, key_dim = keys.shape
    query_heads, query_rows, head_dim = query.shape
    if valid_mask.shape != (batch_size, key_rows) or valid_mask.dtype != torch.bool:
        raise ValueError("LongSpark batched Global16 mask shape does not align")
    if not torch.all(valid_mask.any(dim=-1)):
        raise ValueError("LongSpark batched Global16 requires a key per request")
    if key_dim != head_dim or query_heads % kv_heads:
        raise ValueError("LongSpark Global16 query and target KV heads do not align")

    query_for_scores = query if query.dtype == keys.dtype else query.to(keys.dtype)
    groups = query_heads // kv_heads
    grouped_query = query_for_scores.reshape(kv_heads, groups, query_rows, head_dim)
    scores = torch.einsum("hgqd,bhkd->bhgqk", grouped_query, keys).reshape(
        batch_size, query_heads, query_rows, key_rows
    )
    scores = scores.float().mul_(head_dim**-0.5)
    scores.masked_fill_(~valid_mask[:, None, None], -torch.inf)
    max_logits = scores.max(dim=-1).values
    weights = torch.exp(scores - max_logits[..., None])
    normalizer = weights.sum(dim=-1)
    grouped_weights = weights.to(values.dtype).reshape(
        batch_size, kv_heads, groups, query_rows, key_rows
    )
    weighted_values = torch.einsum(
        "bhgqk,bhkd->bhgqd", grouped_weights, values
    ).reshape(batch_size, query_heads, query_rows, head_dim)
    output = weighted_values.float() / normalizer[..., None]
    logsumexp = max_logits + normalizer.log()
    return output, logsumexp


def normalized_gqa_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized GQA output and logsumexp.

    Shapes are ``query=[Hq, Q, D]`` and ``keys/values=[Hkv, K, D]``.
    Score and accumulator arithmetic follows the LongSpark reference: QK is
    evaluated in the cache dtype and promoted to FP32 before softmax; values
    are accumulated into FP32.
    """
    if query.ndim != 3 or keys.ndim != 3 or values.shape != keys.shape:
        raise ValueError("LongSpark Global16 expects rank-3 query/K/V tensors")
    query_heads, query_rows, head_dim = query.shape
    kv_heads, key_rows, key_dim = keys.shape
    if key_rows == 0:
        raise ValueError("LongSpark Global16 requires at least one target KV row")
    if key_dim != head_dim or query_heads % kv_heads:
        raise ValueError("LongSpark Global16 query and target KV heads do not align")

    query_for_scores = query if query.dtype == keys.dtype else query.to(keys.dtype)
    groups = query_heads // kv_heads
    grouped_query = query_for_scores.reshape(kv_heads, groups, query_rows, head_dim)
    scores = torch.einsum("hgqd,hkd->hgqk", grouped_query, keys).reshape(
        query_heads, query_rows, key_rows
    )
    scores = scores.float().mul_(head_dim**-0.5)
    max_logits = scores.max(dim=-1).values
    weights = torch.exp(scores - max_logits[..., None])
    normalizer = weights.sum(dim=-1)
    grouped_weights = weights.to(values.dtype).reshape(
        kv_heads, groups, query_rows, key_rows
    )
    weighted_values = torch.einsum("hgqk,hkd->hgqd", grouped_weights, values).reshape(
        query_heads, query_rows, head_dim
    )
    output = weighted_values.float() / normalizer[..., None]
    logsumexp = max_logits + normalizer.log()
    return output, logsumexp


def merge_normalized_attention(
    first_output: torch.Tensor,
    first_lse: torch.Tensor,
    second_output: torch.Tensor,
    second_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two normalized attention results without rereading old K/V."""
    if first_output.shape != second_output.shape:
        raise ValueError("LongSpark Global16 outputs must have matching shapes")
    if first_lse.shape != second_lse.shape:
        raise ValueError("LongSpark Global16 logsumexp tensors must match")
    if first_output.shape[:-1] != first_lse.shape:
        raise ValueError("LongSpark Global16 output and logsumexp do not align")

    merged_lse = torch.logaddexp(first_lse.float(), second_lse.float())
    first_weight = torch.exp(first_lse.float() - merged_lse)
    second_weight = torch.exp(second_lse.float() - merged_lse)
    output = (
        first_output.float() * first_weight[..., None]
        + second_output.float() * second_weight[..., None]
    )
    return output, merged_lse
