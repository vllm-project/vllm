# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.attention.head_partition import (
    make_attention_head_partition,
)


def _parts(total_q: int, total_kv: int, tp: int):
    return [
        make_attention_head_partition(
            total_num_heads=total_q,
            total_num_kv_heads=total_kv,
            tp_size=tp,
            tp_rank=rank,
        )
        for rank in range(tp)
    ]


def test_divisible_gqa_partition_matches_existing_layout():
    parts = _parts(total_q=24, total_kv=4, tp=2)

    assert [p.q_head_indices for p in parts] == [
        tuple(range(0, 12)),
        tuple(range(12, 24)),
    ]
    assert [p.unique_kv_head_indices for p in parts] == [(0, 1), (2, 3)]
    assert [p.kv_head_indices for p in parts] == [(0, 1), (2, 3)]
    assert [p.num_heads for p in parts] == [12, 12]
    assert [p.num_kv_heads for p in parts] == [2, 2]


def test_kv_replication_when_tp_exceeds_kv_heads():
    parts = _parts(total_q=24, total_kv=4, tp=8)

    assert [p.kv_head_indices for p in parts] == [
        (0,),
        (0,),
        (1,),
        (1,),
        (2,),
        (2,),
        (3,),
        (3,),
    ]
    assert all(p.num_heads == 3 for p in parts)
    assert all(p.num_kv_heads == 1 for p in parts)


def test_overlapping_gqa_partition_for_qwopus_tp3():
    parts = _parts(total_q=24, total_kv=4, tp=3)

    assert [p.q_head_indices for p in parts] == [
        tuple(range(0, 8)),
        tuple(range(8, 16)),
        tuple(range(16, 24)),
    ]
    assert [p.unique_kv_head_indices for p in parts] == [(0, 1), (1, 2), (2, 3)]
    assert [p.kv_head_indices for p in parts] == [
        (0, 0, 0, 1),
        (1, 1, 2, 2),
        (2, 3, 3, 3),
    ]
    assert [p.q_to_local_kv_indices for p in parts] == [
        (0, 0, 1, 1, 2, 2, 3, 3),
        (0, 0, 1, 1, 2, 2, 3, 3),
        (0, 0, 1, 1, 2, 2, 3, 3),
    ]
    assert all(p.num_heads == 8 for p in parts)
    assert all(p.num_kv_heads == 4 for p in parts)
    assert all(p.local_q_per_kv_slot == 2 for p in parts)
    assert all(p.has_overlapping_kv_partition for p in parts)


def test_slot_expansion_handles_uneven_local_gqa_boundaries():
    parts = _parts(total_q=36, total_kv=6, tp=4)

    assert [p.kv_head_indices for p in parts] == [
        (0, 0, 1),
        (1, 2, 2),
        (3, 3, 4),
        (4, 5, 5),
    ]
    assert all(p.num_heads == 9 for p in parts)
    assert all(p.num_kv_heads == 3 for p in parts)
    assert all(p.local_q_per_kv_slot == 3 for p in parts)


def test_rejects_non_gqa_head_layout():
    with pytest.raises(ValueError, match="total_num_heads .* total_num_kv_heads"):
        make_attention_head_partition(
            total_num_heads=30,
            total_num_kv_heads=8,
            tp_size=3,
            tp_rank=0,
        )


def test_qwopus_tp3_slot_expansion_matches_full_gqa_attention():
    torch.manual_seed(0)
    num_tokens = 5
    num_heads = 24
    num_kv_heads = 4
    head_dim = 16
    scale = head_dim**-0.5

    q = torch.randn(num_tokens, num_heads, head_dim)
    k = torch.randn(num_tokens, num_kv_heads, head_dim)
    v = torch.randn(num_tokens, num_kv_heads, head_dim)

    q_per_kv = num_heads // num_kv_heads
    full_k = k.repeat_interleave(q_per_kv, dim=1)
    full_v = v.repeat_interleave(q_per_kv, dim=1)
    full_scores = torch.einsum("qhd,khd->hqk", q, full_k) * scale
    full_probs = torch.softmax(full_scores, dim=-1)
    full_out = torch.einsum("hqk,khd->qhd", full_probs, full_v)

    local_outputs = []
    for part in _parts(total_q=num_heads, total_kv=num_kv_heads, tp=3):
        local_q = q[:, part.q_head_indices, :]
        local_k = k[:, part.kv_head_indices, :]
        local_v = v[:, part.kv_head_indices, :]
        expanded_k = local_k.repeat_interleave(part.local_q_per_kv_slot, dim=1)
        expanded_v = local_v.repeat_interleave(part.local_q_per_kv_slot, dim=1)
        local_scores = torch.einsum("qhd,khd->hqk", local_q, expanded_k) * scale
        local_probs = torch.softmax(local_scores, dim=-1)
        local_outputs.append(torch.einsum("hqk,khd->qhd", local_probs, expanded_v))

    tp_out = torch.cat(local_outputs, dim=1)
    torch.testing.assert_close(tp_out, full_out)
