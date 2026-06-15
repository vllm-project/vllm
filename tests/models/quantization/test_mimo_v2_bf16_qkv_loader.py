# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the MiMo-V2 fused-QKV BF16 pre-sharded loader.

When the source's fused QKV is the row-wise concatenation of independently-
sharded TP chunks (each ``[Q_chunk, K_chunk, V_chunk]``), vLLM's standard
``QKVParallelLinear`` weight loader assumes a canonical
``[Q_global, K_global, V_global]`` layout and silently produces structurally
wrong per-rank slices. The MTP attention path hits this case after a
derivative export dequantizes the source's FP8 attention back to BF16.

``_mimo_v2_copy_presharded_qkv_bf16`` detects the pre-sharded layout from
the row count and reassembles per-rank Q / K / V from the relevant ckpt
chunks. These tests pin every distinct TP path the helper supports:

  * ``tp_size == ckpt_tp`` direct-narrow fast path
  * ``tp_size < ckpt_tp`` reassembly path (multiple ckpt chunks per local rank)
  * ``tp_size == 1`` single-rank reassembly (the full Q, K, V)
  * SWA layer config with ckpt_tp = swa_num_kv_heads
  * Non pre-sharded shape returns False so the caller can fall through

Any regression in the per-rank Q/K/V reassembly resurfaces immediately
because each ckpt chunk is filled with a distinct numeric fingerprint.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.mimo_v2 import _mimo_v2_copy_presharded_qkv_bf16

HIDDEN = 4096


def _config() -> SimpleNamespace:
    """Mirror XiaomiMiMo/MiMo-V2.5 base for fields the helper reads."""
    return SimpleNamespace(
        head_dim=192,
        v_head_dim=128,
        num_attention_heads=64,
        num_key_value_heads=4,
        swa_head_dim=192,
        swa_v_head_dim=128,
        swa_num_attention_heads=64,
        swa_num_key_value_heads=8,
        hybrid_layer_pattern=[0, 1, 1, 1, 1, 0],
    )


def _build_chunked_qkv(
    n_chunks: int,
    q_heads_per_chunk: int,
    kv_heads_per_chunk: int,
    head_dim: int,
    v_head_dim: int,
    hidden: int,
) -> tuple[torch.Tensor, int]:
    """Build a pre-sharded fused QKV BF16 tensor with per-chunk fingerprints.

    Each ckpt chunk is filled with its index ``c`` so the test can identify
    which chunk a row came from when verifying per-rank reassembly.
    """
    chunk_rows = (
        q_heads_per_chunk * head_dim
        + kv_heads_per_chunk * head_dim
        + kv_heads_per_chunk * v_head_dim
    )
    weight = torch.cat(
        [
            torch.full((chunk_rows, hidden), c, dtype=torch.bfloat16)
            for c in range(n_chunks)
        ],
        dim=0,
    )
    return weight, chunk_rows


def _expected_local_shard(
    weight: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    """Reference per-rank shard by direct slicing of the pre-sharded weight.

    Mirrors the helper's slicing logic so the test fails on any drift, not
    just on the cases the helper explicitly enumerates.
    """
    q_heads_per_ckpt_rank = num_heads // num_kv_heads
    ckpt_q_rows = q_heads_per_ckpt_rank * head_dim
    ckpt_k_rows = head_dim
    ckpt_v_rows = v_head_dim
    ckpt_chunk_rows = ckpt_q_rows + ckpt_k_rows + ckpt_v_rows

    def chunk_slice(ckpt_rank: int, row_start: int, row_count: int) -> torch.Tensor:
        base = ckpt_rank * ckpt_chunk_rows + row_start
        return weight.narrow(0, base, row_count)

    if tp_size == num_kv_heads:
        return weight.narrow(0, tp_rank * ckpt_chunk_rows, ckpt_chunk_rows)

    q_heads_per_rank = num_heads // tp_size
    q_head_start = tp_rank * q_heads_per_rank
    q_head_end = q_head_start + q_heads_per_rank
    q_parts: list[torch.Tensor] = []
    next_q_head = q_head_start
    while next_q_head < q_head_end:
        ckpt_rank = next_q_head // q_heads_per_ckpt_rank
        ckpt_head_start = ckpt_rank * q_heads_per_ckpt_rank
        part_head_end = min(q_head_end, ckpt_head_start + q_heads_per_ckpt_rank)
        part_rows = (part_head_end - next_q_head) * head_dim
        part_start = (next_q_head - ckpt_head_start) * head_dim
        q_parts.append(chunk_slice(ckpt_rank, part_start, part_rows))
        next_q_head = part_head_end

    if tp_size >= num_kv_heads:
        num_replicas = tp_size // num_kv_heads
        kv_head_start = tp_rank // num_replicas
        kv_head_count = 1
    else:
        kv_head_count = num_kv_heads // tp_size
        kv_head_start = tp_rank * kv_head_count
    kv_head_end = kv_head_start + kv_head_count
    k_parts = [
        chunk_slice(ckpt_rank, ckpt_q_rows, ckpt_k_rows)
        for ckpt_rank in range(kv_head_start, kv_head_end)
    ]
    v_parts = [
        chunk_slice(ckpt_rank, ckpt_q_rows + ckpt_k_rows, ckpt_v_rows)
        for ckpt_rank in range(kv_head_start, kv_head_end)
    ]
    return torch.cat([*q_parts, *k_parts, *v_parts], dim=0).contiguous()


def _per_rank_shape(
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
    tp_size: int,
    hidden: int,
) -> tuple[int, int]:
    q_per_rank = (num_heads // tp_size) * head_dim
    if tp_size >= num_kv_heads:
        k_per_rank = head_dim
        v_per_rank = v_head_dim
    else:
        per_rank_kv = num_kv_heads // tp_size
        k_per_rank = per_rank_kv * head_dim
        v_per_rank = per_rank_kv * v_head_dim
    return (q_per_rank + k_per_rank + v_per_rank, hidden)


@pytest.mark.parametrize("tp_rank", [0, 1, 2, 3])
def test_ga_layer_tp_matches_ckpt(tp_rank: int) -> None:
    """tp_size == ckpt_tp: each rank's local weight is one ckpt chunk verbatim.

    Hits the direct-narrow path inside the helper. With per-chunk fingerprints,
    rank ``r``'s local weight must be filled entirely with ``r``.
    """
    config = _config()
    weight, _ = _build_chunked_qkv(
        n_chunks=4,
        q_heads_per_chunk=16,
        kv_heads_per_chunk=1,
        head_dim=config.head_dim,
        v_head_dim=config.v_head_dim,
        hidden=HIDDEN,
    )
    assert weight.shape == (13568, HIDDEN)

    per_rank_shape = _per_rank_shape(
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        v_head_dim=config.v_head_dim,
        tp_size=4,
        hidden=HIDDEN,
    )
    weight_param = torch.nn.Parameter(
        torch.empty(per_rank_shape, dtype=torch.bfloat16),
        requires_grad=False,
    )

    handled = _mimo_v2_copy_presharded_qkv_bf16(
        config=config,
        weight_name="model.layers.0.self_attn.qkv_proj.weight",
        weight_param=weight_param,
        loaded_weight=weight,
        tp_rank=tp_rank,
        tp_size=4,
    )

    assert handled is True
    assert torch.equal(
        weight_param.data,
        torch.full(per_rank_shape, tp_rank, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_ga_layer_tp_below_ckpt(tp_rank: int) -> None:
    """tp_size < ckpt_tp: each rank reassembles Q/K/V from multiple ckpt chunks.

    At TP=2 with ckpt_tp=4 the helper must collect Q from 2 ckpt chunks,
    K from 2 ckpt chunks, and V from 2 ckpt chunks, concatenated as
    [Q, K, V]. The chunk-fingerprint values let us see the right chunks
    landed in each section.
    """
    config = _config()
    weight, _ = _build_chunked_qkv(
        n_chunks=4,
        q_heads_per_chunk=16,
        kv_heads_per_chunk=1,
        head_dim=config.head_dim,
        v_head_dim=config.v_head_dim,
        hidden=HIDDEN,
    )

    per_rank_shape = _per_rank_shape(
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        v_head_dim=config.v_head_dim,
        tp_size=2,
        hidden=HIDDEN,
    )
    weight_param = torch.nn.Parameter(
        torch.empty(per_rank_shape, dtype=torch.bfloat16),
        requires_grad=False,
    )

    handled = _mimo_v2_copy_presharded_qkv_bf16(
        config=config,
        weight_name="model.layers.0.self_attn.qkv_proj.weight",
        weight_param=weight_param,
        loaded_weight=weight,
        tp_rank=tp_rank,
        tp_size=2,
    )

    assert handled is True
    expected = _expected_local_shard(
        weight,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        v_head_dim=config.v_head_dim,
        tp_rank=tp_rank,
        tp_size=2,
    )
    assert torch.equal(weight_param.data, expected)


def test_ga_layer_tp_one() -> None:
    """tp_size == 1: the single rank gets [Q_all, K_all, V_all].

    Verifies the loop end-condition by walking every ckpt chunk for Q and
    every ckpt chunk for K and V.
    """
    config = _config()
    weight, _ = _build_chunked_qkv(
        n_chunks=4,
        q_heads_per_chunk=16,
        kv_heads_per_chunk=1,
        head_dim=config.head_dim,
        v_head_dim=config.v_head_dim,
        hidden=HIDDEN,
    )

    per_rank_shape = _per_rank_shape(
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        v_head_dim=config.v_head_dim,
        tp_size=1,
        hidden=HIDDEN,
    )
    weight_param = torch.nn.Parameter(
        torch.empty(per_rank_shape, dtype=torch.bfloat16),
        requires_grad=False,
    )

    handled = _mimo_v2_copy_presharded_qkv_bf16(
        config=config,
        weight_name="model.layers.0.self_attn.qkv_proj.weight",
        weight_param=weight_param,
        loaded_weight=weight,
        tp_rank=0,
        tp_size=1,
    )

    assert handled is True
    expected = _expected_local_shard(
        weight,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        v_head_dim=config.v_head_dim,
        tp_rank=0,
        tp_size=1,
    )
    assert torch.equal(weight_param.data, expected)


@pytest.mark.parametrize("tp_rank", [0, 1, 2, 3])
def test_swa_layer_tp_below_ckpt(tp_rank: int) -> None:
    """SWA layer at TP=4 with ckpt_tp=8 (swa_num_kv_heads).

    Exercises the SWA branch of ``_mimo_v2_qkv_dims`` and the reassembly
    path together. Each rank takes Q from 2 ckpt chunks, K from 2 chunks,
    V from 2 chunks.
    """
    config = _config()
    weight, _ = _build_chunked_qkv(
        n_chunks=8,
        q_heads_per_chunk=8,
        kv_heads_per_chunk=1,
        head_dim=config.swa_head_dim,
        v_head_dim=config.swa_v_head_dim,
        hidden=HIDDEN,
    )

    per_rank_shape = _per_rank_shape(
        num_heads=config.swa_num_attention_heads,
        num_kv_heads=config.swa_num_key_value_heads,
        head_dim=config.swa_head_dim,
        v_head_dim=config.swa_v_head_dim,
        tp_size=4,
        hidden=HIDDEN,
    )
    weight_param = torch.nn.Parameter(
        torch.empty(per_rank_shape, dtype=torch.bfloat16),
        requires_grad=False,
    )

    handled = _mimo_v2_copy_presharded_qkv_bf16(
        config=config,
        weight_name="model.layers.1.self_attn.qkv_proj.weight",
        weight_param=weight_param,
        loaded_weight=weight,
        tp_rank=tp_rank,
        tp_size=4,
    )

    assert handled is True
    expected = _expected_local_shard(
        weight,
        num_heads=config.swa_num_attention_heads,
        num_kv_heads=config.swa_num_key_value_heads,
        head_dim=config.swa_head_dim,
        v_head_dim=config.swa_v_head_dim,
        tp_rank=tp_rank,
        tp_size=4,
    )
    assert torch.equal(weight_param.data, expected)


def test_non_presharded_shape_returns_false() -> None:
    """A canonical [Q_global, K_global, V_global] layout (no per-chunk
    duplication of K/V) does not match ckpt_tp * chunk_rows for any
    candidate ckpt_tp. The helper must return False so the caller falls
    through to ``param.weight_loader`` for the normal case.
    """
    config = _config()
    canonical_rows = (
        config.num_attention_heads * config.head_dim
        + config.num_key_value_heads * config.head_dim
        + config.num_key_value_heads * config.v_head_dim
    )
    assert canonical_rows == 13568
    # Force a row count that doesn't equal ckpt_tp * chunk_rows for any
    # ckpt_tp the helper can derive. Subtract one row to break the match.
    weight = torch.zeros((canonical_rows - 1, HIDDEN), dtype=torch.bfloat16)

    per_rank_shape = _per_rank_shape(
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        v_head_dim=config.v_head_dim,
        tp_size=4,
        hidden=HIDDEN,
    )
    sentinel = torch.full(per_rank_shape, -1.0, dtype=torch.bfloat16)
    weight_param = torch.nn.Parameter(sentinel.clone(), requires_grad=False)

    handled = _mimo_v2_copy_presharded_qkv_bf16(
        config=config,
        weight_name="model.layers.0.self_attn.qkv_proj.weight",
        weight_param=weight_param,
        loaded_weight=weight,
        tp_rank=0,
        tp_size=4,
    )

    assert handled is False
    assert torch.equal(weight_param.data, sentinel)
