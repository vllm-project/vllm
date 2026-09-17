# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.qwen4_exp.common import qsa_cache
from vllm.models.qwen4_exp.cpu.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.cpu.ops.qsa import (
    qsa_compress_groups_with_ratio,
    qsa_select_paged_tokens,
    qsa_sparse_paged_attention,
    qsa_store_cache_rows,
)
from vllm.models.qwen4_exp.cpu.runtime import has_active_triton_cpu_backend
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu(),
    reason="Qwen4Exp CPU QSA tests require a CPU platform",
)
requires_triton_cpu = pytest.mark.skipif(
    not has_active_triton_cpu_backend(),
    reason="Qwen4Exp sparse QSA requires an active Triton-CPU backend",
)


def test_qsa_metadata_selects_cpu_fallback_with_triton_installed(monkeypatch) -> None:
    monkeypatch.setattr(qsa_cache, "HAS_TRITON", True)

    assert (
        qsa_cache._select_qsa_metadata_fn(torch.device("cpu"))
        is qsa_cache._build_qsa_metadata_torch
    )


@requires_triton_cpu
def test_qsa_sparse_attention_matches_native_checkpoint_shape() -> None:
    torch.manual_seed(2)
    num_query_heads = 24
    num_kv_heads = 2
    head_dim = 256
    page_size = 128
    topk = 2048
    num_blocks = topk // page_size

    query = torch.randn(1, num_query_heads, head_dim, dtype=torch.bfloat16)
    key_cache = torch.randn(
        num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
    )
    value_cache = torch.randn_like(key_cache)
    logical_indices = torch.arange(topk, dtype=torch.int32).view(1, -1)
    block_table = torch.arange(num_blocks, dtype=torch.int32).view(1, -1)
    token_to_req = torch.zeros(1, dtype=torch.int32)

    actual = qsa_sparse_paged_attention(
        query,
        key_cache,
        value_cache,
        logical_indices,
        block_table,
        token_to_req,
    )

    keys = key_cache.flatten(0, 1).repeat_interleave(
        num_query_heads // num_kv_heads, dim=1
    )
    values = value_cache.flatten(0, 1).repeat_interleave(
        num_query_heads // num_kv_heads, dim=1
    )
    scores = torch.einsum("hd,khd->hk", query[0].float(), keys.float())
    probabilities = torch.softmax(scores * head_dim**-0.5, dim=-1)
    expected = torch.einsum("hk,khd->hd", probabilities, values.float()).bfloat16()

    torch.testing.assert_close(actual[0], expected, rtol=2e-2, atol=2e-2)


@requires_triton_cpu
def test_qsa_cpu_selection_handles_short_contexts_and_reuses_output(
    monkeypatch,
) -> None:
    num_heads = 2
    head_dim = 16
    query = torch.ones(3, num_heads, head_dim, dtype=torch.bfloat16)
    key_cache = (
        torch.tensor([1, 3, 2, 6, 4, 5], dtype=torch.bfloat16)
        .view(3, 2, 1, 1)
        .expand(-1, -1, 1, head_dim)
        .contiguous()
    )
    page_table = torch.tensor([[0, -1, -1], [0, 1, 2]], dtype=torch.int32)
    token_to_req = torch.tensor([0, 1, -1], dtype=torch.int32)
    query_positions = torch.tensor([1, 11, 7], dtype=torch.int64)
    sequence_lengths = torch.tensor([2, 12], dtype=torch.int32)
    output = torch.full((3, 5), 99, dtype=torch.int32)
    monkeypatch.setattr(qsa_ops, "_LOGITS_WORKSPACE_BYTES", 8)

    actual = qsa_select_paged_tokens(
        query,
        key_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio=2,
        token_topk=4,
        out=output,
    )

    assert actual is output
    assert actual.tolist() == [
        [0, 1, -1, -1, -1],
        [6, 7, 10, 11, -1],
        [-1, -1, -1, -1, -1],
    ]


@requires_triton_cpu
def test_qsa_cpu_streaming_compression_and_cache_store() -> None:
    raw_keys = torch.arange(6 * 4, dtype=torch.bfloat16).reshape(6, 1, 4)
    raw_positions = torch.arange(2, 8, dtype=torch.int64).view(6, 1, 1).expand(-1, 1, 3)
    token_to_req = torch.zeros(6, dtype=torch.int32)
    logical_positions = torch.arange(2, 8, dtype=torch.int64)
    query_start_loc = torch.tensor([0, 6], dtype=torch.int32)
    state_table = torch.tensor([[0]], dtype=torch.int32)
    state_cache = torch.zeros(1, 4, 1, 4, dtype=torch.bfloat16)
    state_cache[0, 0, 0] = -2
    state_cache[0, 1, 0] = -1
    compressed_slots = torch.full((6,), -1, dtype=torch.int64)
    compressed_slots[[1, 5]] = torch.tensor([0, 1])

    pooled, first_positions = qsa_compress_groups_with_ratio(
        raw_keys,
        raw_positions,
        state_cache,
        state_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        compress_ratio=4,
    )

    expected_first = (
        torch.stack(
            (state_cache[0, 0, 0], state_cache[0, 1, 0], raw_keys[0, 0], raw_keys[1, 0])
        )
        .float()
        .mean(0)
    )
    torch.testing.assert_close(pooled[1, 0], expected_first.bfloat16())
    torch.testing.assert_close(
        pooled[5, 0], raw_keys[2:6, 0].float().mean(0).bfloat16()
    )
    assert first_positions[[1, 5], 0].tolist() == [0, 4]

    slots = torch.tensor([-1, 3, 0, 1, 2, -1])
    qsa_store_cache_rows(state_cache, slots, raw_keys)
    torch.testing.assert_close(state_cache[0, 3, 0], raw_keys[1, 0])
    torch.testing.assert_close(state_cache[0, 0, 0], raw_keys[2, 0])
