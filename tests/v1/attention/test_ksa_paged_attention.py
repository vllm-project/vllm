# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

import vllm.envs as envs
from vllm.models.qwen3_ksa.nvidia.paged_attention import (
    ksa_paged_source_attention,
)


def _reference_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    row_to_request: torch.Tensor,
    kv_start: torch.Tensor,
    kv_end: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(query)
    lse = torch.empty(
        (query.shape[1], query.shape[0]),
        dtype=torch.float32,
        device=query.device,
    )
    group_size = query.shape[1] // key_cache.shape[2]
    kv_head_indices = torch.div(
        torch.arange(query.shape[1], device=query.device),
        group_size,
        rounding_mode="floor",
    )
    states_per_block = key_cache.shape[1]
    for row in range(query.shape[0]):
        start = int(kv_start[row])
        end = int(kv_end[row])
        if start == end:
            output[row].zero_()
            lse[:, row].fill_(-torch.inf)
            continue
        positions = torch.arange(start, end, device=query.device)
        request = int(row_to_request[row])
        physical_blocks = block_table[request, positions // states_per_block].long()
        offsets = positions % states_per_block
        key = key_cache[physical_blocks, offsets].index_select(1, kv_head_indices)
        value = value_cache[physical_blocks, offsets].index_select(1, kv_head_indices)
        scores = torch.einsum(
            "hd,khd->hk",
            query[row].float(),
            key.float(),
        ).mul(scale)
        lse[:, row] = torch.logsumexp(scores, dim=-1)
        probabilities = torch.softmax(scores, dim=-1)
        output[row] = torch.einsum(
            "hk,khd->hd",
            probabilities,
            value.float(),
        ).to(output.dtype)
    return output, lse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_tiled", [False, True])
def test_ksa_paged_source_attention_matches_reference(use_tiled: bool) -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    query = torch.randn(4, 4, 32, dtype=torch.bfloat16, device=device)
    key_cache = torch.randn(6, 4, 2, 32, dtype=torch.bfloat16, device=device)
    value_cache = torch.randn_like(key_cache)
    block_table = torch.tensor(
        [[2, 0, 5], [4, 1, 3]],
        dtype=torch.int32,
        device=device,
    )
    row_to_request = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
    kv_start = torch.tensor([0, 1, 0, 2], dtype=torch.int64, device=device)
    kv_end = torch.tensor([1, 7, 0, 10], dtype=torch.int64, device=device)
    scale = 1.0 / math.sqrt(query.shape[-1])

    tiled_args = {}
    if use_tiled:
        tiled_args = {
            "query_start_loc": torch.tensor(
                [0, 2, 4], dtype=torch.int32, device=device
            ),
            "max_query_len": 2,
        }
    actual_output, actual_lse = ksa_paged_source_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        row_to_request=row_to_request,
        kv_start=kv_start,
        kv_end=kv_end,
        softmax_scale=scale,
        **tiled_args,
    )
    expected_output, expected_lse = _reference_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        row_to_request=row_to_request,
        kv_start=kv_start,
        kv_end=kv_end,
        scale=scale,
    )

    torch.testing.assert_close(actual_output, expected_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ksa_split_k_decode_matches_reference() -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    batch_size = 2
    source_length = 4096
    states_per_block = 16
    blocks_per_request = source_length // states_per_block
    num_blocks = batch_size * blocks_per_request
    query = torch.randn(2, 4, 32, dtype=torch.bfloat16, device=device)
    key_cache = torch.randn(
        num_blocks,
        states_per_block,
        2,
        32,
        dtype=torch.bfloat16,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.arange(
        num_blocks,
        dtype=torch.int32,
        device=device,
    ).view(batch_size, blocks_per_request)
    row_to_request = torch.arange(batch_size, dtype=torch.int32, device=device)
    kv_start = torch.zeros(batch_size, dtype=torch.int64, device=device)
    kv_end = torch.full_like(kv_start, source_length)
    scale = 1.0 / math.sqrt(query.shape[-1])

    actual_output, actual_lse = ksa_paged_source_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        row_to_request=row_to_request,
        kv_start=kv_start,
        kv_end=kv_end,
        softmax_scale=scale,
        query_start_loc=torch.arange(
            batch_size + 1,
            dtype=torch.int32,
            device=device,
        ),
        max_query_len=1,
    )
    expected_output, expected_lse = _reference_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        row_to_request=row_to_request,
        kv_start=kv_start,
        kv_end=kv_end,
        scale=scale,
    )

    torch.testing.assert_close(actual_output, expected_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ksa_prefill_is_bitwise_batch_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    torch.manual_seed(19)
    device = torch.device("cuda")
    row_count = 16
    num_query_heads = 16
    num_kv_heads = 4
    head_dim = 128
    states_per_block = 16
    source_length = 1024
    num_blocks = source_length // states_per_block
    query = torch.randn(
        row_count,
        num_query_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    key_cache = torch.randn(
        num_blocks,
        states_per_block,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.randperm(
        num_blocks,
        dtype=torch.int32,
        device=device,
    ).unsqueeze(0)
    row_to_request = torch.zeros(row_count, dtype=torch.int32, device=device)
    kv_start = torch.zeros(row_count, dtype=torch.int64, device=device)
    kv_end = torch.arange(
        source_length - row_count + 1,
        source_length + 1,
        dtype=torch.int64,
        device=device,
    )
    scale = 1.0 / math.sqrt(head_dim)

    scheduled_output, scheduled_lse = ksa_paged_source_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        row_to_request=row_to_request,
        kv_start=kv_start,
        kv_end=kv_end,
        softmax_scale=scale,
        query_start_loc=torch.tensor([0, row_count], dtype=torch.int32, device=device),
        max_query_len=row_count,
    )
    rowwise_output, rowwise_lse = ksa_paged_source_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        row_to_request=row_to_request,
        kv_start=kv_start,
        kv_end=kv_end,
        softmax_scale=scale,
    )

    assert torch.equal(scheduled_output, rowwise_output)
    assert torch.equal(scheduled_lse, rowwise_lse)
