# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch.nn import functional as F

from vllm.v1.worker.gpu.spec_decode.longspark.kernels import (
    fused_add_argmax,
    fused_markov_argmax,
    longspark_attention,
    longspark_paged_attention,
    update_longspark_global_delta,
)
from vllm.v1.worker.gpu.spec_decode.longspark.state import (
    normalized_gqa_attention,
)


def _gather_paged_rows(cache, block_table, positions):
    block_size = cache.shape[2]
    logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
    offsets = positions.remainder(block_size)
    physical_blocks = torch.gather(block_table, 1, logical_blocks)
    rows = cache[physical_blocks, :, offsets]
    keys, values = rows.split(cache.shape[-1] // 2, dim=-1)
    return keys.transpose(1, 2), values.transpose(1, 2)


def test_longspark_fused_add_argmax_matches_fp32_reference():
    generator = torch.Generator(device="cuda").manual_seed(31)
    batch_size, vocab_size = 3, 10_037
    base = torch.randn(
        batch_size,
        vocab_size,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    bias = torch.randn_like(base, generator=generator)
    chunks = (vocab_size + 8191) // 8192
    partial_values = torch.empty(batch_size, chunks, dtype=torch.float32, device="cuda")
    partial_ids = torch.empty(batch_size, chunks, dtype=torch.int32, device="cuda")
    output_storage = torch.empty(batch_size, 2, dtype=torch.int64, device="cuda")
    output = output_storage[:, 1]

    actual = fused_add_argmax(
        base,
        bias,
        partial_values=partial_values,
        partial_ids=partial_ids,
        output=output,
    )
    expected = (base.float() + bias.float()).argmax(dim=-1)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(("batch_size", "rank"), [(3, 256), (17, 96)])
def test_longspark_fused_markov_argmax_matches_bf16_projection(
    batch_size: int, rank: int
):
    generator = torch.Generator(device="cuda").manual_seed(41)
    vocab_size = 10_037
    base = torch.randn(
        batch_size,
        vocab_size,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    embedding = torch.randn(
        batch_size,
        rank,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    weight = torch.randn(
        vocab_size,
        rank,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    chunks = (vocab_size + 63) // 64
    partial_values = torch.empty(batch_size, chunks, dtype=torch.float32, device="cuda")
    partial_ids = torch.empty(batch_size, chunks, dtype=torch.int32, device="cuda")
    output_storage = torch.empty(batch_size, 2, dtype=torch.int64, device="cuda")
    output = output_storage[:, 1]

    actual = fused_markov_argmax(
        base,
        embedding,
        weight,
        partial_values=partial_values,
        partial_ids=partial_ids,
        output=output,
    )
    bias = F.linear(embedding, weight).to(torch.bfloat16)
    expected = (base.float() + bias.float()).argmax(dim=-1)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    (
        "query_heads",
        "kv_heads",
        "query_rows",
        "global_rows",
        "raw_rows",
        "head_dim",
    ),
    [(32, 8, 7, 16, 256, 128), (6, 2, 5, 9, 96, 64)],
)
def test_longspark_fused_attention_matches_dense_reference(
    query_heads: int,
    kv_heads: int,
    query_rows: int,
    global_rows: int,
    raw_rows: int,
    head_dim: int,
):
    generator = torch.Generator(device="cuda").manual_seed(37)
    batch_size = 2
    query = torch.randn(
        batch_size,
        query_heads,
        query_rows,
        head_dim,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    global_keys = torch.randn(
        batch_size,
        query_heads,
        global_rows,
        head_dim,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    global_values = torch.randn(
        batch_size,
        query_heads,
        global_rows,
        head_dim,
        generator=generator,
        dtype=torch.float32,
        device="cuda",
    )
    raw_keys = torch.randn(
        batch_size,
        kv_heads,
        raw_rows,
        head_dim,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    raw_values = torch.randn_like(raw_keys, generator=generator)
    raw_mask = (
        torch.arange(raw_rows, device="cuda")[None]
        < torch.tensor([raw_rows, max(1, raw_rows - 83)], device="cuda")[:, None]
    )
    local_keys = torch.randn(
        batch_size,
        kv_heads,
        query_rows,
        head_dim,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    local_values = torch.randn_like(local_keys, generator=generator)
    scale = head_dim**-0.5

    actual = longspark_attention(
        query=query,
        global_keys=global_keys,
        global_values=global_values,
        raw_keys=raw_keys,
        raw_values=raw_values,
        raw_mask=raw_mask,
        local_keys=local_keys,
        local_values=local_values,
        scale=scale,
    )

    groups = query_heads // kv_heads
    repeated_raw_keys = raw_keys.repeat_interleave(groups, dim=1)
    repeated_raw_values = raw_values.repeat_interleave(groups, dim=1)
    repeated_local_keys = local_keys.repeat_interleave(groups, dim=1)
    repeated_local_values = local_values.repeat_interleave(groups, dim=1)
    keys = torch.cat((global_keys, repeated_raw_keys, repeated_local_keys), dim=2)
    values = torch.cat(
        (
            global_values.to(torch.bfloat16),
            repeated_raw_values,
            repeated_local_values,
        ),
        dim=2,
    )
    mask = torch.cat(
        (
            torch.ones(batch_size, global_rows, dtype=torch.bool, device="cuda"),
            raw_mask,
            torch.ones(batch_size, query_rows, dtype=torch.bool, device="cuda"),
        ),
        dim=1,
    )
    expected = F.scaled_dot_product_attention(
        query,
        keys,
        values,
        attn_mask=mask[:, None, None],
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    (
        "query_heads",
        "kv_heads",
        "query_rows",
        "global_rows",
        "raw_rows",
        "head_dim",
        "seq_len_values",
    ),
    [
        (32, 8, 7, 16, 256, 128, (300, 173)),
        (6, 2, 5, 9, 96, 64, (140, 73)),
    ],
)
def test_longspark_paged_attention_matches_materialized_window(
    query_heads: int,
    kv_heads: int,
    query_rows: int,
    global_rows: int,
    raw_rows: int,
    head_dim: int,
    seq_len_values: tuple[int, int],
):
    generator = torch.Generator(device="cuda").manual_seed(43)
    batch_size, block_size = 2, 16
    blocks_per_request = (max(seq_len_values) + block_size - 1) // block_size
    num_blocks = batch_size * blocks_per_request
    query = torch.randn(
        batch_size,
        query_heads,
        query_rows,
        head_dim,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    global_keys = torch.randn(
        batch_size,
        query_heads,
        global_rows,
        head_dim,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    global_values = torch.randn(
        batch_size,
        query_heads,
        global_rows,
        head_dim,
        generator=generator,
        dtype=torch.float32,
        device="cuda",
    )
    local_keys = torch.randn(
        batch_size,
        kv_heads,
        query_rows,
        head_dim,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    local_values = torch.randn_like(local_keys, generator=generator)
    cache = torch.randn(
        num_blocks,
        kv_heads,
        block_size,
        2 * head_dim,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    block_table = torch.stack(
        tuple(
            torch.arange(
                request * blocks_per_request,
                (request + 1) * blocks_per_request,
                device="cuda",
            )
            for request in range(batch_size)
        )
    ).int()
    seq_lens = torch.tensor(seq_len_values, dtype=torch.int32, device="cuda")
    counts = seq_lens.long().clamp(max=raw_rows)
    starts = seq_lens.long() - counts
    positions = starts[:, None] + torch.arange(raw_rows, device="cuda")[None]
    raw_mask = positions < seq_lens[:, None]
    safe_positions = torch.minimum(positions, seq_lens[:, None] - 1)
    raw_keys, raw_values = _gather_paged_rows(cache, block_table, safe_positions)

    materialized = longspark_attention(
        query=query,
        global_keys=global_keys,
        global_values=global_values,
        raw_keys=raw_keys,
        raw_values=raw_values,
        raw_mask=raw_mask,
        local_keys=local_keys,
        local_values=local_values,
        scale=head_dim**-0.5,
    )
    paged = longspark_paged_attention(
        query=query,
        global_keys=global_keys,
        global_values=global_values,
        raw_cache=cache,
        raw_block_table=block_table,
        raw_seq_lens=seq_lens,
        raw_rows=raw_rows,
        local_keys=local_keys,
        local_values=local_values,
        scale=head_dim**-0.5,
    )

    torch.testing.assert_close(paged, materialized, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    (
        "query_heads",
        "kv_heads",
        "global_rows",
        "head_dim",
        "start_values",
        "end_values",
        "max_delta_rows",
    ),
    [
        (32, 8, 16, 128, (20, 33, 48), (28, 37, 49), 8),
        (6, 2, 9, 64, (10, 17, 24), (15, 20, 25), 5),
    ],
)
def test_longspark_global_delta_kernel_matches_full_prefix(
    query_heads: int,
    kv_heads: int,
    global_rows: int,
    head_dim: int,
    start_values: tuple[int, int, int],
    end_values: tuple[int, int, int],
    max_delta_rows: int,
):
    generator = torch.Generator(device="cuda").manual_seed(47)
    batch_size, block_size, num_blocks = 3, 16, 64
    query = torch.randn(
        query_heads,
        global_rows,
        head_dim,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    cache = torch.randn(
        num_blocks,
        kv_heads,
        block_size,
        2 * head_dim,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    block_table = torch.stack(
        tuple(
            torch.arange(request * 4, request * 4 + 4, device="cuda")
            for request in range(batch_size)
        )
    ).int()
    starts = torch.tensor(start_values, dtype=torch.int32, device="cuda")
    ends = torch.tensor(end_values, dtype=torch.int32, device="cuda")
    request_slots = torch.tensor([1, 3, 4], dtype=torch.long, device="cuda")
    batch_indices = torch.arange(batch_size, dtype=torch.long, device="cuda")
    state_output = torch.empty(
        6,
        query_heads,
        global_rows,
        head_dim,
        dtype=torch.float32,
        device="cuda",
    )
    state_lse = torch.empty(
        6, query_heads, global_rows, dtype=torch.float32, device="cuda"
    )
    expected = []
    for request, (start, end, slot) in enumerate(
        zip(starts.tolist(), ends.tolist(), request_slots.tolist())
    ):
        prefix_positions = torch.arange(start, device="cuda")[None]
        keys, values = _gather_paged_rows(
            cache, block_table[request : request + 1], prefix_positions
        )
        base_output, base_lse = normalized_gqa_attention(query, keys[0], values[0])
        state_output[slot].copy_(base_output)
        state_lse[slot].copy_(base_lse)

        full_positions = torch.arange(end, device="cuda")[None]
        full_keys, full_values = _gather_paged_rows(
            cache, block_table[request : request + 1], full_positions
        )
        expected.append(normalized_gqa_attention(query, full_keys[0], full_values[0]))

    update_longspark_global_delta(
        query=query,
        cache=cache,
        block_table=block_table,
        batch_indices=batch_indices,
        request_slots=request_slots,
        starts=starts,
        ends=ends,
        state_output=state_output,
        state_lse=state_lse,
        max_delta_rows=max_delta_rows,
    )

    for slot, (expected_output, expected_lse) in zip(request_slots, expected):
        torch.testing.assert_close(
            state_output[slot], expected_output, rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(state_lse[slot], expected_lse, rtol=2e-2, atol=2e-2)
