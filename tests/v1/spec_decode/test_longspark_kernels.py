# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch.nn import functional as F

from vllm.triton_utils import triton
from vllm.v1.worker.gpu.spec_decode.longspark.cudagraph import (
    _prepare_graph_inputs_kernel,
)
from vllm.v1.worker.gpu.spec_decode.longspark.kernels import (
    fused_add_argmax,
    fused_markov_argmax,
    fused_markov_refined_argmax,
    longspark_attention,
    longspark_paged_attention,
    prepare_longspark_shifted_input,
    update_longspark_global_delta,
    update_longspark_global_delta_batched,
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


def test_longspark_shifted_input_fusion_matches_reference():
    generator = torch.Generator(device="cuda").manual_seed(29)
    batch_size, block_size, hidden_size, vocab_size = 3, 7, 96, 10_037
    mask_token_id = vocab_size - 1
    token_ids = torch.randint(
        vocab_size - 1,
        (batch_size,),
        generator=generator,
        dtype=torch.int64,
        device="cuda",
    )
    latest = torch.randn(
        batch_size,
        hidden_size,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    embedding = torch.randn(
        vocab_size,
        hidden_size,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    position = torch.randn(
        block_size,
        hidden_size,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )

    block_token_ids = torch.full(
        (batch_size, block_size),
        mask_token_id,
        dtype=torch.long,
        device="cuda",
    )
    block_token_ids[:, 0] = token_ids
    expected_tokens = embedding[block_token_ids] + position[None]
    expected = torch.cat(
        (expected_tokens, latest[:, None].expand(-1, block_size, -1)),
        dim=-1,
    )
    actual = prepare_longspark_shifted_input(
        token_ids,
        latest,
        embedding,
        position,
        mask_token_id,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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


@pytest.mark.parametrize(
    ("batch_size", "rank"), [(1, 256), (4, 256), (8, 256), (16, 256), (17, 96)]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_longspark_fused_markov_argmax_matches_bf16_projection(
    batch_size: int, rank: int, reverse: bool
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
    token_ids = torch.randint(
        vocab_size,
        (batch_size,),
        generator=generator,
        dtype=torch.int64,
        device="cuda",
    )
    embedding_weight = torch.randn(
        vocab_size,
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
    chunks = (vocab_size + 31) // 32
    partial_values = torch.empty(batch_size, chunks, dtype=torch.float32, device="cuda")
    partial_ids = torch.empty(batch_size, chunks, dtype=torch.int32, device="cuda")
    output_storage = torch.empty(batch_size, 2, dtype=torch.int64, device="cuda")
    output = output_storage[:, 1]
    actual = fused_markov_argmax(
        base,
        token_ids,
        embedding_weight,
        weight,
        reverse=reverse,
        partial_values=partial_values,
        partial_ids=partial_ids,
        output=output,
    )
    embedding = F.embedding(token_ids, embedding_weight)
    bias = F.linear(embedding, weight).to(torch.bfloat16)
    expected = (base.float() + bias.float()).argmax(dim=-1)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
@pytest.mark.parametrize("reverse", [False, True])
def test_longspark_refined_markov_argmax_matches_candidate_reference(
    batch_size: int, reverse: bool
):
    generator = torch.Generator(device="cuda").manual_seed(43)
    vocab_size, candidate_rank, exact_rank, top_k = 10_037, 64, 256, 32
    base = torch.randn(
        batch_size,
        vocab_size,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    token_ids = torch.randint(
        vocab_size,
        (batch_size,),
        generator=generator,
        dtype=torch.int64,
        device="cuda",
    )
    candidate_embedding = torch.randn(
        vocab_size,
        candidate_rank,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    candidate_weight = torch.randn_like(
        candidate_embedding, generator=generator
    )
    exact_embedding = torch.randn(
        vocab_size,
        exact_rank,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    exact_weight = torch.randn_like(exact_embedding, generator=generator)

    block_vocab = 64 if batch_size >= 8 else 32
    chunks = triton.cdiv(vocab_size, block_vocab)
    num_tiles = triton.cdiv(chunks, 512)
    partial_values = torch.empty(
        batch_size, chunks, dtype=torch.float32, device="cuda"
    )
    partial_ids = torch.empty(
        batch_size, chunks, dtype=torch.int32, device="cuda"
    )
    tile_values = torch.empty(
        batch_size, num_tiles, top_k, dtype=torch.float32, device="cuda"
    )
    tile_ids = torch.empty(
        batch_size, num_tiles, top_k, dtype=torch.int32, device="cuda"
    )
    candidate_ids = torch.empty(
        batch_size, top_k, dtype=torch.int32, device="cuda"
    )
    output_storage = torch.empty(
        batch_size, 2, dtype=torch.int64, device="cuda"
    )
    actual = fused_markov_refined_argmax(
        base,
        token_ids,
        candidate_embedding,
        candidate_weight,
        exact_embedding,
        exact_weight,
        reverse=reverse,
        partial_values=partial_values,
        partial_ids=partial_ids,
        tile_values=tile_values,
        tile_ids=tile_ids,
        candidate_ids=candidate_ids,
        output=output_storage[:, 1],
        top_k=top_k,
    )

    approximate_embedding = F.embedding(token_ids, candidate_embedding)
    approximate_bias = F.linear(
        approximate_embedding, candidate_weight
    ).to(torch.bfloat16)
    approximate_scores = base.float() + approximate_bias.float()
    padded_vocab = chunks * block_vocab
    if padded_vocab > vocab_size:
        approximate_scores = F.pad(
            approximate_scores,
            (0, padded_vocab - vocab_size),
            value=-float("inf"),
        )
    chunk_scores, chunk_offsets = approximate_scores.view(
        batch_size, chunks, block_vocab
    ).max(dim=-1)
    chunk_ids = (
        torch.arange(chunks, device="cuda") * block_vocab
    )[None] + chunk_offsets
    top_chunks = chunk_scores.topk(top_k, dim=-1).indices
    reference_candidates = chunk_ids.gather(1, top_chunks)
    exact_queries = F.embedding(token_ids, exact_embedding).float()
    exact_candidates = exact_weight[reference_candidates].float()
    exact_bias = torch.einsum("br,bkr->bk", exact_queries, exact_candidates).to(
        torch.bfloat16
    )
    exact_scores = base.gather(1, reference_candidates).float() + exact_bias.float()
    expected = reference_candidates.gather(
        1, exact_scores.argmax(dim=-1, keepdim=True)
    ).squeeze(1)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(("num_reqs", "captured_num_reqs"), [(3, 4), (4, 4)])
def test_longspark_prepare_graph_inputs(num_reqs: int, captured_num_reqs: int):
    selected_layers, hidden_size, block_size = 2, 8, 5
    selected_row_size = selected_layers * hidden_size
    query_lengths = torch.arange(2, 2 + num_reqs, dtype=torch.int64, device="cuda")
    query_start_loc = torch.cat(
        (torch.zeros(1, dtype=torch.int64, device="cuda"), query_lengths.cumsum(0))
    )
    num_rejected = torch.arange(num_reqs, dtype=torch.int64, device="cuda") % 2
    source_rows = int(query_start_loc[-1].item())
    hidden_states = [
        torch.arange(
            layer * source_rows * hidden_size,
            (layer + 1) * source_rows * hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        ).view(source_rows, hidden_size)
        for layer in range(selected_layers)
    ]
    selected_hidden_ptrs = torch.tensor(
        [state.data_ptr() for state in hidden_states],
        dtype=torch.uint64,
        device="cuda",
    )
    selected_rows = query_start_loc[1:] - num_rejected - 1
    selected_hidden = torch.stack(
        [state.index_select(0, selected_rows) for state in hidden_states], dim=1
    )
    selected_hidden_out = torch.full(
        (captured_num_reqs, selected_layers, hidden_size),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    num_sampled = torch.arange(num_reqs, dtype=torch.int32, device="cuda") % 2
    current_positions = torch.arange(
        100, 100 + num_reqs, dtype=torch.int64, device="cuda"
    )
    request_slots = torch.arange(20, 20 + num_reqs, dtype=torch.int64, device="cuda")
    last_sampled = torch.arange(100, 228, dtype=torch.int64, device="cuda")[:, None]
    next_prefill_tokens = torch.arange(300, 428, dtype=torch.int64, device="cuda")
    anchors = torch.where(
        num_sampled > 0,
        last_sampled[request_slots, 0],
        next_prefill_tokens[request_slots],
    )
    starts = torch.arange(30, 30 + num_reqs, dtype=torch.int32, device="cuda")
    current_positions = starts + 3
    state_committed_lens = torch.full(
        (128,), -1, dtype=torch.int32, device="cuda"
    )
    state_committed_lens[request_slots] = starts
    anchors_out = torch.full((captured_num_reqs,), -1, dtype=torch.int64, device="cuda")
    current_positions_out = torch.full_like(anchors_out, -1)
    position_ids_out = torch.full(
        (captured_num_reqs, block_size), -1, dtype=torch.int64, device="cuda"
    )
    request_slots_out = torch.full_like(anchors_out, -1)
    starts_out = torch.full((captured_num_reqs,), -1, dtype=torch.int32, device="cuda")
    ends_out = torch.full_like(starts_out, -1)
    padding_request_slot = 99
    copy_block = 256
    copy_elements = captured_num_reqs * max(selected_row_size, block_size)

    _prepare_graph_inputs_kernel[(triton.cdiv(copy_elements, copy_block),)](
        selected_hidden_ptrs,
        query_start_loc,
        num_rejected,
        selected_hidden_out,
        num_sampled,
        last_sampled,
        next_prefill_tokens,
        anchors_out,
        current_positions,
        current_positions_out,
        position_ids_out,
        request_slots,
        request_slots_out,
        state_committed_lens,
        starts_out,
        ends_out,
        num_reqs,
        captured_num_reqs,
        padding_request_slot,
        current_positions.stride(0),
        request_slots.stride(0),
        state_committed_lens.stride(0),
        query_start_loc.stride(0),
        num_rejected.stride(0),
        num_sampled.stride(0),
        last_sampled.stride(0),
        next_prefill_tokens.stride(0),
        SELECTED_LAYERS=selected_layers,
        HIDDEN_SIZE=hidden_size,
        BLOCK_SIZE=block_size,
        COPY_BLOCK=copy_block,
        POSITION_BLOCK=triton.next_power_of_2(captured_num_reqs * block_size),
        REQUEST_BLOCK=triton.next_power_of_2(captured_num_reqs),
        num_warps=4,
    )

    torch.testing.assert_close(selected_hidden_out[:num_reqs], selected_hidden)
    torch.testing.assert_close(anchors_out[:num_reqs], anchors)
    torch.testing.assert_close(
        current_positions_out[:num_reqs], current_positions.long()
    )
    torch.testing.assert_close(request_slots_out[:num_reqs], request_slots)
    torch.testing.assert_close(starts_out[:num_reqs], starts)
    torch.testing.assert_close(ends_out[:num_reqs], current_positions)
    torch.testing.assert_close(
        state_committed_lens[request_slots], current_positions
    )
    expected_positions = current_positions.long()[:, None] + torch.arange(
        block_size, dtype=torch.int64, device="cuda"
    )
    torch.testing.assert_close(position_ids_out[:num_reqs], expected_positions)

    if captured_num_reqs > num_reqs:
        assert torch.count_nonzero(selected_hidden_out[num_reqs:]) == 0
        assert torch.count_nonzero(anchors_out[num_reqs:]) == 0
        assert torch.count_nonzero(position_ids_out[num_reqs:]) == 0
        assert torch.all(current_positions_out[num_reqs:] == 1)
        assert torch.all(request_slots_out[num_reqs:] == padding_request_slot)
        assert torch.count_nonzero(starts_out[num_reqs:]) == 0
        assert torch.all(ends_out[num_reqs:] == 1)


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


def test_longspark_batched_global_delta_matches_per_layer_kernels():
    generator = torch.Generator(device="cuda").manual_seed(53)
    layers, batch_size, state_slots = 3, 3, 6
    query_heads, kv_heads, global_rows, head_dim = 8, 2, 4, 64
    block_size, num_blocks, max_delta_rows = 16, 32, 4
    queries = torch.randn(
        layers,
        query_heads,
        global_rows,
        head_dim,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    caches = tuple(
        torch.randn(
            num_blocks,
            kv_heads,
            block_size,
            2 * head_dim,
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        for _ in range(layers)
    )
    block_tables = tuple(
        torch.stack(
            tuple(
                torch.arange(request * 4, request * 4 + 4, device="cuda")
                for request in range(batch_size)
            )
        ).int()
        for _ in range(layers)
    )
    batch_indices = torch.arange(batch_size, dtype=torch.long, device="cuda")
    request_slots = torch.tensor([1, 3, 4], dtype=torch.long, device="cuda")
    starts = torch.tensor([20, 33, 48], dtype=torch.int32, device="cuda")
    ends = torch.tensor([24, 35, 49], dtype=torch.int32, device="cuda")
    initial_output = torch.randn(
        state_slots,
        layers,
        query_heads,
        global_rows,
        head_dim,
        generator=generator,
        dtype=torch.float32,
        device="cuda",
    )
    initial_lse = torch.randn(
        state_slots,
        layers,
        query_heads,
        global_rows,
        generator=generator,
        dtype=torch.float32,
        device="cuda",
    )
    expected_output = initial_output.clone()
    expected_lse = initial_lse.clone()
    for layer in range(layers):
        update_longspark_global_delta(
            query=queries[layer],
            cache=caches[layer],
            block_table=block_tables[layer],
            batch_indices=batch_indices,
            request_slots=request_slots,
            starts=starts,
            ends=ends,
            state_output=expected_output[:, layer],
            state_lse=expected_lse[:, layer],
            max_delta_rows=max_delta_rows,
        )

    actual_output = initial_output.clone()
    actual_lse = initial_lse.clone()
    batch_output = torch.empty(
        batch_size,
        layers,
        query_heads,
        global_rows,
        head_dim,
        dtype=torch.float32,
        device="cuda",
    )
    cache_ptrs = torch.tensor(
        [cache.data_ptr() for cache in caches], dtype=torch.uint64, device="cuda"
    )
    table_ptrs = torch.tensor(
        [table.data_ptr() for table in block_tables],
        dtype=torch.uint64,
        device="cuda",
    )
    update_longspark_global_delta_batched(
        queries=queries,
        caches=caches,
        cache_ptrs=cache_ptrs,
        block_tables=block_tables,
        block_table_ptrs=table_ptrs,
        batch_indices=batch_indices,
        request_slots=request_slots,
        starts=starts,
        ends=ends,
        state_output=actual_output,
        state_lse=actual_lse,
        batch_output=batch_output,
        max_delta_rows=max_delta_rows,
    )

    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=0, atol=0)
    torch.testing.assert_close(
        batch_output, expected_output.index_select(0, request_slots), rtol=0, atol=0
    )
