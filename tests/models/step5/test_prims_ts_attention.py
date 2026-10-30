# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 routing, cache scales, and request-safe PrimTS adapter bindings."""

import pytest
import torch

from vllm.models.step5.nvidia.prims_ts import (
    Step5PrimsTSSparse,
    uses_prims_ts_attention,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata


@pytest.fixture
def full_precision_matmul():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _metadata(capacity, live, query_len, device="cpu"):
    offsets = torch.arange(capacity + 1, device=device, dtype=torch.int32)
    offsets *= query_len
    offsets[live + 1 :] = live * query_len
    lengths = torch.full((capacity,), 64, device=device, dtype=torch.int32)
    lengths[live:] = 0
    slots = torch.arange(capacity * query_len, device=device, dtype=torch.int64)
    slots[live * query_len :] = -1
    metadata = FlashAttentionMetadata(
        num_actual_tokens=capacity * query_len,
        max_query_len=query_len,
        query_start_loc=offsets,
        max_seq_len=64,
        seq_lens=lengths,
        block_table=torch.zeros((capacity, 2), device=device, dtype=torch.int32),
        slot_mapping=slots,
    )
    metadata.num_actual_reqs = capacity
    return metadata


@pytest.mark.parametrize(
    "capacity,live,query_len,spec_tokens,kv_cache_dtype,expected",
    [
        (1, 1, 64, 3, "auto", True),
        (31, 31, 4, 3, "auto", False),
        (32, 31, 4, 3, "auto", True),
        (32, 32, 4, 3, "auto", True),
        (64, 64, 1, 3, "auto", False),
        (32, 32, 2, 3, "auto", False),
        (1, 1, 1, 0, "fp8", True),
        (1, 1, 4, 3, "fp8_e4m3", True),
    ],
)
def test_prims_ts_routes_by_cache_dtype_and_padded_batch_size(
    capacity, live, query_len, spec_tokens, kv_cache_dtype, expected
):
    assert (
        uses_prims_ts_attention(
            _metadata(capacity, live, query_len), spec_tokens, kv_cache_dtype
        )
        == expected
    )


def _require_cuda():
    if not current_platform.is_cuda() or not torch.cuda.is_available():
        pytest.skip("Step5 attention kernels require CUDA")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Step5 PrimTS adapters require SM100, SM103 or SM107")


def _reference(query, cache, table, owners, positions, *, ids):
    """FP32 reference over exact input values and independently selected blocks."""
    output = torch.zeros_like(query, dtype=torch.float32)
    groups, page_size = cache.shape[1:3]
    heads = query.shape[1] // groups
    offsets = torch.arange(8, device=query.device)
    for row, (owner, position) in enumerate(
        zip(owners.cpu().tolist(), positions.cpu().tolist())
    ):
        if position < 0:
            continue
        for group in range(groups):
            blocks = torch.cat((ids[row, group], ids.new_full((1,), position // 8)))
            tokens = (blocks[:, None] * 8 + offsets).flatten()
            tokens = tokens[(tokens >= 0) & (tokens <= position)]
            pages = table[owner, tokens // page_size].long()
            content = cache[pages, group, tokens % page_size].float()
            q = query[row, group * heads : (group + 1) * heads].float()
            probabilities = torch.softmax(
                q @ content[:, :192].T * (1.25 / 192**0.5), -1
            )
            output[row, group * heads : (group + 1) * heads] = (
                probabilities @ content[:, 192:] * 0.75
            )
    return output.to(torch.bfloat16)


def _inputs(dtype, context, query_lengths, kv_lengths):
    page_size, groups, heads, dim = 32, 2, 16, 192
    pages_per_request = (context + page_size - 1) // page_size
    pages = len(query_lengths) * pages_per_request
    query = torch.randn((sum(query_lengths), heads, dim), device="cuda").to(dtype)
    cache = torch.randn((pages, groups, page_size, 2 * dim), device="cuda").to(dtype)
    table = torch.randperm(pages, device="cuda", dtype=torch.int32).view(
        len(query_lengths), pages_per_request
    )
    owner_values: list[int] = []
    position_values: list[int] = []
    starts = [0]
    for request, (qlen, kvlen) in enumerate(zip(query_lengths, kv_lengths)):
        owner_values.extend([request] * qlen)
        position_values.extend(range(kvlen - qlen, kvlen) if kvlen > 0 else [-1] * qlen)
        starts.append(starts[-1] + qlen)
    owners = torch.tensor(owner_values, device="cuda", dtype=torch.int32)
    positions = torch.tensor(position_values, device="cuda", dtype=torch.int64)
    return query, cache, table, owners, positions, starts


def _selected_reference(scores, positions, topk):
    current = positions // 8
    regions = torch.arange(scores.shape[-1], device=scores.device)
    masked = scores.masked_fill(
        regions[None, None] >= current[:, None, None], -torch.inf
    )
    ids = masked.topk(min(topk, scores.shape[-1]), dim=-1).indices
    return torch.where(ids < current[:, None, None], ids, -1)


def _assert_accuracy(output, expected, dtype):
    assert torch.isfinite(output).all()
    error = torch.linalg.vector_norm(output.float() - expected.float())
    error /= torch.linalg.vector_norm(expected.float()).clamp_min(1e-6)
    assert error < (0.04 if dtype == torch.float8_e4m3fn else 0.01)


@pytest.mark.parametrize(
    "dtype,context,topk",
    [
        (torch.float8_e4m3fn, 64, 8),  # Distinct all-history/no-score path.
        (torch.bfloat16, 8192, 512),
        (torch.float8_e4m3fn, 8192, 512),
        (torch.bfloat16, 163840, 512),
        (torch.float8_e4m3fn, 163840, 512),
    ],
)
@pytest.mark.usefixtures("full_precision_matmul")
def test_prims_ts_packed_prefill_matches_reference(dtype, context, topk):
    _require_cuda()
    torch.manual_seed(11)
    # Long KV also exercises grouped prefill with partial request tails.
    if context == 64:
        query_lengths = (3, 5)
    elif context == 163840:
        query_lengths = (1025, 1029)
    else:
        query_lengths = (257, 263)
    query, cache, table, owners, positions, starts = _inputs(
        dtype, context, query_lengths, (context, context - 13)
    )
    regions = (context + 7) // 8
    adapter = Step5PrimsTSSparse(query.device, context, topk)
    plan = adapter.prepare_prefill_attention(
        (query.shape[0], 8, 192),
        dtype,
        query.device,
        cache.shape[2],
        positions,
        min(regions, topk) + 1,
        192**-0.5,
        request_offsets=starts,
        num_kv_groups=2,
    )
    metadata = adapter.prefill_metadata(
        owners, positions, regions, topk, out=plan.ids, history_only=True
    )
    output = torch.empty_like(query, dtype=torch.bfloat16)
    plan.bind(
        query,
        cache,
        table,
        output,
        owners=owners,
        positions=positions,
        qk_scale=1.25,
        v_scale=0.75,
    )
    scores = torch.randn((query.shape[0], 2, regions), device="cuda")
    expected_ids = _selected_reference(scores, positions, topk)
    for group in range(2):
        start = 0
        for chunk in metadata:
            end = start + chunk.current.numel()
            if regions > topk:
                adapter.select_regions(
                    scores[start:end, group].contiguous(),
                    chunk.current,
                    topk,
                    topk_lengths=chunk.topk_lengths,
                    sort=False,
                    out=plan.selection_output(start, end),
                    history_only=True,
                )
            start = end
        plan.finish(query[:, group * 8 : (group + 1) * 8], cache, output)
    expected = _reference(query, cache, table, owners, positions, ids=expected_ids)
    _assert_accuracy(output, expected, dtype)


@pytest.mark.parametrize(
    "dtype,batch,query_len",
    [
        (torch.float8_e4m3fn, 4, 1),
        (torch.bfloat16, 32, 4),
        (torch.float8_e4m3fn, 32, 4),
    ],
)
@pytest.mark.usefixtures("full_precision_matmul")
def test_prims_ts_long_decode_and_mtp_replay(dtype, batch, query_len):
    _require_cuda()
    torch.manual_seed(29)
    context, topk, groups = 163840, 512, 2
    kv_lengths = [context - request * 13 for request in range(batch)]
    kv_lengths[-1] = 0
    query, cache, table, owners, positions, _ = _inputs(
        dtype, context, [query_len] * batch, kv_lengths
    )
    tokens, heads, dim = query.shape
    scores = torch.randn(tokens, groups, context // 8, device="cuda")
    seq_lens = torch.tensor(kv_lengths, device="cuda", dtype=torch.int32)
    active = (positions >= 0).repeat_interleave(groups)
    decode_positions = positions.clamp_min(0).repeat_interleave(groups)
    current = decode_positions // 8
    topk_lengths = current.int()
    output = torch.zeros_like(query, dtype=torch.bfloat16)
    adapter = Step5PrimsTSSparse(query.device, context, topk)

    def run():
        adapter.select_and_run_decode_attention(
            query.view(tokens * groups, heads // groups, dim),
            cache,
            scores.view(tokens * groups, -1),
            current,
            topk,
            table,
            decode_positions,
            active,
            output.view(tokens * groups, heads // groups, dim),
            192**-0.5,
            seq_lens=seq_lens,
            topk_lengths=topk_lengths,
            query_len=query_len,
            k_scale=1.25,
            v_scale=0.75,
        )

    def check():
        ids = _selected_reference(scores, positions, topk)
        expected = _reference(query, cache, table, owners, positions, ids=ids)
        _assert_accuracy(output, expected, dtype)
        assert torch.count_nonzero(output[positions < 0]) == 0

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Replay the same capacity with a different inactive request and KV tails.
    kv_lengths[-1], kv_lengths[0] = context - 21, 0
    seq_lens.copy_(torch.tensor(kv_lengths, device="cuda", dtype=torch.int32))
    local_q = torch.arange(query_len, device="cuda")
    updated = (seq_lens[:, None] - query_len + local_q).flatten()
    positions.copy_(torch.where(seq_lens.repeat_interleave(query_len) > 0, updated, -1))
    active.copy_((positions >= 0).repeat_interleave(groups))
    decode_positions.copy_(positions.clamp_min(0).repeat_interleave(groups))
    current.copy_(decode_positions // 8)
    topk_lengths.copy_(current)
    output.zero_()
    graph.replay()
    torch.accelerator.synchronize()
    check()
