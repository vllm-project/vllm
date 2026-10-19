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


def _require_cuda(sm103=False):
    if not current_platform.is_cuda() or not torch.cuda.is_available():
        pytest.skip("Step5 attention kernels require CUDA")
    if sm103 and torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Step5 PrimTS adapters require SM103")


def _reference(query, cache, table, owners, positions, *, ids):
    """FP32 reference over the exact cache views and selected token positions."""
    output = torch.zeros_like(query, dtype=torch.float32)
    groups, page_size = cache.shape[1:3]
    heads = query.shape[1] // groups
    for row in range(query.shape[0]):
        position = int(positions[row])
        if position < 0:
            continue
        for group in range(groups):
            blocks = torch.cat((ids[row, group], ids.new_tensor([position // 8])))
            tokens = blocks[:, None] * 8 + torch.arange(8, device=query.device)
            tokens = tokens.flatten()
            tokens = tokens[(tokens >= 0) & (tokens <= position)]
            pages = table[owners[row], tokens // page_size].long()
            content = cache[pages, group, tokens % page_size].float()
            q = query[row, group * heads : (group + 1) * heads].float()
            probabilities = torch.softmax(
                q @ content[:, :192].T * (1.25 / 192**0.5), -1
            )
            output[row, group * heads : (group + 1) * heads] = (
                probabilities @ content[:, 192:] * 0.75
            )
    return output.to(torch.bfloat16)


@pytest.mark.parametrize(
    "dtype,topk",
    [(torch.bfloat16, 2), (torch.float8_e4m3fn, 2), (torch.float8_e4m3fn, 8)],
)
def test_prims_ts_prefill_binds_groups_without_crossing_requests(dtype, topk):
    _require_cuda(sm103=True)
    torch.manual_seed(11)
    query = torch.randn((8, 16, 192), device="cuda").to(dtype)
    cache = torch.randn((4, 2, 32, 384), device="cuda").to(dtype)
    table = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)
    owners = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], device="cuda", dtype=torch.int64)
    positions = torch.tensor([61, 62, 63, 59, 60, 61, 62, 63], device="cuda")
    adapter = Step5PrimsTSSparse(query.device, 64, topk)
    plan = adapter.prepare_prefill_attention(
        (8, 8, 192),
        dtype,
        query.device,
        32,
        positions,
        8,
        192**-0.5,
        request_offsets=[0, 3, 8],
        num_kv_groups=2,
    )
    plan.fixup_positions(positions)
    metadata = adapter.prefill_metadata(
        owners, positions, 8, topk, prims_output=plan.metadata_output()
    )
    output = torch.empty_like(query, dtype=torch.bfloat16)
    plan.bind(query, cache, table, output, qk_scale=1.25, v_scale=0.75)
    scores = torch.randn((8, 2, 8), device="cuda")
    expected_ids = (
        torch.topk(scores[:, :, :7], topk, dim=-1).indices
        if topk < 8
        else torch.arange(7, device="cuda").expand(8, 2, 7)
    )
    for group in range(2):
        start = 0
        for chunk in metadata:
            end = start + chunk.current.numel()
            if topk < 8:
                adapter.select_regions(
                    scores[start:end, group].contiguous(),
                    chunk.current,
                    topk,
                    topk_lengths=chunk.topk_lengths,
                    sort=False,
                    prims_output=plan.selection_output(start, end),
                )
            start = end
        plan.finish(query[:, group * 8 : (group + 1) * 8], cache, output)
    expected = _reference(query, cache, table, owners, positions, ids=expected_ids)
    relative_error = torch.linalg.vector_norm(output.float() - expected.float())
    relative_error /= torch.linalg.vector_norm(expected.float())
    assert relative_error < (0.04 if dtype == torch.float8_e4m3fn else 0.01)
    assert plan.ids.dtype == torch.int32
    assert 3 in plan.offsets.tolist()
