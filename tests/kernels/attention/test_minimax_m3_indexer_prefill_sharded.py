# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

CASES = [
    ([1], [0]),
    ([65, 3], [194, 127]),
    ([129], [127]),
    ([3, 1], [32768, 32640]),
    ([1027], [253]),
    ([8193], [127]),
]

requires_sm100 = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="SM100 family required",
)


def _make_case(query_lens, prefixes):
    from vllm.third_party.fmha_sm100.api import fmha_index_score_sharded_plan

    torch.manual_seed(12)
    lengths = [q + p for q, p in zip(query_lens, prefixes)]
    blocks = (max(lengths) + 127) // 128
    total_q = sum(query_lens)
    heads = 4
    table = torch.stack([torch.randperm(blocks, device="cuda") for _ in lengths]).int()
    for row in range(1, len(lengths)):
        table[row, 0] = table[0, 0]
    query = torch.randint(-2, 3, (total_q, heads, 128), device="cuda").to(
        torch.float8_e4m3fn
    )
    query[..., 0] = 16
    cache = torch.randint(-2, 3, (blocks, 128, 128), device="cuda").to(
        torch.float8_e4m3fn
    )
    plan = fmha_index_score_sharded_plan(
        torch.tensor(query_lens, dtype=torch.int32),
        torch.tensor(lengths, dtype=torch.int32),
        heads,
        qo_offset=torch.tensor(prefixes, dtype=torch.int32),
        device=query.device,
    )
    scores = torch.empty((total_q, heads, (blocks + 15) // 16 * 16), device="cuda")
    positions = torch.cat(
        [torch.arange(q, device="cuda") + p for q, p in zip(query_lens, prefixes)]
    )
    return lengths, table, query, cache, plan, scores, positions


def _reference(query, cache, table, rank, query_lens, prefixes, like):
    reference = torch.full_like(like, -float("inf"))
    start = 0
    for req, (qlen, prefix) in enumerate(zip(query_lens, prefixes)):
        positions = torch.arange(qlen, device="cuda") + prefix
        q = query[start : start + qlen].float()
        for block in range((prefix + qlen + 127) // 128):
            keys = cache[table[req, block], rank::2].float()
            dots = torch.einsum("qhd,kd->qhk", q, keys)
            key_positions = block * 128 + rank + 2 * torch.arange(64, device="cuda")
            dots.masked_fill_(
                key_positions[None, None, :] > positions[:, None, None],
                -float("inf"),
            )
            reference[start : start + qlen, :, block] = dots.amax(-1)
        start += qlen
    return reference


@requires_sm100
@pytest.mark.parametrize("query_lens,prefixes", CASES)
@torch.inference_mode()
def test_fmha_prefill_shard_scores(query_lens, prefixes):
    from vllm.third_party.fmha_sm100.api import fmha_index_score_sharded

    lengths, table, query, cache, plan, scores, _ = _make_case(query_lens, prefixes)
    for iteration in range(2):
        query = (-query.float()).to(query.dtype)
        # Future keys must not win before the last query, including page padding.
        for req, length in enumerate(lengths):
            page = table[req, (length - 1) // 128]
            tail = (length - 1) % 128
            cache[page, tail:] = 0
            cache[page, tail:, 0] = -256 if iteration == 0 else 256
        for rank in range(2):
            scores.fill_(float("nan"))
            fmha_index_score_sharded(
                query,
                cache[:, rank::2].contiguous(),
                table,
                plan,
                rank=rank,
                max_score=scores,
                sm_scale=1.0,
            )
            reference = _reference(
                query, cache, table, rank, query_lens, prefixes, scores
            )
            torch.testing.assert_close(scores, reference, rtol=0, atol=0)


@requires_sm100
@pytest.mark.parametrize("query_lens,prefixes", CASES)
@pytest.mark.parametrize("poison", [float("nan"), float("inf")])
@torch.inference_mode()
def test_fmha_prefill_shard_scores_uninitialized(query_lens, prefixes, poison):
    """Without the -inf fill, every query's causal page prefix is still written,
    so the prefill top-k is unchanged even over a poisoned score buffer."""
    from vllm.third_party.fmha_sm100.api import (
        fmha_index_score_sharded,
        sparse_topk_select_prefill,
    )

    _, table, query, cache, plan, scores, positions = _make_case(query_lens, prefixes)
    valid_pages = (positions // 128 + 1).to(torch.int32)
    heads = query.shape[1]
    total_q = query.shape[0]
    for rank in range(2):
        shard = cache[:, rank::2].contiguous()
        fmha_index_score_sharded(
            query, shard, table, plan, rank=rank, max_score=scores, sm_scale=1.0
        )
        expected = torch.empty((heads, total_q, 33), dtype=torch.int32, device="cuda")
        sparse_topk_select_prefill(
            scores, valid_pages, expected, force_begin_blocks=1, force_end_blocks=2
        )

        scores.fill_(poison)
        fmha_index_score_sharded(
            query,
            shard,
            table,
            plan,
            rank=rank,
            max_score=scores,
            sm_scale=1.0,
            initialize=False,
        )
        reference = _reference(query, cache, table, rank, query_lens, prefixes, scores)
        columns = torch.arange(scores.shape[-1], device="cuda")
        in_prefix = (columns[None, :] < valid_pages[:, None])[:, None, :].expand_as(
            scores
        )
        torch.testing.assert_close(
            scores[in_prefix], reference[in_prefix], rtol=0, atol=0
        )
        actual = torch.empty_like(expected)
        sparse_topk_select_prefill(
            scores, valid_pages, actual, force_begin_blocks=1, force_end_blocks=2
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
