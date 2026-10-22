# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="SM100 family required",
)
@pytest.mark.parametrize(
    "query_lens,prefixes",
    [
        ([1], [0]),
        ([65, 3], [194, 127]),
        ([129], [127]),
        ([3, 1], [32768, 32640]),
        ([1027], [253]),
        ([8193], [127]),
    ],
)
@torch.inference_mode()
def test_fmha_prefill_shard_scores(query_lens, prefixes):
    from vllm.third_party.fmha_sm100.api import (
        fmha_index_score_sharded,
        fmha_index_score_sharded_plan,
    )

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
    valid = torch.empty_like(scores, dtype=torch.uint8)
    for _ in range(2):
        query = (-query.float()).to(query.dtype)
        for rank in range(2):
            scores.fill_(float("nan"))
            valid.fill_(255)
            fmha_index_score_sharded(
                query,
                cache[:, rank::2].contiguous(),
                table,
                plan,
                rank=rank,
                max_score=scores,
                valid_score=valid,
                sm_scale=1.0,
            )
            reference = torch.full_like(scores, -float("inf"))
            start = 0
            for req, (qlen, prefix) in enumerate(zip(query_lens, prefixes)):
                positions = torch.arange(qlen, device="cuda") + prefix
                q = query[start : start + qlen].float()
                for block in range((prefix + qlen + 127) // 128):
                    keys = cache[table[req, block], rank::2].float()
                    dots = torch.einsum("qhd,kd->qhk", q, keys)
                    key_positions = (
                        block * 128 + rank + 2 * torch.arange(64, device="cuda")
                    )
                    dots.masked_fill_(
                        key_positions[None, None, :] > positions[:, None, None],
                        -float("inf"),
                    )
                    reference[start : start + qlen, :, block] = dots.amax(-1)
                start += qlen
            torch.testing.assert_close(scores, reference, rtol=0, atol=0)
            torch.testing.assert_close(valid, reference.isfinite().to(torch.uint8))
