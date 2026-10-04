# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 MSA indexer prefill: split-KV score plan and host-side metadata.

The planning tests run on CPU. The score test needs SM100 (fmha_sm100) and
checks that the split plan's scores are bitwise identical to the unsplit plan.
"""

import random

import numpy as np
import pytest
import torch

from vllm.models.minimax_m3.nvidia.indexer_msa import (
    PAGE_SIZE,
    _host_prefill_kv_lens,
    _plan_prefill_segments,
    _segment_page_index,
)
from vllm.platforms import current_platform

NUM_SMS = 148


def _check_layout(qo, kv, heads, num_sms, max_splits):
    seg_qo, seg_kv, seg_req, splits = _plan_prefill_segments(
        qo, kv, heads, num_sms, max_splits
    )
    if splits == 1:
        assert (seg_qo, seg_kv, seg_req) == (qo, kv, list(range(len(qo))))
        return splits
    assert 1 < splits <= max_splits
    assert max(seg_qo) <= 128
    assert not (heads > 1 and max(seg_qo) <= 32)
    # Segments partition each request's queries in order, and each segment's
    # bottom-right causal window ends where the unsplit row's window ends.
    assert seg_req == sorted(seg_req)
    for req, (q, k) in enumerate(zip(qo, kv)):
        idx = [i for i, r in enumerate(seg_req) if r == req]
        assert sum(seg_qo[i] for i in idx) == q
        end = 0
        for i in idx:
            end += seg_qo[i]
            assert seg_kv[i] == k - q + end
    # Every split gets at least one 256-key iteration.
    iters = (np.asarray(seg_kv) + 255) // 256
    pieces = np.minimum(splits, np.maximum(1, iters // 4))
    step = (iters + pieces - 1) // pieces
    assert np.array_equal((iters + step - 1) // step, pieces)
    return splits


@pytest.mark.parametrize(
    ("qo", "kv", "heads", "max_splits", "expect_split"),
    [
        ([172], [65836], 1, 32, True),  # short chunk, deep context
        ([16384], [48677], 1, 32, True),  # 16k chunk: 128 segments
        ([300, 200], [20300, 9000], 2, 32, True),
        ([172], [65836], 1, 1, False),  # disabled
        ([172], [1900], 1, 32, False),  # short context
        ([20, 30], [65536, 40000], 4, 32, False),  # pack-GQA shape
        ([4096] * 40, [65536] * 40, 1, 32, False),  # already fills the GPU
    ],
)
def test_plan_prefill_segments_cases(qo, kv, heads, max_splits, expect_split):
    splits = _check_layout(qo, kv, heads, NUM_SMS, max_splits)
    assert (splits > 1) == expect_split


def test_plan_prefill_segments_random():
    rng = random.Random(0)
    num_split = 0
    for _ in range(3000):
        n = rng.randint(1, 6)
        qo = [
            rng.choice([rng.randint(1, 300), rng.randint(1, 16384)]) for _ in range(n)
        ]
        kv = [q + rng.randint(0, 200_000) for q in qo]
        heads = rng.choice([1, 2, 4])
        num_split += _check_layout(qo, kv, heads, rng.choice([132, 148, 160]), 32) > 1
    assert num_split > 0


@pytest.mark.parametrize("split", [False, True])
def test_segment_page_index_matches_mask(split):
    qo, kv = [172, 300], [65836, 20300]
    seg_qo, seg_kv, seg_req, splits = _plan_prefill_segments(
        qo, kv, 1, NUM_SMS, 32 if split else 1
    )
    assert (splits > 1) == split
    width = (max(kv) + PAGE_SIZE - 1) // PAGE_SIZE + 3
    block_table = torch.randperm(len(qo) * width).reshape(len(qo), width)
    rows, cols = _segment_page_index(seg_req, seg_kv)
    actual = block_table[torch.from_numpy(rows), torch.from_numpy(cols)]
    # Reference: the boolean-mask gather, one segment at a time.
    expected = torch.cat(
        [
            block_table[r, : (k + PAGE_SIZE - 1) // PAGE_SIZE]
            for r, k in zip(seg_req, seg_kv)
        ]
    )
    assert torch.equal(actual, expected)


def test_host_prefill_kv_lens():
    upper = torch.tensor([10, 11, 4000, 9000], dtype=torch.int32)
    qo = torch.tensor([300, 500], dtype=torch.int32)
    # Prefill rows [2, 4): exact host values.
    out = _host_prefill_kv_lens(upper, qo, 2, 4, decode_threshold=4)
    assert out is not None and out.tolist() == [4000, 9000]
    # No host data, a short (possibly spec-decode) row, or an inconsistent
    # value: fall back to the device lengths.
    assert _host_prefill_kv_lens(None, qo, 2, 4, 4) is None
    short = torch.tensor([300, 3], dtype=torch.int32)
    assert _host_prefill_kv_lens(upper, short, 2, 4, 4) is None
    assert _host_prefill_kv_lens(upper, torch.tensor([300, 9001]), 2, 4, 4) is None


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="fmha_sm100 indexer requires SM100 (Blackwell).",
)
@pytest.mark.parametrize("index_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    ("qo", "kv", "heads"),
    [
        ([172], [65836], 1),
        ([1024], [33000], 1),
        ([300, 200], [20300, 9000], 2),
        ([96, 64], [12000, 2720], 4),
    ],
)
def test_split_prefill_score_bitwise(qo, kv, heads, index_dtype):
    from vllm.third_party.fmha_sm100.api import _fmha_sm100, _fmha_sm100_plan

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    head_dim = 128
    pages = [(k + PAGE_SIZE - 1) // PAGE_SIZE for k in kv]
    width = max(pages)
    num_pages = len(kv) * width
    block_table = torch.randperm(num_pages, device=device, dtype=torch.int32)
    block_table = block_table.reshape(len(kv), width)
    k_cache = torch.randn(num_pages, 1, PAGE_SIZE, head_dim, device=device)
    k_cache = k_cache.to(index_dtype)
    q = torch.randn(sum(qo), heads, head_dim, device=device).to(index_dtype)
    max_k_tiles = (width + 127) // 128 * 128

    def score(max_splits):
        seg_qo, seg_kv, seg_req, splits = _plan_prefill_segments(
            qo, kv, heads, num_sms, max_splits
        )
        seg_qo_t = torch.tensor(seg_qo, dtype=torch.int32)
        seg_kv_t = torch.tensor(seg_kv, dtype=torch.int32)
        plan = _fmha_sm100_plan(
            seg_qo_t,
            seg_kv_t,
            heads,
            num_kv_heads=1,
            qo_offset=seg_kv_t - seg_qo_t,
            page_size=PAGE_SIZE,
            output_maxscore=True,
            causal=True,
            num_kv_splits=splits,
        )
        plan["max_k_tiles"] = max_k_tiles
        rows, cols = _segment_page_index(seg_req, seg_kv)
        page_table = block_table[
            torch.from_numpy(rows).to(device), torch.from_numpy(cols).to(device)
        ]
        out = torch.full((sum(qo), heads, max_k_tiles), float("-inf"), device=device)
        _fmha_sm100(
            q,
            k_cache,
            k_cache,
            plan,
            kv_indices=page_table,
            output_o=False,
            output_maxscore=True,
            sm_scale=head_dim**-0.5,
            max_score=out,
        )
        return out, splits

    ref, ref_splits = score(1)
    out, splits = score(32)
    assert ref_splits == 1 and splits > 1
    assert torch.equal(out, ref)
