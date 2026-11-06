# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

CASES = [
    ([1], [0]),
    ([1], [128]),
    ([1], [256]),
    ([65, 3], [194, 127]),
    ([129], [127]),
    ([3, 1], [32768, 32640]),
    ([1027], [253]),
    ([8193], [127]),
    ([0, 65, 0, 3], [127, 194, 0, 127]),
]

requires_sm100 = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="SM100 family required",
)


def _reference(query, cache, table, rank, query_lens, prefixes, like, tp=2):
    reference = torch.full_like(like, -float("inf"))
    start = 0
    for req, (qlen, prefix) in enumerate(zip(query_lens, prefixes)):
        positions = torch.arange(qlen, device="cuda") + prefix
        q = query[start : start + qlen].float()
        for block in range((prefix + qlen + 127) // 128):
            keys = cache[table[req, block], rank::tp].float()
            dots = torch.einsum("qhd,kd->qhk", q, keys)
            key_positions = (
                block * 128 + rank + tp * torch.arange(128 // tp, device="cuda")
            )
            dots.masked_fill_(
                key_positions[None, None, :] > positions[:, None, None],
                -float("inf"),
            )
            reference[start : start + qlen, :, block] = dots.amax(-1)
        start += qlen
    return reference


@requires_sm100
@pytest.mark.parametrize("query_lens,prefixes", CASES)
@pytest.mark.parametrize("tp", [2, 4])
@torch.inference_mode()
def test_indexer_prefill_device_plan_matches_reference(query_lens, prefixes, tp):
    from flashinfer.experimental.cake_msa_icp_indexer.cake_backend import (
        index_score_sharded,
        index_score_sharded_plan_device,
    )

    from vllm.models.minimax_m3.nvidia.ops.index_prefill import prepare_prefill_window
    from vllm.utils.platform_utils import num_compute_units

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
    cu_q = (
        torch.tensor([0] + query_lens, device="cuda", dtype=torch.int32).cumsum(0).int()
    )
    seq_lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    metadata = torch.empty((5, len(lengths) + 1), device="cuda", dtype=torch.int32)
    prepare_prefill_window(cu_q, seq_lens, 0, total_q, metadata)
    plan = index_score_sharded_plan_device(
        metadata[1, :-1],
        metadata[2, :-1],
        metadata[0],
        metadata[4],
        metadata[3, :-1],
        heads,
        total_q=total_q,
        max_query_len=max(query_lens),
        max_seq_len=max(lengths),
        packed_work_range=torch.empty(
            num_compute_units(0), device="cuda", dtype=torch.int64
        ),
        packed_work_info=torch.empty(
            heads * ((total_q + 127) // 128 + len(lengths) - 1),
            device="cuda",
            dtype=torch.int64,
        ),
        workspace=torch.empty(32 << 20, device="cuda", dtype=torch.uint8),
    )
    scores = torch.empty((total_q, heads, (blocks + 15) // 16 * 16), device="cuda")
    for iteration in range(2):
        query = (-query.float()).to(query.dtype)
        # Future keys must not win before the last query, including page padding.
        for req, length in enumerate(lengths):
            if length == 0:
                continue
            page = table[req, (length - 1) // 128]
            tail = (length - 1) % 128
            cache[page, tail:] = 0
            cache[page, tail:, 0] = -256 if iteration == 0 else 256
        for rank in range(tp):
            scores.fill_(float("nan"))
            index_score_sharded(
                query,
                cache[:, rank::tp].contiguous(),
                table,
                plan,
                rank=rank,
                tp=tp,
                max_score=scores,
                sm_scale=1.0,
            )
            reference = _reference(
                query, cache, table, rank, query_lens, prefixes, scores, tp
            )
            torch.testing.assert_close(scores, reference, rtol=0, atol=0)


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("page_size", [128, 256])
@torch.inference_mode()
def test_indexer_device_metadata_replay(tp, page_size):
    """Changing prefixes/windows must overwrite reused metadata and padded slots."""
    from vllm.models.minimax_m3.nvidia.ops.index_prefill import (
        prepare_indexer_metadata,
        prepare_prefill_window,
    )

    cu_q = torch.tensor([0, 4, 133, 133, 136], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([259, 512, 0, 1151], dtype=torch.int32, device="cuda")
    positions = torch.tensor([-1, 0, 127, 128, 511, 0], device="cuda")
    slots = torch.cat(
        (torch.arange(512, device="cuda"), torch.tensor([-1], device="cuda"))
    )
    slots[1] = 0  # A real token owned by a peer still needs causal page counts.
    slots[5] = -1
    context = torch.empty_like(seq_lens)
    valid = torch.empty_like(positions, dtype=torch.int32)
    output_slots = torch.empty_like(slots)
    window = torch.empty((5, 5), dtype=torch.int32, device="cuda")

    def run():
        prepare_indexer_metadata(
            cu_q,
            seq_lens,
            positions,
            slots,
            context,
            valid,
            output_slots,
            tp - 1,
            tp,
            page_size,
        )
        prepare_prefill_window(cu_q, seq_lens, 3, 135, window)

    run()
    torch.testing.assert_close(valid, valid.new_tensor([0, 1, 1, 2, 4, 0]))
    assert output_slots[1] == -1
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    seq_lens.add_(128)
    positions.add_(128)
    slots[4] = -1
    graph.replay()
    torch.testing.assert_close(context, seq_lens - cu_q.diff())
    torch.testing.assert_close(valid, valid.new_tensor([1, 2, 2, 3, 0, 0]))
    begin = (cu_q[:-1] - 3).clamp(0, 132)
    end = (cu_q[1:] - 3).clamp(0, 132)
    prefix = seq_lens - cu_q.diff() + (3 - cu_q[:-1]).clamp(min=0).minimum(cu_q.diff())
    torch.testing.assert_close(window[0], torch.cat((begin, begin.new_tensor([132]))))
    torch.testing.assert_close(window[1, :-1], end - begin)
    torch.testing.assert_close(window[2, :-1], prefix + end - begin)
    torch.testing.assert_close(window[3, :-1], prefix)
    expected_kv = torch.cat(
        (seq_lens.new_zeros(1), (prefix + end - begin).cumsum(0))
    ).int()
    torch.testing.assert_close(window[4], expected_kv)
    offset = slots % page_size
    rows = page_size // tp
    owner = offset // rows if page_size == 256 else offset % tp
    local = offset % rows if page_size == 256 else offset // tp
    expected = slots // page_size * rows + local
    expected[(slots < 0) | (owner != tp - 1)] = -1
    torch.testing.assert_close(output_slots, expected)
