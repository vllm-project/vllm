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
        ([1], [127]),
        ([1], [128]),
        ([1], [256]),
        ([65, 3], [194, 127]),
        ([129], [127]),
        ([3, 1], [32768, 32640]),
        ([1027], [253]),
        ([8193], [127]),
        ([0, 65, 0, 3], [127, 194, 0, 127]),
    ],
)
@pytest.mark.parametrize("page_size", [128, 256])
@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("initialize", [True, False])
@pytest.mark.parametrize("device_plan", [False, True])
@torch.inference_mode()
def test_fmha_prefill_shard_scores(
    query_lens, prefixes, page_size, tp, initialize, device_plan
):
    from vllm.third_party.fmha_sm100.api import (
        fmha_index_score_sharded,
        fmha_index_score_sharded_plan,
        fmha_index_score_sharded_plan_device,
        sparse_topk_select_prefill,
    )

    from vllm.models.minimax_m3.nvidia.ops.index_prefill import prepare_prefill_window
    from vllm.utils.platform_utils import num_compute_units

    torch.manual_seed(12)
    lengths = [q + p for q, p in zip(query_lens, prefixes)]
    blocks = (max(lengths) + 127) // 128
    pages = (max(lengths) + page_size - 1) // page_size
    total_q = sum(query_lens)
    heads = 4
    table = torch.stack([torch.randperm(pages, device="cuda") for _ in lengths]).int()
    for row in range(1, len(lengths)):
        table[row, 0] = table[0, 0]
    query = torch.randint(-2, 3, (total_q, heads, 128), device="cuda").to(
        torch.float8_e4m3fn
    )
    query[..., 0] = 16
    cache = torch.randint(-2, 3, (pages, page_size, 128), device="cuda").to(
        torch.float8_e4m3fn
    )
    plan = fmha_index_score_sharded_plan(
        torch.tensor(query_lens, dtype=torch.int32),
        torch.tensor(lengths, dtype=torch.int32),
        heads,
        qo_offset=torch.tensor(prefixes, dtype=torch.int32),
        device=query.device,
    )
    if device_plan:
        cu_q = (
            torch.tensor([0] + query_lens, device="cuda", dtype=torch.int32)
            .cumsum(0)
            .int()
        )
        seq_lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        metadata = torch.empty((5, len(lengths) + 1), device="cuda", dtype=torch.int32)
        prepare_prefill_window(cu_q, seq_lens, 0, total_q, metadata)
        plan = fmha_index_score_sharded_plan_device(
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
    score_cols = blocks if total_q == 1 else (blocks + 15) // 16 * 16
    storage = torch.empty((total_q, heads, score_cols + 1), device="cuda")
    scores = storage[..., :score_cols]
    for iteration in range(2):
        query = (-query.float()).to(query.dtype)
        # Future keys must not win before the last query, including page padding.
        for req, length in enumerate(lengths):
            if length == 0:
                continue
            page = table[req, (length - 1) // page_size]
            tail = (length - 1) % page_size
            cache[page, tail:] = 0
            cache[page, tail:, 0] = -256 if iteration == 0 else 256
        for rank in range(tp):
            rows = (
                torch.arange(rank, 128, tp, device="cuda")
                if page_size == 128
                else torch.arange(
                    rank * page_size // tp, (rank + 1) * page_size // tp, device="cuda"
                )
            )
            storage.fill_(float("nan"))
            fmha_index_score_sharded(
                query,
                cache[:, rows].contiguous(),
                table,
                plan,
                rank=rank,
                tp=tp,
                max_score=scores,
                sm_scale=1.0,
                initialize=initialize,
            )
            assert torch.isnan(storage[..., -1]).all()
            reference = torch.full_like(scores, -float("inf"))
            start = 0
            for req, (qlen, prefix) in enumerate(zip(query_lens, prefixes)):
                positions = torch.arange(qlen, device="cuda") + prefix
                q = query[start : start + qlen].float()
                for block in range((prefix + qlen + 127) // 128):
                    if page_size == 256 and block % 2 != rank * 2 // tp:
                        continue
                    page = block // (page_size // 128)
                    keys = cache[table[req, page], rows].float()
                    dots = torch.einsum("qhd,kd->qhk", q, keys)
                    key_positions = page * page_size + rows
                    dots.masked_fill_(
                        key_positions[None, None, :] > positions[:, None, None],
                        -float("inf"),
                    )
                    reference[start : start + qlen, :, block] = dots.amax(-1)
                start += qlen
            if initialize or page_size == 256:
                torch.testing.assert_close(scores, reference, rtol=0, atol=0)
            else:
                positions = torch.cat(
                    [
                        torch.arange(q, device="cuda") + p
                        for q, p in zip(query_lens, prefixes)
                    ]
                )
                valid = positions // 128 + 1
                columns = torch.arange(scores.shape[-1], device="cuda")
                mask = (columns[None, :] < valid[:, None])[:, None, :].expand_as(scores)
                torch.testing.assert_close(
                    scores[mask], reference[mask], rtol=0, atol=0
                )

            if not initialize:
                positions = torch.cat(
                    [
                        torch.arange(q, device="cuda") + p
                        for q, p in zip(query_lens, prefixes)
                    ]
                )
                valid = (positions // 128 + 1).to(torch.int32)
                actual = torch.empty(
                    (heads, total_q, 33), device="cuda", dtype=torch.int32
                )
                expected = torch.empty_like(actual)
                for values, carrier in ((scores, actual), (reference, expected)):
                    sparse_topk_select_prefill(
                        values.contiguous(),
                        valid,
                        carrier,
                        force_begin_blocks=1,
                        force_end_blocks=2,
                    )
                torch.testing.assert_close(
                    actual[..., :16].view(torch.float32).sort(-1).values,
                    expected[..., :16].view(torch.float32).sort(-1).values,
                    rtol=0,
                    atol=0,
                )
                ids = actual[..., 16:32]
                live = ids >= 0
                assert torch.all((ids < valid[None, :, None]) | ~live)
                assert torch.all((ids[..., 1:] > ids[..., :-1]) | ~live[..., 1:])
                assert torch.all((ids == 0).any(-1))
                for offset in (1, 2):
                    forced = (valid - offset).clamp_min(0)
                    assert torch.all((ids == forced[None, :, None]).any(-1))
                assert torch.all(actual[..., 32] == 1)


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
