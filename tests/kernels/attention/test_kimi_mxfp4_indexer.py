# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi MXFP4 cache isolation and graph-safe row-bounded selection."""

import pytest
import torch

from vllm.platforms import current_platform


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


# The MXFP4 indexer kernels use gfx950-only scaled MFMA instructions.
pytestmark = pytest.mark.skipif(not _on_gfx950(), reason="needs gfx950 (MI350/MI355)")

# TODO: drop once AITER's row-group FP4 MQA logits kernel accepts physical page
# strides. score_paged reads the index cache in place, whose pages pitch 4352
# bytes (payload + scales), but flydsl_pa_mqa_logits_fp4_rowgroup asserts
# contiguous pages in every AITER release so far (v0.1.29 included).
_needs_strided_rowgroup = pytest.mark.xfail(
    raises=AssertionError,
    strict=True,
    reason="AITER row-group FP4 MQA logits needs contiguous pages",
)


def test_cache_write_preserves_neighboring_packed_layers():
    from vllm.models.kimi_k3.amd.ops.fp4_indexer import store_keys

    stride, offset = 46592, 41984
    storage = torch.full((2, stride), 173, dtype=torch.uint8, device="cuda")
    cache = storage[:, offset : offset + 4352].view(2, 64, 68)
    keys = torch.full((3, 128), 6.0, dtype=torch.bfloat16, device="cuda")
    slots = torch.tensor([-1, 0, 127], dtype=torch.int64, device="cuda")
    store_keys(keys, cache, slots)
    expected = torch.full_like(storage, 173)
    for page, slot in ((0, 0), (1, 63)):
        for group in range(4):
            start = offset + group * 1024 + slot * 16
            expected[page, start : start + 16] = 0x77
            expected[page, offset + 4096 + group * 64 + slot % 16 * 4 + slot // 16] = (
                127
            )
    torch.testing.assert_close(storage, expected, rtol=0, atol=0)


@_needs_strided_rowgroup
@pytest.mark.parametrize("return_scores", [False, True])
def test_indexer_piecewise_replay_refreshes_batch_metadata(return_scores):
    """Replay the registered op across decode, mixed and prefill batches."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.config import CUDAGraphMode, VllmConfig
    from vllm.forward_context import set_forward_context
    from vllm.models.kimi_k3.amd import dsa  # noqa: F401
    from vllm.models.kimi_k3.amd.ops.fp4_indexer import store_keys
    from vllm.models.kimi_k3.amd.ops.fp4_indexer_prefill import prepare_prefill_metadata
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
        DeepseekV32IndexerMetadata,
        DeepseekV32IndexerPrefillChunkMetadata,
        DeepseekV32IndexerPrefillMetadata,
    )

    torch.manual_seed(11)
    rows, heads, width = 6, 32, 4096
    config = VllmConfig()
    q = torch.randn((rows, heads, 128), device="cuda", dtype=torch.bfloat16)
    keys = torch.randn((width, 128), device="cuda", dtype=torch.bfloat16)
    weights = torch.rand((rows, heads), device="cuda", dtype=torch.bfloat16)
    cache = torch.empty((width // 64, 64, 68), device="cuda", dtype=torch.uint8)
    store_keys(keys, cache, torch.arange(width, device="cuda"))
    initial_cache = cache.clone()
    out = torch.empty((rows, 2048), device="cuda", dtype=torch.int32)
    scores = torch.empty_like(out, dtype=torch.float32) if return_scores else None

    def metadata(num_decode, num_prefill, length):
        # Fresh tensors, Python counts and prefill plans on every batch. Keeping
        # only the capture-time pointers valid is insufficient for PIECEWISE.
        table = torch.arange(width // 64, device="cuda", dtype=torch.int32)[None]
        ends = torch.arange(
            length - num_decode + 1, length + 1, device="cuda", dtype=torch.int32
        )
        decode = (
            DeepSeekV32IndexerDecodeMetadata(
                block_table=table,
                seq_lens=ends,
                decode_lens=torch.tensor([num_decode], device="cuda"),
                requires_padding=False,
                schedule_metadata=torch.empty(0, device="cuda", dtype=torch.int32),
                query_start_loc=torch.tensor(
                    [0, num_decode], device="cuda", dtype=torch.int32
                ),
                write_max_decode_len=num_decode,
            )
            if num_decode
            else None
        )
        prefill = None
        if num_prefill:
            chunk = DeepseekV32IndexerPrefillChunkMetadata(
                block_table=table,
                cu_seqlen_ks=torch.zeros(num_prefill, device="cuda", dtype=torch.int32),
                cu_seqlen_ke=torch.arange(
                    length - num_prefill + 1,
                    length + 1,
                    device="cuda",
                    dtype=torch.int32,
                ),
                cu_seq_lens=torch.tensor([0, length], device="cuda", dtype=torch.int32),
                token_to_seq=torch.zeros(length, device="cuda", dtype=torch.int32),
                total_seq_lens=length,
                max_seqlen_k=length,
                token_start=num_decode,
                token_end=num_decode + num_prefill,
                num_reqs=1,
            )
            prefill = prepare_prefill_metadata(
                DeepseekV32IndexerPrefillMetadata([chunk]), 1, 0, 1
            )
        return DeepseekV32IndexerMetadata(
            seq_lens=torch.tensor([length], device="cuda", dtype=torch.int32),
            max_seq_len=length,
            slot_mapping=torch.arange(
                length - num_decode - num_prefill, length, device="cuda"
            ),
            num_decodes=int(num_decode > 0),
            num_decode_tokens=num_decode,
            num_prefills=int(num_prefill > 0),
            num_prefill_tokens=num_prefill,
            decode=decode,
            prefill=prefill,
            block_table=table,
        )

    batches = [
        metadata(*shape)
        for shape in ((6, 0, 3000), (2, 4, 2500), (0, 6, 128), (3, 0, 64), (0, 0, 64))
    ]

    def run():
        torch.ops.vllm.kimi_fp4_indexer(
            q, keys[:rows], weights, cache, out, "indexer", width, scores=scores
        )
        # A downstream graph segment must see the current eager op's writes.
        return out.clone()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for batch in batches:
            with set_forward_context({"indexer": batch}, config):
                run()
        capture = BreakableCUDAGraphCapture()
        with (
            set_forward_context(
                {"indexer": batches[0]},
                config,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            ),
            capture,
        ):
            captured_out = run()
        for batch in batches[1:] + batches[:1]:
            cache.copy_(initial_cache)
            if scores is not None:
                scores.fill_(-123)
            with set_forward_context({"indexer": batch}, config):
                expected = run()
            expected_cache = cache.clone()
            expected_scores = scores.clone() if scores is not None else None
            cache.copy_(initial_cache)
            out.fill_(-2)
            if scores is not None:
                scores.fill_(-123)
            with set_forward_context(
                {"indexer": batch},
                config,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            ):
                capture.replay()
            # Top-k returns an unordered selection; compare the selected sets.
            torch.testing.assert_close(
                captured_out.sort(dim=1).values,
                expected.sort(dim=1).values,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
            if scores is not None:
                assert expected_scores is not None
                torch.testing.assert_close(
                    scores.gather(1, captured_out.argsort(dim=1)),
                    expected_scores.gather(1, expected.argsort(dim=1)),
                    rtol=0,
                    atol=0,
                )
            count = batch.num_decode_tokens + batch.num_prefill_tokens
            assert bool((captured_out[count:] == -1).all())
            assert bool((captured_out[:count] < batch.max_seq_len).all())

        # FULL retains its existing contract: static Python geometry and stable
        # metadata buffers whose contents can change between launches.
        batch = batches[0]
        full = BreakableCUDAGraphCapture()
        with (
            set_forward_context(
                {"indexer": batch},
                config,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
            ),
            full,
        ):
            full_out = run()
        assert full.num_eager_breaks == 0
        batch.decode.seq_lens.fill_(64)
        with set_forward_context({"indexer": batch}, config):
            expected = run()
        out.fill_(-2)
        full.replay()
        torch.testing.assert_close(
            full_out.sort(dim=1).values,
            expected.sort(dim=1).values,
            rtol=0,
            atol=0,
        )
    torch.cuda.current_stream().wait_stream(stream)


@_needs_strided_rowgroup
@pytest.mark.parametrize("return_scores", [False, True])
def test_decode_graph_does_not_select_poisoned_logits_tail(return_scores):
    from vllm.models.kimi_k3.amd.dsa import _select
    from vllm.models.kimi_k3.amd.ops.fp4_indexer import score_paged, store_keys

    torch.manual_seed(7)
    width, rows, heads, k = 4096, 12, 32, 2048
    pages = 2 * width // 64
    backing = torch.empty((pages, 46592), dtype=torch.uint8, device="cuda")
    cache = backing[:, 41984:46336].view(pages, 64, 68)
    keys = torch.randn((pages * 64, 128), dtype=torch.bfloat16, device="cuda")
    store_keys(keys, cache, torch.arange(pages * 64, device="cuda"))
    q = torch.randn((rows, heads, 128), dtype=torch.bfloat16, device="cuda")
    weights = torch.rand((rows, heads), device="cuda") / heads
    table = torch.arange(pages, dtype=torch.int32, device="cuda").view(2, -1)
    starts = torch.tensor([0, 6, 12], dtype=torch.int32, device="cuda")
    ends = torch.tensor(
        [2501 + i for i in range(6)] * 2, dtype=torch.int32, device="cuda"
    )
    selected = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    scores = torch.empty((rows, k), device="cuda") if return_scores else None

    def run():
        logits = score_paged(q, cache, weights, table, starts, ends, 6)
        _select(logits, ends, selected, scores)
        return logits

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = run()
    for length in (3000, 64, 0, 2100):
        ends.fill_(length)
        logits.fill_(1e20)
        if scores is not None:
            scores.fill_(float("nan"))
        graph.replay()
        valid = torch.arange(width, device="cuda")[None, :] < ends[:, None]
        expected = torch.topk(logits.masked_fill(~valid, -float("inf")), k).values
        values = logits.gather(1, selected.clamp_min(0).long())
        values = values.masked_fill(selected < 0, -float("inf"))
        if scores is not None:
            torch.testing.assert_close(scores, values, rtol=0, atol=0)
        torch.testing.assert_close(
            values.sort(-1, descending=True).values, expected, rtol=0, atol=0
        )
        assert bool(((selected < 0) | (selected < ends[:, None])).all())


def test_prefill_reassembly_preserves_keys_across_request_boundaries():
    from vllm.models.kimi_k3.amd.ops.fp4_indexer import cache_views, store_keys
    from vllm.models.kimi_k3.amd.ops.fp4_indexer_prefill import stage_keys
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerPrefillChunkMetadata,
    )

    torch.manual_seed(7)
    lengths = (70, 129)
    keys = torch.randn((sum(lengths), 128), device="cuda", dtype=torch.bfloat16)
    table = torch.tensor([[3, 1, 0], [4, 0, 2]], device="cuda", dtype=torch.int32)
    storage = torch.full((5, 46592), 173, device="cuda", dtype=torch.uint8)
    cache = storage[:, 41984:46336].view(5, 64, 68)
    slots = torch.cat(
        [
            table[i, torch.arange(n, device="cuda") // 64].long() * 64
            + torch.arange(n, device="cuda") % 64
            for i, n in enumerate(lengths)
        ]
    )
    store_keys(keys, cache, slots)
    cu = torch.tensor([0, 70, 199], device="cuda", dtype=torch.int32)
    chunk = DeepseekV32IndexerPrefillChunkMetadata(
        block_table=table,
        cu_seqlen_ks=cu[:-1],
        cu_seqlen_ke=cu[1:],
        cu_seq_lens=cu,
        token_to_seq=torch.cat(
            [
                torch.full((n,), i, device="cuda", dtype=torch.int32)
                for i, n in enumerate(lengths)
            ]
        ),
        total_seq_lens=199,
        max_seqlen_k=129,
        token_start=0,
        token_end=2,
        num_reqs=2,
    )
    payload, scales = stage_keys(cache, chunk, world=1, interleave=1)
    reference = torch.zeros((4, 64, 68), device="cuda", dtype=torch.uint8)
    store_keys(keys, reference, torch.arange(199, device="cuda"))
    ref_payload, ref_scales = cache_views(reference)
    torch.testing.assert_close(payload, ref_payload, rtol=0, atol=0)
    torch.testing.assert_close(scales, ref_scales, rtol=0, atol=0)


def test_prefill_geometry_reuse_keeps_layer_values_and_request_bounds_fresh():
    from vllm.models.kimi_k3.amd.ops.fp4_indexer import store_keys
    from vllm.models.kimi_k3.amd.ops.fp4_indexer_prefill import (
        prefill_indexer,
        prepare_prefill_metadata,
    )
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerPrefillChunkMetadata,
        DeepseekV32IndexerPrefillMetadata,
    )

    torch.manual_seed(7)
    rows, heads = 6, 32
    for length in (4096, 3072):
        chunk = DeepseekV32IndexerPrefillChunkMetadata(
            block_table=torch.arange(length // 64, device="cuda", dtype=torch.int32)[
                None, :
            ],
            cu_seqlen_ks=torch.zeros(rows, device="cuda", dtype=torch.int32),
            cu_seqlen_ke=torch.arange(
                length - rows + 1, length + 1, device="cuda", dtype=torch.int32
            ),
            cu_seq_lens=torch.tensor([0, length], device="cuda", dtype=torch.int32),
            token_to_seq=torch.zeros(length, device="cuda", dtype=torch.int32),
            total_seq_lens=length,
            max_seqlen_k=length,
            token_start=0,
            token_end=rows,
            num_reqs=1,
        )
        plan = prepare_prefill_metadata(
            DeepseekV32IndexerPrefillMetadata([chunk]), 1, 0, 1
        ).plans[0]
        cache = torch.empty((length // 64, 64, 68), device="cuda", dtype=torch.uint8)
        for _ in range(2):
            keys = torch.randn((length, 128), device="cuda", dtype=torch.bfloat16)
            store_keys(keys, cache, torch.arange(length, device="cuda"))
            q = torch.randn((rows, heads, 128), device="cuda", dtype=torch.bfloat16)
            weights = torch.rand((rows, heads), device="cuda", dtype=torch.bfloat16)
            reused = torch.empty((rows, 2048), device="cuda", dtype=torch.int32)
            rebuilt = torch.empty_like(reused)
            prefill_indexer(q, weights, cache, reused, chunk, 1, 1, plan=plan)
            prefill_indexer(q, weights, cache, rebuilt, chunk, 1, 1)
            torch.testing.assert_close(reused, rebuilt, rtol=0, atol=0)
            assert bool((reused < chunk.cu_seqlen_ke[:, None]).all())


@pytest.mark.parametrize("world,interleave", [(1, 1), (8, 1), (8, 64)])
def test_prefill_workspace_covers_staged_keys(world, interleave):
    from types import SimpleNamespace

    from vllm.models.kimi_k3.amd.ops.fp4_indexer_prefill import (
        _local_key_cu,
        prefill_workspace_bytes,
    )

    max_keys, max_reqs = 1 << 20, 16
    run = world * interleave
    # Every request leaves one extra interleave run on rank 0.
    skewed = (max_keys // max_reqs - interleave) // run * run + interleave
    torch.manual_seed(world + interleave)
    uneven = torch.randint(1, max_keys // max_reqs, (max_reqs,))
    bound = prefill_workspace_bytes(max_keys, max_reqs, world, interleave)
    for lengths in (torch.full((max_reqs,), skewed), uneven, torch.tensor([max_keys])):
        lengths = lengths.to(torch.int32)
        total = int(lengths.sum())
        assert total <= max_keys
        cu = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
        local_cu = _local_key_cu(SimpleNamespace(cu_seq_lens=cu), world, interleave)
        local = int(local_cu[-1].max())
        gathered = world * local if world > 1 else 0
        staged = (total + 63) // 64 * 64
        assert 1024**3 + 68 * (local + gathered + staged) <= bound


def _layer_norm_affine(dtype):
    weight = (1 + 0.1 * torch.randn(128, device="cuda")).to(dtype)
    bias = (0.1 * torch.randn(128, device="cuda")).to(dtype)
    return weight, bias


@pytest.mark.parametrize("heads", [16, 32, 64])
@pytest.mark.parametrize(
    "num_queries,num_keys", [(0, 7), (1, 1), (40, 40), (5, 300), (0, 4096)]
)
def test_quantize_index_qk_matches_unfused(heads, num_queries, num_keys):
    from vllm.models.kimi_k3.amd.ops.fp4_indexer import (
        quantize_index_qk,
        quantize_queries,
        store_keys,
    )

    torch.manual_seed(heads + num_queries + num_keys)
    rows = max(num_queries, num_keys)
    stride, offset, pages = 46592, 41984, max(8, num_keys // 32)
    backing = torch.randint(0, 256, (pages, stride), dtype=torch.uint8).cuda()
    backing = torch.stack([backing, backing])
    caches = [b[:, offset : offset + 4352].view(pages, 64, 68) for b in backing]
    hidden = torch.randn((rows, 3808), device="cuda", dtype=torch.bfloat16)
    q = torch.randn((rows, heads, 128), device="cuda", dtype=torch.bfloat16)
    raw_k, raw_w = hidden[:, :128], hidden[:, 128 : 128 + heads]
    slots = torch.randperm(pages * 64, device="cuda")[:num_keys]
    slots[::3] = -1
    weight, bias = _layer_norm_affine(torch.bfloat16)
    scale = (heads**-0.5) * (128**-0.5)

    keys = torch.nn.functional.layer_norm(raw_k, (128,), weight, bias, 1e-5)
    store_keys(keys[:num_keys].contiguous(), caches[0], slots)
    payload, q_scales, weights = quantize_index_qk(
        q, raw_w, num_queries, scale, raw_k, caches[1], slots, (weight, bias, 1e-5)
    )
    torch.testing.assert_close(backing[1], backing[0], rtol=0, atol=0)
    ref_payload, ref_scales = quantize_queries(q[:num_queries])
    torch.testing.assert_close(payload, ref_payload, rtol=0, atol=0)
    torch.testing.assert_close(q_scales, ref_scales, rtol=0, atol=0)
    ref_weights = raw_w[:num_queries].float() * scale
    torch.testing.assert_close(weights, ref_weights, rtol=0, atol=0)


@pytest.mark.parametrize("return_scores", [False, True])
@pytest.mark.parametrize("pad_rows,prefill_rows", [(0, 0), (6, 0), (0, 300)])
def test_raw_key_indexer_matches_normalized_keys(return_scores, pad_rows, prefill_rows):
    """The raw-key path stores and selects what the unfused path does, and the
    call without LayerNorm arguments stays the unfused path."""
    from types import SimpleNamespace

    from vllm.config import VllmConfig
    from vllm.forward_context import set_forward_context
    from vllm.models.kimi_k3.amd import dsa  # noqa: F401
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
        DeepseekV32IndexerMetadata,
    )

    if return_scores and prefill_rows:
        pytest.skip("score merging is decode-only")
    torch.manual_seed(5 + pad_rows + prefill_rows)
    heads, rows_per, k = 32, 5, 2048
    lengths = [150, 300, 90, 2000, 40, 700, 1, 3000]
    reqs = len(lengths)
    decode_rows = reqs * rows_per
    rows = decode_rows + prefill_rows + pad_rows
    pages_per = 48
    pages = reqs * pages_per + 1
    table = torch.arange(1, pages, device="cuda", dtype=torch.int32).view(reqs, -1)
    slots = torch.full((rows,), -1, device="cuda", dtype=torch.int64)
    ends = []
    for r, length in enumerate(lengths):
        for j in range(rows_per):
            pos = length + j
            slots[r * rows_per + j] = table[r, pos // 64].long() * 64 + pos % 64
            ends.append(pos + 1)
    seq_lens = torch.tensor(ends, device="cuda", dtype=torch.int32)
    decode = DeepSeekV32IndexerDecodeMetadata(
        block_table=table,
        seq_lens=seq_lens,
        decode_lens=torch.full((reqs,), rows_per, device="cuda"),
        requires_padding=False,
        schedule_metadata=torch.empty(0, device="cuda", dtype=torch.int32),
        query_start_loc=torch.arange(
            0, decode_rows + 1, rows_per, device="cuda", dtype=torch.int32
        ),
        write_max_decode_len=rows_per,
    )
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32) + rows_per,
        max_seq_len=max(lengths) + rows_per,
        slot_mapping=slots,
        num_decodes=reqs,
        num_decode_tokens=decode_rows,
        num_prefills=1 if prefill_rows else 0,
        num_prefill_tokens=prefill_rows,
        decode=decode,
        prefill=SimpleNamespace(chunks=[], plans=[]) if prefill_rows else None,
        block_table=table,
    )
    hidden = torch.randn((rows, 3808), device="cuda", dtype=torch.bfloat16)
    raw_k, raw_w = hidden[:, :128], hidden[:, 128 : 128 + heads]
    weight, bias = _layer_norm_affine(torch.bfloat16)
    keys = torch.nn.functional.layer_norm(raw_k, (128,), weight, bias, 1e-5)
    q = torch.randn((rows, heads, 128), device="cuda", dtype=torch.bfloat16)
    context = torch.randn((pages * 64, 128), device="cuda", dtype=torch.bfloat16)
    caches: list[torch.Tensor] = []
    outs: list[torch.Tensor] = []
    scores: list[torch.Tensor | None] = []
    for _ in range(2):
        cache = torch.empty((pages, 64, 68), device="cuda", dtype=torch.uint8)
        dsa.store_keys(context, cache, torch.arange(pages * 64, device="cuda"))
        caches.append(cache)
        outs.append(torch.full((rows + 8, k), 777, device="cuda", dtype=torch.int32))
        scores.append(
            torch.zeros((rows + 8, k), device="cuda") if return_scores else None
        )
    scale = (heads**-0.5) * (128**-0.5)
    common = ("layer.indexer", 64 * pages_per, 0, 1, 1, scale)
    with set_forward_context({"layer.indexer": metadata}, VllmConfig()):
        dsa._fp4_indexer(
            q, keys, raw_w.contiguous(), caches[0], outs[0], *common, scores[0]
        )
        dsa._fp4_indexer(
            q, raw_k, raw_w, caches[1], outs[1], *common, scores[1], weight, bias, 1e-5
        )
    torch.testing.assert_close(caches[1], caches[0], rtol=0, atol=0)
    # Rows with more than k candidates come back in an unspecified order.
    torch.testing.assert_close(
        outs[1][:rows].sort(-1).values, outs[0][:rows].sort(-1).values, rtol=0, atol=0
    )
    if return_scores:
        unfused, fused = scores
        assert fused is not None and unfused is not None
        torch.testing.assert_close(
            fused[:rows].sort(-1).values,
            unfused[:rows].sort(-1).values,
            rtol=0,
            atol=0,
            equal_nan=True,
        )
