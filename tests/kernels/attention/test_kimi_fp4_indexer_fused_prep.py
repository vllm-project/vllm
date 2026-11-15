# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi MXFP4 indexer: fused decode prep against the unfused torch path."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm MXFP4 indexer"
)

HEADS, TOPK, EPS = 32, 2048, 1e-5


def _cache(pages: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Index sections inside merged MLA pages, with poisoned neighbors.
    storage = torch.full((pages, 46592), 173, dtype=torch.uint8, device="cuda")
    return storage, storage[:, 41984:46336].view(pages, 64, 68)


def _knorm():
    from vllm.models.kimi_k3.amd.dsa import KimiK3DSAIndexerKNorm

    norm = KimiK3DSAIndexerKNorm(128, EPS).cuda()
    with torch.no_grad():
        norm.weight.copy_(torch.randn(128, device="cuda") * 0.5 + 1)
        norm.bias.copy_(torch.randn(128, device="cuda") * 0.1)
    return norm


@pytest.mark.parametrize("tokens", [4, 16, 64, 192])
@pytest.mark.parametrize("norm", [True, False])
def test_key_store_and_reset_matches_unfused(tokens, norm):
    from vllm.models.kimi_k3.amd.ops.fp4_indexer import (
        store_keys,
        store_keys_and_reset,
    )

    torch.manual_seed(tokens)
    knorm = _knorm()
    pages = 8
    raw = torch.randn((tokens, 128), dtype=torch.bfloat16, device="cuda") * 3 + 0.5
    slots = torch.randperm(pages * 64, device="cuda")[:tokens]
    slots[::5] = -1  # padded tokens
    # Graph padding: the top-k buffer view is longer than the slot mapping.
    num_slots = tokens - tokens // 4
    keys, slots = raw[:num_slots], slots[:num_slots]

    ref_storage, ref_cache = _cache(pages)
    ref_out = torch.full((tokens + 3, TOPK), 7, dtype=torch.int32, device="cuda")
    store_keys(knorm(keys) if norm else keys, ref_cache, slots)
    ref_out[:tokens].fill_(-1)

    storage, cache = _cache(pages)
    out = torch.full_like(ref_out, 7)
    store_keys_and_reset(
        keys,
        cache,
        slots,
        out[:tokens],
        knorm.weight if norm else None,
        knorm.bias if norm else None,
        EPS,
    )
    torch.testing.assert_close(storage, ref_storage, rtol=0, atol=0)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [4, 16, 64, 192])
@pytest.mark.parametrize("tile", [(0, None), (32, 64)])
@pytest.mark.parametrize(
    "heads,heads_per_program", [(32, 8), (32, 1), (16, 32), (64, 8)]
)
def test_query_weights_and_starts_match_unfused(tokens, tile, heads, heads_per_program):
    from vllm.models.kimi_k3.amd.ops.fp4_indexer import (
        quantize_queries,
        quantize_queries_and_weights,
    )

    torch.manual_seed(tokens + 1)
    start, end = tile
    end = tokens if end is None else min(end, tokens)
    if start >= end:
        pytest.skip("tile outside the batch")
    scale = heads**-0.5 * 128**-0.5
    # The wq_index output viewed as heads, then sliced to the decode tile.
    q = torch.randn((tokens, heads * 128), dtype=torch.bfloat16, device="cuda")
    q = q.view(-1, heads, 128)
    weights = torch.randn((tokens, heads), dtype=torch.bfloat16, device="cuda")
    # 4-token verify requests, with a 1-token tail request.
    lens = [4] * (tokens // 4) + ([tokens % 4] if tokens % 4 else [])
    starts = torch.tensor([0] + lens, device="cuda").cumsum(0).to(torch.int32)

    payload, scales, scaled, rel = quantize_queries_and_weights(
        q[start:end],
        weights[start:end],
        scale,
        starts,
        start,
        end - start,
        heads_per_program,
    )
    ref_payload, ref_scales = quantize_queries(q[start:end])
    rel_ref = starts if start == 0 else starts - start
    torch.testing.assert_close(payload, ref_payload, rtol=0, atol=0)
    torch.testing.assert_close(scales, ref_scales, rtol=0, atol=0)
    torch.testing.assert_close(
        scaled, weights[start:end].float() * scale, rtol=0, atol=0
    )
    torch.testing.assert_close(rel, rel_ref.clamp(0, end - start), rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [4, 64, 192])
@pytest.mark.parametrize("return_scores", [False, True])
@pytest.mark.parametrize("prepped", [False, True])
def test_indexer_op_fused_prep_matches_unfused(tokens, return_scores, prepped):
    """The registered op, eager reference vs fused prep under a FULL graph;
    ``prepped``: the caller stored the keys, reset the rows and quantized the
    query (the specialized decode's D1a), the op only scores and selects."""
    from vllm.config import VllmConfig
    from vllm.forward_context import set_forward_context
    from vllm.models.kimi_k3.amd import dsa  # noqa: F401
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
        DeepseekV32IndexerMetadata,
    )

    torch.manual_seed(5)
    width, padded = 4096, tokens + 8
    pages = width // 64
    reqs = tokens // 4
    config = VllmConfig()
    knorm = _knorm()
    q = torch.randn((padded, HEADS, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((padded, 128), device="cuda", dtype=torch.bfloat16)
    weights = torch.rand((padded, HEADS), device="cuda", dtype=torch.bfloat16)
    history = torch.randn((reqs * width, 128), device="cuda", dtype=torch.bfloat16)
    lengths = torch.randint(width // 2, width, (reqs,), device="cuda")
    table = torch.arange(reqs * pages, device="cuda", dtype=torch.int32)
    table = table.view(reqs, pages)
    rows = torch.arange(tokens, device="cuda")
    req = rows // 4
    ends = (lengths[req] - 3 + rows % 4).to(torch.int32)
    slots = (req * width + ends - 1).to(torch.int64)
    starts = torch.arange(0, tokens + 1, 4, device="cuda", dtype=torch.int32)
    batch = DeepseekV32IndexerMetadata(
        seq_lens=lengths.to(torch.int32),
        max_seq_len=width,
        slot_mapping=slots,
        num_decodes=reqs,
        num_decode_tokens=tokens,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=DeepSeekV32IndexerDecodeMetadata(
            block_table=table[req],
            seq_lens=ends,
            decode_lens=torch.full((reqs,), 4, device="cuda"),
            requires_padding=False,
            schedule_metadata=torch.empty(0, device="cuda", dtype=torch.int32),
            query_start_loc=starts,
            write_max_decode_len=4,
        ),
        prefill=None,
        block_table=table,
    )
    scale = HEADS**-0.5 * 128**-0.5

    def run(fused, cache, out, scores):
        prep = {}
        if fused and prepped:
            # D1a's part, here with the Triton kernels it matches bit for bit;
            # the op resets the padding rows past it
            store_keys_and_reset(knorm(k)[:tokens], cache, slots, out[:tokens])
            payload, q_scales, q_weights, _ = quantize_queries_and_weights(
                q[:tokens], weights[:tokens], scale, starts, 0, tokens
            )
            prep = dict(q_payload=payload, q_scales=q_scales, q_weights=q_weights)
        torch.ops.vllm.kimi_fp4_indexer(
            q,
            k if fused else knorm(k),
            weights,
            cache,
            out,
            "indexer",
            width,
            weight_scale=scale,
            scores=scores,
            fused_prep=fused,
            k_norm_weight=knorm.weight if fused else None,
            k_norm_bias=knorm.bias if fused else None,
            k_norm_eps=EPS,
            **prep,
        )

    from vllm.models.kimi_k3.amd.ops.fp4_indexer import (
        quantize_queries_and_weights,
        store_keys,
        store_keys_and_reset,
    )

    initial, cache = _cache(reqs * pages)
    store_keys(knorm(history), cache, torch.arange(reqs * width, device="cuda"))
    results = []
    for fused in (False, True):
        storage = initial.clone()
        cache = storage[:, 41984:46336].view(-1, 64, 68)
        out = torch.full((padded, TOPK), 9, device="cuda", dtype=torch.int32)
        scores = torch.zeros(out.shape, device="cuda") if return_scores else None
        with set_forward_context({"indexer": batch}, config):
            if fused:
                run(fused, cache, out, scores)  # warm up and compile
                storage.copy_(initial)
                out.fill_(9)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(fused, cache, out, scores)
                graph.replay()
            else:
                run(fused, cache, out, scores)
        torch.cuda.synchronize()
        results.append((storage, out, scores))
    (ref_storage, ref_out, ref_scores), (storage, out, scores) = results
    torch.testing.assert_close(storage, ref_storage, rtol=0, atol=0)
    torch.testing.assert_close(
        out.sort(dim=1).values, ref_out.sort(dim=1).values, rtol=0, atol=0
    )
    assert bool((out[tokens:] == -1).all())
    if return_scores:
        torch.testing.assert_close(
            scores.gather(1, out.argsort(dim=1)),
            ref_scores.gather(1, ref_out.argsort(dim=1)),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("width", [131072, 1 << 20])
@pytest.mark.parametrize("length", [100, 2048, 40000, 360000])
@pytest.mark.parametrize("rows", [1, 4, 8, 64])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("pitched", [False, True])
def test_sampled_select_matches_decode(monkeypatch, width, length, rows, tied, pitched):
    """VLLM_K3_INDEXER_SAMPLED_TOPK (under latency_specialized): AITER's
    sampled select under a graph picks the decode kernel's set, including rows
    shorter than k. ``tied``: coarse logits (as MXFP4 scores are) with ties at
    the k-th value, where either select may keep a different tied index: the
    same values then. ``pitched``: the logits a column slice of a wider row
    pitch."""
    from vllm.models.kimi_k3.amd.dsa import _sampled_serves, _select
    from vllm.models.kimi_k3.amd.specialized_decode import levers

    monkeypatch.setattr(levers, "_active", True)  # latency_specialized's gate
    cols = width // 2 if pitched else width
    length = min(length, cols)
    if not _sampled_serves(rows, width, TOPK, torch.accelerator.current_device_index()):
        pytest.skip("sampled top-k not served on this GPU")
    torch.manual_seed(length + rows)
    buf = torch.full((rows, width), 1e20, device="cuda")  # poisoned tail
    if tied:
        buf[:, :length] = torch.randint(0, 8, (rows, length), device="cuda").float()
    else:
        s = torch.randn((rows, length, HEADS), device="cuda").relu_()
        buf[:, :length] = (s * torch.randn((rows, 1, HEADS), device="cuda")).sum(-1)
    logits = buf[:, :cols]
    ends = torch.arange(length - rows + 1, length + 1, device="cuda")
    ends = ends.clamp(min=1).to(torch.int32)

    monkeypatch.setenv("VLLM_K3_INDEXER_SAMPLED_TOPK", "0")
    ref = torch.full((rows, TOPK), 9, device="cuda", dtype=torch.int32)
    _select(logits, ends, ref)
    monkeypatch.setenv("VLLM_K3_INDEXER_SAMPLED_TOPK", "1")
    out = torch.full_like(ref, 9)
    _select(logits, ends, out)  # warm up outside the capture
    out.fill_(9)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _select(logits, ends, out)
    graph.replay()
    torch.accelerator.synchronize()

    # each row: min(end, k) distinct indices below its end, -1 past them
    live = ends.clamp(max=TOPK).long()
    for sel in (out, ref):
        idx = sel.sort(dim=1).values
        valid = idx >= 0
        torch.testing.assert_close(valid.sum(1), live, rtol=0, atol=0)
        assert bool((idx[~valid] == -1).all())
        assert bool((idx < ends[:, None]).all())
        assert bool(((idx[:, 1:] != idx[:, :-1]) | ~valid[:, 1:]).all())
    if not tied:
        # Random logits can still tie at the k-th value (e.g. rows=64 at 128k);
        # compare index sets only on rows whose k-th and (k+1)-th values differ.
        masked = torch.where(
            torch.arange(cols, device="cuda")[None] < ends[:, None],
            logits,
            float("-inf"),
        )
        kk = min(TOPK + 1, cols)
        top = masked.topk(kk, dim=1).values
        untied = (ends <= TOPK) | (top[:, TOPK - 1] != top[:, kk - 1])
        torch.testing.assert_close(
            out.sort(dim=1).values[untied],
            ref.sort(dim=1).values[untied],
            rtol=0,
            atol=0,
        )

    def picked(sel):
        v = logits.gather(1, sel.clamp(min=0).long())
        return v.masked_fill(sel < 0, float("-inf")).sort(dim=1).values

    torch.testing.assert_close(picked(out), picked(ref), rtol=0, atol=0)
