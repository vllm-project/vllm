# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.nvidia.fewhead_prefill import (
    run_fewhead_sparse_prefill,
)
from vllm.models.deepseek_v41.nvidia.triton_fewhead_sparse_prefill import (
    _bf16_fewhead_sparse_fwd,
    fewhead_sparse_mla_fwd,
)
from vllm.triton_utils import triton

D = 512
TOPK = 640
H_LOCAL = 8
H_PAD = 64
VALID_LEN = 512
PAD_SENTINEL = 7.0
MEAN_ABS = 2e-3
MAX_ABS = 2e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [192, 512])
def test_query_union_preserves_multiplicity_when_graph_pairing_changes(width):
    """Singleton and paired branches retain duplicate keys and ignore invalid ones."""
    from vllm.models.deepseek_v41.nvidia.ops.small_head_query_union import _union

    indices = (
        torch.arange(3 * width, device="cuda", dtype=torch.int32).reshape(3, width) % 13
    )
    indices[:, ::17] = -1
    lengths = torch.tensor([width, width - 7, 0], device="cuda", dtype=torch.int32)
    rows = torch.tensor([[0, -1], [1, 2]], device="cuda", dtype=torch.int32)
    count = torch.tensor(2, device="cuda", dtype=torch.int32)
    half = triton.next_power_of_2(width)
    keys = torch.empty((2, 2 * half), device="cuda", dtype=torch.int32)
    weights = torch.empty_like(keys)
    sizes = torch.empty(2, device="cuda", dtype=torch.int32)

    def run():
        _union[(2,)](
            rows,
            count,
            indices,
            lengths,
            keys,
            weights,
            sizes,
            width,
            width,
            half,
            True,
            num_warps=4,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for mapping in ([[0, -1], [1, 2]], [[0, 1], [2, -1]]):
        rows.copy_(torch.tensor(mapping, device="cuda", dtype=torch.int32))
        graph.replay()
        for group, pair in enumerate(mapping):
            for side, row in enumerate(pair):
                expected: Counter[int] = Counter()
                if row >= 0:
                    expected.update(
                        key for key in indices[row, : lengths[row]].tolist() if key >= 0
                    )
                actual: Counter[int] = Counter()
                for key, packed in zip(
                    keys[group, : sizes[group]].tolist(),
                    weights[group, : sizes[group]].tolist(),
                ):
                    weight = (packed >> (16 * side)) & 65535
                    if weight:
                        actual[key] += weight
                assert actual == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pair_warmup_covers_changing_token_counts(monkeypatch):
    """Dynamic batch sizes must preserve grouping without compiling new variants."""
    from vllm.models.deepseek_v41.nvidia.ops.small_head_decode_warmup import PAIR_WARMUP
    from vllm.models.deepseek_v41.nvidia.ops.small_head_query_union import _pairs

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_attention_heads=64)
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=64),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )
    PAIR_WARMUP.warmup(vllm_config=config)
    compilations = []
    monkeypatch.setattr(
        triton.knobs.runtime,
        "jit_post_compile_hook",
        lambda **kwargs: compilations.append(kwargs),
    )
    request_ids = [i // 3 if i % 13 else -1 for i in range(384)]
    valid_rows = [i % 11 not in (0, 1) for i in range(384)]
    req = torch.tensor(request_ids, device="cuda", dtype=torch.int32)
    valid = torch.tensor(valid_rows, device="cuda", dtype=torch.bool)
    rows = torch.empty((384, 2), device="cuda", dtype=torch.int32)
    count = torch.empty((), device="cuda", dtype=torch.int32)
    for tokens in range(17, 385):
        _pairs[(1,)](
            req, valid, rows, count, tokens, triton.next_power_of_2(tokens), num_warps=4
        )
        expected = []
        i = 0
        while i < tokens:
            if not valid_rows[i] or request_ids[i] < 0:
                i += 1
                continue
            second = (
                i + 1
                if i + 1 < tokens
                and valid_rows[i + 1]
                and request_ids[i + 1] == request_ids[i]
                else -1
            )
            expected.append([i, second])
            i += 2 if second >= 0 else 1
        assert count.item() == len(expected)
        assert rows[: len(expected)].tolist() == expected
    assert not compilations


def _packed_decode_cache(pages, page_size):
    from vllm.models.deepseek_v41.common.ops import quantize_and_insert_k_cache
    from vllm.utils.math_utils import round_up

    stride = round_up(page_size * 584, 576)
    storage = torch.empty((pages, stride), dtype=torch.uint8, device="cuda")
    kv = torch.randn((pages * page_size, D), dtype=torch.bfloat16, device="cuda")
    slots = torch.arange(kv.shape[0], dtype=torch.int64, device="cuda")
    quantize_and_insert_k_cache(kv, storage, slots, block_size=page_size)
    return storage.as_strided((pages, page_size, 584), (stride, 584, 1))


def _unpack_decode_cache(cache):
    pages, page_size, _ = cache.shape
    raw = cache.as_strided((pages, page_size * 584), (cache.stride(0), 1))
    data = raw[:, : page_size * 576].reshape(-1, 576)
    scales = raw[:, page_size * 576 :].reshape(-1, 8)[:, :7].float()
    nope = data[:, :448].contiguous().view(torch.float8_e4m3fn).float()
    nope = (nope.reshape(-1, 7, 64) * torch.exp2(scales - 127)[..., None]).flatten(1)
    rope = data[:, 448:].contiguous().view(torch.bfloat16).float()
    return torch.cat((nope, rope), dim=-1).to(torch.bfloat16).float()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "tokens,heads,swa_width,extra_page",
    [(1, 8, 128, 0), (5, 8, 192, 64), (17, 8, 128, 32), (17, 16, 192, 64)],
)
def test_direct_decode_matches_fp32_with_mutable_graph_inputs(
    tokens, heads, swa_width, extra_page
):
    """Direct decode handles TP4, draft windows, empty rows and strided buffers."""
    from vllm.models.deepseek_v41.nvidia.ops.small_head_sparse_decode import (
        small_head_sparse_decode,
    )

    torch.manual_seed(42)
    q = torch.randn((tokens + 2, heads + 1, D), device="cuda", dtype=torch.bfloat16)[
        1:-1, :heads
    ]
    storage = torch.full(
        (tokens + 2, heads + 1, D), PAD_SENTINEL, device="cuda", dtype=torch.bfloat16
    )
    out = storage[1:-1, :heads]
    sink = torch.linspace(-4, 4, heads, device="cuda", dtype=torch.float32)
    caches, indices, lengths, keys = [], [], [], []
    for page_size, width in [(64, swa_width)] + (
        [(extra_page, 512)] if extra_page else []
    ):
        cache = _packed_decode_cache(4, page_size)
        index = torch.randint(
            0, 4 * page_size, (tokens + 2, width + 8), device="cuda", dtype=torch.int32
        )[:, 4:-4]
        index[:, 11::13] = -1
        caches.append(cache)
        indices.append(index)
        lengths.append(
            torch.full((tokens + 2,), width, device="cuda", dtype=torch.int32)
        )
        keys.append(_unpack_decode_cache(cache))

    def run():
        small_head_sparse_decode(
            q,
            caches[0],
            indices[0],
            lengths[0],
            caches[1] if extra_page else None,
            indices[1] if extra_page else None,
            lengths[1] if extra_page else None,
            sink,
            D**-0.5,
            out,
            heads,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for replay in range(2):
        if replay:
            q.neg_()
            for index, length in zip(indices, lengths):
                index.copy_(index.flip(0))
                length[:tokens:2] = 0
        storage.fill_(PAD_SENTINEL)
        graph.replay()
        gathered, masks = [], []
        for key, index, length in zip(keys, indices, lengths):
            index = index[:tokens]
            gathered.append(key[index.clamp_min(0).long()])
            masks.append(
                (index >= 0)
                & (torch.arange(index.shape[1], device="cuda") < length[:tokens, None])
            )
        values = torch.cat(gathered, dim=1)
        valid = torch.cat(masks, dim=1)
        scores = torch.einsum("thd,tkd->thk", q.float(), values) * D**-0.5
        scores.masked_fill_(~valid[:, None, :], -float("inf"))
        logits = torch.cat((scores, sink[None, :, None].expand(tokens, -1, -1)), dim=-1)
        expected = torch.einsum("thk,tkd->thd", logits.softmax(-1)[..., :-1], values)
        _assert_close(out, expected)
        assert torch.all(storage[[0, -1]] == PAD_SENTINEL)
        assert torch.all(storage[:, heads:] == PAD_SENTINEL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [24, 64, 184])
@pytest.mark.parametrize(
    "ratio,piecewise,separate_index_source",
    [(1, False, False), (2, False, False), (1, True, False), (2, False, True)],
)
def test_paired_decode_refreshes_shared_unions_and_scratch_on_graph_replay(
    monkeypatch, tokens, ratio, piecewise, separate_index_source
):
    """Mutable requests/indices and shared scratch must preserve every layer output."""
    from vllm.config import CUDAGraphMode
    from vllm.models.deepseek_v41.nvidia.paired_decode import PairedDecode
    from vllm.v1.attention.ops.flashmla import flash_mla_with_kvcache, get_mla_metadata
    from vllm.v1.worker import workspace

    _require_flashmla_sparse()
    torch.manual_seed(42)
    manager = workspace.WorkspaceManager(torch.device("cuda"))
    monkeypatch.setattr(workspace, "_manager", manager)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=5, use_dspark=lambda: True
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=64),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_hidden_layers=40, index_topk=512)
        ),
    )
    context = SimpleNamespace(
        additional_kwargs={},
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE
        if piecewise
        else CUDAGraphMode.FULL,
    )
    decoders = [
        PairedDecode(
            SimpleNamespace(
                layer_id=i + 2,
                n_local_heads=8,
                compress_ratio=ratio,
                kv_source_layer_id=2,
                index_source_layer_id=3 if i and separate_index_source else 2,
                window_size=128,
            ),
            config,
        )
        for i in range(2)
    ]
    for decoder in decoders:
        decoder.reserve()
    manager.lock()
    sc = _packed_decode_cache(128, 64)
    ec = _packed_decode_cache(128, 64 // ratio)
    stride = sc.stride(0) + ec.stride(0)
    backing = torch.empty(128 * stride, dtype=torch.uint8, device="cuda")
    sc_view = backing.as_strided(sc.shape, (stride, 584, 1))
    ec_view = backing.as_strided(ec.shape, (stride, 584, 1), sc.stride(0))
    sc_view.copy_(sc)
    ec_view.copy_(ec)
    sc, ec = sc_view, ec_view
    req = torch.arange(tokens, dtype=torch.int32, device="cuda") // 2
    valid = torch.ones(tokens, dtype=torch.bool, device="cuda")
    valid[-2:] = False
    valid[tokens // 2] = False
    sl = torch.where(valid, 192, 0).to(torch.int32)
    el = torch.where(valid, 512, 0).to(torch.int32)
    si = (
        torch.arange(192, dtype=torch.int32, device="cuda")[None, :]
        .expand(tokens, -1)
        .clone()
    )
    sis = [si, si + 512]
    ei = torch.randint(0, 3900, (tokens, 512), dtype=torch.int32, device="cuda")
    extra_indices = [ei, ei + 64 if separate_index_source else ei]
    qs = [
        torch.randn(tokens, H_PAD, D, dtype=torch.bfloat16, device="cuda")
        for _ in decoders
    ]
    refs = [torch.empty_like(q) for q in qs]
    outputs = [torch.full_like(q, PAD_SENTINEL) for q in qs]
    sink = torch.zeros(H_PAD, device="cuda")
    sink[8:] = -float("inf")

    def baseline():
        for i in range(2):
            flash_mla_with_kvcache(
                q=qs[i].unsqueeze(1),
                k_cache=sc.unsqueeze(-2),
                block_table=None,
                head_dim_v=512,
                # Lengths change between replays; FlashMLA plans cannot be reused.
                tile_scheduler_metadata=get_mla_metadata()[0],
                cache_seqlens=None,
                is_fp8_kvcache=True,
                indices=sis[i].unsqueeze(1),
                topk_length=sl,
                softmax_scale=D**-0.5,
                attn_sink=sink,
                extra_k_cache=ec.unsqueeze(-2),
                extra_indices_in_kvcache=extra_indices[i].unsqueeze(1),
                extra_topk_length=el,
                out=refs[i].unsqueeze(1),
            )

    def candidate():
        for i, decoder in enumerate(decoders):
            assert decoder.run(
                context,
                qs[i][:, :8],
                sc,
                sis[i],
                sl,
                ec,
                extra_indices[i],
                el,
                sink[:8],
                D**-0.5,
                outputs[i][:, :8],
                req,
                valid,
            )

    def check():
        for actual, expected in zip(outputs, refs):
            _assert_close(actual[valid, :8], expected[valid, :8])
            assert torch.count_nonzero(actual[~valid, :8]) == 0
            assert torch.all(actual[:, 8:] == PAD_SENTINEL)

    baseline()
    candidate()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate()
    for step in range(2):
        req.copy_(torch.arange(tokens, dtype=torch.int32, device="cuda") // (step + 1))
        valid.fill_(True)
        sl.fill_(192)
        el.fill_(512)
        for q, ix in zip(qs, sis):
            q.add_(0.125)
            ix.add_(1)
        ei.add_(1)
        if separate_index_source:
            extra_indices[1].add_(1)
        baseline()
        graph.replay()
        check()


def _require_flashmla_sparse() -> None:
    from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported

    ok, reason = is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason or "FlashMLA sparse unavailable")


def _inputs(s_q: int, *, seed: int = 0):
    torch.manual_seed(seed)
    device = "cuda"
    s_kv = max(s_q, TOPK)
    sm_scale = D**-0.5
    kv = torch.randn((s_kv, 1, D), dtype=torch.bfloat16, device=device)
    q64 = torch.randn((s_q, H_PAD, D), dtype=torch.bfloat16, device=device)
    indices = torch.randint(0, s_kv, (s_q, TOPK), dtype=torch.int32, device=device)
    indices[:, VALID_LEN:] = -1
    lens = torch.full((s_q,), VALID_LEN, dtype=torch.int32, device=device)
    sink64 = torch.zeros((H_PAD,), dtype=torch.float32, device=device)
    sink64[H_LOCAL:] = float("-inf")
    return q64, kv, indices, lens, sink64, sm_scale


def _flash_local_heads(q64, kv, indices, lens, sink64, sm_scale):
    from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

    flash_out = flash_mla_sparse_fwd(
        q=q64,
        kv=kv,
        indices=indices.unsqueeze(1),
        sm_scale=sm_scale,
        attn_sink=sink64,
        topk_length=lens,
    )
    flash_q = flash_out[0] if isinstance(flash_out, (tuple, list)) else flash_out
    return flash_q[:, :H_LOCAL]


def _assert_close(got: torch.Tensor, ref: torch.Tensor) -> None:
    err = (got.float() - ref.float()).abs()
    assert bool(torch.isfinite(got).all())
    assert float(err.mean()) < MEAN_ABS
    assert float(err.max()) < MAX_ABS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("s_q", [64, 2048, 8192])
def test_fewhead_matches_padded_flashmla_noncontiguous_out(s_q: int):
    _require_flashmla_sparse()
    q64, kv, indices, lens, sink64, sm_scale = _inputs(s_q)
    ref = _flash_local_heads(q64, kv, indices, lens, sink64, sm_scale)

    q_view = q64[:, :H_LOCAL]
    out = torch.full_like(q64, PAD_SENTINEL)
    out_view = out[:, :H_LOCAL]
    assert not q_view.is_contiguous()
    assert not out_view.is_contiguous()

    tri = fewhead_sparse_mla_fwd(
        q_view,
        kv,
        indices,
        sm_scale,
        attn_sink=sink64[:H_LOCAL],
        topk_length=lens,
        out=out_view,
    )
    assert tri.data_ptr() == out_view.data_ptr()
    _assert_close(out[:, :H_LOCAL], ref)
    padded = out[:, H_LOCAL:]
    assert torch.equal(padded, torch.full_like(padded, PAD_SENTINEL))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_run_fewhead_sparse_prefill_writes_local_heads_only():
    _require_flashmla_sparse()
    s_q = 2048
    pad = 128
    q_full = torch.randn((s_q + pad, H_PAD, D), dtype=torch.bfloat16, device="cuda")
    q64, kv, indices, lens, sink64, sm_scale = _inputs(s_q, seed=1)
    q_full[pad : pad + s_q].copy_(q64)
    q_chunk = q_full[pad : pad + s_q]
    ref = _flash_local_heads(q_chunk.contiguous(), kv, indices, lens, sink64, sm_scale)

    out_full = torch.full_like(q_full, PAD_SENTINEL)
    out_chunk = out_full[pad : pad + s_q]
    returned = run_fewhead_sparse_prefill(
        q_chunk,
        kv,
        indices,
        sm_scale,
        attn_sink=sink64,
        topk_length=lens,
        out=out_chunk,
        n_local_heads=H_LOCAL,
    )
    assert returned.data_ptr() == out_chunk.data_ptr()
    assert not q_chunk[:, :H_LOCAL].is_contiguous()
    _assert_close(out_chunk[:, :H_LOCAL], ref)
    assert torch.equal(
        out_chunk[:, H_LOCAL:],
        torch.full_like(out_chunk[:, H_LOCAL:], PAD_SENTINEL),
    )
    assert torch.equal(
        out_full[:pad],
        torch.full_like(out_full[:pad], PAD_SENTINEL),
    )
    assert torch.equal(
        out_full[pad + s_q :],
        torch.full_like(out_full[pad + s_q :], PAD_SENTINEL),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fewhead_kv_offsets_cross_int32_boundary():
    """Large KV offsets must not wrap into the guard before the view."""
    boundary = 2**31 // D
    required_bytes = (2 * boundary + 1) * D * 2
    if torch.accelerator.get_memory_info()[0] < required_bytes + 1024**3:
        pytest.skip("Large-offset regression requires 9 GiB of free GPU memory")

    # The guard keeps the old wrapped address mapped, so failure is an assertion
    # rather than an illegal access that poisons subsequent CUDA tests.
    storage = torch.empty((2 * boundary + 1, D), device="cuda", dtype=torch.bfloat16)
    storage[0].fill_(-3)
    kv = storage[boundary:]
    kv[boundary - 1].fill_(2)
    kv[boundary].fill_(1)
    q = torch.zeros((2, H_PAD, D), device="cuda", dtype=torch.bfloat16)
    indices = torch.full((2, TOPK), -1, device="cuda", dtype=torch.int32)
    indices[:, 0] = torch.tensor([boundary - 1, boundary], device="cuda")
    lengths = torch.ones(2, device="cuda", dtype=torch.int32)
    sink = torch.zeros(H_LOCAL, device="cuda", dtype=torch.float32)

    got = fewhead_sparse_mla_fwd(
        q[:, :H_LOCAL], kv, indices, D**-0.5, attn_sink=sink, topk_length=lengths
    )
    expected = torch.tensor([1, 0.5], device="cuda", dtype=torch.bfloat16)
    assert torch.equal(got, expected[:, None, None].expand_as(got))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_startup_warmup_covers_strided_prefill_without_runtime_compilation(monkeypatch):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                num_attention_heads=64,
                head_dim=D,
                sliding_window=128,
                index_topk=512,
                compress_ratios=[0, 1, 2],
            )
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
    )
    allocated = torch.accelerator.memory_allocated()
    _bf16_fewhead_sparse_fwd.warmup(vllm_config=config)
    assert torch.accelerator.memory_allocated() == allocated
    compilations = []
    monkeypatch.setattr(
        triton.knobs.runtime,
        "jit_post_compile_hook",
        lambda **kwargs: compilations.append(kwargs),
    )
    for s_q in (1, 17, 2048, 2063):
        q, kv, indices, lengths, sink, scale = _inputs(s_q)
        out = torch.empty_like(q)
        for topk in (128, TOPK):
            for native in (False, True):
                query = q[:, :H_LOCAL]
                result = out[:, :H_LOCAL]
                if native:
                    query = query.contiguous()
                    result = torch.empty_like(query)
                fewhead_sparse_mla_fwd(
                    query,
                    kv,
                    indices[:, :topk],
                    scale,
                    attn_sink=sink[:H_LOCAL],
                    topk_length=lengths,
                    out=result,
                )
    torch.accelerator.synchronize()
    assert not compilations


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("tokens", [1, 17, 129])
def test_native_head_short_prefill_matches_fp32_reference(heads, tokens):
    """The independent backend cannot fall back for short or partially empty rows."""
    torch.manual_seed(42)
    q = torch.randn((tokens + 1, heads, D), device="cuda", dtype=torch.bfloat16)[1:]
    kv = torch.randn((257, D), device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, 257, (tokens, 128), device="cuda", dtype=torch.int32)
    lengths = torch.arange(tokens, device="cuda", dtype=torch.int32) % 128
    if tokens == 1:
        lengths.fill_(113)
    indices[:, 11::13] = -1
    sink = torch.linspace(-4, 4, heads, device="cuda", dtype=torch.float32)
    keys = kv[indices.clamp_min(0).long()].float()
    scores = torch.einsum("thd,tkd->thk", q.float(), keys) * D**-0.5
    valid = (indices >= 0) & (torch.arange(128, device="cuda") < lengths[:, None])
    scores.masked_fill_(~valid[:, None, :], -float("inf"))
    weights = torch.softmax(
        torch.cat((scores, sink[None, :, None].expand(tokens, -1, -1)), dim=-1),
        dim=-1,
    )[..., :-1]
    expected = torch.einsum("thk,tkd->thd", weights, keys)
    out = torch.empty_like(q)
    fewhead_sparse_mla_fwd(
        q, kv, indices, D**-0.5, attn_sink=sink, topk_length=lengths, out=out
    )
    _assert_close(out, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fewhead_sparse_mla_fwd(
            q, kv, indices, D**-0.5, attn_sink=sink, topk_length=lengths, out=out
        )
    out.zero_()
    graph.replay()
    _assert_close(out, expected)
