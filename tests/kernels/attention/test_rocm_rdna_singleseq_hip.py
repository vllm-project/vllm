# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA3.5 decode and prefill attention: accuracy against a reference, and
fallback to Triton for unsupported calls."""

import itertools

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")

hip = pytest.importorskip("vllm.v1.attention.ops.rocm_rdna_singleseq_hip")

HQ, HKV, HEAD_DIM = 32, 16, 256
# Relative error bound per dtype.
TOL = {torch.float16: 1e-3, torch.bfloat16: 8e-3}
DTYPES = list(TOL)


def _skip_unless_gfx1151():
    from vllm.platforms.rocm import on_gfx1151

    if not on_gfx1151():
        pytest.skip("kernel is built for gfx1151")


def _built(dtype=torch.float16, **want):
    """The first built variant matching `want`, and its module."""
    _skip_unless_gfx1151()
    for v in hip.variants():
        if v.dtype == dtype and all(pred(getattr(v, k)) for k, pred in want.items()):
            module = hip.load(v)
            assert module is not None, f"{v.name} is not in _rocm_C"
            return v, module
    raise AssertionError(f"no built-in variant with {sorted(want)}")


def _paged_inputs(seq_len, v, seed=0):
    """An HND KV cache whose last page is partly used."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    num_blocks = -(-seq_len // v.page_size)
    kv = torch.randn(
        num_blocks,
        v.num_kv_heads,
        v.page_size,
        2 * v.head_size,
        device=dev,
        dtype=v.dtype,
    )
    kv = kv * 0.5
    q = torch.randn(
        v.max_query_len, v.num_q_heads, v.head_size, device=dev, dtype=v.dtype
    )
    return q * 0.5, kv, torch.arange(num_blocks, device=dev, dtype=torch.int32)


def _reference(q, kv, seq_len, window=0, mm_ranges=()):
    m, hq, hkv, hd = q.shape[0], q.shape[1], kv.shape[1], q.shape[2]
    flat = kv.transpose(1, 2).reshape(-1, hkv, 2 * hd)[:seq_len]
    k, v = flat[..., :hd], flat[..., hd:]
    gqa = hq // hkv
    qf = q.float().permute(1, 0, 2)
    kf = k.float().permute(1, 0, 2).repeat_interleave(gqa, 0)
    vf = v.float().permute(1, 0, 2).repeat_interleave(gqa, 0)
    scores = torch.bmm(qf, kf.transpose(1, 2)) * (hd**-0.5)
    pos = torch.arange(seq_len, device=q.device).view(1, seq_len)
    lim = (seq_len - m + torch.arange(m, device=q.device)).view(m, 1)
    masked = pos > lim
    for start, end in mm_ranges:
        masked &= ~((lim >= start) & (lim <= end) & (pos >= start) & (pos <= end))
    if window:
        masked |= pos < lim - (window - 1)
    scores = scores.masked_fill(masked.view(1, m, seq_len), float("-inf"))
    return torch.bmm(torch.softmax(scores, -1), vf).permute(1, 0, 2)


def _max_rel(got, ref) -> float:
    """Relative error with a floor."""
    return ((got - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()


def _launch(module, q, kv, block_table, out, scratch, seq_len):
    seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
    module.decode_attn(q, kv, block_table, out, *scratch, seq_lens, q.shape[2] ** -0.5)


def _check_one(v, module, lens):
    """Launches sharing one scratch, each on new inputs."""
    scratch = hip.make_scratch(v, torch.device("cuda"))
    for seed, seq_len in enumerate(lens):
        q, kv, block_table = _paged_inputs(seq_len, v, seed)
        out = torch.full_like(q, float("nan"))
        _launch(module, q, kv, block_table, out, scratch, seq_len)
        torch.accelerator.synchronize()
        ref = _reference(q, kv, seq_len, v.window)
        got = out.float().nan_to_num(1e9)
        assert _max_rel(got, ref) <= TOL[v.dtype], f"{v.name}, S={seq_len}"


def _unit(v):
    """A test case: the variants of one shape and dtype."""
    dt = "bf16" if v.dtype == torch.bfloat16 else "fp16"
    return f"q{v.num_q_heads}_kv{v.num_kv_heads}_d{v.head_size}_w{v.window}_{dt}"


def test_decode_configs_are_well_formed():
    """No two decode builds share a shape, and every built shape covers every M
    in both dtypes."""
    variants = hip.variants()
    if not variants:
        pytest.skip("_rocm_C carries no RDNA3.5 variants")
    assert len({hip._lookup_key(v) for v in variants}) == len(variants), (
        "two builds share a shape"
    )
    configs = {
        (v.num_q_heads, v.num_kv_heads, v.head_size, v.window, v.page_size)
        for v in variants
    }
    for hq, hkv, d, window, bs in configs:
        assert d in hip.SUPPORTED_HEAD_SIZES and hq % hkv == 0
        for m, dtype in itertools.product(range(1, hip.MAX_M + 1), DTYPES):
            v = hip.variant_for(hq, hkv, d, m, bs, 1, window, dtype)
            assert v is not None, (hq, hkv, d, window, bs, m, dtype)


def test_every_listed_variant_is_built_in():
    _skip_unless_gfx1151()
    # Raises if a build has no instantiation unit.
    assert torch.ops._rocm_C.rdna_singleseq_decode_configs()
    missing = [v.name for v in hip.variants() if hip.load(v) is None]
    assert not missing, f"{len(missing)} variants not in _rocm_C, e.g. {missing[0]}"


@pytest.mark.parametrize("unit", sorted({_unit(v) for v in hip.variants()}))
def test_built_in_variants_match_reference(unit):
    """Every built decode variant matches the reference."""
    _skip_unless_gfx1151()
    for v in hip.variants():
        if _unit(v) != unit:
            continue
        module = hip.load(v)
        assert module is not None, v.name
        s = max(1024, v.window + 64, v.page_size + 1000)
        _check_one(v, module, [48, s, s + 1000])
        # An image ending among the query tokens: rows on each side differ.
        m = v.max_query_len
        ranges = [(s - m - 40, s - m + m // 2)]
        q, kv, block_table = _paged_inputs(s, v)
        buf = _decode_mm_buf(ranges, q.device)
        got = _decode_mm(module, v, q, kv, block_table, s, buf)
        ref = _reference(q, kv, s, v.window, ranges)
        assert _max_rel(got, ref) <= TOL[v.dtype], f"{v.name}, image"
        # At M=1 the image ends at the only query token, so nothing changes.
        if m > 1:
            causal = _reference(q, kv, s, v.window)
            assert _max_rel(causal, ref) > 10 * TOL[v.dtype], f"{v.name}, image"


@pytest.mark.parametrize(
    "want",
    [
        {
            "window": bool,
            "dot_product": lambda x: not x,
            "max_segments": lambda x: x > 1,
        },
        {
            "window": bool,
            "dot_product": lambda x: not x,
            "max_segments": lambda x: x == 1,
        },
        {"window": bool, "dot_product": bool},
    ],
    ids=["wmma segments", "wmma", "dot_product"],
)
def test_sliding_window_never_reads_freed_pages(want):
    """Freed pages before the window do not reach the output."""
    v, module = _built(**want)
    seq_len = 2016
    q, kv, block_table = _paged_inputs(seq_len, v)
    ref = _reference(q, kv, seq_len, v.window)
    first = max(0, seq_len - v.max_query_len - (v.window - 1)) // v.page_size
    kv = torch.cat([kv, torch.full_like(kv[:1], float("nan"))])
    block_table = block_table.clone()
    block_table[:first] = kv.shape[0] - 1
    out = torch.empty_like(q)
    _launch(module, q, kv, block_table, out, hip.make_scratch(v, q.device), seq_len)
    torch.accelerator.synchronize()
    assert torch.isfinite(out).all()
    assert _max_rel(out.float(), ref) <= TOL[v.dtype]


def test_graph_replay_follows_seq_lens():
    """A captured graph follows the sequence length on the device."""
    v, module = _built(
        window=lambda x: not x,
        dot_product=lambda x: not x,
        max_segments=lambda x: x > 1,
    )
    long_s = 1024
    q, kv, block_table = _paged_inputs(long_s, v)
    scratch = hip.make_scratch(v, q.device)
    out = torch.empty_like(q)
    seq_lens = torch.tensor([48], device=q.device, dtype=torch.int32)

    def launch():
        module.decode_attn(
            q, kv, block_table, out, *scratch, seq_lens, q.shape[2] ** -0.5
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for s in (48, 208, long_s):
        seq_lens.fill_(s)
        out.zero_()
        graph.replay()
        torch.accelerator.synchronize()
        ref = _reference(q, kv, s)
        assert _max_rel(out.float(), ref) <= TOL[v.dtype], f"S={s}"


@pytest.mark.parametrize("dtype", DTYPES)
def test_other_dtype_is_refused_not_miscomputed(dtype):
    """A build refuses the other 16-bit dtype."""
    other = next(d for d in DTYPES if d != dtype)
    v, module = _built(dtype=other)
    q, kv, block_table = _paged_inputs(1024, v)
    q, kv = q.to(dtype), kv.to(dtype)
    with pytest.raises(RuntimeError, match="kernel built for"):
        _launch(
            module,
            q,
            kv,
            block_table,
            torch.empty_like(q),
            hip.make_scratch(v, q.device),
            1024,
        )


def _impl():
    from vllm.v1.attention.backends.rocm_rdna_singleseq_attn import (
        RocmRdnaSingleSeqAttentionImpl as Impl,
    )

    return Impl.__new__(Impl)


def test_unsupported_head_size_falls_back_to_triton():
    """An unbuilt head size falls back to Triton."""
    from vllm.v1.kv_cache_interface import KVQuantMode

    impl = _impl()
    impl._rejected = None
    impl.head_size, impl.num_heads, impl.num_kv_heads = 80, HQ, HKV

    fits = impl._prepare(
        kv_cache=torch.empty(0, dtype=torch.float16),
        q=torch.empty(0, dtype=torch.float16),
        out=torch.empty(0, dtype=torch.float16),
        alibi_slopes=None,
        sinks=None,
        softcap=0,
        causal=True,
        window_size=None,
        kv_quant_mode=KVQuantMode.NONE,
        seqused_k=torch.zeros(1),
        max_seqlen_q=0,
    )
    assert not fits
    assert "head_size 80" in impl._rejected


def test_strided_query_falls_back_to_triton():
    """A strided Q falls back to Triton."""
    from vllm.v1.kv_cache_interface import KVQuantMode

    impl = _impl()
    impl._rejected = None
    impl.head_size, impl.num_heads, impl.num_kv_heads = HEAD_DIM, HQ, HKV
    qkv = torch.empty(2, (HQ + 2 * HKV) * HEAD_DIM, dtype=torch.float16)
    q = qkv[:, : HQ * HEAD_DIM].view(-1, HQ, HEAD_DIM)
    fits = impl._prepare(
        kv_cache=torch.empty(0, dtype=torch.float16),
        q=q,
        out=torch.empty(2, HQ, HEAD_DIM, dtype=torch.float16),
        alibi_slopes=None,
        sinks=None,
        softcap=0,
        causal=True,
        window_size=None,
        kv_quant_mode=KVQuantMode.NONE,
        seqused_k=torch.zeros(1),
        max_seqlen_q=2,
    )
    assert not fits
    assert "not contiguous" in impl._rejected


@pytest.mark.parametrize(
    "nseq, tokens, max_q, reason",
    [
        (3, 3, 1, "3 sequences"),
        (1, 64, 64, "more than"),
    ],
)
def test_non_decode_batches_fall_back_to_triton(nseq, tokens, max_q, reason):
    """Only one sequence of at most MAX_M tokens reaches the decode kernel."""
    from vllm.v1.kv_cache_interface import KVQuantMode

    impl = _impl()
    impl._rejected = None
    impl.head_size, impl.num_heads, impl.num_kv_heads = HEAD_DIM, HQ, HKV
    fits = impl._prepare(
        kv_cache=torch.empty(0, dtype=torch.float16),
        q=torch.empty(tokens, HQ, HEAD_DIM, dtype=torch.float16),
        alibi_slopes=None,
        sinks=None,
        softcap=0,
        causal=True,
        window_size=None,
        kv_quant_mode=KVQuantMode.NONE,
        seqused_k=torch.zeros(nseq),
        max_seqlen_q=max_q,
    )
    assert not fits
    assert reason in impl._rejected


@pytest.mark.parametrize("gfx1151", [True, False])
def test_singleseq_is_opt_in(gfx1151, monkeypatch):
    """The backend is never selected automatically."""
    import vllm.platforms.rocm as rocm
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as B

    monkeypatch.setattr(rocm, "on_gfx1151", lambda: gfx1151)
    order = rocm._get_backend_priorities(use_mla=False, use_sparse=False)
    assert B.ROCM_RDNA_SINGLESEQ_ATTN not in order


@pytest.mark.parametrize("max_num_seqs", [1, 2])
def test_singleseq_requires_one_sequence(max_num_seqs, monkeypatch):
    """The backend requires max_num_seqs=1 and uses Triton's KV cache."""
    import types

    from vllm.v1.attention.backends import rocm_rdna_singleseq_attn as rdna
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheLayout

    config = types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_seqs=max_num_seqs)
    )
    monkeypatch.setattr(rdna, "get_current_vllm_config_or_none", lambda: config)
    backend = rdna.RocmRdnaSingleSeqAttentionBackend
    reason = backend.supports_combination(
        HEAD_DIM, torch.float16, "auto", 16, False, False, False, False, None
    )
    assert reason == (None if max_num_seqs == 1 else "max_num_seqs must be 1")
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=HKV, head_size=HEAD_DIM, dtype=torch.float16
    )
    assert backend.customize_spec(spec) == TritonAttentionBackend.customize_spec(spec)
    assert backend.supported_kv_cache_layouts() == (KVCacheLayout.LBHNC,)
    assert 512 in backend.get_supported_head_sizes()


# ---------------------------------------------------------------- prefill


def _prefill_reference(q, kv, block_table, seq_len, window=0, mm_ranges=()):
    """Float reference of a prefill; a token in a multimodal range [start, end]
    also sees the range's later keys."""
    m, hq, hkv, hd = q.shape[0], q.shape[1], kv.shape[1], q.shape[2]
    pages = kv[block_table.long()]
    flat = pages.transpose(1, 2).reshape(-1, hkv, 2 * hd)[:seq_len].float()
    gqa = hq // hkv
    pos = torch.arange(seq_len, device=q.device).view(1, seq_len)
    lim = (seq_len - m + torch.arange(m, device=q.device)).view(m, 1)
    masked = pos > lim
    for start, end in mm_ranges:
        q_in = (lim >= start) & (lim <= end)
        k_in = (pos >= start) & (pos <= end)
        masked &= ~(q_in & k_in)
    if window:
        masked |= pos < lim - (window - 1)
    out = torch.empty(m, hq, hd, device=q.device)
    for h0 in range(0, hq, 4):
        heads = range(h0, min(hq, h0 + 4))
        kvh = [h // gqa for h in heads]
        qf = q[:, h0 : h0 + len(heads)].float().permute(1, 0, 2)
        k = flat[:, kvh, :hd].permute(1, 0, 2)
        v = flat[:, kvh, hd:].permute(1, 0, 2)
        scores = torch.bmm(qf, k.transpose(1, 2)) * (hd**-0.5)
        scores = scores.masked_fill(masked.view(1, m, seq_len), float("-inf"))
        out[:, h0 : h0 + len(heads)] = torch.bmm(torch.softmax(scores, -1), v).permute(
            1, 0, 2
        )
    return out


def _mm_ranges_tensor(mm_ranges, device):
    """One sequence's ranges as TRITON_ATTN's (1, ranges, 2), padded with an
    empty range."""
    pairs = [*mm_ranges, (0, 0)]
    return torch.tensor([pairs], device=device, dtype=torch.int32)


def _triton_prefill(q, kv, block_table, seq_len, window=0, mm_ranges=()):
    """Triton's unified attention on the same cache."""
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    m, hd, hkv = q.shape[0], q.shape[2], kv.shape[1]
    k, v = kv.transpose(1, 2).split(hd, dim=-1)
    out = torch.empty_like(q)
    one = torch.ones(1, hkv, device=q.device)
    unified_attention(
        q,
        k,
        v,
        out,
        torch.tensor([0, m], device=q.device, dtype=torch.int32),
        m,
        torch.tensor([seq_len], device=q.device, dtype=torch.int32),
        seq_len,
        hd**-0.5,
        True,
        (window - 1, 0) if window else (-1, -1),
        block_table.view(1, -1),
        0,
        None,
        one,
        one,
        mm_prefix_range=(_mm_ranges_tensor(mm_ranges, q.device) if mm_ranges else None),
        mm_prefix_clamp_sliding_window=True,
    )
    return out


def _assert_prefill_close(got, q, kv, block_table, seq_len, window, what, mm_ranges=()):
    """Within twice Triton's error against the float reference."""
    ref = _prefill_reference(q, kv, block_table, seq_len, window, mm_ranges)
    err = (got.float().nan_to_num(1e9) - ref).abs().max().item()
    triton = _triton_prefill(q, kv, block_table, seq_len, window, mm_ranges)
    bound = 2 * (triton.float() - ref).abs().max().item() + 1e-4
    assert err <= bound, f"{what}: max error {err:.2e}, Triton's x2 {bound:.2e}"


def _prefill_inputs(v, m, seq_len, seed=0):
    """Q for m tokens and a shuffled HND cache of seq_len keys."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    num_blocks = -(-seq_len // v.page_size)
    kv = torch.randn(
        num_blocks + 2, v.num_kv_heads, v.page_size, 2 * v.head_size, device=dev
    )
    q = torch.randn(m, v.num_q_heads, v.head_size, device=dev)
    block_table = torch.randperm(num_blocks + 2, device=dev)[:num_blocks]
    return (
        (q * 0.5).to(v.dtype),
        (kv * 0.5).to(v.dtype),
        block_table.to(torch.int32).contiguous(),
    )


def _prefill_unit(v):
    dt = "bf16" if v.dtype == torch.bfloat16 else "fp16"
    return (
        f"prefill_q{v.num_q_heads}_kv{v.num_kv_heads}_d{v.head_size}_w{v.window}_{dt}"
    )


def _prefill_rows():
    return [v for rows in hip._prefill_table().values() for v in rows]


def test_prefill_configs_are_well_formed():
    """No two prefill builds share a shape and step, every decode configuration
    has prefill builds, and the build of the largest step reached runs."""
    if not _prefill_rows():
        pytest.skip("_rocm_C carries no RDNA3.5 variants")
    keys = {
        (v.num_q_heads, v.num_kv_heads, v.head_size, v.page_size, v.window)
        + (v.dtype, v.min_query_len)
        for v in _prefill_rows()
    }
    assert len(keys) == len(_prefill_rows()), "two builds share a step"
    configs = {
        (v.num_q_heads, v.num_kv_heads, v.head_size, v.window)
        for v in hip.variants()
        if v.page_size == 16
    }
    for (hq, hkv, d, window), dtype in itertools.product(configs, DTYPES):
        steps = hip.prefill_variants(hq, hkv, d, 16, window, dtype)
        assert steps, (hq, hkv, d, window, dtype)
        starts = [v.min_query_len for v in steps]
        assert starts == sorted(set(starts))
        for v in steps:
            for m in (v.min_query_len, v.min_query_len + 1):
                assert hip.prefill_variant_for(hq, hkv, d, 16, 1, window, dtype, m) == v
    v = steps[0]
    args = (v.num_q_heads, v.num_kv_heads, v.head_size, 16, 1, v.window, v.dtype)
    assert hip.prefill_variant_for(*args, hip.PREFILL_MIN_M - 1) is None
    assert hip.prefill_variant_for(*args, hip.PREFILL_MIN_M) == steps[0]
    assert hip.prefill_variant_for(*args, 1 << 20) == steps[-1]


def test_every_listed_prefill_variant_is_built_in():
    _skip_unless_gfx1151()
    # Raises if a build has no instantiation unit.
    assert torch.ops._rocm_C.rdna_singleseq_prefill_configs()
    missing = [v for v in _prefill_rows() if hip.load_prefill(v) is None]
    assert not missing, f"{len(missing)} prefill variants not in _rocm_C"


@pytest.mark.parametrize("unit", sorted({_prefill_unit(v) for v in _prefill_rows()}))
def test_built_in_prefill_variants_match_reference(unit):
    """Every built prefill variant matches the reference, with and without a
    prefix."""
    _skip_unless_gfx1151()
    for v in _prefill_rows():
        if _prefill_unit(v) != unit:
            continue
        module = hip.load_prefill(v)
        assert module is not None, v
        args = (v.num_q_heads, v.num_kv_heads, v.head_size, v.page_size, v.window)
        scratch = hip.make_prefill_scratch(
            hip.prefill_variants(*args, v.dtype), torch.device("cuda")
        )
        prefix = max(4096, 2 * v.window)
        cases = [(v.min_query_len, 0), (v.min_query_len + 37, prefix)]
        if v == hip.prefill_variants(*args, v.dtype)[0]:
            cases += [(hip.PREFILL_MIN_M, prefix), (hip.PREFILL_MIN_M + 50, 0)]
        for seed, (m, p) in enumerate(cases):
            seq_len = m + p
            q, kv, block_table = _prefill_inputs(v, m, seq_len, seed)
            out = torch.full_like(q, float("nan"))
            seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
            module.prefill_attn(
                q,
                kv,
                block_table,
                out,
                *scratch,
                seq_lens,
                seq_len,
                v.head_size**-0.5,
            )
            torch.accelerator.synchronize()
            _assert_prefill_close(
                out, q, kv, block_table, seq_len, v.window, f"{v}, M={m}, prefix={p}"
            )


def test_prefill_past_the_last_step():
    """M past the last step runs the last row."""
    _skip_unless_gfx1151()
    v = next(
        rows[-1]
        for rows in hip._prefill_table().values()
        if rows[-1].dtype == torch.float16 and rows[-1].head_size == 128
    )
    module = hip.load_prefill(v)
    args = (v.num_q_heads, v.num_kv_heads, v.head_size, v.page_size, v.window)
    scratch = hip.make_prefill_scratch(
        hip.prefill_variants(*args, v.dtype), torch.device("cuda")
    )
    for seed, (m, p) in enumerate([(8193, 0), (12000, 1000)]):
        seq_len = m + p
        q, kv, block_table = _prefill_inputs(v, m, seq_len, seed)
        out = torch.full_like(q, float("nan"))
        seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
        module.prefill_attn(
            q, kv, block_table, out, *scratch, seq_lens, seq_len, v.head_size**-0.5
        )
        torch.accelerator.synchronize()
        _assert_prefill_close(
            out, q, kv, block_table, seq_len, v.window, f"{v}, M={m}, prefix={p}"
        )


def test_prefill_window_never_reads_freed_pages():
    """Freed pages before the window do not reach the output."""
    _skip_unless_gfx1151()
    v = next(v for v in _prefill_rows() if v.window and v.dtype == torch.float16)
    module = hip.load_prefill(v)
    m, seq_len = v.min_query_len, v.min_query_len + 3 * v.window
    q, kv, block_table = _prefill_inputs(v, m, seq_len)
    clean = (kv.clone(), block_table.clone())
    first = (seq_len - m - (v.window - 1)) // v.page_size
    kv = torch.cat([kv, torch.full_like(kv[:1], float("nan"))])
    block_table = block_table.clone()
    block_table[:first] = kv.shape[0] - 1
    out = torch.empty_like(q)
    args = (v.num_q_heads, v.num_kv_heads, v.head_size, v.page_size, v.window)
    scratch = hip.make_prefill_scratch(hip.prefill_variants(*args, v.dtype), q.device)
    seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
    module.prefill_attn(
        q, kv, block_table, out, *scratch, seq_lens, seq_len, v.head_size**-0.5
    )
    torch.accelerator.synchronize()
    assert torch.isfinite(out).all()
    _assert_prefill_close(out, q, *clean, seq_len, v.window, "freed pages")


@pytest.mark.parametrize(
    "shape",
    [(32, 16, 256, 1024, torch.float16), (32, 4, 512, 0, torch.float16)]
    + [(32, 16, 256, 1024, torch.bfloat16)],
    ids=["swa_fp16", "global_fp16", "swa_bf16"],
)
@pytest.mark.parametrize(
    "m, prefix, mm_ranges",
    [
        (600, 0, [(40, 295), (400, 469)]),
        (700, 3000, [(3100, 3355)]),
        (300, 500, [(450, 560)]),
        # Longer than the 1024-key window: the window bounds earlier keys.
        (2048, 512, [(600, 2099)]),
    ],
    ids=["two_images", "split_kv", "from_prefix", "longer_than_window"],
)
def test_prefill_mm_prefix_matches_reference(shape, m, prefix, mm_ranges):
    """Tokens of a multimodal range see the whole range (Gemma4's shapes)."""
    _skip_unless_gfx1151()
    hq, hkv, d, window, dtype = shape
    v = hip.prefill_variant_for(hq, hkv, d, 16, 1, window, dtype, m)
    assert v is not None, shape
    module = hip.load_prefill(v)
    assert module is not None, v
    scratch = hip.make_prefill_scratch(
        hip.prefill_variants(hq, hkv, d, 16, window, dtype), torch.device("cuda")
    )
    seq_len = m + prefix
    q, kv, block_table = _prefill_inputs(v, m, seq_len)
    out = torch.full_like(q, float("nan"))
    seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
    module.prefill_attn(
        q,
        kv,
        block_table,
        out,
        *scratch,
        seq_lens,
        seq_len,
        d**-0.5,
        _mm_ranges_tensor(mm_ranges, q.device)[0],
    )
    torch.accelerator.synchronize()
    _assert_prefill_close(
        out, q, kv, block_table, seq_len, window, f"{v}, M={m}", mm_ranges
    )


@pytest.mark.parametrize("bad", ["batch", "triples", "cpu"])
def test_prefill_refuses_malformed_mm_ranges(bad):
    """Only one sequence's (ranges, 2) on the device reaches the kernel: the
    batch's (sequences, ranges, 2) would unmask other sequences' keys."""
    _skip_unless_gfx1151()
    v = hip.prefill_variant_for(32, 16, 256, 16, 1, 1024, torch.float16, 64)
    module = hip.load_prefill(v)
    scratch = hip.make_prefill_scratch([v], torch.device("cuda"))
    q, kv, block_table = _prefill_inputs(v, 64, 64)
    good = _mm_ranges_tensor([(10, 40)], q.device)
    ranges = {
        "batch": good,
        "triples": torch.zeros(2, 3, dtype=torch.int32, device=q.device),
        "cpu": good[0].cpu(),
    }[bad]
    seq_lens = torch.tensor([64], device=q.device, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="mm_ranges must be"):
        module.prefill_attn(
            q, kv, block_table, torch.empty_like(q), *scratch, seq_lens, 64,
            256**-0.5, ranges,
        )  # fmt: skip


def _decode_mm_buf(ranges, device, capacity=4):
    """The decode kernel's [count, start0, end0, ...], empty slots zeroed."""
    buf = torch.zeros(1 + 2 * capacity, dtype=torch.int32)
    flat = [len(ranges), *(x for r in ranges for x in r)]
    buf[: len(flat)] = torch.tensor(flat, dtype=torch.int32)
    return buf.to(device)


def _decode_mm(module, v, q, kv, block_table, seq_len, buf):
    out = torch.full_like(q, float("nan"))
    seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
    module.decode_attn(
        q, kv, block_table, out, *hip.make_scratch(v, q.device), seq_lens,
        q.shape[2] ** -0.5, buf,
    )  # fmt: skip
    torch.accelerator.synchronize()
    return out.float().nan_to_num(1e9)


@pytest.mark.parametrize(
    "want, min_m",
    [
        (
            {
                "window": bool,
                "dot_product": lambda x: not x,
                "max_segments": lambda x: x > 1,
            },
            4,
        ),
        (
            {
                "window": lambda x: not x,
                "dot_product": lambda x: not x,
                "max_segments": lambda x: x > 1,
            },
            4,
        ),
        (
            {
                "window": lambda x: not x,
                "dot_product": lambda x: not x,
                "max_segments": lambda x: x == 1,
            },
            4,
        ),
        # Dot-product builds go up to M=3; an image starting among M=2's query
        # tokens would only hold the last one.
        ({"dot_product": bool}, 3),
    ],
    ids=["wmma_window segments", "wmma segments", "wmma", "dot_product"],
)
@pytest.mark.parametrize(
    "image",
    [
        # Over every query token, beside a past image.
        lambda s, m, w: [(100, 355), (s - m - 40, s - 1)],
        # Ending, or starting, among the query tokens: the rows after it, or
        # before it, stay causal.
        lambda s, m, w: [(s - m - 40, s - m + m // 2)],
        lambda s, m, w: [(s - m + m // 2, s - 1)],
        # Past the sequence, as in a chunk ending inside the image.
        lambda s, m, w: [(s - m - 40, s + 200)],
        # Longer than the window, which still bounds earlier keys.
        lambda s, m, w: [(s - (w or 1000) - 500, s - 1)],
        # Two images over the query tokens: rows end at either one.
        lambda s, m, w: [(s - m - 40, s - m + 1), (s - m + 2, s - 1)],
    ],
    ids=[
        "covers_all",
        "ends_mid",
        "starts_mid",
        "past_seq_len",
        "longer_than_window",
        "two_in_span",
    ],
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_decode_mm_prefix_matches_reference(want, min_m, image, dtype):
    """Decode-sized calls inside an image see the image's later keys, and the
    count comes from device memory."""
    v, module = _built(dtype, max_query_len=lambda m: m >= min_m, **want)
    m, seq_len = v.max_query_len, max(2000, v.window + 1000) + 7
    q, kv, block_table = _paged_inputs(seq_len, v)
    # Keys past the sequence must not be read, even inside an image.
    kv[-1, :, seq_len % v.page_size :] = float("nan")
    ranges = image(seq_len, m, v.window)
    got = _decode_mm(
        module, v, q, kv, block_table, seq_len, _decode_mm_buf(ranges, q.device)
    )
    ref = _reference(q, kv, seq_len, v.window, ranges)
    assert _max_rel(got, ref) <= TOL[v.dtype], v.name
    causal = _reference(q, kv, seq_len, v.window)
    assert _max_rel(causal, ref) > 10 * TOL[v.dtype], "the image changed nothing"
    # With several images over the query tokens, each one shows.
    cut = [r for r in ranges if r[0] < r[1] and r[1] >= seq_len - m]
    for r in cut if len(cut) > 1 else []:
        rest = [x for x in ranges if x != r]
        without = _reference(q, kv, seq_len, v.window, rest)
        assert _max_rel(without, ref) > 10 * TOL[v.dtype], f"{r} changed nothing"


@pytest.mark.parametrize("case", ["zero", "over_capacity"])
def test_decode_mm_ranges_stop_at_count_and_capacity(case):
    """The kernel reads min(count, capacity) ranges: those it reads apply, and
    a range past them, over every query token, changes nothing."""
    v, module = _built(
        window=lambda x: not x,
        dot_product=lambda x: not x,
        max_query_len=lambda m: m >= 4,
    )
    m, seq_len = v.max_query_len, 2000
    q, kv, block_table = _paged_inputs(seq_len, v)
    trap = (seq_len - m - 40, seq_len - 1)
    if case == "zero":
        # The trap in the buffer, past count.
        ranges, count, size = [], 0, 3
    else:
        # count 5 in a 2-range buffer, the trap just past its end. The second
        # range ends among the query tokens, so the trap would extend it.
        ranges = [(100, 355), (seq_len - m - 40, seq_len - m + m // 2)]
        count, size = 5, 5
    flat = [count, *(x for r in ranges for x in r), *trap]
    buf = torch.tensor(flat, dtype=torch.int32, device=q.device)[:size]
    got = _decode_mm(module, v, q, kv, block_table, seq_len, buf)
    ref = _reference(q, kv, seq_len, 0, ranges)
    assert _max_rel(got, ref) <= TOL[v.dtype], v.name
    # Reading the trap, or ignoring the ranges that fit, must show.
    for wrong in (ranges + [trap], []):
        if wrong != ranges:
            wrong_ref = _reference(q, kv, seq_len, 0, wrong)
            assert _max_rel(wrong_ref, ref) > 10 * TOL[v.dtype], wrong


def test_graph_replay_follows_mm_ranges():
    """A captured graph reads the ranges the builder rewrites in place each
    step, and none are left over when they go back to empty."""
    v, module = _built(
        window=lambda x: not x,
        dot_product=lambda x: not x,
        max_query_len=lambda m: m >= 4,
    )
    seq_len = 1024
    q, kv, block_table = _paged_inputs(seq_len, v)
    scratch = hip.make_scratch(v, q.device)
    out = torch.empty_like(q)
    seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
    buf = _decode_mm_buf([], q.device)

    def launch():
        module.decode_attn(
            q, kv, block_table, out, *scratch, seq_lens, q.shape[2] ** -0.5, buf
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    m = v.max_query_len
    for ranges in ([], [(seq_len - m - 40, seq_len - 1)], []):
        # As the builder: a smaller count leaves earlier ranges past it.
        flat = _decode_mm_buf(ranges, q.device)[: 1 + 2 * len(ranges)]
        buf[: flat.numel()].copy_(flat)
        out.zero_()
        graph.replay()
        torch.accelerator.synchronize()
        ref = _reference(q, kv, seq_len, 0, ranges)
        assert _max_rel(out.float(), ref) <= TOL[v.dtype], f"ranges={ranges}"


@pytest.mark.parametrize("bad", ["even", "int64", "cpu"])
def test_decode_refuses_malformed_mm_ranges(bad):
    """Only an odd-length int32 [count, start0, end0, ...] on the device
    reaches the kernel, which sizes its capacity from it."""
    v, module = _built()
    q, kv, block_table = _paged_inputs(64, v)
    good = _decode_mm_buf([(10, 40)], q.device)
    ranges = {
        "even": good[:4],
        "int64": good.long(),
        "cpu": good.cpu(),
    }[bad]
    seq_lens = torch.tensor([64], device=q.device, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="mm_ranges must be"):
        module.decode_attn(
            q, kv, block_table, torch.empty_like(q), *hip.make_scratch(v, q.device),
            seq_lens, q.shape[2] ** -0.5, ranges,
        )  # fmt: skip


@pytest.mark.parametrize(
    "ranges, seq_len, query_len, expected",
    [
        # Decode after the image: no query token in a range.
        ([(100, 355)], 500, 1, (False, -1)),
        # The image's last token recomputed alone: no later keys.
        ([(100, 355)], 356, 1, (False, 255)),
        # A short call inside a long image (e.g. after a prefix-cache hit).
        ([(100, 355)], 300, 5, (True, 199)),
        # A short image long past.
        ([(10, 14)], 500, 1, (False, -1)),
        # An empty range over the query tokens.
        ([(495, 495)], 500, 8, (False, -1)),
        # Text after a 2000-token image.
        ([(0, 1999)], 2200, 100, (False, -1)),
        # A prefill chunk ending inside the image.
        ([(100, 355)], 200, 150, (True, 99)),
    ],
    ids=[
        "after",
        "last_token",
        "short_call_inside",
        "short_past",
        "empty",
        "text_after_long",
        "prefill_into",
    ],  # fmt: skip
)
def test_mm_query_overlap(ranges, seq_len, query_len, expected):
    """Only ranges holding the step's query tokens count."""
    import types

    from vllm.v1.attention.backends.rocm_rdna_singleseq_attn import mm_query_overlap

    md = types.SimpleNamespace(
        mm_req_doc_ranges={0: ranges},
        seq_lens_cpu_upper_bound=torch.tensor([seq_len]),
        query_start_loc_cpu=torch.tensor([0, query_len]),
    )
    assert mm_query_overlap(md) == expected


# 40 ten-token images, [100 + 20i, 109 + 20i]: more than MM_RANGES_CAPACITY.
_MANY = [(100 + 20 * i, 109 + 20 * i) for i in range(40)]


@pytest.mark.parametrize(
    "ranges, seq_len, query_len, expected",
    [
        # Query tokens 202..204, inside image 5.
        (_MANY, 205, 3, [(200, 209)]),
        # Text after the images.
        (_MANY, 2000, 3, []),
        # No sequence lengths: every range, which do not fit.
        (_MANY, None, 3, None),
        # Query tokens 98..99, widened to 97 for rejected drafts: a range ending
        # at 97 or starting at 99 is kept; one ending at 96, starting at 100 or
        # empty is not.
        ([(80, 96), (90, 97), (98, 98), (99, 120), (100, 110)], 100, 2,
         [(90, 97), (99, 120)]),
    ],
    ids=["inside_one", "text_after", "no_seq_lens", "edges"],
)  # fmt: skip
def test_mm_ranges_decode_buffer(ranges, seq_len, query_len, expected):
    """The decode buffer holds only the ranges that can hold the step's query
    tokens, so any number of images fits; when they do not fit it is emptied,
    so a replayed CUDA graph never reads an earlier step's ranges."""
    import types

    from vllm.v1.attention.backends.rocm_rdna_singleseq_attn import (
        RocmRdnaSingleSeqMetadataBuilder,
        mm_ranges_buffer,
    )

    device = torch.device("cpu")
    buf = mm_ranges_buffer(device)
    buf.fill_(7)
    md = types.SimpleNamespace(
        mm_req_doc_ranges={0: ranges},
        seq_lens_cpu_upper_bound=None if seq_len is None else torch.tensor([seq_len]),
        query_start_loc_cpu=torch.tensor([0, query_len]),
    )
    out = RocmRdnaSingleSeqMetadataBuilder._mm_ranges_decode(
        types.SimpleNamespace(device=device), md
    )
    if expected is None:
        assert out is None
        assert buf[0] == 0
    else:
        flat = [x for span in expected for x in span]
        assert buf[: 1 + len(flat)].tolist() == [len(expected), *flat]
        assert out is buf


@pytest.mark.parametrize("overshoot", [False, True], ids=["exact", "async_spec"])
def test_mm_ranges_decode_builder_feeds_kernel(overshoot):
    """The builder's buffer gives the kernel the mask of all the request's
    ranges, also when seq_lens_cpu_upper_bound overshoots by M - 1 rejected
    drafts: the image then ends before the estimated query tokens."""
    import types

    from vllm.v1.attention.backends.rocm_rdna_singleseq_attn import (
        RocmRdnaSingleSeqMetadataBuilder,
    )

    v, module = _built(
        window=lambda x: not x,
        dot_product=lambda x: not x,
        max_query_len=lambda m: m >= 4,
    )
    m, seq_len = v.max_query_len, 2007
    q, kv, block_table = _paged_inputs(seq_len, v)
    # 40 past images, more than the buffer holds, and one ending among the
    # query tokens.
    ranges = [(10 + 40 * i, 30 + 40 * i) for i in range(40)]
    ranges.append((seq_len - m - 40, seq_len - 2))
    md = types.SimpleNamespace(
        mm_req_doc_ranges={0: ranges},
        seq_lens_cpu_upper_bound=torch.tensor([seq_len + overshoot * (m - 1)]),
        query_start_loc_cpu=torch.tensor([0, m]),
    )
    buf = RocmRdnaSingleSeqMetadataBuilder._mm_ranges_decode(
        types.SimpleNamespace(device=q.device), md
    )
    assert buf is not None
    got = _decode_mm(module, v, q, kv, block_table, seq_len, buf)
    ref = _reference(q, kv, seq_len, 0, ranges)
    assert _max_rel(got, ref) <= TOL[v.dtype], v.name
    causal = _reference(q, kv, seq_len)
    assert _max_rel(causal, ref) > 10 * TOL[v.dtype], "the image changed nothing"


def test_forward_passes_mm_ranges_to_the_decode_kernel():
    """Forward hands the builder's buffer to the decode kernel."""
    import types

    from vllm.v1.attention.backends.rocm_rdna_singleseq_attn import (
        RocmRdnaSingleSeqMetadata,
    )

    impl = _impl()
    impl.scale, impl.logits_soft_cap, impl.kernel_calls = HEAD_DIM**-0.5, 0, 0
    impl.sliding_window = impl.alibi_slopes = impl.sinks = None
    impl.chunk_lookback = impl._kv_quant_mode = None
    passed = []
    module = types.SimpleNamespace(decode_attn=lambda *args: passed.append(args[-1]))
    impl._prepare = lambda kv_cache, **kwargs: (module, (None,) * 4)
    buf = torch.zeros(3, dtype=torch.int32)
    md = RocmRdnaSingleSeqMetadata.__new__(RocmRdnaSingleSeqMetadata)
    for name, value in {
        "num_actual_tokens": 2,
        "seq_lens": torch.tensor([100]),
        "block_table": None,
        "max_query_len": 2,
        "causal": True,
        "mm_prefix_range_tensor": None,
        "mm_future_keys": True,
        "mm_reach": 4,
        "mm_ranges_decode": buf,
        "rswa_prefix_lens": None,
    }.items():
        setattr(md, name, value)
    q = torch.zeros(2, HQ, HEAD_DIM, dtype=torch.float16)
    impl.forward(types.SimpleNamespace(), q, q, q, torch.empty(0), md, q.clone())
    assert len(passed) == 1 and passed[0] is buf


@pytest.mark.parametrize(
    "future_keys, ranges, reach, clamp, reason",
    [
        (True, False, 4, True, "do not fit"),
        (True, True, 4, True, None),
        (False, False, 1500, False, "past the window"),
        (False, False, 1500, True, None),
        (False, False, 1000, False, None),
    ],
    ids=[
        "later_keys_no_buffer",
        "later_keys_buffer",
        "unclamped_reach",
        "clamped_reach",
        "within_window",
    ],  # fmt: skip
)
def test_mm_ranges_decode_decision(future_keys, ranges, reach, clamp, reason):
    """Decode runs Triton when a query token's range has later keys, or when an
    unclamped range reaches past the window; otherwise the ranges do not
    stop it."""
    from vllm.v1.kv_cache_interface import KVQuantMode

    impl = _impl()
    impl._rejected = None
    impl.head_size, impl.num_heads, impl.num_kv_heads = HEAD_DIM, HQ, HKV
    fits = impl._prepare(
        kv_cache=torch.empty(0, dtype=torch.float16),
        q=torch.empty(1, HQ, HEAD_DIM, dtype=torch.float16),
        out=torch.empty(1, HQ, HEAD_DIM, dtype=torch.float16),
        alibi_slopes=None,
        sinks=None,
        softcap=0,
        causal=True,
        window_size=(1023, 0),
        kv_quant_mode=KVQuantMode.NONE,
        seqused_k=torch.zeros(1),
        max_seqlen_q=2,
        mm_future_keys=future_keys,
        mm_reach=reach,
        mm_prefix_clamp=clamp,
        mm_ranges_decode=torch.zeros(3, dtype=torch.int32) if ranges else None,
    )
    assert not fits
    if reason is None:
        # Refused later, on the tensors, not for the ranges.
        assert "multimodal" not in impl._rejected
    else:
        assert reason in impl._rejected
