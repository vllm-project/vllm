# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing, fallback and accuracy tests for the ROCM_AITER_FA FP8 prefill path."""

import functools
import importlib
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm AITER FP8 prefill tests", allow_module_level=True)

from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms.rocm import get_cdna_version

pytest.importorskip("aiter.ops.triton.attention.mha")
from vllm.v1.attention.backends.rocm_aiter_fa import (
    _CP_TOKENS_PER_ITER_ROCM as WORKSPACE_ROWS,
)

gfx942 = pytest.mark.skipif(get_cdna_version() != 3, reason="gfx942 only")
pytestmark = pytest.mark.usefixtures("workspace_init")

HQ, HKV, HD = 24, 4, 256
SEQ_LENS = [37, 64, 19]

FP8 = current_platform.fp8_dtype()
FP8_MAX = torch.finfo(FP8).max


@pytest.fixture
def fp8():
    mod = importlib.import_module(
        "vllm.v1.attention.backends.rocm_aiter_triton_fp8_prefill"
    )
    mod._load_kernel.cache_clear()
    yield mod
    mod._load_kernel.cache_clear()


@pytest.fixture
def kernel(monkeypatch):
    import aiter.ops.triton.attention.mha as mha

    calls = []

    @functools.wraps(mha._flash_attn_forward)
    def fake(q, k, v, **kw):
        calls.append(SimpleNamespace(q=q, k=k, v=v, **kw))
        o = torch.full(q.shape[:-1] + v.shape[-1:], 0.5, dtype=torch.float32)
        lse = torch.arange(q.shape[0] * q.shape[1], dtype=torch.float32).view(
            q.shape[0], q.shape[1]
        )
        return o.to(q.device), lse.to(q.device), None, 0, 0

    monkeypatch.setattr(mha, "_flash_attn_forward", fake)
    return calls


class _Orig:
    def __init__(self):
        self.calls = []
        self.result = object()

    def __call__(self, **kw):
        self.calls.append(kw)
        return self.result


def _cu(seq_lens):
    return torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32
    ).cuda()


def _inputs(
    hq=HQ,
    hkv=HKV,
    hd=HD,
    dtype=torch.bfloat16,
    k_dtype=None,
    v_heads=None,
    k_hd=None,
    seq_lens=SEQ_LENS,
):
    t = sum(seq_lens)
    cu = _cu(seq_lens)
    q = torch.randn(t, hq, hd, device="cuda").to(dtype)
    k = torch.randn(t, hkv, k_hd or hd, device="cuda").to(k_dtype or dtype)
    v = torch.randn(t, v_heads or hkv, k_hd or hd, device="cuda").to(k_dtype or dtype)
    return dict(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max(seq_lens),
        max_seqlen_k=max(seq_lens),
        min_seqlen_q=1,
        dropout_p=0.0,
        softmax_scale=hd**-0.5,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        return_lse=True,
        sink_ptr=None,
    )


def _assert_same_args(call, args):
    assert call.keys() == args.keys()
    for name, value in args.items():
        assert call[name] is value, name


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize(
    "switch, kv_cache_dtype, head_size, plain, wanted",
    [
        (True, "fp8", HD, True, True),
        (True, "fp8_e4m3", HD, True, True),
        (False, "fp8", HD, True, False),
        (True, "auto", HD, True, False),
        (True, "fp8_e5m2", HD, True, False),
        (True, "fp8", 128, True, False),
        (True, "fp8", HD, False, False),
    ],
)
def test_switch_selects_the_path(fp8, switch, kv_cache_dtype, head_size, plain, wanted):
    on = wanted and get_cdna_version() == 3
    assert fp8.enabled(switch, head_size, kv_cache_dtype, plain) is on
    assert fp8._load_kernel.cache_info().currsize == int(on)


@pytest.mark.parametrize("config", ["on", "off", "none"])
def test_impl_reads_the_attention_config(monkeypatch, fp8, config):
    from vllm.config import AttentionConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends import rocm_aiter_fa

    seen = []

    def fake_enabled(*a):
        seen.append(a)
        return a[0]

    monkeypatch.setattr(fp8, "enabled", fake_enabled)

    def make():
        return rocm_aiter_fa.AiterFlashAttentionImpl(
            num_heads=HQ,
            head_size=HD,
            scale=HD**-0.5,
            num_kv_heads=HKV,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8",
        )

    if config == "none":
        impl = make()
    else:
        switch = AttentionConfig(use_prefill_query_quantization=config == "on")
        with set_current_vllm_config(VllmConfig(attention_config=switch)):
            impl = make()
    assert seen == [(config == "on", HD, "fp8", True)]
    assert impl.fp8_prefill is (config == "on")
    if impl.fp8_prefill:
        assert impl._prefill_attn.func is fp8.triton_fp8_prefill_attn
        # Every other test passes orig= explicitly, so this is the only check
        # that an ineligible call falls back to CK in production.
        assert impl._prefill_attn.keywords == {
            "orig": rocm_aiter_ops.flash_attn_varlen_func
        }
    else:
        assert impl._prefill_attn is rocm_aiter_ops.flash_attn_varlen_func


@gfx942
@pytest.mark.parametrize("layer", ["plain", "alibi", "sliding_window", "sinks"])
def test_layers_triton_cannot_serve_stay_on_ck(fp8, layer):
    from vllm.config import AttentionConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends import rocm_aiter_fa

    kwargs: dict[str, Any] = dict(alibi_slopes=None, sliding_window=None)
    if layer == "alibi":
        kwargs["alibi_slopes"] = [1.0] * HQ
    elif layer == "sliding_window":
        kwargs["sliding_window"] = 128
    elif layer == "sinks":
        kwargs["sinks"] = torch.zeros(HQ, device="cuda")
    switch = AttentionConfig(use_prefill_query_quantization=True)
    with set_current_vllm_config(VllmConfig(attention_config=switch)):
        impl = rocm_aiter_fa.AiterFlashAttentionImpl(
            num_heads=HQ,
            head_size=HD,
            scale=HD**-0.5,
            num_kv_heads=HKV,
            kv_cache_dtype="fp8",
            **kwargs,
        )
    assert impl.fp8_prefill is (layer == "plain")
    if not impl.fp8_prefill:
        assert impl._prefill_attn is rocm_aiter_ops.flash_attn_varlen_func


ACCEPT = {
    "gqa_bf16": {},
    "gqa_fp16": dict(make=dict(dtype=torch.float16)),
    "mha_equal_heads": dict(make=dict(hq=8, hkv=8)),
    "mqa_single_kv_head": dict(make=dict(hq=8, hkv=1)),
    "non_causal": dict(call=dict(causal=False)),
    "non_causal_separate_offsets": dict(call=dict(causal=False), separate_k=True),
    "no_lse": dict(call=dict(return_lse=False)),
    "window_none": dict(call=dict(window_size=None)),
    "single_sequence": dict(make=dict(seq_lens=[129])),
}

DECLINE: dict[str, dict[str, Any]] = {
    "q_not_3d": dict(q_2d=True),
    "head_dim_128": dict(make=dict(hd=128)),
    "kv_head_dim_mismatch": dict(make=dict(k_hd=128)),
    "dtype_fp32": dict(make=dict(dtype=torch.float32)),
    "kv_dtype_differs": dict(make=dict(k_dtype=torch.float16)),
    "kv_already_fp8": dict(make=dict(k_dtype=current_platform.fp8_dtype())),
    "heads_not_divisible": dict(make=dict(hq=24, hkv=5)),
    "k_v_head_count_differs": dict(make=dict(v_heads=8)),
    "empty_k": dict(empty_k=True),
    "cu_seqlens_int64": dict(cu_int64=True),
    "alibi": dict(call=dict(alibi_slopes=torch.ones(HQ))),
    "sink": dict(call=dict(sink_ptr=torch.zeros(HQ))),
    "sliding_window": dict(call=dict(window_size=(127, 0))),
    "dropout": dict(call=dict(dropout_p=0.1)),
    "causal_separate_offsets": dict(separate_k=True),
}


def _case(spec):
    args = _inputs(**spec.get("make", {}))
    for name, value in spec.get("call", {}).items():
        args[name] = value.cuda() if isinstance(value, torch.Tensor) else value
    if spec.get("separate_k"):
        args["cu_seqlens_k"] = args["cu_seqlens_q"].clone()
    if spec.get("q_2d"):
        args["q"] = args["q"].flatten(1)
    if spec.get("empty_k"):
        args.update(
            k=args["k"][:0],
            v=args["v"][:0],
            causal=False,
            cu_seqlens_k=torch.zeros_like(args["cu_seqlens_q"]),
        )
    if spec.get("cu_int64"):
        args["cu_seqlens_q"] = args["cu_seqlens_k"] = args["cu_seqlens_q"].long()
    return args


@pytest.mark.parametrize("name", sorted(ACCEPT))
def test_eligible_calls_reach_triton(fp8, kernel, name):
    args = _case(ACCEPT[name])
    orig = _Orig()
    fp8.triton_fp8_prefill_attn(orig=orig, **args)
    assert orig.calls == []
    assert len(kernel) == 1
    assert kernel[0].q.dtype == kernel[0].k.dtype == kernel[0].v.dtype
    assert kernel[0].q.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn)
    assert kernel[0].causal is bool(args["causal"])
    assert kernel[0].config == fp8._TUNED_CONFIG
    assert kernel[0].max_seqlen_q % 16 == 0 and kernel[0].max_seqlen_k % 16 == 0


@pytest.mark.parametrize("name", sorted(DECLINE))
def test_ineligible_calls_fall_back_unchanged(fp8, kernel, name):
    args = _case(DECLINE[name])
    orig = _Orig()
    assert fp8.triton_fp8_prefill_attn(orig=orig, **args) is orig.result
    assert kernel == []
    assert len(orig.calls) == 1
    _assert_same_args(orig.calls[0], args)


@pytest.mark.parametrize("failure", ["import_error", "signature_drift", "dao_ai_impl"])
def test_aiter_api_drift_keeps_ck(monkeypatch, fp8, failure):
    import aiter.ops.triton.attention.mha as mha

    if failure == "import_error":
        monkeypatch.setitem(sys.modules, "aiter.ops.triton.attention.mha", None)
    elif failure == "signature_drift":

        def old_api(q, k, v, dropout_p, softmax_scale, causal):
            raise AssertionError("an incompatible kernel must not be launched")

        monkeypatch.setattr(mha, "_flash_attn_forward", old_api)
    else:
        monkeypatch.setattr(mha, "_MHA_IMPL", "dao_ai")

    assert fp8._load_kernel() is None
    assert fp8.enabled(True, HD, "fp8", True) is False


def test_kernel_errors_are_not_swallowed(monkeypatch, fp8):
    """The path is opt-in, so a launch or OOM failure surfaces instead of
    silently running the call a second time on CK. ``profile`` reserves the
    FP8 buffers so the OOM is accounted for rather than caught."""
    import aiter.ops.triton.attention.mha as mha

    @functools.wraps(mha._flash_attn_forward)
    def broken(*a, **kw):
        raise torch.OutOfMemoryError("HIP OOM")

    monkeypatch.setattr(mha, "_flash_attn_forward", broken)
    orig = _Orig()
    with pytest.raises(torch.OutOfMemoryError):
        fp8.triton_fp8_prefill_attn(orig=orig, **_inputs())
    assert orig.calls == []


def _fp8_kv(args):
    for i, name in enumerate(("k", "v")):
        args[name] = args[name].to(FP8)
        # One scale per tensor, as the cache holds it.
        args[f"descale_{name}"] = torch.full((1,), 0.5 + i, device="cuda")
    return args


def test_fp8_kv_is_passed_through(fp8, kernel):
    args = _fp8_kv(_inputs())
    fp8.triton_fp8_prefill_attn(orig=_Orig(), **args)
    (call,) = kernel
    batch = args["cu_seqlens_q"].numel() - 1
    for name in ("k", "v"):
        assert getattr(call, name) is args[name]
        descale = getattr(call, f"descale_{name}")
        assert descale.shape == (batch, HKV)
        assert bool((descale == args[f"descale_{name}"]).all())


def test_ineligible_fp8_kv_raises(fp8, kernel):
    args = _fp8_kv(_inputs())
    args["alibi_slopes"] = torch.ones(HQ, device="cuda")
    orig = _Orig()
    with pytest.raises(ValueError, match="alibi"):
        fp8.triton_fp8_prefill_attn(orig=orig, **args)
    assert orig.calls == [] and kernel == []


def test_profile_run_covers_a_full_prefill(fp8, kernel):
    from vllm.v1.worker.workspace import lock_workspace

    args = _inputs()
    t = args["q"].shape[0]
    # A narrower profile query than the prefill, as microbatching produces.
    fp8.profile(args["q"][: t // 2], HQ, HKV, t, t)
    lock_workspace()
    fp8.triton_fp8_prefill_attn(orig=_Orig(), **args)
    assert len(kernel) == 1


def test_profile_run_short_of_the_prefill_is_caught(fp8, kernel):
    """A reservation the prefill outgrows must fail, so the test above can pass."""
    from vllm.v1.worker.workspace import lock_workspace

    args = _inputs()
    fp8.profile(args["q"], HQ, HKV, 1, 1)
    lock_workspace()
    with pytest.raises(AssertionError):
        fp8.triton_fp8_prefill_attn(orig=_Orig(), **args)


def test_unexpected_kwarg_falls_back_unchanged(fp8, kernel):
    args = _inputs()
    args["softcap"] = 30.0
    orig = _Orig()
    assert fp8.triton_fp8_prefill_attn(orig=orig, **args) is orig.result
    assert kernel == []
    _assert_same_args(orig.calls[0], args)


def test_descales_are_one_value_broadcast_to_batch_and_head(fp8, kernel):
    """Descales are read at [b, head] with an implicit unit head stride, so the
    per-tensor scale is materialised to that shape rather than passed as a
    scalar."""
    fp8.triton_fp8_prefill_attn(orig=_Orig(), **_inputs())
    (call,) = kernel
    batch = len(SEQ_LENS)
    assert tuple(call.descale_q.shape) == (batch, HQ)
    assert tuple(call.descale_k.shape) == (batch, HKV)
    assert tuple(call.descale_v.shape) == (batch, HKV)
    for d in (call.descale_q, call.descale_k, call.descale_v):
        assert d.dtype == torch.float32
        assert d.is_contiguous()
        assert d.unique().numel() == 1


def test_softmax_scale_none_default(fp8, kernel):
    args = _inputs()
    args["softmax_scale"] = None
    fp8.triton_fp8_prefill_attn(orig=_Orig(), **args)
    assert kernel[0].softmax_scale == HD**-0.5


def test_lse_is_returned_heads_by_tokens(fp8, kernel):
    args = _inputs()
    o, lse = fp8.triton_fp8_prefill_attn(orig=_Orig(), **args)
    t = args["q"].shape[0]
    assert o.dtype == args["q"].dtype
    assert tuple(o.shape) == (t, HQ, HD)
    assert tuple(lse.shape) == (HQ, t)
    expected = torch.arange(t * HQ, dtype=torch.float32, device=lse.device)
    torch.testing.assert_close(lse, expected.view(t, HQ).t())


def test_out_is_written_in_place(fp8, kernel):
    args = _inputs()
    args["return_lse"] = False
    out = torch.zeros_like(args["q"])
    res = fp8.triton_fp8_prefill_attn(orig=_Orig(), out=out, **args)
    assert res is out
    assert bool((out == 0.5).all())


def _use_fp8(fp8, impl, on=True):
    """Resolve the state the constructor reaches on gfx942.

    ``enabled()`` refuses off gfx942, so the backend tests set the outcome
    directly and let ``test_impl_reads_the_attention_config`` cover the real
    constructor path.
    """
    impl.fp8_prefill = on
    impl._prefill_attn = (
        fp8.prefill_attn() if on else rocm_aiter_ops.flash_attn_varlen_func
    )


def _make_impl(monkeypatch, fp8, on, shuffle=False):
    from vllm.v1.attention.backends import rocm_aiter_fa

    monkeypatch.setattr(
        rocm_aiter_fa.rocm_aiter_ops, "is_shuffle_kv_cache_enabled", lambda: shuffle
    )
    # An FP8 cache is what enabled() requires, and it is what decides the
    # dequant flag the context gather is given.
    impl = rocm_aiter_fa.AiterFlashAttentionImpl(
        num_heads=HQ,
        head_size=HD,
        scale=HD**-0.5,
        num_kv_heads=HKV,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8",
    )
    _use_fp8(fp8, impl, on)
    return impl


def _run_backend_forward(
    monkeypatch, fp8, on, workspace=None, ctx=48, stub_merge=True, shuffle=False
):
    from vllm.v1.attention.backends import rocm_aiter_fa

    gathered = []

    def gather(**kw):
        gathered.append(kw)

    monkeypatch.setattr(rocm_aiter_fa, "cp_mha_gather_cache", gather)
    if stub_merge:
        monkeypatch.setattr(rocm_aiter_fa, "merge_attn_states", lambda **kw: None)

    impl = _make_impl(monkeypatch, fp8, on, shuffle)
    seen = []
    attn = impl._prefill_attn

    def spy(**kw):
        seen.append(kw)
        return attn(**kw)

    impl._prefill_attn = spy

    n_ext, n_pre = 16, 32
    t = n_ext + n_pre
    g = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=g).bfloat16()

    def i32(*xs):
        return torch.tensor(xs, dtype=torch.int32, device="cuda")

    if workspace is None:
        workspace = randn(2, ctx, HKV, HD)
    chunk = SimpleNamespace(
        num_chunks=1,
        workspace=workspace,
        cu_seq_lens_chunk=[i32(0, ctx)],
        max_seq_lens=[ctx],
        chunk_starts=[i32(0)],
        token_to_batch=[torch.zeros(ctx, dtype=torch.int32, device="cuda")],
        total_token_per_batch=[ctx],
        swa_metadata=None,
    )
    md = SimpleNamespace(
        num_actual_tokens=t,
        num_decodes=0,
        num_prefills=1,
        num_extends=1,
        num_decode_tokens=0,
        num_extend_tokens=n_ext,
        use_cascade=False,
        causal=True,
        block_table=torch.zeros(2, 4, dtype=torch.int32, device="cuda"),
        slot_mapping=torch.arange(t, device="cuda"),
        prefill_metadata=SimpleNamespace(
            query_start_loc=i32(0, n_pre), max_query_len=n_pre, max_seq_len=n_pre
        ),
        extend_metadata=SimpleNamespace(
            query_start_loc=i32(0, n_ext),
            max_query_len=n_ext,
            max_seq_len=n_ext + ctx,
            chunk_context_metadata=chunk,
        ),
    )
    layer = SimpleNamespace(
        _k_scale=torch.ones(1, device="cuda"), _v_scale=torch.ones(1, device="cuda")
    )
    kv_shape = (4, 2, 16, HKV * HD) if shuffle else (4, HKV, 16, 2 * HD)
    kv_cache = torch.zeros(kv_shape, device="cuda").bfloat16()
    q, k, v = randn(t, HQ, HD), randn(t, HKV, HD), randn(t, HKV, HD)
    output = torch.zeros(t, HQ, HD, device="cuda").bfloat16()
    impl.forward(layer, q, k, v, kv_cache, md, output)
    return seen, output, gathered


def _ck_stub(monkeypatch):
    from vllm.v1.attention.backends import rocm_aiter_fa

    ck = []

    def fake_ck(**kw):
        ck.append(kw)
        q = kw["q"]
        o = torch.zeros_like(q)
        lse = torch.zeros(q.shape[1], q.shape[0], device=q.device)
        if kw.get("out") is not None:
            kw["out"].copy_(o)
        return (o, lse) if kw.get("return_lse") else o

    monkeypatch.setattr(rocm_aiter_fa.rocm_aiter_ops, "flash_attn_varlen_func", fake_ck)
    return ck


SITES = [
    # (causal, return_lse, writes into out=)
    (True, True, False),
    (False, True, False),
    (True, False, True),
]


def test_all_three_call_sites_dispatch(monkeypatch, fp8, kernel):
    ck = _ck_stub(monkeypatch)
    seen, _, _ = _run_backend_forward(monkeypatch, fp8, on=True)
    shapes = sorted(
        (bool(s["causal"]), bool(s.get("return_lse")), s.get("out") is not None)
        for s in seen
    )
    assert shapes == sorted(SITES)
    assert len(kernel) == 3
    assert ck == []


def test_call_sites_run_ck_when_off(monkeypatch, fp8, kernel):
    ck = _ck_stub(monkeypatch)
    seen, _, _ = _run_backend_forward(monkeypatch, fp8, on=False)
    assert len(seen) == 3
    assert kernel == []
    assert len(ck) == 3


def _profile_forward(monkeypatch, fp8, on, shuffle=False, num_tokens=64):
    """Drive forward()'s profiling branch and record the reserved KV width."""
    impl = _make_impl(monkeypatch, fp8, on, shuffle)
    reserved = []
    real_profile = fp8.profile

    def spy(query, num_q_heads, num_kv_heads, num_kv_tokens, num_q_tokens):
        reserved.append((num_kv_tokens, num_q_tokens))
        real_profile(query, num_q_heads, num_kv_heads, num_kv_tokens, num_q_tokens)

    monkeypatch.setattr(fp8, "profile", spy)
    q = torch.randn(num_tokens, HQ, HD, device="cuda").bfloat16()
    k = torch.randn(num_tokens, HKV, HD, device="cuda").bfloat16()
    kv_cache = torch.zeros(4, HKV, 16, 2 * HD, device="cuda").bfloat16()
    layer = SimpleNamespace(
        _k_scale=torch.ones(1, device="cuda"), _v_scale=torch.ones(1, device="cuda")
    )
    output = torch.ones(num_tokens, HQ, HD, device="cuda").bfloat16()
    impl.forward(layer, q, k, k, kv_cache, None, output)
    return reserved, output


@pytest.mark.parametrize("shuffle", [False, True], ids=["nhd", "shuffle"])
def test_profile_branch_reserves_for_the_widest_context(monkeypatch, fp8, shuffle):
    """The shuffle layout still quantises the gathered context, so the profile
    run has to reserve a full _CP_TOKENS_PER_ITER_ROCM of it."""
    n = 64
    reserved, output = _profile_forward(
        monkeypatch, fp8, on=True, shuffle=shuffle, num_tokens=n
    )
    assert reserved == [(max(n, WORKSPACE_ROWS) if shuffle else n, n)]
    assert bool((output == 0).all())


def test_profile_reserves_the_scheduler_width_not_the_dummy_batch(monkeypatch, fp8):
    """Microbatching splits the profile batch, so the dummy query is narrower
    than a later unsplit step on the same workspace slot."""
    from vllm.config import AttentionConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends import rocm_aiter_fa

    monkeypatch.setattr(
        rocm_aiter_fa.rocm_aiter_ops, "is_shuffle_kv_cache_enabled", lambda: False
    )
    cfg = VllmConfig(
        attention_config=AttentionConfig(use_prefill_query_quantization=True)
    )
    with set_current_vllm_config(cfg):
        impl = rocm_aiter_fa.AiterFlashAttentionImpl(
            num_heads=HQ,
            head_size=HD,
            scale=HD**-0.5,
            num_kv_heads=HKV,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8",
        )
    wide = cfg.scheduler_config.max_num_batched_tokens
    assert impl.max_num_batched_tokens == wide
    _use_fp8(fp8, impl, True)

    reserved = []
    monkeypatch.setattr(
        fp8, "profile", lambda q, hq, hkv, kv, qt: reserved.append((kv, qt))
    )
    half = max(1, wide // 2)
    q = torch.randn(half, HQ, HD, device="cuda").bfloat16()
    k = torch.randn(half, HKV, HD, device="cuda").bfloat16()
    kv_cache = torch.zeros(4, HKV, 16, 2 * HD, device="cuda").bfloat16()
    layer = SimpleNamespace(
        _k_scale=torch.ones(1, device="cuda"), _v_scale=torch.ones(1, device="cuda")
    )
    out = torch.ones(half, HQ, HD, device="cuda").bfloat16()
    impl.forward(layer, q, k, k, kv_cache, None, out)
    assert reserved == [(wide, wide)], reserved


def test_profile_branch_is_skipped_when_the_path_is_off(monkeypatch, fp8):
    reserved, output = _profile_forward(monkeypatch, fp8, on=False)
    assert reserved == []
    assert bool((output == 0).all())


@pytest.mark.parametrize("shuffle", [False, True], ids=["nhd", "shuffle"])
def test_profiled_workspace_survives_a_full_width_extend(
    monkeypatch, fp8, kernel, shuffle
):
    """What the profile run reserves must cover the widest real extend, since a
    short reservation now raises at serving time instead of falling back."""
    from vllm.v1.worker.workspace import lock_workspace

    _profile_forward(monkeypatch, fp8, on=True, shuffle=shuffle)
    lock_workspace()
    seen, _, _ = _run_backend_forward(
        monkeypatch, fp8, on=True, ctx=WORKSPACE_ROWS, shuffle=shuffle
    )
    assert len(seen) == 3


def _workspace(ctx, stale):
    g = torch.Generator(device="cuda").manual_seed(1)
    ws = torch.full((2, WORKSPACE_ROWS, HKV, HD), stale, device="cuda").bfloat16()
    ws[:, :ctx] = torch.randn(2, ctx, HKV, HD, device="cuda", generator=g)
    return ws


@pytest.mark.parametrize("on", [True, False])
def test_chunk_call_site_slices_to_total_tokens(monkeypatch, fp8, kernel, on):
    _ck_stub(monkeypatch)
    ctx = 48
    ws = _workspace(ctx, 0.0)
    seen, _, gathered = _run_backend_forward(
        monkeypatch, fp8, on=on, workspace=ws, ctx=ctx
    )
    (chunk_call,) = [s for s in seen if not s["causal"]]
    # The FP8 view packs K and V into the first half of the workspace bytes.
    v_offset = ws[0].numel() if on else ws[0].nbytes
    for name, ptr in (("k", ws.data_ptr()), ("v", ws.data_ptr() + v_offset)):
        assert chunk_call[name].shape[0] == ctx, name
        assert chunk_call[name].data_ptr() == ptr, name
        assert (chunk_call[name].dtype == FP8) is on, name
        assert (f"descale_{name}" in chunk_call) is on, name
    # The FP8 arm reads the cache as-is; the CK arm needs it dequantised.
    assert [kw["dequant"] for kw in gathered] == [not on]


@gfx942
@pytest.mark.parametrize("stale", [1e4, float("nan")], ids=["huge", "nan"])
def test_chunk_amax_ignores_stale_workspace_rows(monkeypatch, fp8, stale):
    # Only the shuffled layout still requantises the gathered context.
    ctx = 48
    clean = _workspace(ctx, 0.0)[:, :ctx].clone()
    _, ref, _ = _run_backend_forward(
        monkeypatch,
        fp8,
        on=True,
        workspace=clean,
        ctx=ctx,
        stub_merge=False,
        shuffle=True,
    )
    _, got, _ = _run_backend_forward(
        monkeypatch,
        fp8,
        on=True,
        workspace=_workspace(ctx, stale),
        ctx=ctx,
        stub_merge=False,
        shuffle=True,
    )
    assert torch.isfinite(ref).all()
    assert torch.equal(got, ref)


def _head_scaled(args, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def factors(n, lo, hi):
        return 2 ** (torch.rand(n, device="cuda", generator=g) * (hi - lo) + lo)

    k_s, v_s = factors(HKV, -2, 2), factors(HKV, -2, 2)
    q_s = factors(HQ, -0.5, 0.5) / k_s.repeat_interleave(HQ // HKV)
    for name, s in (("q", q_s), ("k", k_s), ("v", v_s)):
        t = args[name]
        args[name] = (t.float() * s[None, :, None]).to(t.dtype)
    return args


def _run_triton(monkeypatch, fp8, args):
    import aiter.ops.triton.attention.mha as mha

    launched = []
    real = mha._flash_attn_forward

    @functools.wraps(real)
    def counted(*a, **kw):
        launched.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(mha, "_flash_attn_forward", counted)
    res = fp8.triton_fp8_prefill_attn(
        orig=rocm_aiter_ops.flash_attn_varlen_func, **args
    )
    monkeypatch.setattr(mha, "_flash_attn_forward", real)
    assert launched == [1]
    return res


HEAD_REL_TOL = 0.08
LSE_REL_TOL = 4e-3


@gfx942
@pytest.mark.parametrize("causal", [True, False])
def test_triton_matches_ck_per_head(monkeypatch, fp8, causal):
    torch.manual_seed(0)
    args = _head_scaled(_inputs(seq_lens=[173, 517, 64]))
    args["causal"] = causal
    ck_o, ck_lse = rocm_aiter_ops.flash_attn_varlen_func(**args)
    o, lse = _run_triton(monkeypatch, fp8, args)
    assert lse.shape == ck_lse.shape
    errs = [_rel(o[:, h], ck_o[:, h]) for h in range(HQ)]
    assert max(errs) < HEAD_REL_TOL, errs
    lse_errs = [_rel(lse[h], ck_lse[h]) for h in range(HQ)]
    assert max(lse_errs) < LSE_REL_TOL, lse_errs


@gfx942
def test_wrong_descales_are_not_silently_ignored(monkeypatch, fp8):
    """Negative control for _descale: every other descale assertion is on shape
    and dtype, so corrupt the value the module computes and require the output
    to diverge from CK. Guards the context-chunk and KV-sharing sites too,
    which share _descale."""
    torch.manual_seed(0)
    args = _head_scaled(_inputs(seq_lens=[173, 517, 64]))
    ck_o, _ = rocm_aiter_ops.flash_attn_varlen_func(**args)
    real_descale = fp8._descale
    monkeypatch.setattr(
        fp8,
        "_descale",
        lambda scale, n_heads, batch: real_descale(scale, n_heads, batch) * 4.0,
    )
    o, _ = _run_triton(monkeypatch, fp8, args)
    assert max(_rel(o[:, h], ck_o[:, h]) for h in range(HQ)) > 0.5


@gfx942
def test_noncontiguous_qkv(monkeypatch, fp8):
    torch.manual_seed(0)
    args = _inputs(seq_lens=[173, 64])
    qkv = torch.randn(237, HQ + 2 * HKV, HD, device="cuda").bfloat16()
    q, k, v = qkv.split([HQ, HKV, HKV], dim=1)
    assert not q.is_contiguous()
    args.update(q=q, k=k, v=v)
    o, lse = _run_triton(monkeypatch, fp8, args)
    args.update(q=q.contiguous(), k=k.contiguous(), v=v.contiguous())
    ref_o, ref_lse = _run_triton(monkeypatch, fp8, args)
    assert torch.equal(o, ref_o)
    assert torch.equal(lse, ref_lse)


@gfx942
def test_outlier_sequence_does_not_corrupt_its_neighbour(monkeypatch, fp8):
    """One scale per tensor per call, so a 50x larger sequence shifts the other
    one. E4M3 is floating point, so the larger scale costs range and not
    mantissa, and both sequences stay within the CK tolerance."""
    torch.manual_seed(0)
    alone = _inputs(seq_lens=[256])
    both = _inputs(seq_lens=[256, 256])
    for name in ("q", "k", "v"):
        both[name][:256] = alone[name]
        both[name][256:] *= 50
    for args in (alone, both):
        args["causal"] = False
    o_alone, _ = _run_triton(monkeypatch, fp8, alone)
    o_both, _ = _run_triton(monkeypatch, fp8, both)
    ck_o, _ = rocm_aiter_ops.flash_attn_varlen_func(**alone)
    assert _rel(o_alone, ck_o) < HEAD_REL_TOL
    assert _rel(o_both[:256], ck_o) < HEAD_REL_TOL


@gfx942
@pytest.mark.parametrize("shuffle", [False, True], ids=["nhd", "shuffle"])
def test_extend_with_context_matches_ck(monkeypatch, fp8, shuffle):
    from vllm.v1.attention.backends import rocm_aiter_fa

    q_lens, chunk_lens = [40, 25], [[64, 30], [36, 0]]
    g = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=g).bfloat16()

    t = sum(q_lens)
    q, k, v = randn(t, HQ, HD), randn(t, HKV, HD), randn(t, HKV, HD)
    ctx = [(randn(sum(c), HKV, HD), randn(sum(c), HKV, HD)) for c in chunk_lens]
    k_scale, v_scale = (
        (torch.stack([c[i].abs().amax() for c in ctx]).max().float() / FP8_MAX)
        .view(1)
        .cuda()
        for i in (0, 1)
    )
    ctx8 = [((ck / k_scale).to(FP8), (cv / v_scale).to(FP8)) for ck, cv in ctx]
    ws = torch.full((2, WORKSPACE_ROWS, HKV, HD), float("nan"), device="cuda")
    ws = ws.bfloat16()
    gathered: list[tuple[int, bool]] = []

    def gather(key, value, total_tokens, dequant, **kw):
        ck8, cv8 = ctx8[len(gathered)]
        gathered.append((total_tokens, dequant))
        if dequant:
            key[:total_tokens] = (ck8.float() * k_scale).to(key.dtype)
            value[:total_tokens] = (cv8.float() * v_scale).to(value.dtype)
        else:
            key[:total_tokens] = ck8
            value[:total_tokens] = cv8

    monkeypatch.setattr(rocm_aiter_fa, "cp_mha_gather_cache", gather)
    monkeypatch.setattr(
        rocm_aiter_fa.rocm_aiter_ops, "is_shuffle_kv_cache_enabled", lambda: shuffle
    )
    chunk = SimpleNamespace(
        num_chunks=2,
        workspace=ws,
        cu_seq_lens_chunk=[_cu(c) for c in chunk_lens],
        max_seq_lens=[max(c) for c in chunk_lens],
        chunk_starts=[_cu(c)[:-1] for c in chunk_lens],
        token_to_batch=[torch.zeros(sum(c), dtype=torch.int32) for c in chunk_lens],
        total_token_per_batch=[sum(c) for c in chunk_lens],
        swa_metadata=None,
    )
    md = SimpleNamespace(extend_metadata=SimpleNamespace(chunk_context_metadata=chunk))
    impl = rocm_aiter_fa.AiterFlashAttentionImpl(
        num_heads=HQ,
        head_size=HD,
        scale=HD**-0.5,
        num_kv_heads=HKV,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8",
    )

    import aiter.ops.triton.attention.mha as mha

    real_fwd = mha._flash_attn_forward
    launches = []

    @functools.wraps(real_fwd)
    def counted(*a, **kw):
        launches.append(kw)
        return real_fwd(*a, **kw)

    monkeypatch.setattr(mha, "_flash_attn_forward", counted)

    def run(on):
        _use_fp8(fp8, impl, on)
        gathered.clear()
        launches.clear()
        out = torch.zeros(t, HQ, HD, device="cuda").bfloat16()
        impl.extend_forward(
            attn_metadata=md,
            query=q,
            key=k,
            value=v,
            key_cache=None,
            value_cache=None,
            output=out,
            cu_seqlens_q=_cu(q_lens),
            max_seqlen_q=max(q_lens),
            min_seqlen_q=1,
            max_seqlen_k=max(q_lens) + 100,
            block_table=None,
            slot_mapping=None,
            k_scale=k_scale,
            v_scale=v_scale,
            kv_descales={},
        )
        return out

    ref = run(False)
    assert launches == [], "the CK arm must not reach the Triton kernel"
    assert gathered == [(94, True), (36, True)]
    got = run(True)
    # One suffix call plus one per chunk, so both arms cannot both be CK.
    assert len(launches) == 3, launches
    assert gathered == [(94, shuffle), (36, shuffle)]
    if not shuffle:
        for kw in launches:
            if kw["causal"]:
                continue
            assert kw["descale_k"].shape == (len(q_lens), HKV)
            assert bool((kw["descale_k"] == k_scale).all())
            assert bool((kw["descale_v"] == v_scale).all())
    assert torch.isfinite(ref).all() and torch.isfinite(got).all()
    errs = [_rel(got[:, h], ref[:, h]) for h in range(HQ)]
    assert max(errs) < HEAD_REL_TOL, errs


@gfx942
@pytest.mark.parametrize("shuffle", [False, True], ids=["nhd", "shuffle"])
def test_kv_sharing_prefill_reads_fp8_cache(monkeypatch, fp8, shuffle):
    """A shared layer hands the kernel cache FP8 K/V, not a requantised copy."""
    from vllm.v1.attention.backends import rocm_aiter_fa

    q_lens = [40, 25]
    t = sum(q_lens)
    g = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=g).bfloat16()

    k_ref, v_ref = randn(t, HKV, HD), randn(t, HKV, HD)
    k_scale = (k_ref.abs().amax().float() / FP8_MAX).view(1)
    v_scale = (v_ref.abs().amax().float() / FP8_MAX).view(1)
    k8, v8 = (k_ref / k_scale).to(FP8), (v_ref / v_scale).to(FP8)
    gathered: list[tuple[bool, torch.dtype]] = []

    def gather(key, value, total_tokens, dequant, **kw):
        gathered.append((dequant, key.dtype))
        if dequant:
            key[:total_tokens] = (k8.float() * k_scale).to(key.dtype)
            value[:total_tokens] = (v8.float() * v_scale).to(value.dtype)
        else:
            key[:total_tokens] = k8
            value[:total_tokens] = v8

    monkeypatch.setattr(rocm_aiter_fa, "cp_mha_gather_cache", gather)
    monkeypatch.setattr(
        rocm_aiter_fa.rocm_aiter_ops, "is_shuffle_kv_cache_enabled", lambda: shuffle
    )
    impl = rocm_aiter_fa.AiterFlashAttentionImpl(
        num_heads=HQ,
        head_size=HD,
        scale=HD**-0.5,
        num_kv_heads=HKV,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8",
        kv_sharing_target_layer_name="layer.0",
    )
    cu = torch.tensor([0, q_lens[0], t], dtype=torch.int32, device="cuda")
    # Wider than the batch, so the token slice of the workspace is not contiguous.
    ws = torch.full((2, 4 * t, HKV, HD), float("nan"), device="cuda").bfloat16()
    md = SimpleNamespace(
        num_actual_tokens=t,
        num_decodes=0,
        num_prefills=len(q_lens),
        num_extends=0,
        num_decode_tokens=0,
        num_extend_tokens=0,
        use_cascade=False,
        causal=True,
        block_table=torch.zeros(len(q_lens), 4, dtype=torch.int32, device="cuda"),
        slot_mapping=torch.arange(t, device="cuda"),
        prefill_metadata=SimpleNamespace(
            query_start_loc=cu, max_query_len=max(q_lens), max_seq_len=max(q_lens)
        ),
        extend_metadata=None,
        kv_sharing_metadata=SimpleNamespace(
            workspace=ws,
            query_start_loc=cu,
            token_to_batch=torch.zeros(t, dtype=torch.int32, device="cuda"),
            seq_starts=torch.zeros(len(q_lens), dtype=torch.int32, device="cuda"),
        ),
    )
    layer = SimpleNamespace(_k_scale=k_scale.cuda(), _v_scale=v_scale.cuda())
    kv_shape = (4, 2, 16, HKV * HD) if shuffle else (4, HKV, 16, 2 * HD)
    kv_cache = torch.zeros(kv_shape, device="cuda").bfloat16()
    q = randn(t, HQ, HD)

    import aiter.ops.triton.attention.mha as mha

    real_fwd = mha._flash_attn_forward
    launches = []

    @functools.wraps(real_fwd)
    def counted(*a, **kw):
        launches.append(kw)
        return real_fwd(*a, **kw)

    monkeypatch.setattr(mha, "_flash_attn_forward", counted)

    def run(on):
        _use_fp8(fp8, impl, on)
        gathered.clear()
        launches.clear()
        out = torch.zeros(t, HQ, HD, device="cuda").bfloat16()
        impl.forward(layer, q, None, None, kv_cache, md, out)
        return out

    ref = run(False)
    assert launches == [], "the CK arm must not reach the Triton kernel"
    assert gathered == [(True, torch.bfloat16)]
    got = run(True)
    assert len(launches) == 1, launches
    direct = not shuffle
    assert gathered == [(not direct, FP8 if direct else torch.bfloat16)]
    if direct:
        kw = launches[0]
        assert kw["descale_k"].shape == (len(q_lens), HKV)
        assert bool((kw["descale_k"] == k_scale.cuda()).all())
        assert bool((kw["descale_v"] == v_scale.cuda()).all())
    assert torch.isfinite(ref).all() and torch.isfinite(got).all()
    errs = [_rel(got[:, h], ref[:, h]) for h in range(HQ)]
    assert max(errs) < HEAD_REL_TOL, errs
