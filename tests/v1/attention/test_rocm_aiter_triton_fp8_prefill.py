# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ROCM_AITER_FA FP8 Triton prefill path.

The dispatcher either runs aiter's Triton MHA on FP8 Q/K/V or hands the call
back to CK unchanged. Routing tests replace the kernel with a recorder; the
tests marked ``gfx942`` run the real kernel against CK.
"""

import builtins
import functools
import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm AITER FP8 prefill tests", allow_module_level=True)

from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms.rocm import on_gfx942

# Every test drives aiter's Triton MHA, as a recorder or for real.
pytest.importorskip("aiter.ops.triton.attention.mha")

# is_fp8_fnuz() is also true on gfx90a/gfx940/gfx941, where _TUNED_CONFIG was
# never searched and is_triton_fp8_prefill_enabled() refuses anyway.
gfx942 = pytest.mark.skipif(not on_gfx942(), reason="gfx942 only")

FLAG = "VLLM_ROCM_USE_AITER_TRITON_FP8_PREFILL"
HQ, HKV, HD = 24, 4, 256
SEQ_LENS = [37, 64, 19]
WORKSPACE_ROWS = 32 * 1024  # rocm_aiter_fa._CP_TOKENS_PER_ITER_ROCM


@pytest.fixture
def fp8():
    return importlib.import_module(
        "vllm.v1.attention.backends.rocm_aiter_triton_fp8_prefill"
    )


def _force_enabled(monkeypatch, fp8, on=True):
    """Set the resolved state directly, so routing tests also run off gfx942.
    Resolution itself is tested separately."""
    import aiter.ops.triton.attention.mha as mha

    monkeypatch.setenv(FLAG, "1" if on else "0")
    monkeypatch.setattr(rocm_aiter_ops, "_TRITON_FP8_PREFILL", on, raising=False)
    monkeypatch.setattr(fp8, "_enabled", on, raising=False)
    monkeypatch.setattr(fp8, "_mha", mha if on else None, raising=False)


@pytest.fixture
def kernel(monkeypatch):
    """Replace the Triton kernel with a recorder returning well-formed outputs."""
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
    """Stands in for rocm_aiter_ops.flash_attn_varlen_func and records calls."""

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


# --------------------------------------------------------------------------
# Flags
# --------------------------------------------------------------------------


@pytest.mark.parametrize("flag", [FLAG])
def test_flag_is_registered_and_defaults_off(monkeypatch, flag):
    import vllm.envs as envs

    assert flag in envs.environment_variables
    getter = envs.environment_variables[flag]
    monkeypatch.delenv(flag, raising=False)
    assert getter() is False
    for value, expect in (("1", True), ("true", True), ("0", False)):
        monkeypatch.setenv(flag, value)
        assert getter() is expect


def test_adds_exactly_one_env_var_and_it_validates(monkeypatch):
    """This change registers one flag and no more. A flag that nothing reads is
    public surface that cannot easily be withdrawn, so the follow-up PRs that
    build on this path register their own."""
    import os

    import vllm.envs as envs

    assert FLAG in envs.environment_variables
    for name in (
        "VLLM_ROCM_FP8_PREFILL_PROLOGUE",
        "VLLM_ROCM_FP8_PREFILL_VERIFY_BITWISE",
        "VLLM_ROCM_FP8_PREFIX_KV",
        "VLLM_ROCM_USE_AITER_TRANSPOSED_SCALE_QUANT",
    ):
        assert name not in envs.environment_variables, (
            f"{name} belongs to a follow-up PR, not this one"
        )

    for name in [n for n in os.environ if n.startswith("VLLM_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv(FLAG, "1")
    current_platform.validate_environ(hard_fail=True)


@pytest.mark.parametrize("value, expect", [("1", True), ("0", False), ("", False)])
def test_flag_drives_rocm_aiter_ops(monkeypatch, fp8, value, expect):
    from vllm.platforms.rocm import on_gfx942

    monkeypatch.setattr(fp8, "_enabled", None, raising=False)
    monkeypatch.setenv(FLAG, value)
    rocm_aiter_ops.refresh_env_variables()
    try:
        assert bool(rocm_aiter_ops.is_triton_fp8_prefill_enabled()) is (
            expect and on_gfx942()
        )
        assert fp8._resolve() is (expect and on_gfx942())
        assert fp8._enabled is (expect and on_gfx942())
    finally:
        monkeypatch.delenv(FLAG, raising=False)
        rocm_aiter_ops.refresh_env_variables()


def test_flag_off_is_a_pass_through(monkeypatch, fp8, kernel):
    _force_enabled(monkeypatch, fp8, on=False)
    orig = _Orig()
    args = _inputs()
    assert fp8.maybe_triton_prefill_attn(orig=orig, **args) is orig.result
    assert kernel == []
    assert len(orig.calls) == 1
    _assert_same_args(orig.calls[0], args)


def test_flag_resolved_once(monkeypatch, fp8):
    monkeypatch.setattr(fp8, "_enabled", None, raising=False)
    asked = []
    monkeypatch.setattr(
        rocm_aiter_ops, "is_triton_fp8_prefill_enabled", lambda: asked.append(1)
    )
    for _ in range(3):
        fp8.maybe_triton_prefill_attn(orig=_Orig(), **_inputs())
    assert asked == [1]


class _Opaque:
    """Fails on any attribute access, so any tensor op on it raises."""

    def __getattribute__(self, name):
        raise AssertionError(f"off path touched .{name}")


def test_flag_off_cost(monkeypatch, fp8):
    """Once resolved off, a call is one bool check and the original call: no
    flag re-read, no import, and no tensor op on any argument."""
    monkeypatch.setattr(fp8, "_enabled", False, raising=False)

    def boom():
        raise AssertionError("flag re-read on the off path")

    monkeypatch.setattr(rocm_aiter_ops, "is_triton_fp8_prefill_enabled", boom)
    imports = []
    real_import = builtins.__import__

    def spy_import(name, *a, **kw):
        imports.append(name)
        return real_import(name, *a, **kw)

    args = {name: _Opaque() for name in _inputs()}
    orig = _Orig()
    builtins.__import__ = spy_import
    try:
        res = fp8.maybe_triton_prefill_attn(orig=orig, **args)
    finally:
        builtins.__import__ = real_import
    assert res is orig.result
    assert imports == []
    _assert_same_args(orig.calls[0], args)


# --------------------------------------------------------------------------
# Eligibility matrix: every condition, one row each
# --------------------------------------------------------------------------

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

DECLINE = {
    "q_not_3d": dict(q_2d=True, why="must be [tokens"),
    "head_dim_128": dict(make=dict(hd=128), why="head_dim 128"),
    "head_dim_64": dict(make=dict(hd=64), why="head_dim 64"),
    "kv_head_dim_mismatch": dict(make=dict(k_hd=128), why="k/v head_dim"),
    "dtype_fp32": dict(make=dict(dtype=torch.float32), why="not bf16/fp16"),
    "kv_dtype_differs": dict(make=dict(k_dtype=torch.float16), why="k/v dtype"),
    "kv_already_fp8": dict(
        make=dict(k_dtype=current_platform.fp8_dtype()), why="k/v dtype"
    ),
    "heads_not_divisible": dict(make=dict(hq=24, hkv=5), why="not a multiple"),
    "k_v_head_count_differs": dict(
        make=dict(v_heads=8), why="num_k_heads 4 != num_v_heads 8"
    ),
    "empty_k": dict(empty_k=True, why="token counts"),
    "cu_seqlens_int64": dict(cu_int64=True, why="int64"),
    "alibi": dict(call=dict(alibi_slopes=torch.ones(HQ)), why="alibi"),
    "sink": dict(call=dict(sink_ptr=torch.zeros(HQ)), why="sink"),
    "sliding_window": dict(call=dict(window_size=(127, 0)), why="sliding window"),
    "dropout": dict(call=dict(dropout_p=0.1), why="dropout"),
    # K shorter than Q under causal leaves rows with no key; see _eligible.
    "causal_separate_offsets": dict(separate_k=True, why="causal with separate"),
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


def _decide(fp8, args):
    return fp8._eligible(
        args["q"],
        args["k"],
        args["v"],
        args["alibi_slopes"],
        args["sink_ptr"],
        args["window_size"],
        args["dropout_p"],
        args["causal"],
        args["cu_seqlens_q"],
        args["cu_seqlens_k"],
    )


@pytest.mark.parametrize("name", sorted(ACCEPT))
def test_eligible_calls_reach_triton(monkeypatch, fp8, kernel, name):
    _force_enabled(monkeypatch, fp8)
    args = _case(ACCEPT[name])
    orig = _Orig()
    fp8.maybe_triton_prefill_attn(orig=orig, **args)
    assert orig.calls == []
    assert len(kernel) == 1
    assert kernel[0].q.dtype == kernel[0].k.dtype == kernel[0].v.dtype
    assert kernel[0].q.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn)
    assert kernel[0].causal is bool(args["causal"])
    assert kernel[0].config == fp8._TUNED_CONFIG


@pytest.mark.parametrize("name", sorted(DECLINE))
def test_ineligible_calls_fall_back_unchanged(monkeypatch, fp8, kernel, name):
    _force_enabled(monkeypatch, fp8)
    args = _case(DECLINE[name])
    orig = _Orig()
    assert fp8.maybe_triton_prefill_attn(orig=orig, **args) is orig.result
    assert kernel == []
    assert len(orig.calls) == 1
    _assert_same_args(orig.calls[0], args)


@pytest.mark.parametrize("name", sorted(ACCEPT))
def test_eligible_reports_no_reason(fp8, name):
    assert _decide(fp8, _case(ACCEPT[name])) == (True, "")


@pytest.mark.parametrize("name", sorted(DECLINE))
def test_refusal_reason_names_the_condition(fp8, name):
    ok, why = _decide(fp8, _case(DECLINE[name]))
    assert not ok
    assert DECLINE[name]["why"] in why


# --------------------------------------------------------------------------
# aiter API drift: never raise inside forward, fall back to CK for good
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failure", ["import_error", "signature_drift", "dao_ai_impl", "launch_error"]
)
def test_aiter_api_drift_falls_back(monkeypatch, fp8, failure):
    import aiter.ops.triton.attention.mha as mha

    monkeypatch.setattr(fp8, "_enabled", None, raising=False)
    monkeypatch.setattr(rocm_aiter_ops, "is_triton_fp8_prefill_enabled", lambda: True)
    launches = []
    if failure == "import_error":
        monkeypatch.setitem(sys.modules, "aiter.ops.triton.attention.mha", None)
    elif failure == "signature_drift":

        def old_api(q, k, v, dropout_p, softmax_scale, causal):
            launches.append(1)

        monkeypatch.setattr(mha, "_flash_attn_forward", old_api)
    elif failure == "dao_ai_impl":
        monkeypatch.setattr(mha, "_MHA_IMPL", "dao_ai")
    else:

        @functools.wraps(mha._flash_attn_forward)
        def broken(*a, **kw):
            launches.append(1)
            raise RuntimeError("triton compile error: out of LDS")

        monkeypatch.setattr(mha, "_flash_attn_forward", broken)

    for _ in range(2):
        orig = _Orig()
        args = _inputs()
        assert fp8.maybe_triton_prefill_attn(orig=orig, **args) is orig.result
        assert len(orig.calls) == 1
        _assert_same_args(orig.calls[0], args)
        assert fp8._enabled is False
    assert len(launches) == (1 if failure == "launch_error" else 0)


def test_oom_falls_back_but_keeps_the_path(monkeypatch, fp8):
    """An OOM says this call did not fit, not that the kernel is broken. The
    call falls back to CK unchanged and the next one still tries Triton."""
    import aiter.ops.triton.attention.mha as mha

    _force_enabled(monkeypatch, fp8)
    launches = []

    @functools.wraps(mha._flash_attn_forward)
    def oom(*a, **kw):
        launches.append(1)
        raise torch.OutOfMemoryError("HIP out of memory")

    monkeypatch.setattr(mha, "_flash_attn_forward", oom)
    for _ in range(2):
        orig = _Orig()
        args = _inputs()
        assert fp8.maybe_triton_prefill_attn(orig=orig, **args) is orig.result
        _assert_same_args(orig.calls[0], args)
        assert fp8._enabled is True
    assert len(launches) == 2, "an OOM must not disable the path"


def test_unexpected_kwarg_falls_back_unchanged(monkeypatch, fp8, kernel):
    """A kwarg this module does not know may change what CK computes, so it
    must reach CK rather than be dropped on the way to the Triton kernel."""
    _force_enabled(monkeypatch, fp8)
    args = _inputs()
    args["softcap"] = 30.0
    orig = _Orig()
    assert fp8.maybe_triton_prefill_attn(orig=orig, **args) is orig.result
    assert kernel == []
    _assert_same_args(orig.calls[0], args)


# --------------------------------------------------------------------------
# What the kernel receives, and the output contract
# --------------------------------------------------------------------------


def test_kernel_gets_per_head_descale_shapes(monkeypatch, fp8, kernel):
    """descale_q is read at [b, q_head] and descale_k/v at [b, kv_head] with no
    bounds check and an implicit unit head stride, so the shapes must be
    exactly these, contiguous."""
    _force_enabled(monkeypatch, fp8)
    fp8.maybe_triton_prefill_attn(orig=_Orig(), **_inputs())
    (call,) = kernel
    batch = len(SEQ_LENS)
    assert tuple(call.descale_q.shape) == (batch, HQ)
    assert tuple(call.descale_k.shape) == (batch, HKV)
    assert tuple(call.descale_v.shape) == (batch, HKV)
    for d in (call.descale_q, call.descale_k, call.descale_v):
        assert d.dtype == torch.float32
        assert d.is_contiguous()


def test_softmax_scale_none_default(monkeypatch, fp8, kernel):
    _force_enabled(monkeypatch, fp8)
    args = _inputs()
    args["softmax_scale"] = None
    fp8.maybe_triton_prefill_attn(orig=_Orig(), **args)
    assert kernel[0].softmax_scale == HD**-0.5


def test_lse_is_returned_heads_by_tokens(monkeypatch, fp8, kernel):
    _force_enabled(monkeypatch, fp8)
    args = _inputs()
    o, lse = fp8.maybe_triton_prefill_attn(orig=_Orig(), **args)
    t = args["q"].shape[0]
    assert o.dtype == args["q"].dtype
    assert tuple(o.shape) == (t, HQ, HD)
    assert tuple(lse.shape) == (HQ, t)
    expected = torch.arange(t * HQ, dtype=torch.float32, device=lse.device)
    torch.testing.assert_close(lse, expected.view(t, HQ).t())


def test_out_is_written_in_place(monkeypatch, fp8, kernel):
    _force_enabled(monkeypatch, fp8)
    args = _inputs()
    args["return_lse"] = False
    out = torch.zeros_like(args["q"])
    res = fp8.maybe_triton_prefill_attn(orig=_Orig(), out=out, **args)
    assert res is out
    assert bool((out == 0.5).all())


# --------------------------------------------------------------------------
# The three call sites in rocm_aiter_fa
# --------------------------------------------------------------------------


def _run_backend_forward(monkeypatch, fp8, workspace=None, ctx=48, stub_merge=True):
    """Run AiterFlashAttentionImpl.forward on one prefill and one extend request
    with one context chunk of ``ctx`` tokens. The gather is stubbed: the
    context K/V must already be in the first ``ctx`` rows of ``workspace``."""
    from vllm.v1.attention.backends import rocm_aiter_fa

    monkeypatch.setattr(rocm_aiter_fa, "cp_mha_gather_cache", lambda **kw: None)
    if stub_merge:
        monkeypatch.setattr(rocm_aiter_fa, "merge_attn_states", lambda **kw: None)
    monkeypatch.setattr(
        rocm_aiter_fa.rocm_aiter_ops, "is_shuffle_kv_cache_enabled", lambda: False
    )

    seen = []
    real = fp8.maybe_triton_prefill_attn

    def spy(**kw):
        seen.append(kw)
        return real(**kw)

    monkeypatch.setattr(fp8, "maybe_triton_prefill_attn", spy)

    impl = rocm_aiter_fa.AiterFlashAttentionImpl(
        num_heads=HQ,
        head_size=HD,
        scale=HD**-0.5,
        num_kv_heads=HKV,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
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
    kv_cache = torch.zeros(4, HKV, 16, 2 * HD, device="cuda").bfloat16()
    q, k, v = randn(t, HQ, HD), randn(t, HKV, HD), randn(t, HKV, HD)
    output = torch.zeros(t, HQ, HD, device="cuda").bfloat16()
    impl.forward(layer, q, k, v, kv_cache, md, output)
    return seen, output


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
    return ck, fake_ck


SITES = [
    # (causal, return_lse, writes into out=)
    (True, True, False),  # extend_forward: new tokens against themselves
    (False, True, False),  # extend_forward: new tokens against a context chunk
    (True, False, True),  # forward: pure prefill
]


def test_all_three_call_sites_dispatch(monkeypatch, fp8, kernel):
    ck, fake_ck = _ck_stub(monkeypatch)
    _force_enabled(monkeypatch, fp8)
    seen, _ = _run_backend_forward(monkeypatch, fp8)
    shapes = sorted(
        (bool(s["causal"]), bool(s.get("return_lse")), s.get("out") is not None)
        for s in seen
    )
    assert shapes == sorted(SITES)
    assert all(s["orig"] is fake_ck for s in seen)
    assert len(kernel) == 3
    assert ck == []


def test_call_sites_keep_ck_when_flag_off(monkeypatch, fp8, kernel):
    ck, _ = _ck_stub(monkeypatch)
    _force_enabled(monkeypatch, fp8, on=False)
    seen, _ = _run_backend_forward(monkeypatch, fp8)
    assert len(seen) == 3
    assert kernel == []
    assert len(ck) == 3


def _workspace(ctx, stale):
    """The 32768-row extend workspace: ``ctx`` rows of context, then ``stale``."""
    g = torch.Generator(device="cuda").manual_seed(1)
    ws = torch.full((2, WORKSPACE_ROWS, HKV, HD), stale, device="cuda").bfloat16()
    ws[:, :ctx] = torch.randn(2, ctx, HKV, HD, device="cuda", generator=g)
    return ws


@pytest.mark.parametrize("flag_on", [True, False])
def test_chunk_call_site_slices_to_total_tokens(monkeypatch, fp8, kernel, flag_on):
    _ck_stub(monkeypatch)
    _force_enabled(monkeypatch, fp8, on=flag_on)
    ctx = 48
    ws = _workspace(ctx, 0.0)
    seen, _ = _run_backend_forward(monkeypatch, fp8, workspace=ws, ctx=ctx)
    (chunk_call,) = [s for s in seen if not s["causal"]]
    for name, src in (("k", ws[0]), ("v", ws[1])):
        assert chunk_call[name].shape[0] == ctx, name
        assert chunk_call[name].data_ptr() == src.data_ptr(), name


@gfx942
@pytest.mark.parametrize("stale", [1e4, float("nan")], ids=["huge", "nan"])
def test_chunk_amax_ignores_stale_workspace_rows(monkeypatch, fp8, stale):
    """Rows of the workspace past total_tokens hold whatever an earlier request
    left there (or uninitialised memory). They must not reach the K/V scale."""
    _force_enabled(monkeypatch, fp8)
    ctx = 48
    clean = _workspace(ctx, 0.0)[:, :ctx].clone()
    _, ref = _run_backend_forward(
        monkeypatch, fp8, workspace=clean, ctx=ctx, stub_merge=False
    )
    _, got = _run_backend_forward(
        monkeypatch, fp8, workspace=_workspace(ctx, stale), ctx=ctx, stub_merge=False
    )
    assert torch.isfinite(ref).all()
    assert torch.equal(got, ref)
    assert fp8._enabled is True


# --------------------------------------------------------------------------
# Real kernel against CK
# --------------------------------------------------------------------------


def _head_scaled(args, seed=0):
    """Give every head its own magnitude, so that a head read with the wrong
    descale, KV head or LSE row shows up. K and V heads get 0.25x-4x; each Q
    head gets the inverse of its KV head's K factor times 0.7x-1.4x, which keeps
    the logits, and so the softmax sharpness, close to unscaled."""
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
    res = fp8.maybe_triton_prefill_attn(
        orig=rocm_aiter_ops.flash_attn_varlen_func, **args
    )
    monkeypatch.setattr(mha, "_flash_attn_forward", real)
    assert launched == [1]
    return res


# Measured on MI325X with _head_scaled inputs: at most 0.058 per head on the
# output and 2.8e-3 on LSE. A descale read from the wrong head is off by O(1).
HEAD_REL_TOL = 0.08
LSE_REL_TOL = 4e-3


@gfx942
@pytest.mark.parametrize("causal", [True, False])
def test_triton_matches_ck_per_head(monkeypatch, fp8, causal):
    _force_enabled(monkeypatch, fp8)
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
def test_kernel_descale_indexing_contract(fp8):
    """Pin the aiter indexing the descale shapes rely on: with a genuinely
    different scale per head, descale_q must be read at [b, q_head] and
    descale_k/v at [b, kv_head]."""
    import aiter.ops.triton.attention.mha as mha

    torch.manual_seed(0)
    args = _head_scaled(_inputs(seq_lens=[173, 517, 64]))
    batch = len(args["cu_seqlens_q"]) - 1

    def per_head(t):
        amax = t.abs().amax(dim=(0, 2)).float()
        scale = amax / fp8._FP8_MAX
        t8 = (t.float() / scale[None, :, None]).to(current_platform.fp8_dtype())
        return t8, scale[None, :].repeat(batch, 1).contiguous()

    (q8, dq), (k8, dk), (v8, dv) = (per_head(args[n]) for n in ("q", "k", "v"))

    def run(dq, dk, dv):
        return mha._flash_attn_forward(
            q8,
            k8,
            v8,
            dropout_p=0.0,
            softmax_scale=HD**-0.5,
            causal=True,
            window_size_left=-1,
            window_size_right=-1,
            bias=None,
            alibi_slopes=None,
            return_lse=True,
            return_softmax=False,
            max_seqlen_q=args["max_seqlen_q"],
            max_seqlen_k=args["max_seqlen_k"],
            cu_seqlens_q=args["cu_seqlens_q"],
            cu_seqlens_k=args["cu_seqlens_k"],
            descale_q=dq,
            descale_k=dk,
            descale_v=dv,
            config=dict(fp8._TUNED_CONFIG),
        )[0]

    ck_o, _ = rocm_aiter_ops.flash_attn_varlen_func(**args)
    o = run(dq, dk, dv)
    errs = [_rel(o[:, h], ck_o[:, h]) for h in range(HQ)]
    assert max(errs) < HEAD_REL_TOL, errs
    # The same check must see a K/V descale read from the wrong KV head.
    o = run(dq, dk.flip(1).contiguous(), dv.flip(1).contiguous())
    assert max(_rel(o[:, h], ck_o[:, h]) for h in range(HQ)) > 0.5


@gfx942
def test_noncontiguous_qkv(monkeypatch, fp8):
    """Q/K/V as views of one packed QKV tensor, as the attention layer passes
    them, must give the same result as contiguous copies."""
    _force_enabled(monkeypatch, fp8)
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
def test_batch_invariance_of_scale(monkeypatch, fp8):
    """One scale per call means a sequence's output depends on what it is
    batched with: next to a 50x larger sequence it is rounded on a different
    FP8 grid. This bounds the change; accuracy against CK stays the same."""
    _force_enabled(monkeypatch, fp8)
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
    shift = _rel(o_both[:256], o_alone)
    err_alone = _rel(o_alone, ck_o)
    err_both = _rel(o_both[:256], ck_o)
    # Measured: shift 0.070, error against CK 0.051 alone and co-batched.
    assert 0 < shift < 0.1
    assert err_alone < HEAD_REL_TOL
    assert err_both < HEAD_REL_TOL


@gfx942
def test_extend_with_context_matches_ck(monkeypatch, fp8):
    """extend_forward with the real kernel, real merge_attn_states and a stale
    workspace: two context chunks, non-causal with Q and K lengths that differ,
    and one sequence with zero context tokens in the second chunk."""
    from vllm.v1.attention.backends import rocm_aiter_fa

    q_lens, chunk_lens = [40, 25], [[64, 30], [36, 0]]
    g = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=g).bfloat16()

    t = sum(q_lens)
    q, k, v = randn(t, HQ, HD), randn(t, HKV, HD), randn(t, HKV, HD)
    ctx = [(randn(sum(c), HKV, HD), randn(sum(c), HKV, HD)) for c in chunk_lens]
    ws = torch.full((2, WORKSPACE_ROWS, HKV, HD), float("nan"), device="cuda")
    ws = ws.bfloat16()
    gathered = []

    def gather(key, value, total_tokens, **kw):
        ck_, cv_ = ctx[len(gathered)]
        gathered.append(total_tokens)
        key[:total_tokens] = ck_
        value[:total_tokens] = cv_

    monkeypatch.setattr(rocm_aiter_fa, "cp_mha_gather_cache", gather)
    monkeypatch.setattr(
        rocm_aiter_fa.rocm_aiter_ops, "is_shuffle_kv_cache_enabled", lambda: False
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
        kv_cache_dtype="auto",
    )

    import aiter.ops.triton.attention.mha as mha

    real_fwd = mha._flash_attn_forward
    launches = []

    def counted(*a, **kw):
        launches.append(1)
        return real_fwd(*a, **kw)

    monkeypatch.setattr(mha, "_flash_attn_forward", counted)

    def run(on):
        _force_enabled(monkeypatch, fp8, on=on)
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
            k_scale=None,
            v_scale=None,
        )
        return out

    ref = run(False)
    assert launches == [], "flag off must not reach the Triton kernel"
    got = run(True)
    # suffix call + one per context chunk: without this the comparison below
    # would pass trivially if an eligibility condition regressed and both arms
    # ran CK.
    assert len(launches) == 3, launches
    assert gathered == [94, 36]
    assert torch.isfinite(ref).all() and torch.isfinite(got).all()
    errs = [_rel(got[:, h], ref[:, h]) for h in range(HQ)]
    assert max(errs) < HEAD_REL_TOL, errs
    assert fp8._enabled is True
