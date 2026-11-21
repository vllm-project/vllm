# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL routes of rocm_aiter_ops.triton_sparse_mla_fwd: each must match the
Triton kernel on its call shapes and fall back to it whenever FlyDSL cannot
take a call.

VLLM_ROCM_USE_AITER_FLYDSL_SPARSE_MLA: the in-tree kernel on the Kimi-K3.1
fp8_ds_mla calls. VLLM_ROCM_USE_AITER_FLYDSL_SPARSE_MLA_FP8_DECODE: aiter's
fp8 decode kernel on the DeepSeek-V4.1 per-tensor fp8 decode call."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-only aiter path"
)

KV_LORA_RANK, ROPE_DIM, TOPK = 512, 64, 2048
# fp8_ds_mla record: 512 fp8 latent, 4 fp32 per-128 scales, 64 bf16 rope.
ROW_BYTES = KV_LORA_RANK + 4 * 4 + ROPE_DIM * 2

# DeepSeek-V4.1 decode: 64 heads x 512 per rank, 128-key window + top-512.
H, D = 64, 512
WINDOW, TOPK_V41 = 128, 512
MAX_KEYS = WINDOW + TOPK_V41


@pytest.fixture
def flydsl_ops(monkeypatch):
    """rocm_aiter_ops with the FlyDSL flag on; yields the FlyDSL launch log."""
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("FlyDSL sparse MLA is gfx950-only")
    flydsl = pytest.importorskip("vllm.models.kimi_k3.amd.ops.flydsl_sparse_mla")
    pytest.importorskip("aiter.ops.triton.attention.sparse_mla")
    from vllm._aiter_ops import rocm_aiter_ops

    launches: list[tuple[int, bool, bool]] = []

    def fwd(q, *args, has_invalid=False, return_lse=False, **kwargs):
        launches.append((q.shape[1], has_invalid, return_lse))
        return flydsl.flydsl_sparse_mla_fp8ds_fwd(
            q, *args, has_invalid=has_invalid, return_lse=return_lse, **kwargs
        )

    monkeypatch.setattr(
        rocm_aiter_ops,
        "_flydsl_sparse_mla_ops",
        staticmethod(lambda: (flydsl.flydsl_sparse_mla_fp8ds_supported, fwd)),
    )
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FLYDSL_SPARSE_MLA", "1")
    # Exercise FlyDSL on small batches too; the gate has its own test.
    monkeypatch.setenv("VLLM_ROCM_AITER_FLYDSL_SPARSE_MLA_MIN_TOKENS", "1")
    # Keep aiter's own FlyDSL route off so the Triton reference is Gluon.
    monkeypatch.delenv("AITER_FLYDSL_SPARSE_MLA", raising=False)
    rocm_aiter_ops.refresh_env_variables()
    try:
        yield rocm_aiter_ops, launches
    finally:
        monkeypatch.undo()
        rocm_aiter_ops.refresh_env_variables()


def _fp8_ds_mla_cache(num_slots: int, gen: torch.Generator) -> torch.Tensor:
    latent = (torch.randn(num_slots, KV_LORA_RANK, generator=gen) * 0.3).to(
        torch.float8_e4m3fn
    )
    scales = torch.exp2(torch.randint(-6, 1, (num_slots, 4), generator=gen).float())
    rope = (torch.randn(num_slots, ROPE_DIM, generator=gen) * 0.5).to(torch.bfloat16)
    cache = torch.cat(
        [latent.view(torch.uint8), scales.view(torch.uint8), rope.view(torch.uint8)],
        dim=1,
    )
    assert cache.shape[1] == ROW_BYTES
    return cache.cuda()


def _indices(case: str, num_tokens: int, num_slots: int, gen: torch.Generator):
    """(indptr, indices, has_invalid) as the K3.1 fp8_ds_mla call sites build them.

    fixed: DCP non-compact prefill, TOPK per row, -1 past the valid count.
    ragged: non-DCP / compact DCP prefill, min(pos + 1, TOPK) valid slots per row.
    """
    positions = torch.randint(0, 3 * TOPK, (num_tokens,), generator=gen)
    edges = torch.tensor([0, 1, TOPK - 1, TOPK])[:num_tokens]
    positions[: len(edges)] = edges
    lens = torch.clamp(positions + 1, max=TOPK)
    if case == "fixed":
        indices = torch.full((num_tokens, TOPK), -1, dtype=torch.int32)
        for row, n in enumerate(lens.tolist()):
            indices[row, :n] = torch.randint(0, num_slots, (n,), generator=gen)
        indptr = torch.arange(num_tokens + 1, dtype=torch.int32) * TOPK
        return indptr.cuda(), indices.flatten().cuda(), True
    indptr = torch.zeros(num_tokens + 1, dtype=torch.int32)
    indptr[1:] = lens.cumsum(0)
    indices = torch.randint(
        0, num_slots, (int(indptr[-1]),), generator=gen, dtype=torch.int32
    )
    return indptr.cuda(), indices.cuda(), False


def _run(ops, q, kv, indptr, indices, has_invalid, return_lse, scale):
    out = torch.empty(
        (q.shape[0], q.shape[1], KV_LORA_RANK), dtype=torch.bfloat16, device=q.device
    )
    lse = ops.triton_sparse_mla_fwd(
        q,
        kv,
        out,
        scale,
        indptr,
        indices,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=ROPE_DIM,
        has_invalid=has_invalid,
        return_lse=return_lse,
    )
    return out, lse


@pytest.mark.parametrize("case", ["fixed", "ragged"])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("num_heads,num_tokens", [(12, 300), (16, 37), (4, 1)])
@torch.inference_mode()
def test_flydsl_matches_triton(
    flydsl_ops, case: str, return_lse: bool, num_heads: int, num_tokens: int
) -> None:
    ops, launches = flydsl_ops
    gen = torch.Generator().manual_seed(num_heads * 1000 + num_tokens)
    num_slots = 3 * TOPK
    cache = _fp8_ds_mla_cache(num_slots, gen)
    # As forward_mqa hands it over: one row per page.
    kv = cache.unsqueeze(1)
    indptr, indices, has_invalid = _indices(case, num_tokens, num_slots, gen)
    q = (
        torch.randn(num_tokens, num_heads, KV_LORA_RANK + ROPE_DIM, generator=gen) * 0.5
    ).to("cuda", torch.bfloat16)
    scale = (KV_LORA_RANK + ROPE_DIM) ** -0.5

    out, lse = _run(ops, q, kv, indptr, indices, has_invalid, return_lse, scale)
    assert launches == [(num_heads, has_invalid, return_lse)]
    ops._FLYDSL_SPARSE_MLA = False
    ref_out, ref_lse = _run(ops, q, kv, indptr, indices, has_invalid, return_lse, scale)
    assert len(launches) == 1

    assert not out.isnan().any()
    torch.testing.assert_close(out, ref_out, atol=3e-2, rtol=2e-2)
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref_out.float().flatten(), dim=0
    )
    assert cos > 0.9999
    if return_lse:
        assert lse is not None and ref_lse is not None
        torch.testing.assert_close(lse, ref_lse.view_as(lse), atol=1e-3, rtol=1e-4)
    else:
        assert lse is None


@torch.inference_mode()
def test_flydsl_cache_past_4gib(flydsl_ops) -> None:
    """The non-DCP K3.1 call hands FlyDSL its whole packed block-row space,
    which spans more than 4 GiB on a real server: slots past that line must
    be addressed in 64 bits (DMA, scale and rope reads alike)."""
    ops, launches = flydsl_ops
    span = 5 * 2**30
    num_slots = span // ROW_BYTES
    free, _ = torch.accelerator.get_memory_info()
    if free < span + 2**30:
        pytest.skip("needs 6 GiB of free device memory")
    gen = torch.Generator().manual_seed(3)
    num_tokens, num_heads = 64, 16
    small = _fp8_ds_mla_cache(3 * TOPK, gen)
    n = small.shape[0]
    # The rows that get read, scattered to either side of the 4 GiB line.
    line = 2**32 // ROW_BYTES + 1
    slots = torch.cat(
        [
            torch.randperm(line, generator=gen)[: n // 2],
            line + torch.randperm(num_slots - line, generator=gen)[: n - n // 2],
        ]
    ).to("cuda", torch.int64)
    big = torch.empty(num_slots, ROW_BYTES, dtype=torch.uint8, device="cuda")
    big[slots] = small
    indptr, local, has_invalid = _indices("ragged", num_tokens, n, gen)
    indices = slots[local.long()].to(torch.int32)
    assert int(indices.max()) * ROW_BYTES > 2**32
    q = (
        torch.randn(num_tokens, num_heads, KV_LORA_RANK + ROPE_DIM, generator=gen) * 0.5
    ).to("cuda", torch.bfloat16)
    scale = (KV_LORA_RANK + ROPE_DIM) ** -0.5

    out, lse = _run(ops, q, big.unsqueeze(1), indptr, indices, has_invalid, True, scale)
    assert launches == [(num_heads, has_invalid, True)]
    # The same kernel over the compact cache with local ids is the same work
    # in the same order, so it is bit-exact.
    out_s, lse_s = _run(
        ops, q, small.unsqueeze(1), indptr, local, has_invalid, True, scale
    )
    assert torch.equal(out, out_s) and torch.equal(lse, lse_s)
    # The Triton kernel gathers with 64-bit offsets, so it is a reference too.
    ops._FLYDSL_SPARSE_MLA = False
    ref_out, ref_lse = _run(
        ops, q, big.unsqueeze(1), indptr, indices, has_invalid, True, scale
    )
    assert len(launches) == 2
    assert not out.isnan().any()
    torch.testing.assert_close(out, ref_out, atol=3e-2, rtol=2e-2)
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref_out.float().flatten(), dim=0
    )
    assert cos > 0.9999
    torch.testing.assert_close(lse, ref_lse.view_as(lse), atol=1e-3, rtol=1e-4)


@torch.inference_mode()
def test_flydsl_falls_back_to_triton(flydsl_ops, monkeypatch) -> None:
    ops, launches = flydsl_ops
    gen = torch.Generator().manual_seed(0)
    cache = _fp8_ds_mla_cache(TOPK, gen)
    indptr, indices, _ = _indices("ragged", 8, TOPK, gen)
    q = torch.randn(8, 16, KV_LORA_RANK + ROPE_DIM, generator=gen).to(
        "cuda", torch.bfloat16
    )

    def run():
        return _run(ops, q, cache.unsqueeze(1), indptr, indices, False, True, 0.04)

    ops._FLYDSL_SPARSE_MLA = False
    ref_out, ref_lse = run()
    ops._FLYDSL_SPARSE_MLA = True

    # aiter without the FlyDSL op.
    with monkeypatch.context() as m:
        m.setattr(ops, "_flydsl_sparse_mla_ops", staticmethod(lambda: None))
        out, lse = run()
    assert torch.equal(out, ref_out) and torch.equal(lse, ref_lse)
    assert not launches

    # A config that never ran eagerly is not compiled mid CUDA-graph capture.
    ops._FLYDSL_SPARSE_MLA_COMPILED.discard((16, False, True))
    with monkeypatch.context() as m:
        m.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        out, lse = run()
    assert torch.equal(out, ref_out) and torch.equal(lse, ref_lse)
    assert not launches
    run()
    assert launches == [(16, False, True)]
    with monkeypatch.context() as m:
        m.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        run()
    assert len(launches) == 2


def _run_sink(ops, q, kv, indptr, indices, scale, sink):
    out = torch.empty(
        (q.shape[0], q.shape[1], KV_LORA_RANK), dtype=torch.bfloat16, device=q.device
    )
    ops.triton_sparse_mla_fwd(
        q,
        kv,
        out,
        scale,
        indptr,
        indices,
        has_invalid=False,
        attn_sink=sink,
        return_lse=True,
    )
    return out


@torch.inference_mode()
def test_flydsl_rejects_unsupported_calls(flydsl_ops) -> None:
    """Calls FlyDSL cannot take (attention sink here) stay on Triton."""
    ops, launches = flydsl_ops
    gen = torch.Generator().manual_seed(1)
    cache = _fp8_ds_mla_cache(TOPK, gen)
    indptr, indices, _ = _indices("ragged", 8, TOPK, gen)
    q = torch.randn(8, 16, KV_LORA_RANK + ROPE_DIM, generator=gen).to(
        "cuda", torch.bfloat16
    )
    sink = torch.zeros(16, device="cuda")
    out = _run_sink(ops, q, cache.unsqueeze(1), indptr, indices, 0.04, sink)
    assert not launches
    ops._FLYDSL_SPARSE_MLA = False
    ref = _run_sink(ops, q, cache.unsqueeze(1), indptr, indices, 0.04, sink)
    assert torch.equal(out, ref)


IN_TREE_MODULE = "vllm.models.kimi_k3.amd.ops.flydsl_sparse_mla"
AITER_MODULE = "aiter.ops.flydsl.sparse_mla"


def _resolve(monkeypatch, broken: tuple[str, ...]):
    """_flydsl_sparse_mla_ops() with the named modules failing to import."""
    import importlib

    from vllm._aiter_ops import rocm_aiter_ops

    real = importlib.import_module

    def fake(name, *args, **kwargs):
        if name in broken:
            raise ImportError(f"no {name}")
        return real(name, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(importlib, "import_module", fake)
        rocm_aiter_ops._flydsl_sparse_mla_ops.cache_clear()
        try:
            return rocm_aiter_ops._flydsl_sparse_mla_ops()
        finally:
            rocm_aiter_ops._flydsl_sparse_mla_ops.cache_clear()


def test_flydsl_ops_prefer_in_tree(monkeypatch) -> None:
    """The lookup takes vLLM's own FlyDSL kernel over aiter's copy."""
    in_tree = pytest.importorskip(IN_TREE_MODULE)

    ops = _resolve(monkeypatch, ())
    assert ops is not None
    assert ops == (
        in_tree.flydsl_sparse_mla_fp8ds_supported,
        in_tree.flydsl_sparse_mla_fp8ds_fwd,
    )


def test_flydsl_ops_fall_back_to_aiter(monkeypatch) -> None:
    """Without the in-tree kernel the lookup still finds aiter's copy."""
    aiter_flydsl = pytest.importorskip(AITER_MODULE)

    assert _resolve(monkeypatch, (IN_TREE_MODULE,)) == (
        aiter_flydsl.flydsl_sparse_mla_fp8ds_supported,
        aiter_flydsl.flydsl_sparse_mla_fp8ds_fwd,
    )


def test_flydsl_ops_none_without_flydsl(monkeypatch) -> None:
    """A box with no FlyDSL at all keeps running on the Triton kernel.

    The lookup must report absence instead of raising; triton_sparse_mla_fwd
    treats None as "not handled" and falls through. Needs no FlyDSL install,
    so it runs wherever the ROCm kernel suite runs.
    """
    assert _resolve(monkeypatch, (IN_TREE_MODULE, AITER_MODULE)) is None


@torch.inference_mode()
def test_flydsl_min_tokens_gate(flydsl_ops, monkeypatch) -> None:
    """Batches below the min-tokens gate (decode) stay on Triton."""
    ops, launches = flydsl_ops
    gen = torch.Generator().manual_seed(2)
    cache = _fp8_ds_mla_cache(TOPK, gen)
    q = torch.randn(64, 12, KV_LORA_RANK + ROPE_DIM, generator=gen).to(
        "cuda", torch.bfloat16
    )
    indptr, indices, _ = _indices("ragged", 64, TOPK, gen)
    monkeypatch.setenv("VLLM_ROCM_AITER_FLYDSL_SPARSE_MLA_MIN_TOKENS", "65")
    ops.refresh_env_variables()
    _run(ops, q, cache.unsqueeze(1), indptr, indices, False, True, 0.04)
    assert not launches
    monkeypatch.setenv("VLLM_ROCM_AITER_FLYDSL_SPARSE_MLA_MIN_TOKENS", "64")
    ops.refresh_env_variables()
    _run(ops, q, cache.unsqueeze(1), indptr, indices, False, True, 0.04)
    assert launches == [(12, False, True)]


@pytest.fixture
def fp8_decode_env(monkeypatch):
    """rocm_aiter_ops with the FlyDSL fp8 decode flag on and no kernel warm."""
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FLYDSL_SPARSE_MLA_FP8_DECODE", "1")
    rocm_aiter_ops.refresh_env_variables()
    monkeypatch.setattr(rocm_aiter_ops, "_FLYDSL_SPARSE_MLA_FP8_DECODE_WARM", {})
    try:
        yield rocm_aiter_ops
    finally:
        monkeypatch.undo()
        rocm_aiter_ops.refresh_env_variables()


def _fp8_decode_meta(
    ops, rows: int, extra: bool = True, mxfp8: bool = True, capturing: bool = False
) -> bool:
    """The V4.1 decode call on meta tensors; True if FlyDSL took it."""
    meta = {"device": "meta"}
    with pytest.MonkeyPatch.context() as m:
        m.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
        return ops._flydsl_sparse_mla_fp8_decode(
            torch.empty(rows, H, D, dtype=torch.float8_e4m3fn, **meta),
            torch.empty(4, 64, D, dtype=torch.float8_e4m3fn, **meta),
            D**-0.5,
            torch.empty(rows + 1, dtype=torch.int32, **meta),
            torch.empty(rows, dtype=torch.int32, **meta),
            MAX_KEYS,
            extra_kv=torch.empty(4, 64, D, **meta) if extra else None,
            out=None if mxfp8 else torch.empty(rows, H, D, **meta),
            out_mxfp8=(torch.empty(1, **meta), torch.empty(1, **meta))
            if mxfp8
            else None,
        )


def test_fp8_decode_gate_and_warmup(fp8_decode_env, monkeypatch) -> None:
    """Runs without aiter's op (a stand-in for its can_use / fwd / warmup).

    The env flag, the row range and aiter's check gate the route. Each
    (extra list, output mode) compiles once, on an eager call: under graph
    capture an uncompiled one records Triton, and a failed compile keeps
    Triton without retrying.
    """
    ops = fp8_decode_env
    launches: list[int] = []
    warmups: list[tuple[bool, bool]] = []
    state = {"supported": True, "compiles": True}

    def warmup(device, *, mxfp8, has_extra):
        warmups.append((mxfp8, has_extra))
        if not state["compiles"]:
            raise RuntimeError("compile failed")

    monkeypatch.setattr(
        ops,
        "_flydsl_sparse_mla_fp8_decode_ops",
        staticmethod(
            lambda: (
                lambda *a, **k: state["supported"],
                lambda q, *a, **k: launches.append(q.shape[0]),
                warmup,
            )
        ),
    )
    lo, hi = ops._FLYDSL_SPARSE_MLA_FP8_DECODE_ROWS
    assert not _fp8_decode_meta(ops, 16, capturing=True)
    assert [_fp8_decode_meta(ops, r) for r in (lo - 1, lo, hi, hi + 1)] == [
        False,
        True,
        True,
        False,
    ]
    assert _fp8_decode_meta(ops, 96, capturing=True)
    assert not _fp8_decode_meta(ops, 96, mxfp8=False, capturing=True)
    assert not _fp8_decode_meta(ops, 96, extra=False, capturing=True)
    assert warmups == [(True, True)]

    state["compiles"] = False
    assert not _fp8_decode_meta(ops, 16, extra=False)
    assert not _fp8_decode_meta(ops, 16, extra=False)
    assert warmups == [(True, True), (True, False)]
    state["compiles"] = True
    assert _fp8_decode_meta(ops, 16, mxfp8=False)
    assert warmups[-1] == (False, True)

    state["supported"] = False
    assert not _fp8_decode_meta(ops, lo)
    state["supported"] = True
    ops._FLYDSL_SPARSE_MLA_FP8_DECODE = False
    assert not _fp8_decode_meta(ops, lo)
    ops._FLYDSL_SPARSE_MLA_FP8_DECODE = True
    monkeypatch.setattr(
        ops, "_flydsl_sparse_mla_fp8_decode_ops", staticmethod(lambda: None)
    )
    assert not _fp8_decode_meta(ops, lo)
    assert launches == [lo, hi, 96, 16]


def test_fp8_decode_ops_none_without_aiter_op(monkeypatch) -> None:
    """An aiter without the FlyDSL fp8 decode op keeps the Triton kernel."""
    import sys
    import types

    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl", types.ModuleType("x"))
    rocm_aiter_ops._flydsl_sparse_mla_fp8_decode_ops.cache_clear()
    try:
        assert rocm_aiter_ops._flydsl_sparse_mla_fp8_decode_ops() is None
    finally:
        rocm_aiter_ops._flydsl_sparse_mla_fp8_decode_ops.cache_clear()


@pytest.fixture
def fp8_decode_kernel(fp8_decode_env, monkeypatch):
    """fp8_decode_env with aiter's real kernel; yields the launched row counts."""
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("FlyDSL fp8 sparse MLA decode is gfx950-only")
    pytest.importorskip("aiter.ops.triton.attention.sparse_mla")
    ops = fp8_decode_env
    real = ops._flydsl_sparse_mla_fp8_decode_ops()
    if real is None:
        pytest.skip("aiter has no flydsl_sparse_mla_decode_fp8_h64")
    can_use, fwd, warmup = real
    launches: list[int] = []

    def logged_fwd(q, *args, **kwargs):
        launches.append(q.shape[0])
        fwd(q, *args, **kwargs)

    monkeypatch.setattr(
        ops,
        "_flydsl_sparse_mla_fp8_decode_ops",
        staticmethod(lambda: (can_use, logged_fwd, warmup)),
    )
    return ops, launches


def _ragged(rows: int, lo: int, hi: int, slots: int, empty: int, gen):
    lens = torch.randint(lo, hi + 1, (rows,), generator=gen)
    lens[0], lens[-1], lens[empty] = hi, lo, 0
    indptr = torch.zeros(rows + 1, dtype=torch.int32)
    indptr[1:] = lens.cumsum(0)
    indices = torch.randint(0, slots, (int(indptr[-1]),), generator=gen)
    return indptr.cuda(), indices.to("cuda", torch.int32)


def _v41_decode_call(rows: int, seed: int, extra: bool = True) -> dict:
    """The DeepSeek-V4.1 decode call of _plain_fp8_sparse_mla: fp8 q and
    caches with unit per-tensor scales, a SWA list and (unless SWA-only) a
    top-k list per row, a sink, inverse RoPE + MXFP8 output. Row rows // 2
    has no key in either list, like a CUDA-graph padding row."""
    gen = torch.Generator().manual_seed(seed)

    def e4m3(*shape):
        return torch.randn(*shape, generator=gen).to("cuda", torch.float8_e4m3fn)

    empty = rows // 2
    swa, comp = e4m3(64, 64, D), e4m3(256, 64, D)
    swa_indptr, swa_indices = _ragged(rows, 1, WINDOW, 64 * 64, empty, gen)
    topk_indptr, topk_indices = _ragged(rows, 0, TOPK_V41, 256 * 64, empty, gen)
    angles = torch.rand(4096, D // 16, generator=gen) * 6.3
    one = torch.ones(1, device="cuda")
    return dict(
        q=e4m3(rows, H, D),
        kv_buffer=swa,
        o=None,
        sm_scale=D**-0.5,
        kv_indptr=swa_indptr,
        kv_indices=swa_indices,
        kv_lora_rank=D,
        qk_rope_head_dim=0,
        q_scale=one,
        kv_scale=one,
        attn_sink=torch.randn(H, generator=gen).cuda(),
        extra_kv_buffer=comp if extra else None,
        extra_kv_indptr=topk_indptr if extra else None,
        extra_kv_indices=topk_indices if extra else None,
        has_invalid=False,
        dot_precision="fp8",
        inv_rope_positions=torch.randint(0, 4096, (rows,), generator=gen).cuda(),
        inv_rope_cos_sin_cache=torch.cat([angles.cos(), angles.sin()], 1).cuda(),
        out_mxfp8=(
            torch.empty(rows, H * D, dtype=torch.float8_e4m3fn, device="cuda"),
            torch.empty(rows, H * D // 32, dtype=torch.uint8, device="cuda"),
        ),
    )


def _mxfp8_rows(out_mxfp8) -> torch.Tensor:
    codes, scales = out_mxfp8
    rows = codes.shape[0]
    scale = torch.exp2(scales.float() - 127).view(rows, -1, 1)
    return (codes.float().view(rows, -1, 32) * scale).view(rows, -1)


def _run_fp8_decode(ops, call: dict, max_keys_per_row: int | None) -> torch.Tensor:
    for t in call["out_mxfp8"]:
        t.view(torch.uint8).fill_(0xFF)  # NaN codes and scales
    ops.triton_sparse_mla_fwd(**call, max_keys_per_row=max_keys_per_row)
    return _mxfp8_rows(call["out_mxfp8"])


@pytest.mark.parametrize(
    "rows,extra", [(6, True), (96, True), (2048, True), (96, False)]
)
@torch.inference_mode()
def test_fp8_decode_matches_triton(fp8_decode_kernel, rows: int, extra: bool) -> None:
    """Every head-group kernel (16, 32 and 64 heads per workgroup) matches the
    Triton kernel, with and without the top-k list and on a row with no
    keys, and a CUDA graph replays the eager result."""
    ops, launches = fp8_decode_kernel
    call = _v41_decode_call(rows, seed=rows, extra=extra)
    max_keys = MAX_KEYS if extra else WINDOW
    out = _run_fp8_decode(ops, call, max_keys)
    assert launches == [rows]
    ops._FLYDSL_SPARSE_MLA_FP8_DECODE = False
    ref = _run_fp8_decode(ops, call, max_keys)
    assert len(launches) == 1
    assert out.isfinite().all()
    empty = rows // 2
    assert not out[empty].any() and not ref[empty].any()
    keyed = torch.arange(rows, device=out.device) != empty
    # Both outputs are MXFP8-rounded and Triton's P is one e4m3 term (row
    # cosine ~0.9997 to fp64); observed minimum here 0.99901.
    cos = torch.nn.functional.cosine_similarity(out[keyed], ref[keyed], dim=1)
    assert cos.min() > 0.998

    ops._FLYDSL_SPARSE_MLA_FP8_DECODE = True
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.triton_sparse_mla_fwd(**call, max_keys_per_row=max_keys)
    for t in call["out_mxfp8"]:
        t.zero_()
    graph.replay()
    assert len(launches) == 2
    assert torch.equal(_mxfp8_rows(call["out_mxfp8"]), out)


@torch.inference_mode()
def test_fp8_decode_falls_back_to_triton(fp8_decode_kernel) -> None:
    """A call without the key bound (prefill), a bound above the kernel's
    1024 keys, or a single row runs the Triton kernel."""
    ops, launches = fp8_decode_kernel
    call = _v41_decode_call(16, seed=0)
    ops._FLYDSL_SPARSE_MLA_FP8_DECODE = False
    ref = _run_fp8_decode(ops, call, MAX_KEYS)
    ops._FLYDSL_SPARSE_MLA_FP8_DECODE = True
    for bound in (None, 1025):
        assert torch.equal(_run_fp8_decode(ops, call, bound), ref)
    single = _v41_decode_call(1, seed=1)
    _run_fp8_decode(ops, single, MAX_KEYS)
    assert not launches
