# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 x MXFP8 decode GEMV (tcgen05, block-scaled) for already-quantized activations
of at most 64 tokens (MAX_TOKENS).

Compiled on the first call per (N, K, split, ring, early trigger, PDL, push, MMA width),
which must happen outside
CUDA-graph capture, or ahead of any call by `precompile`; once `freeze` closes
compiles, a call whose kernel is not
compiled declines instead.
"""

from __future__ import annotations

import threading
from typing import Any

import torch


def _kernel():
    """The TP4 kernel module (imports the CuTe DSL; see check_cute_dsl)."""
    check_cute_dsl()
    from . import kernel

    return kernel


# Every cutlass.experimental.primitives name the kernel uses.
_PRIMS = (
    "barrier_cluster_arrive_relaxed",
    "barrier_cluster_wait",
    "barrier_cta_sync",
    "cp_async_bulk_tensor_prefetch",
    "cp_async_bulk_tensor_shared_cta_global",
    "CTAGroup",
    "elect_sync",
    "fence_mbarrier_init",
    "fence_proxy",
    "GridDepAction",
    "griddepcontrol",
    "mapa",
    "mbarrier_arrive",
    "mbarrier_arrive_expect_tx",
    "mbarrier_init",
    "mbarrier_try_wait_parity",
    "MBarrierScope",
    "MemScope",
    "MMABlockScaleKind",
    "prefetch_tensormap",
    "S2TCopyMode",
    "SharedSpace",
    "tcgen05_alloc",
    "tcgen05_commit",
    "tcgen05_cp",
    "tcgen05_dealloc",
    "tcgen05_fence",
    "tcgen05_ld",
    "tcgen05_mma_block_scale",
    "tcgen05_relinquish_alloc_permit",
    "tcgen05_wait",
    "Tcgen05CpShape",
    "Tcgen05Fence",
    "Tcgen05MxInstrDesc",
    "Tcgen05SmemDesc",
    "Tcgen05Wait",
)


class CuteDslUnavailable(RuntimeError):
    pass


def check_cute_dsl() -> None:
    """The kernel uses `cutlass.experimental.primitives` (tcgen05 block-scaled MMA,
    TMEM copies, griddepcontrol)
    and `cutlass.experimental.cuda` tensor maps. vLLM pins nvidia-cutlass-dsl 4.8.0
    (requirements/cuda.txt);
    raise a clear error if this install does not expose them."""
    try:
        import cutlass.experimental.cuda
        from cutlass.experimental import primitives as prims
    except ImportError as e:
        raise CuteDslUnavailable(
            "o_proj stage needs cutlass.experimental.primitives / "
            "cutlass.experimental.cuda from "
            f"nvidia-cutlass-dsl (vLLM pins 4.8.0): {e}"
        ) from e
    missing = [n for n in _PRIMS if not hasattr(prims, n)]
    missing += [
        f"cuda.{n}"
        for n in ("create_tensor_map_tiled", "TensorMap", "TensorMapSwizzle")
        if not hasattr(cutlass.experimental.cuda, n)
    ]
    if missing:
        raise CuteDslUnavailable(
            f"cutlass.experimental.primitives lacks {missing} (nvidia-cutlass-dsl "
            "drift)"
        )


MAX_TOKENS = 64
MMA_WIDTHS = (
    8,
    16,
    32,
    48,
    64,
    # token columns of the compiled MMA; a call uses the smallest that holds M (64: the
    # executor's widest step,
)
# 8 requests x 8 rows; the M128 x N64 block-scaled MMA reads each weight tile once for
# all rows)
_SMEM_LIMIT = 220 * 1024


def _mma_n(m: int) -> int:
    return next(w for w in MMA_WIDTHS if m <= w)


def _smem_bytes(ring: int, tiles: int, split: int, mma_n: int) -> int:
    return (
        ring * (128 * 128 + 512)
        + tiles * (mma_n * 128 + 512)
        + split * (128 // split) * (mma_n + 4) * 4
        + 1024
    )


_lock = threading.Lock()
_compiled: dict[tuple, Any] = {}

# (N, K) -> (split, ring): GB300 exposed time behind a 4.25 us PDL predecessor at M 1 /
# 8, vs the fused low-M GEMM:
# QKV 2.78 / 3.16 vs 4.58 / 6.21 us, o_proj 2.23 / 2.41 vs 4.40 / 5.09, shared gate_up
# 2.43 / 2.74 vs 3.56 / 4.30,
# shared down 1.74 / 1.66 vs 2.92 / 3.38. A config is (split, ring[, contig, nacc,
# a_tmem, st_async]); the optional
# fields default to _CONTIG, _NACC, _A_TMEM, _ST_ASYNC.
_CONFIGS: dict[tuple[int, int], tuple] = {
    (2560, 6144): (4, 12),
    (6144, 2048): (2, 8),
    # Shared-expert gate_up: split 2, ring 12, single accumulator, A operand from shared
    # memory.
    (1536, 6144): (2, 12, 1, 1, 0, 1),
    # Shared-expert down: unsplit, ring 3, single accumulator, A operand from shared
    # memory.
    (6144, 768): (1, 3, 1, 1, 0, 1),
    (2304, 6144): (4, 12),
    (6144, 3072): (2, 12),
    # Dense-layer gate_up.
    (6144, 6144): (2, 12),
}
# Per-token-count configs (N, K, M): the grid that suits the concurrent routed-expert
# engine depends on how long that engine runs. At 2 tokens the shared-expert gate_up
# splits K over 4 CTAs.
_CONFIGS_BY_M: dict[tuple[int, int, int], tuple] = {
    (1536, 6144, 2): (4, 12, 1, 1, 0, 1),
}
# Above 8 tokens (spec-decode verify rows): the shared-expert gate_up splits K over 4
# CTAs.
_CONFIGS_WIDE: dict[tuple[int, int], tuple] = {
    (1536, 6144): (4, 12, 1, 1, 0, 1),
}

# Defaults of the optional config fields: contiguous k-tile ranges per split rank, one
# TMEM accumulator, the A operand staged in TMEM (when its columns fit), st.async
# split-K partial exchange.
_CONTIG = True
_NACC = 1
_A_TMEM = True
_ST_ASYNC = True
_ensured = set()


def _arg(t: torch.Tensor, align: int = 16):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(
        leading_dim=t.dim() - 1
    )


def get_config(n: int, k: int, m: int | None = None) -> tuple | None:
    cfg = _CONFIGS_BY_M.get((n, k, m)) if m is not None else None
    if cfg is None and m is not None and m > 8:
        cfg = _CONFIGS_WIDE.get((n, k))
    cfg = cfg or _CONFIGS.get((n, k))
    # The split-K epilogue gives each of the split CTAs 128 // split rows of a 128-row
    # output tile.
    if (
        cfg is None
        or n % 128 != 0
        or k % 128 != 0
        or (k // 128) % cfg[0] != 0
        or 128 % cfg[0] != 0
    ):
        return None
    return cfg


def _declines(n: int, k: int, m: int, x_bf16: bool, cfg: tuple) -> bool:
    """Whether a (split, ring, ...) config declines an [M, K] x [N, K]^T call on shape
    grounds."""
    split = cfg[0]
    if not 0 < m <= MAX_TOKENS or 128 % split != 0 or (k // 128) % split != 0:
        return True
    tiles = k // 128 // split
    mma_n = _mma_n(m)
    # GB300 exposed time vs the FlashInfer prequantized GEMM (behind a PDL predecessor),
    # at 24 / 48 tokens: QKV 3.76 /
    # 4.63 vs 7.46 / 11.44 us, o_proj 2.74 / 3.33 vs 4.46 / 5.84, shared gate_up 4.57 /
    # 5.71 vs 7.16 / 11.25. Behind
    # only for K = 768 (unsplit shared down) at 17-32 tokens: 3.1 vs 2.9.
    # The in-kernel bf16 -> MXFP8 quantization (x_bf16) costs one 32-element block per
    # thread and round on 128 threads;
    # beyond 8 rounds per CTA, mxfp8_quantize + this GEMV (or the fallback GEMM) is
    # faster.
    if (16 < m <= 32 and k < 1024) or (x_bf16 and tiles * 4 * mma_n > 8 * 128):
        return True
    return _smem_bytes(min(cfg[1], tiles, 2), tiles, split, mma_n) > _SMEM_LIMIT


def serves(n: int, k: int, m: int, x_bf16: bool) -> bool:
    """Whether the GEMV with its default config takes an [M, K] activation (bf16 or
    MXFP8) against an [N, K] weight."""
    cfg = get_config(n, k, m)
    return cfg is not None and not _declines(n, k, m, x_bf16, cfg)


def _resolve(
    m: int,
    n: int,
    k: int,
    x_bf16: bool,
    trigger_early: bool,
    cfg: tuple | None,
    use_pdl: bool | None,
    ar: tuple | None,
    ra_tab: int = 0,
) -> tuple | None:
    """The compile key of an [M, K] x [N, K]^T call and its runtime push words, or None
    when the call is declined.

    `ar` is the push tuple of `_run`; only its (world, bucket tokens, mode) enter
    the key, the addresses and the
    rank are runtime arguments."""
    cfg = cfg or get_config(n, k, m)
    if cfg is None or _declines(n, k, m, x_bf16, cfg):
        return None
    split, ring = cfg[0], cfg[1]
    contig = bool(cfg[2]) if len(cfg) > 2 else _CONTIG
    nacc = int(cfg[3]) if len(cfg) > 3 else _NACC
    a_tmem = bool(cfg[4]) if len(cfg) > 4 else _A_TMEM
    st_async = bool(cfg[5]) if len(cfg) > 5 else _ST_ASYNC
    stagger = int(cfg[6]) if len(cfg) > 6 else 0
    early_w = bool(cfg[7]) if len(cfg) > 7 else False
    qfirst = int(cfg[8]) if len(cfg) > 8 else 0
    burst = int(cfg[9]) if len(cfg) > 9 else 0
    ptrig = (
        int(cfg[10]) if len(cfg) > 10 else 0
    )  # push modes keep the wait-exit trigger (post resident, polling)
    pfx = (
        bool(cfg[11]) if len(cfg) > 11 else False
    )  # split-K push 4: flags / peer bases before the MMA wait
    ra = (
        int(cfg[12]) if len(cfg) > 12 else 0
    )  # router CTAs ahead of the GEMV grid (ra_tab required)
    ra_ck = int(cfg[13]) if len(cfg) > 13 else 0  # 1: trap on an unpaired call
    assert not ra or ra_tab, "ra: ra_tab (device int64[RA_TAB_WORDS]) required"
    ra_tok = m if ra else 0
    assert burst != 2 or contig, "burst 2 (per-row prefetch) needs contiguous k-ranges"
    ring = min(ring, k // 128 // split)
    tiles = k // 128 // split
    mma_n = _mma_n(m)
    while ring > 2 and _smem_bytes(ring, tiles, split, mma_n) > _SMEM_LIMIT:
        ring -= 1
    if a_tmem:
        tmem_cols = _kernel().tmem_cols

        if tmem_cols(ring, nacc, True, tiles, mma_n) > 512:
            a_tmem = False
    # ar = (multicast workspace address, flags address, rank, world, bucket tokens,
    # mode): push the output rows into the next m3_mc_allreduce call's exchange slots
    # (mode 2 also defers the dependents' launch until after the push).
    ar_ws, ar_flags, ar_rank, ar_mode, ar_world, ar_maxt = 0, 0, 0, 0, 1, 8
    if ar is not None and split <= 4 and n % 2 == 0:
        ar_ws, ar_flags, ar_rank, ar_world, ar_maxt, ar_mode = ar
        if ar_mode >= 2 and not ptrig:
            trigger_early = False
    if use_pdl is None:
        use_pdl = True
    key = (
        n,
        k,
        split,
        ring,
        trigger_early,
        contig,
        nacc,
        a_tmem,
        st_async,
        x_bf16,
        ar_mode,
        ar_world,
        ar_maxt,
        mma_n,
        use_pdl,
        stagger,
        early_w,
        qfirst,
        burst,
        pfx,
        ra,
        ra_tok,
        ra_ck,
    )
    return key, (ar_ws, ar_flags, ar_rank, ra_tab)


def _compile(key: tuple, args: tuple, m: int, runtime_ar: tuple, stream, extra: tuple):
    """cute.compile the kernel of `key` (see `_resolve`) on these argument views;
    cached."""
    import cutlass.cute as cute

    kernel = _kernel()

    (
        n,
        k,
        split,
        ring,
        trigger_early,
        contig,
        nacc,
        a_tmem,
        st_async,
        x_bf16,
        ar_mode,
        ar_world,
        ar_maxt,
        mma_n,
        use_pdl,
        stagger,
        early_w,
        qfirst,
        burst,
        pfx,
        ra,
        ra_tok,
        ra_ck,
    ) = key
    ar_ws, ar_flags, ar_rank, ra_tab = runtime_ar
    with _lock:
        fn = _compiled.get(key)
        if fn is None:
            fn = _compiled[key] = cute.compile(
                kernel.m3_mx_gemv,
                *args,
                m,
                n,
                k,
                split,
                ring,
                trigger_early,
                contig,
                nacc,
                a_tmem,
                st_async,
                x_bf16,
                ar_ws,
                ar_flags,
                ar_rank,
                ar_mode,
                ar_world,
                ar_maxt,
                mma_n,
                use_pdl,
                stream,
                *extra,
                False,
                False,
                stagger,
                early_w,
                qfirst,
                burst,
                pfx,
                ra,
                ra_tab,
                ra_tok,
                ra_ck,
            )
    return fn


def _views(
    x: torch.Tensor,
    xsf: torch.Tensor | None,
    weight,
    weight_scale,
    y,
    x_bf16: bool,
) -> tuple:
    k = x.shape[1]
    if x_bf16:
        x_bytes = x.view(torch.uint8)
        xsf = _placeholder_scales(k, x.device)
        x_words = x.view(torch.int32)
    else:
        x_bytes = x.view(torch.uint8)
        x_words = _placeholder_words(x.device)
    return (
        _arg(weight.view(torch.uint8)),
        _arg(weight_scale.view(torch.uint8).reshape(-1)),
        _arg(x_bytes),
        _arg(xsf.view(torch.uint8).reshape(-1)),  # type: ignore[union-attr]
        _arg(
            y.view(-1)
            if y.numel()
            else _placeholder_words(x.device).view(torch.bfloat16)
        ),
        _arg(x_words.reshape(-1)),
    )


def _run(
    x: torch.Tensor,
    xsf: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_bf16: bool,
    trigger_early: bool,
    cfg: tuple | None,
    use_pdl: bool | None,
    ar: tuple | None = None,
    ra_tab: int = 0,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    global last_pushed, last_uncompiled
    last_pushed = False
    last_uncompiled = None
    import cuda.bindings.driver as cuda_driver

    m, k = x.shape
    n = weight.shape[0]
    if weight.shape[1] != k:
        return None
    resolved = _resolve(m, n, k, x_bf16, trigger_early, cfg, use_pdl, ar, ra_tab)
    if resolved is None:
        return None
    key, runtime_ar = resolved
    fn = _compiled.get(key)
    if fn is None:
        if _frozen[0]:
            # Compiles are closed (the model built every kernel it dispatches to before
            # warmup): decline, the caller reports the miss.
            last_uncompiled = key
            return None
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "m3_mx_gemv must run once per shape outside CUDA-graph capture first"
            )
    y = (
        out
        if out is not None
        else torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    )
    assert y.shape == (m, n) and y.dtype == torch.bfloat16 and y.is_contiguous()
    args = _views(x, xsf, weight, weight_scale, y, x_bf16)
    extra = _extra_args(x.device)
    stream = cuda_driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    if fn is None:
        fn = _compile(key, args, m, runtime_ar, stream, extra)
    ar_ws, ar_flags, ar_rank, ra_tab = runtime_ar
    fn(*args, m, ar_ws, ar_flags, ar_rank, stream, *extra, ra_tab)
    last_pushed = key[10] > 0
    return y


def precompile(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    m: int,
    x_bf16: bool,
    use_pdl: bool | None = None,
    ar: tuple[int, int, int] | None = None,
) -> tuple | None:
    """Compile, without launching, the kernel an M-row call against `weight` takes;
    returns its key, or None when
    such a call is declined. `ar` = (world, bucket tokens, mode) of a pushing call
    (its addresses and rank are
    runtime arguments, so none is needed here). Runs outside capture, even while
    compiles are frozen."""
    import cuda.bindings.driver as cuda_driver

    n, k = weight.shape
    push = None if ar is None else (0, 0, 0, *ar)
    resolved = _resolve(m, n, k, x_bf16, True, None, use_pdl, push)
    if resolved is None:
        return None
    key, runtime_ar = resolved
    if key in _compiled:
        return key
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "m3_mx_gemv.precompile allocates and compiles: call it outside capture"
        )
    dev = weight.device
    x = torch.zeros(
        m, k, dtype=torch.bfloat16 if x_bf16 else torch.float8_e4m3fn, device=dev
    )
    xsf = (
        None
        if x_bf16
        else torch.full((128 * (k // 32),), 127, dtype=torch.uint8, device=dev)
    )
    y = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
    args = _views(x, xsf, weight, weight_scale, y, x_bf16)
    stream = cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    _compile(key, args, m, runtime_ar, stream, _extra_args(dev))
    return key


def freeze(frozen: bool = True) -> None:
    """Close (or reopen) compiles: while frozen, a call whose kernel is not compiled yet
    declines (returns None and
    sets `last_uncompiled`) instead of compiling. `precompile` and
    `ensure_compiled` still compile."""
    _frozen[0] = bool(frozen)


def frozen() -> bool:
    return _frozen[0]


def is_compiled(key: tuple) -> bool:
    return key in _compiled


def _extra_args(device) -> tuple:
    """Placeholder arguments of the kernel's draft-input prologue and RoPE / KV epilogue
    (compiled off)."""
    i32 = _placeholder_words(device)
    f32 = i32.view(torch.float32)
    u8 = i32.view(torch.uint8)
    d = (
        _arg(i32),
        _arg(i32),
        _arg(i32),
        _arg(i32),
        _arg(f32),
        0,
        _arg(i32),
        _arg(i32),
        0,
    )
    r = (_arg(f32), _arg(i32), _arg(i32), _arg(u8), _arg(u8), _arg(f32), _arg(i32))
    return d + r


def m3_mx_gemv(
    xq: torch.Tensor,
    xsf: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    trigger_early: bool = True,
    cfg: tuple | None = None,
    use_pdl: bool | None = None,
    ar: tuple | None = None,
    ra_tab: int = 0,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """`xq` E4M3 [M <= 64, K] with `xsf` its 128x4-swizzled UE8M0 scales; `weight`
    E4M3 [N, K] with
    `weight_scale` swizzled. Returns bf16 [M, N], or None when the shape is not
    served."""
    return _run(
        xq,
        xsf,
        weight,
        weight_scale,
        False,
        trigger_early,
        cfg,
        use_pdl,
        ar,
        ra_tab,
        out,
    )


def m3_mx_gemv_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    trigger_early: bool = True,
    cfg: tuple | None = None,
    use_pdl: bool | None = None,
    ar: tuple | None = None,
) -> torch.Tensor | None:
    """The same GEMV for a bf16 `x` [M <= 8, K] (contiguous), quantized to MXFP8
    inside the kernel exactly as
    torch.ops.trtllm.mxfp8_quantize(x, True) does."""
    if (
        x.dtype != torch.bfloat16
        or x.dim() != 2
        or not x.is_contiguous()
        or x.data_ptr() % 16 != 0
    ):
        return None
    return _run(x, None, weight, weight_scale, True, trigger_early, cfg, use_pdl, ar)


_PLACEHOLDERS: dict[tuple, torch.Tensor] = {}
# Whether the last call pushed its output into an all-reduce (see _run's ``ar``).
last_pushed = False
# The key of the last call declined because compiles are frozen and its kernel was not
# compiled (see ``freeze``).
last_uncompiled: tuple | None = None
_frozen = [False]


def _placeholder_scales(k: int, device) -> torch.Tensor:
    key = ("sf", k, device)
    t = _PLACEHOLDERS.get(key)
    if t is None:
        t = _PLACEHOLDERS[key] = torch.zeros(
            k // 128 * 512, dtype=torch.uint8, device=device
        )
    return t


def _placeholder_words(device) -> torch.Tensor:
    key = ("w", device)
    t = _PLACEHOLDERS.get(key)
    if t is None:
        t = _PLACEHOLDERS[key] = torch.zeros(16, dtype=torch.int32, device=device)
    return t


def ensure_compiled(
    weight: torch.Tensor, weight_scale: torch.Tensor, use_pdl: bool
) -> None:
    """Compiles the kernel of this weight's shape outside capture (a one-token dummy
    call)."""
    if (
        weight.dim() != 2
        or weight.dtype != torch.float8_e4m3fn
        or get_config(*weight.shape) is None
    ):
        return
    n, k = weight.shape
    if (n, k, use_pdl) in _ensured:
        return
    _ensured.add((n, k, use_pdl))
    xq = torch.zeros(1, k, dtype=torch.float8_e4m3fn, device=weight.device)
    xsf = torch.full((128 * (k // 32),), 127, dtype=torch.uint8, device=weight.device)
    m3_mx_gemv(xq, xsf, weight, weight_scale, use_pdl=use_pdl)
    for width in MMA_WIDTHS[1:]:
        m3_mx_gemv(
            torch.zeros(width, k, dtype=torch.float8_e4m3fn, device=weight.device),
            xsf,
            weight,
            weight_scale,
            use_pdl=use_pdl,
        )
    m3_mx_gemv_bf16(
        torch.zeros(1, k, dtype=torch.bfloat16, device=weight.device),
        weight,
        weight_scale,
        use_pdl=use_pdl,
    )


def compile_call(
    xq: torch.Tensor,
    xsf: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    cfg: tuple,
    ar: tuple,
    ra_tab: int,
    out: torch.Tensor,
    use_pdl: bool = True,
) -> tuple:
    """Compile (without launching) exactly the kernel `m3_mx_gemv(xq, xsf, weight,
    weight_scale, cfg=cfg,
    use_pdl=use_pdl, ar=ar, ra_tab=ra_tab, out=out)` will run, so the first live call
    never compiles: a live
    call that compiles lazily while peers already poll its pushes can run past their
    deadlines. Returns the key."""
    import cuda.bindings.driver as cuda_driver

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("compile_call: outside CUDA-graph capture only")
    m, k = xq.shape
    n = weight.shape[0]
    resolved = _resolve(m, n, k, False, True, cfg, use_pdl, ar, ra_tab)
    if resolved is None:
        raise ValueError(f"o_proj GEMV declines M={m} N={n} K={k} cfg={cfg}")
    key, runtime_ar = resolved
    if key in _compiled:
        return key
    args = _views(xq, xsf, weight, weight_scale, out, False)
    stream = cuda_driver.CUstream(torch.cuda.current_stream(xq.device).cuda_stream)
    _compile(key, args, m, runtime_ar, stream, _extra_args(xq.device))
    return key
