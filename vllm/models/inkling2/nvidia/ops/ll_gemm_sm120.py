# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-M BF16 tensor-core projections for SM120/SM121: batched `x @ w.T` with
deterministic split-K for narrow outputs, and a grouped q/k/v projection."""

import functools
from typing import NamedTuple

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import current_stream

MAX_M = 16
_MP = 16
_MAX_TILES = 4096


@triton.jit
def _ll_gemm_kernel(
    X,
    W,
    Bias,
    Y,
    P,
    C,
    M,
    N,
    K,
    SXB,
    SXM,
    SWB,
    SWN,
    SYB,
    SYM,
    BN: tl.constexpr,
    BK: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
    SPLIT: tl.constexpr,
    MP: tl.constexpr,
):
    """Program `(n, b, s)` computes column tile `n` of batch `b` over K range
    `s`. With `SPLIT > 1` it writes fp32 partials to `P` `[SPLIT, B, MP, N]`;
    the last of the tile's programs to finish (counted in `C`, reset to 0
    after) sums them in split order, so results are deterministic."""
    tl.extra.cuda.gdc_wait()
    pid_n = tl.program_id(0)
    b = tl.program_id(1)
    s = tl.program_id(2)
    rows = tl.arange(0, MP)
    cols = pid_n * BN + tl.arange(0, BN)
    ks = s * K_PER_SPLIT + tl.arange(0, BK)
    x_ptrs = X + (b * SXB + rows[:, None] * SXM + ks[None, :])
    w_ptrs = W + (b * SWB + cols[None, :] * SWN + ks[:, None])
    acc = tl.zeros((MP, BN), tl.float32)
    for _ in range(0, tl.minimum(K - s * K_PER_SPLIT, K_PER_SPLIT), BK):
        # Issuing W's copies before X's measured faster.
        w = tl.load(w_ptrs, (cols[None, :] < N) & (ks[:, None] < K), other=0)
        x = tl.load(x_ptrs, (rows[:, None] < M) & (ks[None, :] < K), other=0)
        acc += tl.dot(x, w)
        x_ptrs += BK
        w_ptrs += BK
        ks += BK

    is_last = True
    if SPLIT > 1:
        num_b = tl.num_programs(1)
        tile = b * tl.num_programs(0) + pid_n
        offs = rows[:, None] * N + cols[None, :]
        mask = (rows[:, None] < M) & (cols < N)[None, :]
        tl.store(P + ((s * num_b + b) * MP * N + offs), acc, mask)
        tl.debug_barrier()
        is_last = tl.atomic_add(C + tile, 1, sem="acq_rel", scope="gpu") == SPLIT - 1
        if is_last:
            acc = tl.zeros((MP, BN), tl.float32)
            for i in tl.static_range(SPLIT):
                acc += tl.load(
                    P + ((i * num_b + b) * MP * N + offs),
                    mask,
                    cache_modifier=".cg",
                )
            tl.atomic_xchg(C + tile, 0, sem="relaxed", scope="gpu")
    if is_last:
        if Bias is not None:
            acc += tl.load(Bias + cols, cols < N, other=0).to(tl.float32)[None, :]
        tl.store(
            Y + (b * SYB + rows[:, None] * SYM + cols[None, :]),
            acc.to(Y.dtype.element_ty),
            (rows[:, None] < M) & (cols[None, :] < N),
        )
    tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _segment_input(b, XQ, XK, XV, SXQ, SXK, SXV):
    if b == 0:
        x, stride = XQ, SXQ
    elif b == 1:
        x, stride = XK, SXK
    else:
        x, stride = XV, SXV
    return x, stride


@triton.jit
def _qkv_bmm_kernel(
    XQ,
    XK,
    XV,
    W,
    Y,
    M,
    N,
    K,
    S1,
    S2,
    SXQ,
    SXK,
    SXV,
    SWN,
    SYM,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    """Columns `[0, S1)` of `Y` are `XQ @ W.T`, `[S1, S2)` `XK` and `[S2, N)`
    `XV`. A tile spanning a boundary runs both inputs' dots and picks per
    column (the second input's loads hit L2)."""
    tl.extra.cuda.gdc_wait()
    n0 = tl.program_id(0) * BN
    n1 = tl.minimum(n0 + BN, N) - 1
    b_lo = (n0 >= S1).to(tl.int32) + (n0 >= S2).to(tl.int32)
    b_hi = (n1 >= S1).to(tl.int32) + (n1 >= S2).to(tl.int32)
    x_lo, sx_lo = _segment_input(b_lo, XQ, XK, XV, SXQ, SXK, SXV)
    x_hi, sx_hi = _segment_input(b_hi, XQ, XK, XV, SXQ, SXK, SXV)
    rows = tl.arange(0, MP)
    cols = n0 + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    x_lo += rows[:, None] * sx_lo + ks[None, :]
    x_hi += rows[:, None] * sx_hi + ks[None, :]
    w_ptrs = W + (cols[None, :] * SWN + ks[:, None])
    acc_lo = tl.zeros((MP, BN), tl.float32)
    acc_hi = tl.zeros((MP, BN), tl.float32)
    for _ in range(0, K, BK):
        x_mask = (rows[:, None] < M) & (ks[None, :] < K)
        w = tl.load(w_ptrs, (cols[None, :] < N) & (ks[:, None] < K), other=0)
        acc_lo += tl.dot(tl.load(x_lo, x_mask, other=0), w)
        acc_hi += tl.dot(tl.load(x_hi, x_mask, other=0), w)
        x_lo += BK
        x_hi += BK
        w_ptrs += BK
        ks += BK
    segment = (cols >= S1).to(tl.int32) + (cols >= S2).to(tl.int32)
    acc = tl.where((segment == b_lo)[None, :], acc_lo, acc_hi)
    tl.store(
        Y + (rows[:, None] * SYM + cols[None, :]),
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )
    tl.extra.cuda.gdc_launch_dependents()


class _Config(NamedTuple):
    bn: int
    bk: int
    split: int
    k_per_split: int
    num_warps: int
    num_stages: int


@functools.cache
def _config(nb: int, n: int, k: int) -> _Config:
    """Split K into about one wave of programs when that gives at least 3
    splits, preferring 16-column tiles; otherwise one 32-column tile over all
    of K."""
    num_sms = num_compute_units()
    bk = min(128, max(16, triton.next_power_of_2(k)))
    k_tiles = triton.cdiv(k, bk)
    for bn in (16, 32):
        split = min(32, k_tiles, num_sms // (nb * triton.cdiv(n, bn)))
        if split >= 3:
            tiles_per_split = triton.cdiv(k_tiles, split)
            split = triton.cdiv(k_tiles, tiles_per_split)
            return _Config(bn, bk, split, tiles_per_split * bk, 4, 4)
    bk = min(256, max(16, triton.next_power_of_2(k)))
    return _Config(32, bk, 1, triton.cdiv(k, bk) * bk, 4, 3)


@functools.cache
def _counter(stream: int) -> torch.Tensor:
    """Split-K tile counters for one stream; each launch leaves them zeroed."""
    return torch.zeros(_MAX_TILES, dtype=torch.int32, device="cuda")


def _ll_gemm(
    x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None, out: torch.Tensor
) -> None:
    nb, m, k = x.shape
    n = w.shape[1]
    assert 0 < m <= MAX_M and w.shape == (nb, n, k)
    assert x.stride(2) == w.stride(2) == 1
    assert bias is None or (bias.shape == (n,) and bias.stride(0) == 1)
    assert out.shape == (nb, m, n) and out.stride(2) == 1
    assert max(nb * w.stride(0), nb * x.stride(0), nb * out.stride(0)) < 2**31
    cfg = _config(nb, n, k)
    n_tiles = triton.cdiv(n, cfg.bn)
    partials = counter = None
    if cfg.split > 1:
        assert nb * n_tiles <= _MAX_TILES
        partials = torch.empty(
            (cfg.split, nb, _MP, n), dtype=torch.float32, device=x.device
        )
        counter = _counter(current_stream().cuda_stream)
    _ll_gemm_kernel[(n_tiles, nb, cfg.split)](
        x,
        w,
        bias,
        out,
        partials,
        counter,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        BN=cfg.bn,
        BK=cfg.bk,
        K_PER_SPLIT=cfg.k_per_split,
        SPLIT=cfg.split,
        MP=_MP,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
        launch_pdl=True,
    )


def use_qkv_bmm(
    x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor, w: torch.Tensor
) -> bool:
    """Whether `qkv_bmm` serves bf16 `x_q`, `x_k`, `x_v` `[M, K]` and `w`."""
    return (
        x_q.is_cuda
        and x_q.ndim == 2
        and 0 < x_q.shape[0] <= MAX_M
        and x_q.shape == x_k.shape == x_v.shape
        and x_q.dtype == x_k.dtype == x_v.dtype == w.dtype == torch.bfloat16
        and x_q.stride(1) == x_k.stride(1) == x_v.stride(1) == w.stride(1) == 1
        and current_platform.is_device_capability_family(120)
        and not envs.VLLM_BATCH_INVARIANT
    )


def qkv_bmm(
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    w: torch.Tensor,
    k_start: int,
    v_start: int,
) -> torch.Tensor:
    """One launch of three linears on consecutive row ranges of `w` `[N, K]`:
    output columns `[0, k_start)` from `x_q`, `[k_start, v_start)` from `x_k`
    and the rest from `x_v`, each `[M, K]` with M <= 16. Returns `[M, N]`
    with fp32 accumulation."""
    xs = (x_q, x_k, x_v)
    m, k = x_q.shape
    n = w.shape[0]
    s1, s2 = k_start, v_start
    bn = 32
    assert 0 < m <= MAX_M and w.shape == (n, k)
    assert all(x.shape == (m, k) and x.stride(1) == 1 for x in xs)
    assert w.stride(1) == 1 and s1 > 0 and s1 + bn <= s2 <= n  # one boundary per tile
    assert max(n * w.stride(0), *(m * x.stride(0) for x in xs)) < 2**31
    out = torch.empty((m, n), dtype=x_q.dtype, device=x_q.device)
    # About 1.6 waves of 32-column tiles at Inkling's 8608 columns; wider
    # tiles or split-K measured no faster.
    _qkv_bmm_kernel[(triton.cdiv(n, bn),)](
        x_q,
        x_k,
        x_v,
        w,
        out,
        m,
        n,
        k,
        s1,
        s2,
        x_q.stride(0),
        x_k.stride(0),
        x_v.stride(0),
        w.stride(0),
        out.stride(0),
        BN=bn,
        BK=128,
        MP=_MP,
        num_warps=4,
        num_stages=4,
        launch_pdl=True,
    )
    return out


def sm12x_bf16_ll_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Low-latency `x @ weight.T` (+ `bias` `[N]`) for at most `MAX_M` rows,
    with fp32 accumulation, into `out` if given. `weight` `[N, K]` takes `x`
    `[..., K]` -> `[..., N]`; `weight` `[B, N, K]` takes `x` `[B, M, K]` ->
    `[B, M, N]`."""
    n = weight.shape[-2]
    if weight.ndim == 3:
        if out is None:
            out = x.new_empty((*x.shape[:2], n), dtype=out_dtype or x.dtype)
        _ll_gemm(x, weight, bias, out)
        return out
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if out is None:
        out = x.new_empty((x2.shape[0], n), dtype=out_dtype or x.dtype)
    if x2.shape[0]:
        _ll_gemm(x2[None], weight[None], bias, out[None])
    return out.view(*lead, n)


def use_sm12x_bf16_ll_gemm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> bool:
    """Whether `sm12x_bf16_ll_gemm` serves these operands."""
    if weight.ndim == 2:
        shapes_ok = x.ndim >= 2
        m = x.numel() // max(x.shape[-1], 1)
    else:
        shapes_ok = weight.ndim == 3 and x.ndim == 3 and x.shape[0] == weight.shape[0]
        m = x.shape[-2] if x.ndim >= 2 else 0
    return (
        shapes_ok
        and x.is_cuda
        and weight.device == x.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.shape[-1] == weight.shape[-1]
        and x.stride(-1) == weight.stride(-1) == 1
        and 0 < m <= MAX_M
        and (
            bias is None
            or (
                bias.dtype == torch.bfloat16
                and bias.device == x.device
                and bias.stride(0) == 1
            )
        )
        and current_platform.is_device_capability_family(120)
        and not envs.VLLM_BATCH_INVARIANT
    )
