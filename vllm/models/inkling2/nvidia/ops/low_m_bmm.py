# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-M BF16 batched `x @ w.T` for SM120/SM121, with deterministic split-K."""

import functools
from typing import NamedTuple

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import current_stream

MAX_M = 16
_MP = 16
_MAX_TILES = 4096


@triton.jit(do_not_specialize=["M"])
def _low_m_bmm_kernel(
    X,
    W,
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
        x = tl.load(x_ptrs, (rows[:, None] < M) & (ks[None, :] < K), other=0)
        w = tl.load(w_ptrs, (cols[None, :] < N) & (ks[:, None] < K), other=0)
        acc += tl.dot(x, w)
        x_ptrs += BK
        w_ptrs += BK
        ks += BK

    is_last = True
    if SPLIT > 1:
        num_b = tl.num_programs(1)
        tile = b * tl.num_programs(0) + pid_n
        offs = rows[:, None] * N + cols[None, :]
        col_mask = (cols < N)[None, :]
        tl.store(P + ((s * num_b + b) * MP * N + offs), acc, col_mask)
        tl.debug_barrier()
        is_last = tl.atomic_add(C + tile, 1, sem="acq_rel", scope="gpu") == SPLIT - 1
        if is_last:
            acc = tl.zeros((MP, BN), tl.float32)
            for i in tl.static_range(SPLIT):
                acc += tl.load(
                    P + ((i * num_b + b) * MP * N + offs),
                    col_mask,
                    cache_modifier=".cg",
                )
            tl.atomic_xchg(C + tile, 0, sem="relaxed", scope="gpu")
    if is_last:
        tl.store(
            Y + (b * SYB + rows[:, None] * SYM + cols[None, :]),
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
    splits, preferring 16-column tiles; otherwise one 32-column tile each."""
    num_sms = num_compute_units()
    bk = min(128, max(16, triton.next_power_of_2(k)))
    k_tiles = triton.cdiv(k, bk)
    for bn in (16, 32):
        split = min(32, k_tiles, num_sms // (nb * triton.cdiv(n, bn)))
        if split >= 3:
            tiles_per_split = triton.cdiv(k_tiles, split)
            split = triton.cdiv(k_tiles, tiles_per_split)
            return _Config(bn, bk, split, tiles_per_split * bk, 4, 4)
    return _Config(32, bk, 1, k_tiles * bk, 4, 4)


@functools.cache
def _counter(stream: int) -> torch.Tensor:
    """Split-K tile counters for one stream; each launch leaves them zeroed."""
    return torch.zeros(_MAX_TILES, dtype=torch.int32, device="cuda")


def use_low_m_bmm(x: torch.Tensor, w: torch.Tensor) -> bool:
    """Whether `low_m_bmm` serves bf16 `x` `[..., M, K]` and `w` `[..., N, K]`."""
    return (
        x.is_cuda
        and x.dtype == w.dtype == torch.bfloat16
        and 0 < x.shape[-2] <= MAX_M
        and x.stride(-1) == w.stride(-1) == 1
        and current_platform.is_device_capability_family(120)
    )


def splits_k(n: int, k: int) -> bool:
    """Whether one `[N, K]` weight runs split-K, where `low_m_bmm` beats
    `_gemv`; elsewhere they match."""
    return _config(1, n, k).split > 1


def low_m_bmm(
    x: torch.Tensor, w: torch.Tensor, out_dtype: torch.dtype | None = None
) -> torch.Tensor:
    """`x` `[B, M, K]` @ `w` `[B, N, K]`.T -> `[B, M, N]` for M <= 16, with
    fp32 accumulation."""
    nb, m, k = x.shape
    n = w.shape[1]
    assert 0 < m <= MAX_M and w.shape == (nb, n, k)
    assert x.stride(2) == w.stride(2) == 1
    assert nb * max(n * k, m * x.stride(1)) < 2**31
    out = torch.empty((nb, m, n), dtype=out_dtype or x.dtype, device=x.device)
    cfg = _config(nb, n, k)
    n_tiles = triton.cdiv(n, cfg.bn)
    partials = counter = None
    if cfg.split > 1:
        assert nb * n_tiles <= _MAX_TILES
        partials = torch.empty(
            (cfg.split, nb, _MP, n), dtype=torch.float32, device=x.device
        )
        counter = _counter(current_stream().cuda_stream)
    _low_m_bmm_kernel[(n_tiles, nb, cfg.split)](
        x,
        w,
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
    return out
