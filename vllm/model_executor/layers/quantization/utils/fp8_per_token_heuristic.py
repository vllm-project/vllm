# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA4 per-token FP8 launch choices and specialized contiguous kernels."""

from typing import Any

import torch

from vllm.model_executor.layers.quantization.compressed_tensors import (
    triton_scaled_mm as shared_mm,
)
from vllm.model_executor.layers.quantization.utils import (
    fp8_per_token_heuristic_rules as heuristic_rules,
)
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.triton_utils import tl, triton

knobs = triton.knobs
_LAUNCH_CACHE: dict[tuple, Any] = {}
get_heuristic_config = heuristic_rules.get_heuristic_config


def can_use_fp8_per_token_heuristic(a, b, sa, sb, out_dtype, bias) -> bool:
    from vllm import envs

    m, k = a.shape
    n = b.shape[1]
    return (
        m > 0
        and n > 0
        and k > 0
        and m * n < 2**31
        and k % 16 == n % 16 == 0
        and a.dtype == b.dtype == torch.float8_e4m3fn
        and out_dtype in (torch.bfloat16, torch.float16)
        and a.stride() == (k, 1)
        and b.stride() == (1, k)
        and sa.dtype == sb.dtype == torch.float32
        and sa.numel() == m
        and sb.numel() == n
        and sa.is_contiguous()
        and sb.is_contiguous()
        and (
            bias is None
            or (bias.dtype == out_dtype and bias.is_contiguous() and bias.numel() == n)
        )
        and not envs.VLLM_TRITON_USE_TD
    )


def _launch(kernel, grid, args, options, key, tensors, *, warmup=False):
    """Reuse specialization after checking its pointer attributes and options."""
    if warmup:
        compiled = kernel.warmup(*args, grid=grid, **options)
        if compiled is not None:
            _ = compiled.run
        return compiled
    enter = knobs.runtime.launch_enter_hook
    exit_hook = knobs.runtime.launch_exit_hook
    if (
        kernel.debug
        or kernel.pre_run_hooks
        or knobs.runtime.debug
        or knobs.compilation.instrumentation_mode
        or getattr(enter, "calls", enter)
        or getattr(exit_hook, "calls", exit_hook)
        or knobs.runtime.add_stages_inspection_hook
    ):
        return kernel[grid](*args, **options)
    specialized = (
        tensors[:2]
        if kernel is _fp8_per_token_gemm or kernel is _fp8_per_token_gemv
        else tensors
    )
    attrs = tuple(
        (t.dtype, t.data_ptr() % 16, t.untyped_storage().size() <= 2**31 - 1)
        for t in specialized
        if t is not None
    )
    dtypes = tuple(t.dtype if t is not None else None for t in tensors)
    key = (
        kernel,
        tensors[0].device.index,
        knobs.amd.use_buffer_ops,
        key,
        attrs,
        dtypes,
    )
    compiled = _LAUNCH_CACHE.get(key)
    if compiled is None:
        compiled = kernel[grid](*args, **options)
        if len(_LAUNCH_CACHE) >= 4096:
            _LAUNCH_CACHE.clear()
        _LAUNCH_CACHE[key] = compiled
    else:
        stream = triton.runtime.driver.active.get_current_stream(
            tensors[0].device.index
        )
        compiled.run(
            grid[0],
            grid[1] if len(grid) > 1 else 1,
            1,
            stream,
            compiled.function,
            compiled.packed_metadata,
            None,
            None,
            None,
            *args,
        )
    return compiled


@triton.jit
def _tile_coords(
    pid,
    M,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    GROUP: tl.constexpr,
    STRIPE: tl.constexpr,
):
    nn = tl.cdiv(N, BN)
    nm = tl.cdiv(M, BM)
    if STRIPE > 0:
        span = nm * STRIPE
        start = pid // span * STRIPE
        width = tl.minimum(STRIPE, nn - start)
        p = pid % span
        return p // width, start + p % width
    elif GROUP > 1:
        span = GROUP * nn
        start = pid // span * GROUP
        size = tl.minimum(GROUP, nm - start)
        p = pid % span
        return start + p % size, p // size
    else:
        return pid // nn, pid % nn


@triton.jit(do_not_specialize=["SA", "SB", "C", "Bias"])
def _fp8_per_token_gemm(
    A,
    W,
    SA,
    SB,
    C,
    Bias,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    STRIPE: tl.constexpr,
    UNMASKED: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PERSISTENT: tl.constexpr,
):
    C = tl.multiple_of(C, 16)
    pid = tl.program_id(0)
    split = tl.program_id(1)
    tiles = tl.cdiv(M, BM) * tl.cdiv(N, BN)
    step = tl.num_programs(0) if PERSISTENT else tiles
    for tile in tl.range(pid, tiles, step, num_stages=1):
        mi, ni = _tile_coords(tile, M, N, BM, BN, GROUP, STRIPE)
        rows = mi * BM + tl.arange(0, BM)
        cols = ni * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for block in range(split, tl.cdiv(K, BK), SPLIT_K):
            ki = block * BK + kk
            ap = A + rows[:, None].to(tl.int64) * K + ki[None, :]
            wp = W + cols[None, :].to(tl.int64) * K + ki[:, None]
            if UNMASKED:
                a = tl.load(ap)
                b = tl.load(wp)
            else:
                a = tl.load(ap, (rows[:, None] < M) & (ki[None, :] < K), 0.0)
                b = tl.load(wp, (cols[None, :] < N) & (ki[:, None] < K), 0.0)
            acc = tl.dot(a, b, acc)
        offsets = rows[:, None].to(tl.int64) * N + cols[None, :]
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        if SPLIT_K > 1:
            tl.store(C + split.to(tl.int64) * M * N + offsets, acc, mask)
        else:
            sa = tl.load(SA + rows, rows < M, 0)
            sb = tl.load(SB + cols, cols < N, 0)
            out = (acc * sa[:, None] * sb[None, :]).to(C.dtype.element_ty)
            if Bias is not None:
                bias = tl.load(Bias + cols, cols < N, 0)
                out = out + bias[None, :]
            tl.store(C + offsets, out, mask)
        if PERSISTENT:
            tl.debug_barrier()


@triton.jit
def _fp8_split_epilogue(
    P, SA, SB, C, Bias, M, N: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr
):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows, cols = ids // N, ids % N
    acc = tl.full((BLOCK,), 0, tl.float32)
    for split in range(SPLIT_K):
        acc += tl.load(P + tl.cast(split, tl.int64) * M * N + ids, rows < M, 0)
    sa = tl.load(SA + rows, rows < M, 0)
    sb = tl.load(SB + cols, rows < M, 0)
    out = (acc * sa * sb).to(C.dtype.element_ty)
    if Bias is not None:
        out += tl.load(Bias + cols, rows < M, 0)
    tl.store(C + ids, out, rows < M)


@triton.jit(do_not_specialize=["SA", "SB", "C", "Bias"])
def _fp8_per_token_gemv(
    A,
    W,
    SA,
    SB,
    C,
    Bias,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    C = tl.multiple_of(C, 16)
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    row = tl.program_id(1)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        ki = block * BK + kk
        a = tl.load(A + row * K + ki, ki < K, 0.0).to(tl.float32)
        w = tl.load(
            W + cols[:, None].to(tl.int64) * K + ki[None, :],
            (cols[:, None] < N) & (ki[None, :] < K),
            0.0,
        ).to(tl.float32)
        acc += a[None, :] * w
    value = tl.sum(acc, 1)
    sa = tl.load(SA + row)
    sb = tl.load(SB + cols, cols < N, 0)
    out = (value * sa * sb).to(C.dtype.element_ty)
    if Bias is not None:
        out += tl.load(Bias + cols, cols < N, 0)
    tl.store(C + row * N + cols, out, cols < N)


def launch_fp8_per_token(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None,
    config: dict,
    *,
    _warmup: bool = False,
) -> torch.Tensor:
    """Launch already-validated contiguous FP8 inputs with an explicit choice."""
    m, k = a.shape
    n = b.shape[1]
    if _warmup:
        from vllm.model_executor.warmup.jit_warmup_triton_helper import (
            TritonWarmupTensor,
        )

        result = TritonWarmupTensor(out_dtype)
    else:
        result = torch.empty((m, n), device=a.device, dtype=out_dtype)
    bm, bn, bk = (
        config[key] for key in ("block_size_m", "block_size_n", "block_size_k")
    )
    warps, stages = config.get("num_warps", 4), config.get("num_stages", 2)
    unmasked = (
        config.get("unmasked_loads", False)
        and m % bm == n % bn == k % bk == 0
        and a.data_ptr() % 16 == b.data_ptr() % 16 == 0
    )
    group, stripe = config.get("group_m", 1), config.get("n_stripe", 0)
    if group != 1 or stripe:
        nm, nn = (m + bm - 1) // bm, (n + bn - 1) // bn
        if group * nn > 2**31 - 1:
            group = 1
        if stripe and nm * max(nn, stripe) > 2**31 - 1:
            stripe = 0
        if stripe:
            group = 1
    key = (
        m,
        n,
        k,
        out_dtype,
        bias is not None,
        bm,
        bn,
        bk,
        warps,
        stages,
        group,
        stripe,
        unmasked,
    )
    options = dict(num_warps=warps, num_stages=stages)
    variant = config.get("variant", "gemm")
    args: tuple
    if config.get("engine") == "shared":
        grid = (((m + bm - 1) // bm) * ((n + bn - 1) // bn),)
        args = (
            a,
            b,
            sa,
            sb,
            result,
            bias,
            m,
            n,
            k,
            k,
            1,
            1,
            k,
            n,
            1,
            tl.float32,
            bm,
            bn,
            bk,
            1 if m == 1 else bm,
            1 if n == 1 else bn,
            False,
            True,
            group,
            stripe,
            unmasked,
        )
        _launch(
            shared_mm.scaled_mm_kernel,
            grid,
            args,
            options,
            key,
            (a, b, sa, sb, result, bias),
            warmup=_warmup,
        )
        return result
    if variant == "gemv":
        args = (a, b, sa, sb, result, bias, m, n, k, bn, bk)
        _launch(
            _fp8_per_token_gemv,
            ((n + bn - 1) // bn, m),
            args,
            options,
            key,
            (a, b, sa, sb, result, bias),
            warmup=_warmup,
        )
        return result
    split = config.get("split_k", 1)
    persistent = variant == "persistent"
    grid = ((m + bm - 1) // bm) * ((n + bn - 1) // bn)
    if persistent:
        grid = min(grid, config.get("programs", 32))
    target = result
    if split > 1:
        target = (
            TritonWarmupTensor(torch.float32)
            if _warmup
            else torch.empty((split, m, n), device=a.device, dtype=torch.float32)
        )
    args = (
        a,
        b,
        sa,
        sb,
        target,
        bias,
        m,
        n,
        k,
        bm,
        bn,
        bk,
        group,
        stripe,
        unmasked,
        split,
        persistent,
    )
    _launch(
        _fp8_per_token_gemm,
        (grid, split),
        args,
        options,
        key + (split, persistent),
        (a, b, sa, sb, target, bias),
        warmup=_warmup,
    )
    if split > 1:
        _launch(
            _fp8_split_epilogue,
            ((m * n + 255) // 256,),
            (target, sa, sb, result, bias, m, n, split, 256),
            {"num_warps": 4, "num_stages": 2},
            (m, n, split, out_dtype, bias is not None, 256, 4, 2),
            (target, sa, sb, result, bias),
            warmup=_warmup,
        )
    return result


class _Fp8PerTokenWarmup(VllmJitKernel[tuple]):
    def get_warmup_keys(self, *, n, k, out_dtype, max_tokens):
        from vllm.model_executor.warmup.jit_warmup_triton_helper import (
            triton_scalar_specialization_rep,
        )

        choices: dict[tuple, int] = {}
        for m in range(1, max_tokens + 1):
            config = get_heuristic_config(m, n, k)
            bm, bn, bk = (
                config[key] for key in ("block_size_m", "block_size_n", "block_size_k")
            )
            unmasked = (
                config.get("unmasked_loads", False) and m % bm == n % bn == k % bk == 0
            )
            key = (
                tuple(sorted(config.items())),
                triton_scalar_specialization_rep(m),
                unmasked,
            )
            choices.setdefault(key, m)
        return [
            (m, n, k, out_dtype, bias)
            for m in choices.values()
            for bias in (False, True)
        ]

    def compile(self, compile_key):
        from vllm.model_executor.warmup.jit_warmup_triton_helper import (
            TritonWarmupTensor,
        )

        m, n, k, out_dtype, bias = compile_key
        a = TritonWarmupTensor(torch.float8_e4m3fn, shape=(m, k))
        b = TritonWarmupTensor(torch.float8_e4m3fn, shape=(k, n), strides=(1, k))
        scale = TritonWarmupTensor(torch.float32)
        launch_fp8_per_token(
            a,
            b,
            scale,
            scale,
            out_dtype,
            TritonWarmupTensor(out_dtype) if bias else None,
            get_heuristic_config(m, n, k),
            _warmup=True,
        )


fp8_per_token_warmup = _Fp8PerTokenWarmup()
