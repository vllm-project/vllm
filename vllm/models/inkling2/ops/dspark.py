# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernels of the Inkling2 DSpark draft, matching the reference implementation."""

import torch
import torch.distributed as dist

from vllm.triton_utils import tl, tldevice, triton

from .allreduce import _multicast_memory


@triton.jit
def _mix_rmsnorm_kernel(
    X,
    ALPHA,
    WEIGHT,
    OUT,
    ROW_STRIDE: tl.constexpr,
    TAP_STRIDE: tl.constexpr,
    HIDDEN_STRIDE: tl.constexpr,
    TAPS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BLOCK)
    mixed = tl.full((BLOCK,), 0, tl.float32)
    for tap in tl.static_range(TAPS):
        x = tl.load(
            X + row * ROW_STRIDE + tap * TAP_STRIDE + h * HIDDEN_STRIDE,
            h < HIDDEN,
            0,
        ).to(tl.float32)
        alpha = tl.load(ALPHA + tap).to(tl.float32)
        mixed += x * alpha
    inv_rms = tl.rsqrt(tl.sum(mixed * mixed, 0) / HIDDEN + EPS)
    weight = tl.load(WEIGHT + h, h < HIDDEN, 0).to(tl.float32)
    tl.store(OUT + row * HIDDEN + h, mixed * inv_rms * weight, h < HIDDEN)


def mix_rmsnorm(
    targets: torch.Tensor, alpha: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """RMSNorm of the `alpha`-weighted sum of `targets[:, i]` (fp32 math).

    `targets` is `(rows, taps, hidden)`; the output has `targets.dtype`.
    """
    rows, taps, hidden = targets.shape
    assert alpha.shape == (taps,) and alpha.is_contiguous()
    assert weight.shape == (hidden,) and weight.is_contiguous()
    out = targets.new_empty((rows, hidden))
    if rows:
        _mix_rmsnorm_kernel[(rows,)](
            targets,
            alpha,
            weight,
            out,
            *targets.stride(),
            taps,
            hidden,
            eps,
            triton.next_power_of_2(hidden),
            num_warps=8,
        )
    return out


@triton.jit
def _qk_norm_rope_kernel(
    X,
    Q_WEIGHT,
    K_WEIGHT,
    COS_SIN,
    POS,
    row_stride,
    q_size,
    NHQ: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    NORM: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    pos = tl.load(POS + t).to(tl.int64)

    HALF: tl.constexpr = D // 2
    half_ar = tl.arange(0, HALF)
    d_ar = tl.arange(0, D)
    cos = tl.load(COS_SIN + pos * D + half_ar).to(tl.float32)
    sin = tl.load(COS_SIN + pos * D + HALF + half_ar).to(tl.float32)

    is_q = h < NHQ
    col0 = tl.where(is_q, h * D, q_size + (h - NHQ) * D).to(tl.int64)
    row = X + t * row_stride + col0
    if NORM:
        w_ptr = tl.where(is_q, Q_WEIGHT.to(tl.int64), K_WEIGHT.to(tl.int64)).to(
            tl.pointer_type(tl.bfloat16)
        )
        x = tl.load(row + d_ar).to(tl.float32)
        ms = tl.sum(x * x, 0) / D
        inv = 1.0 / tl.sqrt(ms + EPS)
        w1 = tl.load(w_ptr + half_ar).to(tl.float32)
        w2 = tl.load(w_ptr + HALF + half_ar).to(tl.float32)
        x1 = tl.load(row + half_ar).to(tl.float32) * inv * w1
        x2 = tl.load(row + HALF + half_ar).to(tl.float32) * inv * w2
        x1 = x1.to(tl.bfloat16).to(tl.float32)
        x2 = x2.to(tl.bfloat16).to(tl.float32)
    else:
        x1 = tl.load(row + half_ar).to(tl.float32)
        x2 = tl.load(row + HALF + half_ar).to(tl.float32)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    tl.store(row + half_ar, o1.to(tl.bfloat16))
    tl.store(row + HALF + half_ar, o2.to(tl.bfloat16))


def qk_norm_rope_(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    eps: float,
) -> None:
    """In-place per-head Q/K RMSNorm (rounded to bf16) + neox RoPE on the
    leading `num_q_heads + num_k_heads` heads of the fused `qkv`."""
    head_dim = cos_sin_cache.shape[-1]
    if qkv.shape[0]:
        _qk_norm_rope_kernel[(qkv.shape[0], num_q_heads + num_k_heads)](
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            qkv.stride(0),
            num_q_heads * head_dim,
            NHQ=num_q_heads,
            D=head_dim,
            EPS=eps,
            NORM=True,
        )


def rope_(x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor):
    """In-place neox RoPE on `x` of shape `(T, H, D)` (fp32 math)."""
    num_tokens, num_heads, head_dim = x.shape
    assert x.stride(1) == head_dim and x.stride(2) == 1
    if num_tokens:
        _qk_norm_rope_kernel[(num_tokens, num_heads)](
            x,
            x,
            x,
            cos_sin_cache,
            positions,
            x.stride(0),
            0,
            NHQ=0,
            D=head_dim,
            EPS=0.0,
            NORM=False,
        )


@triton.jit
def _silu_and_mul_kernel(X, Y, stride_x, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    gate = tl.load(X + row * stride_x + cols, mask, other=0).to(tl.float32)
    up = tl.load(X + row * stride_x + N + cols, mask, other=0).to(tl.float32)
    silu = tl.div_rn(gate, 1.0 + tldevice.exp(-gate))
    tl.store(Y + row * N + cols, (silu * up).to(Y.dtype.element_ty), mask)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """`silu(gate) * up` in fp32 (precise exp/div), rounded once."""
    num_tokens, width = x.shape
    n = width // 2
    out = x.new_empty((num_tokens, n))
    if num_tokens:
        _silu_and_mul_kernel[(num_tokens, triton.cdiv(n, 1024))](
            x, out, x.stride(0), n, 1024, num_warps=4
        )
    return out


@triton.jit
def _add_softcap_kernel(A, B, OUT, stride_a, stride_b, softcap, N, BLOCK: tl.constexpr):
    row = tl.program_id(1).to(tl.int64)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    a = tl.load(A + row * stride_a + cols, mask, other=0).to(tl.float32)
    b = tl.load(B + row * stride_b + cols, mask, other=0).to(tl.float32)
    x = (a + b).to(tl.bfloat16)
    x = x / softcap
    x = tldevice.tanh(x)
    x = x * softcap
    tl.store(OUT + row * N + cols, x.to(tl.bfloat16), mask)


def add_softcap(a: torch.Tensor, b: torch.Tensor, softcap: float) -> torch.Tensor:
    """bf16 `softcap * tanh(bf16(a + b) / softcap)` of two bf16 2D tensors."""
    rows, n = a.shape
    assert a.stride(1) == 1 and b.stride(1) == 1
    out = a.new_empty((rows, n))
    if rows:
        _add_softcap_kernel[(triton.cdiv(n, 1024), rows)](
            a, b, out, a.stride(0), b.stride(0), softcap, n, 1024
        )
    return out


@triton.jit
def _round_bf16(x):
    # fp32 -> bf16 -> fp32 (round to nearest even) on the bits: the compiler
    # folds an extf(truncf(x)) pair, and with it fuses mul + add into an FMA.
    bits = x.to(tl.uint32, bitcast=True)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _markov_gate_kernel(RAW, STATE, OUT, LATENT, RANK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, RANK)
    raw = RAW + row * 3 * RANK + c
    gate = _round_bf16(tl.load(raw))
    cand = _round_bf16(tl.load(raw + RANK))
    out = _round_bf16(tl.load(raw + 2 * RANK))
    state = tl.load(STATE + row * RANK + c).to(tl.float32)
    # Each torch op of the reference rounds to bf16; its sigmoid is
    # 1 / (1 + expf(-x)) with IEEE division.
    gate = _round_bf16(tldevice.div_rn(1.0, 1.0 + tldevice.exp(-gate)))
    cand = _round_bf16(tldevice.tanh(cand))
    keep = _round_bf16(gate * state)
    update = _round_bf16(_round_bf16(1.0 - gate) * cand)
    tl.store(OUT + row * RANK + c, (keep + update).to(tl.bfloat16))
    tl.store(LATENT + row * RANK + c, tldevice.tanh(out).to(tl.bfloat16))


def markov_gate(
    raw: torch.Tensor, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The recurrent head's gating of the fp32 `rnn_proj` output `raw`
    (rounded to bf16 first), as the reference's bf16 torch ops: returns
    `(gate * state + (1 - gate) * tanh(cand), tanh(out))`."""
    rows, rank = state.shape
    assert raw.shape == (rows, 3 * rank) and raw.is_contiguous()
    assert state.is_contiguous() and rank & (rank - 1) == 0
    out, latent = torch.empty_like(state), torch.empty_like(state)
    if rows:
        _markov_gate_kernel[(rows,)](raw, state, out, latent, rank)
    return out, latent


class Inkling2DSparkArgmax:
    """Lamport slots of `dspark_argmax`: each row's per-rank, per-block
    (max, index) pairs, multicast to every rank, in two buffers that a row
    alternates between; and each row's arrival count and buffer parity."""

    def __init__(self, group: dist.ProcessGroup, max_rows: int, num_cols: int):
        self.rank, self.world_size = dist.get_rank(group), group.size()
        self.num_cols = num_cols
        num_blocks = triton.cdiv(num_cols, 1024)
        num_bytes = 2 * max_rows * self.world_size * num_blocks * 8
        device = torch.accelerator.current_device_index()
        if self.world_size == 1:
            self.buffer = torch.full((num_bytes // 4,), -0.0, device=device)
            self.multicast_ptr = self.local_ptr = self.buffer.data_ptr()
        else:
            from flashinfer.comm.mnnvl import is_multicast_supported

            assert is_multicast_supported(device)
            self.buffer = _multicast_memory(num_bytes, group)
            self.multicast_ptr = self.buffer.get_multicast_ptr()
            assert self.multicast_ptr, "Inkling2 DSpark requires NVLink multicast"
            self.local_ptr = self.buffer.get_unicast_ptr(self.rank)
            self.buffer.lamport_initialize(self.rank, torch.float32)
        self.row_state = torch.zeros(max_rows, dtype=torch.int32, device=device)
        torch.accelerator.synchronize()
        dist.barrier(group=group)


def dspark_argmax(
    workspace: Inkling2DSparkArgmax,
    base: torch.Tensor,
    bias: torch.Tensor,
    softcap: float,
    valid: int,
    offset: int,
) -> torch.Tensor:
    """First argmax (NaN as -inf) of every rank's `add_softcap(base, bias,
    softcap)[:, :valid]` (its vocabulary shard at `offset`), gathered."""
    assert base.dtype == bias.dtype == torch.bfloat16 and base.shape == bias.shape
    assert base.shape[1] == workspace.num_cols
    out = torch.empty(base.shape[0], dtype=torch.int64, device=base.device)
    torch.ops._C.inkling2_dspark_argmax(
        base,
        bias,
        softcap,
        valid,
        offset,
        workspace.multicast_ptr,
        workspace.local_ptr,
        workspace.row_state,
        workspace.rank,
        workspace.world_size,
        out,
    )
    return out
