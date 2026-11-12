# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram: the IPC all-gather (kernels/eg_ag.py) across the 4 TP
processes. `dst` and the flags are hipMalloc'd per rank and shared through
IPC handles (ar_runner.ipc_alloc); `dst` is persistent, so its rows can be
read by a later graph segment.

    # once per process, outside graph capture
    ag = IpcAllGather(rank, cpu_group, row_bytes, max_m)
    # src [m, ...] (row_bytes per row) on this rank
    # → out [m, 4 · row_bytes / es], same dtype
    out = ag(src)

Every rank must make the same calls in the same order (seq counters, as
ar_runner). `out` is a view of the persistent buffer: the next call
overwrites it.
"""

import torch

from vllm.models.deepseek_v41.amd.asm.ar_runner import flag_state, ipc_alloc
from vllm.models.deepseek_v41.amd.asm.kernels import eg_ag
from vllm.models.deepseek_v41.amd.asm.kernels.kmod import Kernel
from vllm.triton_utils import tl, triton


class _Cai:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": (nbytes,),
            "typestr": "|u1",
            "version": 3,
        }


def device_bytes(ptr: int, nbytes: int) -> torch.Tensor:
    """uint8 tensor over raw device memory (no copy, not owned)."""
    return torch.as_tensor(_Cai(ptr, nbytes), device="cuda")


class IpcAllGather:
    def __init__(
        self,
        rank: int,
        cpu_group,
        row_bytes: int,
        max_m: int = 256,
        world: int = 4,
        g: int = eg_ag.G,
        split: bool = False,
    ):
        assert world == 4, "eg_ag is written for TP4"
        self.rank, self.b, self.max_m, self.g = rank, row_bytes, max_m, g
        mine, every = ipc_alloc(
            dict(dst=max_m * 4 * row_bytes, flags=eg_ag.flag_bytes(g)),
            rank,
            cpu_group,
            world,
        )
        self.ptrs = {f"dst{d}": every["dst"][d] for d in range(4)}
        self.ptrs.update({f"flags{d}": every["flags"][d] for d in range(4)})
        self.flags_mine = mine["flags"]
        self.flag_layout = (
            eg_ag.flag_bytes(g),
            1,
        )  # (bytes, hops), for IPC diagnostics
        self.dst = device_bytes(mine["dst"], max_m * 4 * row_bytes)
        if not split:
            k = eg_ag.source(rank, row_bytes, g=g)
            self.k = Kernel(k["name"], k["src"], k["ka"])
        else:  # the push split: push() on one stream, wait() on another
            self.k_push, self.k_wait = (
                Kernel(d["name"], d["src"], d["ka"])
                for d in (
                    eg_ag.source(rank, row_bytes, g=g, phase=ph)
                    for ph in ("push", "wait")
                )
            )
        torch.distributed.barrier(group=cpu_group)

    def __call__(self, src: torch.Tensor) -> torch.Tensor:
        m = src.shape[0]
        assert (
            src.is_contiguous()
            and src[0].numel() * src.element_size() == self.b
            and 1 <= m <= self.max_m
        ), (src.shape, src.dtype, self.b)
        self.k(self.g, 64, **self.ptrs, src=src, m=m)
        return (
            self.dst[: m * 4 * self.b]
            .view(src.dtype)
            .view(m, 4 * self.b // src.element_size())
        )

    def push(self, src: torch.Tensor) -> None:
        """Every peer's dst gets src (and its flag); returns without waiting for the
        peers."""
        m = src.shape[0]
        assert (
            src.is_contiguous()
            and src[0].numel() * src.element_size() == self.b
            and 1 <= m <= self.max_m
        )
        self.k_push(self.g, 64, **self.ptrs, src=src, m=m)

    def wait(self, m: int, like: torch.Tensor) -> torch.Tensor:
        """Waits for the peers' pushes of the launch this rank's last push() counted; →
        the gathered rows (as __call__). Must run after that push (stream order or an
        event)."""
        self.k_wait(self.g, 64, **self.ptrs, src=like, m=m)
        return (
            self.dst[: m * 4 * self.b]
            .view(like.dtype)
            .view(m, 4 * self.b // like.element_size())
        )

    def errors(self) -> int:
        """Poll timeouts so far (CTAs with a nonzero err word)."""
        return sum(e != 0 for e in self.state()[1])

    def state(self) -> tuple[list, list]:
        """(ctr, err) per CTA: launches so far, and the seq of the last timed-out launch
        (0: none). Syncs."""
        return flag_state(self.flags_mine, eg_ag.flag_bytes(self.g), self.g)


# ---------------------------------------------------------------- resharded wkv, gate
# exchange

RANK_K, RANK_ROW = (
    1536,
    1584,
)  # per rank: 6 columns × 256 MXFP8 codes, then their 48 E8M0 scales


@triton.jit(do_not_specialize=["M"])
def _side_gemm_packed_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    RK: tl.constexpr,
    RROW: tl.constexpr,
    RANKS: tl.constexpr,
    PLANE_N: tl.constexpr,
    PLANE_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NPROG: tl.constexpr,
):
    """VLLM's _block32_tiled_kernel loop (MXFP8 x, 32×32 block-scaled e4m3 w, no split
    K), persistent (program p takes output tiles p, p + NPROG, .., column tiles fastest,
    so the grid stays small next to the main stream), on the all-gathered packed rows:
    row = RANKS × [RK codes | RK/32 scales] (x_ptr / xs_ptr are fp8 / uint8 views of the
    same buffer), K = RANKS · RK in the same order as vLLM's gathered input. Output
    column n goes to plane n // PLANE_N (rows of PLANE_N, planes PLANE_STRIDE apart)."""
    grid_n: tl.constexpr = N // BLOCK_N
    tiles = tl.cdiv(M, BLOCK_M) * grid_n
    for tile in range(tl.program_id(0), tiles, NPROG):
        pid_m = tile // grid_n
        pid_n = tile % grid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        offs_k = tl.arange(0, BLOCK_K)
        offs_sk = tl.arange(0, BLOCK_K // 32)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d in tl.static_range(RANKS):
            x_ptrs = (
                x_ptr + offs_m[:, None] * (RANKS * RROW) + d * RROW + offs_k[None, :]
            )
            xs_ptrs = (
                xs_ptr
                + offs_m[:, None] * (RANKS * RROW)
                + d * RROW
                + RK
                + offs_sk[None, :]
            )
            w_ptrs = w_ptr + offs_n[:, None] * K + d * RK + offs_k[None, :]
            ws_ptrs = (
                ws_ptr
                + (offs_n[:, None] // 32) * (K // 32)
                + d * (RK // 32)
                + offs_sk[None, :]
            )
            for _ in range(0, RK, BLOCK_K):
                x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
                xs = tl.load(xs_ptrs, mask=m_mask[:, None], other=0)
                w = tl.load(w_ptrs)
                ws = tl.load(ws_ptrs)
                acc = tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3", acc=acc)
                x_ptrs += BLOCK_K
                w_ptrs += BLOCK_K
                xs_ptrs += BLOCK_K // 32
                ws_ptrs += BLOCK_K // 32
        plane = pid_n * BLOCK_N // PLANE_N
        cols = offs_n - plane * PLANE_N
        tl.store(
            out_ptr + plane * PLANE_STRIDE + offs_m[:, None] * PLANE_N + cols[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=m_mask[:, None],
        )


def side_gemm_packed(
    g: torch.Tensor,
    w: torch.Tensor,
    ws: torch.Tensor,
    out: torch.Tensor,
    nprog: int = 0,
    block_n: int = 0,
    block_k: int = 256,
    block_m: int = 0,
    num_warps: int = 0,
    num_stages: int = 2,
) -> None:
    """Out [M, N] bf16 = gathered MXFP8 rows g (uint8 [M, 4 · RANK_ROW]) @ w.T
    (e4m3 [N, 4 · RANK_K], block scales ws [N/32, K/32]); or out [P, ≥ M, N/P]
    (planes, e.g. [key | value]: column n in plane n // (N/P)). nprog 0: one program
    per output tile (short CTAs, vLLM-like); else persistent on nprog. Default
    tiles from eg_gemm_tune.py (N 10240): M ≤ 128 64 × 64 × 256, 4 warps;
    else 128 × 128 × 256, 8 warps (23 / 28 / 37 µs at M 64 / 128 / 256)."""
    M = g.shape[0]
    N, K = w.shape
    ranks = g.shape[1] // RANK_ROW
    big = M > 128
    block_n = block_n or (128 if big else 64)
    num_warps = num_warps or (8 if big else 4)
    block_m = block_m or (128 if big else min(64, max(16, triton.next_power_of_2(M))))
    if out.dim() == 2:
        assert out.shape == (M, N) and out.is_contiguous()
        plane_n, plane_stride = N, 0
    else:
        plane_n, plane_stride = out.shape[2], out.stride(0)
        assert (
            out.shape[0] * plane_n == N
            and out.shape[1] >= M
            and out.stride(1) == plane_n
            and out.stride(2) == 1
        )
    assert ranks * RANK_K == K and plane_n % block_n == 0 and RANK_K % block_k == 0
    tiles = triton.cdiv(M, block_m) * (N // block_n)
    nprog = tiles if nprog <= 0 else min(nprog, tiles)
    _side_gemm_packed_kernel[(nprog,)](
        g.view(torch.float8_e4m3fn),
        g,
        w,
        ws,
        out,
        M,
        N,
        K,
        RANK_K,
        RANK_ROW,
        ranks,
        plane_n,
        plane_stride,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        NPROG=nprog,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@triton.jit
def _eg_gate_kernel(
    hidden_states,
    key,
    q_weight,
    k_weight,
    token_mask,
    gate_out,
    hidden_stride_t,
    key_stride_t,
    eps,
    clamp_value,
    HC: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """VLLM's _fused_engram_post_wkv_kernel up to the gate, for hc copy HC
    (this rank's key) only; the same arithmetic, so the same gate bits."""
    token_idx = tl.program_id(0).to(tl.int64)
    dim_offsets = tl.arange(0, BLOCK_SIZE)
    dim_valid = dim_offsets < DIM
    hidden = tl.load(
        hidden_states + token_idx * hidden_stride_t + HC * DIM + dim_offsets,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)
    key = tl.load(
        key + token_idx * key_stride_t + dim_offsets, mask=dim_valid, other=0.0
    ).to(tl.float32)
    q = tl.load(q_weight + HC * DIM + dim_offsets, mask=dim_valid, other=0.0).to(
        tl.float32
    )
    k = tl.load(k_weight + HC * DIM + dim_offsets, mask=dim_valid, other=0.0).to(
        tl.float32
    )
    hidden_rms = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / DIM + eps)
    key_rms = tl.rsqrt(tl.sum(key * key, axis=0) / DIM + eps)
    dot = tl.sum(hidden * q * k * key, axis=0)
    dot *= hidden_rms * key_rms * tl.rsqrt(DIM * 1.0)
    gate_input = tl.sqrt(tl.maximum(tl.abs(dot), clamp_value))
    gate_input = tl.where(dot < 0.0, -gate_input, gate_input)
    gate = tl.sigmoid(gate_input)
    if HAS_MASK:
        gate = tl.where(tl.load(token_mask + token_idx), gate, 0.0)
    tl.store(gate_out + token_idx * 4, gate)


@triton.jit
def _eg_out_kernel(
    hidden_states,
    value,
    gates,
    output,
    hidden_stride_t,
    value_stride_t,
    HC_MULT: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """VLLM's kernel after the gate: out[t, h] = copy + gate[t, h] · value[t]
    (gates: all-gathered, copy h's at [t, 4 h])."""
    program_idx = tl.program_id(0)
    token_idx = (program_idx // HC_MULT).to(tl.int64)
    hc_idx = program_idx % HC_MULT
    dim_offsets = tl.arange(0, BLOCK_SIZE)
    dim_valid = dim_offsets < DIM
    hidden = tl.load(
        hidden_states + token_idx * hidden_stride_t + hc_idx * DIM + dim_offsets,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(gates + token_idx * (4 * HC_MULT) + 4 * hc_idx)
    v = tl.load(
        value + token_idx * value_stride_t + dim_offsets, mask=dim_valid, other=0.0
    ).to(tl.float32)
    tl.store(
        output + token_idx * hidden_stride_t + hc_idx * DIM + dim_offsets,
        hidden + gate * v,
        mask=dim_valid,
    )


def eg_gate(
    hidden: torch.Tensor,
    key: torch.Tensor,
    q_weight,
    k_weight,
    token_mask,
    gate_out: torch.Tensor,
    hc: int,
    eps: float,
    clamp: float,
) -> None:
    n, hc_mult, dim = hidden.shape
    assert (
        hidden.stride(2) == 1
        and hidden.stride(1) == dim
        and key.stride(1) == 1
        and gate_out.shape[1] == 4
    )
    assert q_weight.is_contiguous() and k_weight.is_contiguous()
    block = triton.next_power_of_2(dim)
    mask = token_mask if token_mask is not None else hidden
    _eg_gate_kernel[(n,)](
        hidden,
        key,
        q_weight,
        k_weight,
        mask,
        gate_out,
        hidden.stride(0),
        key.stride(0),
        eps,
        clamp,
        HC=hc,
        DIM=dim,
        BLOCK_SIZE=block,
        HAS_MASK=token_mask is not None,
        num_warps=8 if block >= 2048 else 4,
    )


def eg_out(
    hidden: torch.Tensor, value: torch.Tensor, gates: torch.Tensor
) -> torch.Tensor:
    n, hc_mult, dim = hidden.shape
    assert (
        hidden.is_contiguous()
        and value.stride(1) == 1
        and gates.shape == (n, 4 * hc_mult)
        and gates.is_contiguous()
    )
    out = torch.empty_like(hidden)
    block = triton.next_power_of_2(dim)
    _eg_out_kernel[(n * hc_mult,)](
        hidden,
        value,
        gates,
        out,
        hidden.stride(0),
        value.stride(0),
        HC_MULT=hc_mult,
        DIM=dim,
        BLOCK_SIZE=block,
        num_warps=8 if block >= 2048 else 4,
    )
    return out
