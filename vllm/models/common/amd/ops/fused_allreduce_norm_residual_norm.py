# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""All-reduce plus Muse Spark's post-norm / gated residual / pre-norm.

AITER's fused QuickReduce epilogue
(``QuickAllReduceInt4.allreduce_rmsnorm``) computes
``rmsnorm(allreduce(x) + residual)`` on a bf16 residual. Muse Spark's
boundary is a different function: post-norm the reduced branch, blend it
with the fp32 stream, then pre-norm that sum. This kernel does the sum and
that boundary in one launch, so the reduced branch is never stored.

The sum is a bf16 partial sum in fp32. It is not the INT4 QuickReduce codec,
and publishing the partials is a synchronous copy plus a host barrier, so
the op stays off unless ``VLLM_ROCM_MUSE_FUSED_AR_BOUNDARY`` is set.
"""

from __future__ import annotations

import ctypes

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.distributed import get_tp_group
from vllm.distributed.device_communicators.cuda_wrapper import (
    CudaRTLibrary,
    cudaIpcMemHandle_t,
)
from vllm.logger import init_logger
from vllm.models.muse_spark.common.layers import (
    MuseSparkNormPreservingResidualConnection,
    MuseSparkResidualStream,
    MuseSparkRMSNorm,
)
from vllm.models.muse_spark.common.ops.fused_rmsnorm import supports_norm_residual_norm
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# One row of the boundary has to stay live for the second norm.
_BOUNDARY_TILE_ELEMS = 8192


@triton.jit
def _effective_weight(w_ptr, d, dm, gain_center):
    """``weight + gain_center`` in fp32; the norm rounds only at its store."""
    return tl.load(w_ptr + d, mask=dm, other=0.0).to(tl.float32) + gain_center


@triton.jit
def _allreduce_norm_residual_norm_kernel(
    peer_ptrs,
    stream_ptr,
    new_stream_ptr,
    out_ptr,
    post_w_ptr,
    pre_w_ptr,
    alpha_ptr,
    beta_ptr,
    T,
    D: tl.constexpr,
    WORLD: tl.constexpr,
    post_eps,
    pre_eps,
    post_gain_center,
    pre_gain_center,
    alpha_scalar,
    POST_HAS_WEIGHT: tl.constexpr,
    POST_INPUT_SCALE: tl.constexpr,
    PRE_HAS_WEIGHT: tl.constexpr,
    PRE_INPUT_SCALE: tl.constexpr,
    GATE_STREAM: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
    bT: tl.constexpr,
    bD: tl.constexpr,
):
    """Sum each rank's bf16 partial, then one residual boundary."""
    t = tl.program_id(0) * bT + tl.arange(0, bT)
    tm = t < T
    d = tl.arange(0, bD)
    dm = d < D
    idx = t[:, None] * D + d[None, :]
    mask = tm[:, None] & dm[None, :]

    x = tl.zeros((bT, bD), dtype=tl.float32)
    for rank in range(WORLD):
        base = tl.load(peer_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
        x += tl.load(base + idx, mask=mask, other=0.0).to(tl.float32)
    stream = tl.load(stream_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    if PER_CHANNEL:
        beta = tl.load(beta_ptr + d, mask=dm, other=0.0)[None, :]
    else:
        beta = tl.load(beta_ptr)
    if GATE_STREAM:
        if PER_CHANNEL:
            alpha = tl.load(alpha_ptr + d, mask=dm, other=0.0)[None, :]
        else:
            alpha = tl.load(alpha_ptr)
    if POST_HAS_WEIGHT:
        post_w = _effective_weight(post_w_ptr, d, dm, post_gain_center)
    if PRE_HAS_WEIGHT:
        pre_w = _effective_weight(pre_w_ptr, d, dm, pre_gain_center)

    if POST_INPUT_SCALE and POST_HAS_WEIGHT:
        x = x * post_w[None, :]
    sum_sq = tl.sum(x * x, axis=1)
    y = x * tl.rsqrt(sum_sq * (1.0 / D) + post_eps)[:, None]
    if POST_HAS_WEIGHT and not POST_INPUT_SCALE:
        y = y * post_w[None, :]

    if GATE_STREAM:
        new_stream = alpha * stream + beta * y
    else:
        new_stream = alpha_scalar * stream + beta * y
    tl.store(
        new_stream_ptr + idx,
        new_stream.to(new_stream_ptr.dtype.element_ty),
        mask=mask,
    )

    z = new_stream
    if PRE_INPUT_SCALE and PRE_HAS_WEIGHT:
        z = z * pre_w[None, :]
    pre_sum_sq = tl.sum(z * z, axis=1)
    z = z * tl.rsqrt(pre_sum_sq * (1.0 / D) + pre_eps)[:, None]
    if PRE_HAS_WEIGHT and not PRE_INPUT_SCALE:
        z = z * pre_w[None, :]
    tl.store(out_ptr + idx, z.to(out_ptr.dtype.element_ty), mask=mask)


def _launch_boundary(
    peer_ptrs: torch.Tensor,
    world: int,
    stream: torch.Tensor,
    post_weight: torch.Tensor | None,
    pre_weight: torch.Tensor | None,
    post_eps: float,
    pre_eps: float,
    *,
    alpha: torch.Tensor | float,
    beta: torch.Tensor,
    post_gain_center: float,
    pre_gain_center: float,
    post_input_scale: bool,
    pre_input_scale: bool,
    out_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the boundary over partials addressed by ``peer_ptrs``."""
    rows, hidden = stream.shape
    for weight, gain_center, input_scale, name in (
        (post_weight, post_gain_center, post_input_scale, "post"),
        (pre_weight, pre_gain_center, pre_input_scale, "pre"),
    ):
        if weight is None:
            if gain_center != 0.0 or input_scale:
                raise ValueError(f"{name} scale requires a weight")
        else:
            if weight.shape != (hidden,) or weight.stride(-1) != 1:
                raise ValueError(f"{name} weight must be a contiguous [{hidden}]")

    if beta.numel() not in (1, hidden) or not beta.is_contiguous():
        raise ValueError(f"beta must be a contiguous [1] or [{hidden}]")
    per_channel = beta.numel() > 1
    gate_stream = isinstance(alpha, torch.Tensor)
    alpha_scalar = 1.0
    if gate_stream:
        if alpha.shape != beta.shape or not alpha.is_contiguous():
            raise ValueError("alpha must be contiguous and match beta")
    else:
        alpha_scalar = float(alpha)

    out = torch.empty(
        stream.shape,
        dtype=out_dtype or torch.bfloat16,
        device=stream.device,
    )
    new_stream = torch.empty_like(stream)
    if rows == 0:
        return out, new_stream

    tile = triton.next_power_of_2(hidden)
    rows_per_program = max(1, _BOUNDARY_TILE_ELEMS // tile)
    num_warps = 32 if tile >= 8192 else (16 if tile >= 4096 else 8)
    if current_platform.is_rocm():
        # 64-thread wavefronts: 32 warps would exceed the 1024-thread limit.
        num_warps = min(num_warps, 16)
    _allreduce_norm_residual_norm_kernel[(triton.cdiv(rows, rows_per_program),)](
        peer_ptrs,
        stream,
        new_stream,
        out,
        post_weight if post_weight is not None else stream,
        pre_weight if pre_weight is not None else stream,
        alpha if gate_stream else beta,
        beta,
        rows,
        D=hidden,
        WORLD=world,
        post_eps=post_eps,
        pre_eps=pre_eps,
        post_gain_center=post_gain_center,
        pre_gain_center=pre_gain_center,
        alpha_scalar=alpha_scalar,
        POST_HAS_WEIGHT=post_weight is not None,
        POST_INPUT_SCALE=post_input_scale,
        PRE_HAS_WEIGHT=pre_weight is not None,
        PRE_INPUT_SCALE=pre_input_scale,
        GATE_STREAM=gate_stream,
        PER_CHANNEL=per_channel,
        bT=rows_per_program,
        bD=tile,
        num_warps=num_warps,
    )
    return out, new_stream


def fused_allreduce_norm_residual_norm_partials(
    partials: torch.Tensor,
    stream: torch.Tensor,
    post_weight: torch.Tensor | None,
    pre_weight: torch.Tensor | None,
    post_eps: float,
    pre_eps: float,
    *,
    alpha: torch.Tensor | float,
    beta: torch.Tensor,
    post_gain_center: float = 0.0,
    pre_gain_center: float = 0.0,
    post_input_scale: bool = False,
    pre_input_scale: bool = False,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same boundary as the collective, over a stacked ``[world, rows, hidden]``.

    The stack is this process's stand-in for peer buffers. The kernel still
    loads each rank through a pointer, which is the launch the collective uses.
    """
    if partials.ndim != 3 or partials.dtype is not torch.bfloat16:
        raise ValueError("partials must be bf16 [world, rows, hidden]")
    if not partials.is_contiguous() or stream.shape != partials.shape[1:]:
        raise ValueError("partials must be contiguous and match stream's rows")
    world = partials.shape[0]
    peer_ptrs = torch.tensor(
        [partials[rank].data_ptr() for rank in range(world)],
        dtype=torch.int64,
        device=partials.device,
    )
    return _launch_boundary(
        peer_ptrs,
        world,
        stream,
        post_weight,
        pre_weight,
        post_eps,
        pre_eps,
        alpha=alpha,
        beta=beta,
        post_gain_center=post_gain_center,
        pre_gain_center=pre_gain_center,
        post_input_scale=post_input_scale,
        pre_input_scale=pre_input_scale,
        out_dtype=out_dtype,
    )


class _PeerPartials:
    """One IPC buffer a rank, mapped by every rank of the tensor-parallel group."""

    def __init__(self, nbytes: int, device: torch.device) -> None:
        group = get_tp_group()
        self.nbytes = nbytes
        self.cpu_group = group.cpu_group
        self.rank = group.rank_in_group
        self.lib = CudaRTLibrary()
        self.lib.cudaSetDevice(device.index)
        self.local = self.lib.cudaMalloc(nbytes)
        self.lib.cudaMemset(self.local, 0, nbytes)
        self.lib.cudaDeviceSynchronize()
        self._opened: list[ctypes.c_void_p] = []
        handle = self.lib.cudaIpcGetMemHandle(self.local)
        handles: list[bytes | None] = [None] * group.world_size
        dist.all_gather_object(handles, bytes(handle), group=self.cpu_group)
        pointers: list[int] = []
        for rank, raw in enumerate(handles):
            if rank == self.rank:
                pointer = self.local
            else:
                assert raw is not None
                pointer = self.lib.cudaIpcOpenMemHandle(
                    cudaIpcMemHandle_t.from_buffer_copy(raw)
                )
                self._opened.append(pointer)
            assert pointer.value is not None
            pointers.append(pointer.value)
        self.peer_ptrs = torch.tensor(pointers, dtype=torch.int64, device=device)

    def publish(self, partial: torch.Tensor) -> None:
        torch.cuda.current_stream().synchronize()
        self.lib.cudaMemcpy(
            self.local, ctypes.c_void_p(partial.data_ptr()), partial.nbytes
        )
        dist.barrier(group=self.cpu_group)

    def close(self) -> None:
        while self._opened:
            self.lib.cudaIpcCloseMemHandle(self._opened.pop())
        if self.local is not None:
            self.lib.cudaFree(self.local)
            self.local = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            return


_peers: _PeerPartials | None = None


def _peers_for(partial: torch.Tensor) -> _PeerPartials:
    global _peers
    grown = _peers is not None and (
        _peers.nbytes < partial.nbytes or _peers.peer_ptrs.device != partial.device
    )
    if _peers is None or grown:
        if _peers is not None:
            _peers.close()
        _peers = _PeerPartials(partial.nbytes, partial.device)
    return _peers


def _boundary_ready(
    boundary: MuseSparkResidualStream, pre_norm: MuseSparkRMSNorm
) -> bool:
    post_norm, connection = boundary.post_norm, boundary.connection
    return (
        isinstance(connection, MuseSparkNormPreservingResidualConnection)
        and getattr(connection, "_coefficients_ready", False)
        and post_norm.supports_fused_kernel(boundary.branch)
        and pre_norm.supports_fused_kernel(boundary.stream)
        and supports_norm_residual_norm(boundary.branch, boundary.stream)
        and boundary.branch.dtype is torch.bfloat16
        and triton.next_power_of_2(boundary.branch.shape[-1]) <= _BOUNDARY_TILE_ELEMS
    )


def fused_allreduce_norm_residual_then_norm(
    boundary: MuseSparkResidualStream,
    pre_norm: MuseSparkRMSNorm,
    *,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Sum this rank's o_proj partial with its peers and apply the boundary.

    Returns ``None`` when the fused collective cannot run. The caller
    all-reduces the partial and uses the unfused boundary.
    """
    if not _boundary_ready(boundary, pre_norm):
        return None
    # Decode steps are CUDA-graphed and too small for the host barrier.
    # Prefill and mixed steps (the 8k rows this is aimed at) are eager.
    if boundary.branch.shape[0] < 1024 or torch.cuda.is_current_stream_capturing():
        return None
    partial = boundary.branch
    if not partial.is_contiguous():
        partial = partial.contiguous()
    stream = boundary.stream.reshape(-1, partial.shape[-1])
    partial = partial.reshape(-1, partial.shape[-1])
    try:
        peers = _peers_for(partial)
        peers.publish(partial)
    except Exception:
        logger.exception("Fused all-reduce boundary unavailable; using all-reduce")
        return None
    post_norm = boundary.post_norm
    connection = boundary.connection
    out, new_stream = _launch_boundary(
        peers.peer_ptrs,
        peers.peer_ptrs.numel(),
        stream,
        post_norm.weight if post_norm.has_weight else None,
        pre_norm.weight,
        post_norm.variance_epsilon,
        pre_norm.variance_epsilon,
        alpha=connection.alpha,
        beta=connection.beta,
        post_gain_center=post_norm.fused_gain_center,
        pre_gain_center=pre_norm.fused_gain_center,
        post_input_scale=(
            post_norm.applies_weight_before_norm and post_norm.has_weight
        ),
        pre_input_scale=(pre_norm.applies_weight_before_norm and pre_norm.has_weight),
        out_dtype=out_dtype,
    )
    return out.view(boundary.branch.shape), new_stream.view(boundary.stream.shape)


def fuse_allreduce_norm_residual() -> bool:
    """Whether Muse should skip ``o_proj``'s all-reduce and use this op."""
    return envs.VLLM_ROCM_MUSE_FUSED_AR_BOUNDARY
