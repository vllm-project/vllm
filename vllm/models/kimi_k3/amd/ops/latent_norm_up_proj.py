# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 latent-MoE decode tail: routed-expert RMSNorm folded into the
sharded up-projection.

After the latent all-reduce the TP tail runs the RMSNorm of the [M, 3584]
latent and then a hipBLASLt addmm into this rank's [M, 896] hidden shard of the
shared expert output. For decode batches that becomes a split-K Triton GEMM
plus a small per-row finish kernel:

    part[k]   = bf16(x * w)[:, chunk k] @ W[:, chunk k].T        (fp32)
    shared[:, shard] = bf16(shared[:, shard] + inv_rms(x) * sum_k part[k])

The per-row 1/rms commutes with the GEMM, so it is computed in the finish
kernel and the normalized latent never goes to memory. The GEMM input is
rounded to bf16 before the row scale instead of after it, which stays within
bf16 noise of the unfused chain. Split-K spreads the 6.4 MB weight read over
the whole GPU at small M; partials go through memory rather than atomics,
which are slow across XCDs.
"""

import torch

from vllm import ir
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

# Largest decode batch (rows) that takes the fused kernel; larger batches keep
# the RMSNorm + hipBLASLt addmm.
MAX_TOKENS = 192


@triton.jit
def _latent_up_proj_partial_kernel(
    latent_ptr,  # [M, K]
    norm_w_ptr,  # [K]
    w_ptr,  # [N, K]
    part_ptr,  # [SPLIT_K, M, N] fp32
    M,
    stride_lm,
    stride_wn,
    K: tl.constexpr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """part[k] = bf16(x * w)[:, chunk k] @ W[:, chunk k].T for one tile."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    k_chunk: tl.constexpr = K // SPLIT_K

    # The weight tile is the MFMA A operand in its natural [N, K] layout, so
    # small M only pads the B operand.
    acc = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
    for k0 in tl.range(pid_k * k_chunk, (pid_k + 1) * k_chunk, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        xt = tl.load(
            latent_ptr + offs_m[None, :] * stride_lm + offs_k[:, None],
            mask=mask_m[None, :],
            other=0.0,
        ).to(tl.float32)
        nw = tl.load(norm_w_ptr + offs_k).to(tl.float32)
        # The weight is streamed once; skip L1 so it does not evict the latent.
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=mask_n[:, None],
            other=0.0,
            cache_modifier=".cg",
        )
        acc += tl.dot(w, (xt * nw[:, None]).to(w.dtype))

    tl.store(
        part_ptr + (pid_k * M + offs_m[None, :]) * N + offs_n[:, None],
        acc,
        mask=mask_m[None, :] & mask_n[:, None],
    )


@triton.jit
def _latent_up_proj_finish_kernel(
    latent_ptr,  # [M, K]
    part_ptr,  # [SPLIT_K, M, N] fp32
    out_ptr,  # [M, N] view, accumulated in place
    M,
    eps,
    stride_lm,
    stride_om,
    K: tl.constexpr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """out[m] = bf16(out[m] + inv_rms(x[m]) * sum_k part[k, m]), one row."""
    row = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    x = tl.load(latent_ptr + row * stride_lm + offs_k, mask=offs_k < K, other=0.0).to(
        tl.float32
    )
    inv_rms = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / K + eps)
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    total = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        total += tl.load(part_ptr + (k * M + row) * N + offs_n, mask=mask_n, other=0.0)
    out_ptrs = out_ptr + row * stride_om + offs_n
    out = total * inv_rms + tl.load(out_ptrs, mask=mask_n, other=0.0).to(tl.float32)
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=mask_n)


def _launch_config(num_tokens: int) -> tuple[int, int, int, int, int, int]:
    """(BLOCK_M, BLOCK_N, SPLIT_K, BLOCK_K, num_warps, num_stages) of the
    partial GEMM, tuned for an N = 896, K = 3584 shard on gfx950."""
    block_m = min(max(16, triton.next_power_of_2(num_tokens)), 64)
    if num_tokens <= 16:
        return block_m, 32, 14, 256, 2, 1
    if num_tokens <= 128:
        return block_m, 32, 7, 128, 4, 3
    return block_m, 32, 4, 128, 4, 2


def latent_norm_up_proj_supported(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    if latent.dim() != 2 or up_weight.dim() != 2 or out.dim() != 2:
        return False
    M, K = latent.shape
    N = up_weight.shape[0]
    if not 0 < M <= MAX_TOKENS:
        return False
    _, _, split_k, block_k, _, _ = _launch_config(M)
    return (
        latent.dtype == torch.bfloat16
        and up_weight.dtype == torch.bfloat16
        and out.dtype == torch.bfloat16
        and latent.stride(-1) == 1
        and up_weight.stride(-1) == 1
        and out.stride(-1) == 1
        and norm_weight.is_contiguous()
        and tuple(norm_weight.shape) == (K,)
        and up_weight.shape[1] == K
        and tuple(out.shape) == (M, N)
        # Every K chunk is a whole number of BLOCK_K steps (no K masking), and
        # the finish kernel holds a full row of the latent and of the output.
        and K % (split_k * block_k) == 0
        and K <= 8192
        and N <= 4096
    )


def latent_norm_up_proj_triton(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    up_weight: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """In place: out += rms_norm(latent, norm_weight, eps) @ up_weight.T."""
    M, K = latent.shape
    N = up_weight.shape[0]
    block_m, block_n, split_k, block_k, num_warps, num_stages = _launch_config(M)
    part = torch.empty((split_k, M, N), dtype=torch.float32, device=latent.device)
    _latent_up_proj_partial_kernel[
        (triton.cdiv(N, block_n), split_k, triton.cdiv(M, block_m))
    ](
        latent,
        norm_weight,
        up_weight,
        part,
        M,
        latent.stride(0),
        up_weight.stride(0),
        K=K,
        N=N,
        SPLIT_K=split_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        matrix_instr_nonkdim=16,
    )
    _latent_up_proj_finish_kernel[(M,)](
        latent,
        part,
        out,
        M,
        eps,
        latent.stride(0),
        out.stride(0),
        K=K,
        N=N,
        SPLIT_K=split_k,
        BLOCK_K=triton.next_power_of_2(K),
        BLOCK_N=triton.next_power_of_2(N),
        num_warps=4,
    )


def _latent_norm_up_proj_impl(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    up_weight: torch.Tensor,
    shared_output: torch.Tensor,
    shard_start: int,
) -> None:
    # The token gate is taken here, at run time: the caller is traced with a
    # symbolic batch size, so a Python branch there would be fixed at trace time.
    out = shared_output.narrow(-1, shard_start, up_weight.shape[0])
    if latent_norm_up_proj_supported(latent, norm_weight, up_weight, out):
        latent_norm_up_proj_triton(latent, norm_weight, eps, up_weight, out)
        return
    out.addmm_(ir.ops.rms_norm(latent, norm_weight, eps), up_weight.t())


def _latent_norm_up_proj_fake(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    up_weight: torch.Tensor,
    shared_output: torch.Tensor,
    shard_start: int,
) -> None:
    return None


direct_register_custom_op(
    op_name="k3_latent_norm_up_proj",
    op_func=_latent_norm_up_proj_impl,
    mutates_args=["shared_output"],
    fake_impl=_latent_norm_up_proj_fake,
)
