# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-token router GEMM with FP32 output for gfx950 (FP32 or BF16 weights)."""

import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950
from vllm.triton_utils import tl, triton

_MAX_TOKENS = 32
# From this many multiply-adds on, F.linear wins on MI355.
_MAX_MACS = 1 << 27
# Above this num_tokens * num_experts, hipBLASLt wins for BF16 weights.
_MAX_BF16_TOKEN_EXPERTS = 8192


@triton.jit
def _rocm_fp32_router_gemm_kernel(
    hidden_states_ptr,
    router_weight_ptr,
    output_ptr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // N
    pid_n = pid - pid_m * N
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_k = tl.arange(0, BLOCK_K)
    partials = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offsets_k_block = k_start + offsets_k
        if EVEN_K:
            hidden_states = tl.load(
                hidden_states_ptr + offsets_m[:, None] * K + offsets_k_block[None, :],
                mask=offsets_m[:, None] < M,
                other=0.0,
            ).to(tl.float32)
            router_weight = tl.load(router_weight_ptr + pid_n * K + offsets_k_block).to(
                tl.float32
            )
        else:
            k_mask = offsets_k_block < K
            hidden_states = tl.load(
                hidden_states_ptr + offsets_m[:, None] * K + offsets_k_block[None, :],
                mask=(offsets_m[:, None] < M) & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            router_weight = tl.load(
                router_weight_ptr + pid_n * K + offsets_k_block,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
        partials += hidden_states * router_weight[None, :]

    accumulator = tl.sum(partials, axis=1)
    tl.store(
        output_ptr + offsets_m * N + pid_n,
        accumulator,
        mask=offsets_m < M,
    )


def _launch_config(
    num_experts: int, num_tokens: int, bf16_weight: bool
) -> tuple[int, int, int]:
    """Return (BLOCK_M, BLOCK_K, num_warps), swept on MI355 for M<=32.

    One program computes BLOCK_M tokens for one expert, so narrow routers keep
    BLOCK_M=1 to fill the GPU and wide routers grow BLOCK_M to reuse weights.
    """
    if bf16_weight:
        return (1 if num_tokens <= 4 else 2), 1024, 2
    if num_experts < 64:
        return 1, 1024, 4
    wide = num_experts > 256
    if num_tokens <= 2:
        return 1, 512, 2
    if num_tokens <= 4:
        return (2, 1024, 2) if wide else (2, 2048, 4)
    if num_tokens <= 8:
        return (4, 512, 2) if wide else (4, 1024, 4)
    if num_tokens <= 16:
        return (8, 512, 2) if wide else (4, 1024, 4)
    if num_experts >= 768:
        return 16, 256, 2
    return (8, 256, 2) if wide else (4, 512, 2)


def can_use_rocm_fp32_router_gemm(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
) -> bool:
    """Return whether the tensors can use the gfx950 fast path."""
    try:
        _validate_inputs(hidden_states, router_weight)
    except (RuntimeError, ValueError):
        return False
    return True


def _validate_inputs(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
) -> None:
    if not current_platform.is_rocm() or not on_gfx950():
        raise RuntimeError("rocm_fp32_router_gemm requires ROCm gfx950")
    if hidden_states.dim() != 2 or router_weight.dim() != 2:
        raise ValueError("hidden_states and router_weight must be 2D tensors")
    if hidden_states.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("hidden_states must have dtype bfloat16 or float32")
    if router_weight.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("router_weight must have dtype float32 or bfloat16")
    if hidden_states.device.type != "cuda" or router_weight.device.type != "cuda":
        raise ValueError("hidden_states and router_weight must be GPU tensors")
    if hidden_states.device != router_weight.device:
        raise ValueError("hidden_states and router_weight must be on the same device")
    if not hidden_states.is_contiguous() or not router_weight.is_contiguous():
        raise ValueError("hidden_states and router_weight must be contiguous")
    if router_weight.shape[1] != hidden_states.shape[1]:
        raise ValueError("router_weight and hidden_states hidden sizes differ")
    if not 0 <= hidden_states.shape[0] <= _MAX_TOKENS:
        raise ValueError(f"num_tokens must be in [0, {_MAX_TOKENS}]")
    if hidden_states.shape[0] * router_weight.numel() >= _MAX_MACS:
        raise ValueError(
            f"num_tokens * num_experts * hidden_size must be < {_MAX_MACS}"
        )
    if (
        router_weight.dtype == torch.bfloat16
        and hidden_states.shape[0] * router_weight.shape[0] > _MAX_BF16_TOKEN_EXPERTS
    ):
        raise ValueError(
            f"num_tokens * num_experts must be <= {_MAX_BF16_TOKEN_EXPERTS} "
            "for bfloat16 weights"
        )


def rocm_fp32_router_gemm(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute ``hidden_states @ router_weight.T`` with FP32 accumulation."""
    _validate_inputs(hidden_states, router_weight)
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    num_experts = router_weight.shape[0]
    output = hidden_states.new_empty((num_tokens, num_experts), dtype=torch.float32)
    if num_tokens == 0:
        return output

    block_m, block_k, num_warps = _launch_config(
        num_experts, num_tokens, router_weight.dtype == torch.bfloat16
    )
    block_k = min(block_k, triton.next_power_of_2(hidden_size))
    grid = (triton.cdiv(num_tokens, block_m) * num_experts,)
    _rocm_fp32_router_gemm_kernel[grid](
        hidden_states,
        router_weight,
        output,
        M=num_tokens,
        K=hidden_size,
        N=num_experts,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        EVEN_K=hidden_size % block_k == 0,
        num_warps=num_warps,
        num_stages=1,
    )
    return output
