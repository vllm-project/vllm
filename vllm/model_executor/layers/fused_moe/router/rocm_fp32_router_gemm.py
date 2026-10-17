# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-token FP32 router GEMM for gfx950."""

import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950
from vllm.triton_utils import tl, triton

_MAX_TOKENS = 32
ROCM_FP32_ROUTER_GEMM_SUPPORTED_SHAPES = frozenset(
    {
        (3072, 256),
        (4096, 8),
        (4096, 192),
        (6144, 128),
        (6144, 256),
    }
)


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
):
    pid = tl.program_id(0)
    pid_m = pid // N
    pid_n = pid - pid_m * N
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_k = tl.arange(0, BLOCK_K)
    partials = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offsets_k_block = k_start + offsets_k
        hidden_states = tl.load(
            hidden_states_ptr + offsets_m[:, None] * K + offsets_k_block[None, :],
            mask=offsets_m[:, None] < M,
            other=0.0,
        ).to(tl.float32)
        router_weight = tl.load(router_weight_ptr + pid_n * K + offsets_k_block)
        partials += hidden_states * router_weight[None, :]

    accumulator = tl.sum(partials, axis=1)
    tl.store(
        output_ptr + offsets_m * N + pid_n,
        accumulator,
        mask=offsets_m < M,
    )


def _launch_config(
    hidden_size: int,
    num_experts: int,
    num_tokens: int,
) -> tuple[int, int, int]:
    if (hidden_size, num_experts) == (4096, 8):
        return 1, 1024, 4

    if num_tokens <= 4:
        return 1, 1024, 8
    if num_tokens <= 16:
        block_m = 8 if (hidden_size, num_experts) == (6144, 256) else 4
        return block_m, 1024, 8
    if (hidden_size, num_experts) == (3072, 256):
        return 8, 1024, 8
    return 4, 2048, 8


def can_use_rocm_fp32_router_gemm(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
) -> bool:
    """Return whether the tensors match the tuned gfx950 fast path."""
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
    if router_weight.dtype != torch.float32:
        raise ValueError("router_weight must have dtype float32")
    if hidden_states.device.type != "cuda" or router_weight.device.type != "cuda":
        raise ValueError("hidden_states and router_weight must be GPU tensors")
    if hidden_states.device != router_weight.device:
        raise ValueError("hidden_states and router_weight must be on the same device")
    if not hidden_states.is_contiguous() or not router_weight.is_contiguous():
        raise ValueError("hidden_states and router_weight must be contiguous")
    shape = (hidden_states.shape[1], router_weight.shape[0])
    if (
        shape not in ROCM_FP32_ROUTER_GEMM_SUPPORTED_SHAPES
        or router_weight.shape[1] != shape[0]
    ):
        raise ValueError(
            "supported (hidden_size, num_experts) shape pairs are "
            "(3072, 256), (4096, 8), (4096, 192), (6144, 128), "
            "and (6144, 256)"
        )
    if not 0 <= hidden_states.shape[0] <= _MAX_TOKENS:
        raise ValueError(f"num_tokens must be in [0, {_MAX_TOKENS}]")


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

    block_m, block_k, num_warps = _launch_config(hidden_size, num_experts, num_tokens)
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
        num_warps=num_warps,
        num_stages=1,
    )
    return output
