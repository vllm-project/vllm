# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-token FP32 router GEMM for gfx950."""

import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950
from vllm.triton_utils import tl, triton

_MAX_TOKENS = 128
_NUM_XCDS = 8
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
    num_m_tiles,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_REMAP: tl.constexpr,
):
    pid = tl.program_id(0)
    # Workgroups are dispatched round-robin over the 8 XCDs, so consecutive
    # launch ids land on different dies. Remapping them into per-XCD contiguous
    # blocks, combined with the expert-major decode below, keeps each die
    # reading N / 8 weight rows instead of all N -- the weight is the only
    # sizeable operand here (N * K fp32), so this is what the L2s hold.
    if XCD_REMAP:
        per_xcd = (N * num_m_tiles) // NUM_XCDS
        pid = (pid % NUM_XCDS) * per_xcd + pid // NUM_XCDS
    pid_n = pid // num_m_tiles
    pid_m = pid - pid_n * num_m_tiles
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_k = tl.arange(0, BLOCK_K)
    partials = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for k_start in tl.static_range(0, K, BLOCK_K):
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
) -> tuple[int, int, int, int, bool]:
    """(BLOCK_M, BLOCK_K, num_warps, num_stages, xcd_remap) for this problem.

    Swept over the five supported shapes for M in 1..256 on MI355X; see the PR
    for the table. Two rules carry most of it:

    * BLOCK_M grows with the token count, but never past what keeps the grid
      wide: one program per (expert, row tile) is only ``num_experts *
      ceil(M / BLOCK_M)`` programs, and MI355X wants a few hundred of them.
      This is what keeps the 8-expert shape on BLOCK_M=1 while the 256-expert
      shapes move up to 16.
    * The XCD remap pays off once there are enough experts to give every die a
      distinct slice of the weight; at 8 experts it only serializes each
      expert onto one die and loses.
    """
    block_k = 2048 if hidden_size % 2048 == 0 else 1024
    if num_tokens <= 4:
        block_m, block_k, num_warps, num_stages = 1, 512, 4, 2
    elif num_tokens <= 8:
        block_m, num_warps, num_stages = 2, 4, 2
    elif num_tokens <= 16:
        block_m, num_warps, num_stages = 4, 4, 1
    elif num_tokens <= 32:
        block_m, num_warps, num_stages = 8, 4, 2
    elif num_tokens <= 64:
        block_m, block_k, num_warps, num_stages = 8, 1024, 2, 2
    else:
        block_m, block_k, num_warps, num_stages = 16, 1024, 4, 1

    # BLOCK_M rows share one pass over the weight. Once the weight no longer
    # fits a single XCD's L2 (4 MiB), that pass is the expensive part, so widen
    # the row tile; among the supported shapes only (6144, 256) is that large.
    if num_tokens > 8 and hidden_size * num_experts * 4 > 4 * 1024 * 1024:
        block_m *= 2

    # One program per (expert, row tile), so the grid is num_experts *
    # ceil(M / BLOCK_M); with few experts that is not enough to fill 256 CUs.
    # Cap BLOCK_M (a power of two, which tl.arange requires) to keep it wide.
    max_block_m = max(1, (num_tokens * num_experts) // 512)
    block_m = min(block_m, 1 << (max_block_m.bit_length() - 1))
    if block_m == 1 and num_tokens > 4:
        # Narrow grid: one row per program leaves the partials tile as the only
        # register pressure, and a slimmer program runs more waves per CU.
        block_k, num_warps, num_stages = 512, 2, 1
    return block_m, block_k, num_warps, num_stages, num_experts >= 64


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

    block_m, block_k, num_warps, num_stages, xcd_remap = _launch_config(
        hidden_size, num_experts, num_tokens
    )
    num_m_tiles = triton.cdiv(num_tokens, block_m)
    grid = (num_m_tiles * num_experts,)
    _rocm_fp32_router_gemm_kernel[grid](
        hidden_states,
        router_weight,
        output,
        M=num_tokens,
        num_m_tiles=num_m_tiles,
        K=hidden_size,
        N=num_experts,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        NUM_XCDS=_NUM_XCDS,
        XCD_REMAP=xcd_remap and grid[0] % _NUM_XCDS == 0,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
