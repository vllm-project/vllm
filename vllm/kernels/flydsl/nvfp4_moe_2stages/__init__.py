# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16-by-NVFP4 two-stage MoE kernels for gfx942 and gfx950.

This module intentionally contains no routing, tuning, or model-specific
configuration. Callers provide route metadata and the selected tile shape.
Weights must already use the FlyDSL NVFP4 preshuffle layout.
"""

from __future__ import annotations

from typing import Literal

import torch
from aiter.ops.activation import silu_and_mul
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg

from .stage1 import compile_moe_gemm1
from .stage2 import compile_moe_gemm2


def _empty(device: torch.device, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    return torch.empty(0, device=device, dtype=dtype)


def _validate_common(
    activations: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    *,
    expected_k: int,
) -> None:
    if activations.dtype != torch.bfloat16:
        raise ValueError("FlyDSL NVFP4 MoE requires bfloat16 activations")
    if weight.dtype != torch.uint8 or weight_scale.dtype != torch.uint8:
        raise ValueError("FlyDSL NVFP4 weights and block scales must be uint8")
    if global_scale.dtype != torch.float32:
        raise ValueError("FlyDSL NVFP4 global scale must be float32")
    if activations.ndim not in (2, 3):
        raise ValueError("activations must have shape [M, K] or [M, topk, K]")
    if weight.ndim != 3 or weight_scale.ndim != 3:
        raise ValueError("weight and weight_scale must be three-dimensional")
    if weight.shape[0] != weight_scale.shape[0]:
        raise ValueError("weight and weight_scale must have the same expert count")
    if global_scale.numel() != weight.shape[0]:
        raise ValueError("global_scale must have one value per local expert")
    if activations.shape[-1] != expected_k:
        raise ValueError(
            "activation K "
            f"({activations.shape[-1]}) does not match expected K ({expected_k})"
        )
    if sorted_token_ids.dtype != torch.int32:
        raise ValueError("sorted_token_ids must be int32")
    if sorted_expert_ids.dtype != torch.int32:
        raise ValueError("sorted_expert_ids must be int32")
    if num_valid_ids.dtype != torch.int32 or num_valid_ids.numel() != 1:
        raise ValueError("num_valid_ids must be a one-element int32 tensor")
    tensors = (
        weight,
        weight_scale,
        global_scale,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
    )
    if any(t.device != activations.device for t in tensors):
        raise ValueError("all FlyDSL NVFP4 inputs must be on the same device")


def _validate_tile(tile_m: int, tile_n: int, tile_k: int) -> None:
    if tile_m not in (16, 32, 64, 128):
        raise ValueError("tile_m must be one of 16, 32, 64, or 128")
    if tile_n not in (64, 128):
        raise ValueError("tile_n must be 64 or 128")
    if tile_k not in (64, 128, 256):
        raise ValueError("tile_k must be 64, 128, or 256")


def nvfp4_moe_stage1(
    activations: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    *,
    topk: int,
    inter_dim: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    output: torch.Tensor | None = None,
    sorted_weights: torch.Tensor | None = None,
    k_batch: int = 1,
) -> torch.Tensor:
    """Run the BF16 × NVFP4 gate/up projection.

    Args:
        activations: BF16 tensor shaped ``[M, model_dim]``.
        weight: Preshuffled packed NVFP4 tensor ``[E, 2*I, model_dim/2]``.
        weight_scale: Preshuffled uint8 E4M3 block scales
            ``[E, model_dim/16, 2*I]``.
        global_scale: Per-expert FP32 scale ``[E]``.
        sorted_token_ids: Int32 route IDs, padded with the token count.
        sorted_expert_ids: Int32 expert ID for each route tile.
        num_valid_ids: One-element int32 count of valid routes.
        topk: Number of routed experts per token.
        inter_dim: Per-expert intermediate dimension.
        tile_m: Route tile height.
        tile_n: Output tile width.
        tile_k: K tile width.
        output: Optional BF16 ``[M, topk, inter_dim]`` destination.
        sorted_weights: Optional FP32 router weights aligned with route IDs.
        k_batch: Split-K factor.

    Returns:
        The stage-one BF16 output.
    """
    _validate_tile(tile_m, tile_n, tile_k)
    model_dim = activations.shape[-1]
    _validate_common(
        activations,
        weight,
        weight_scale,
        global_scale,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        expected_k=model_dim,
    )
    if weight.shape[1] != 2 * inter_dim or weight.shape[2] * 2 != model_dim:
        raise ValueError(
            "stage-one NVFP4 weight shape does not match activations/inter_dim"
        )
    if output is None:
        output = torch.empty(
            (activations.shape[0], topk, inter_dim),
            dtype=torch.bfloat16,
            device=activations.device,
        )
    if output.shape != (activations.shape[0], topk, inter_dim):
        raise ValueError("stage-one output must have shape [M, topk, inter_dim]")
    if k_batch < 1:
        raise ValueError("k_batch must be positive")
    weights = (
        sorted_weights
        if sorted_weights is not None
        else _empty(activations.device, torch.float32)
    )
    is_split_k = k_batch > 1
    # Split-K atomically accumulates separate gate/up projections. Allocate the
    # required initialized temporary directly rather than empty() followed by
    # zero_(). The activation below reduces its last dimension back to I.
    partial_output = (
        torch.zeros(
            (activations.shape[0], topk, 2 * inter_dim),
            dtype=torch.bfloat16,
            device=activations.device,
        )
        if is_split_k
        else None
    )
    kernel_output = partial_output if partial_output is not None else output

    executable = compile_moe_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=weight.shape[0],
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=sorted_weights is not None,
        group_size=16,
        out_dtype="bf16",
        use_cshuffle_epilog=None if is_split_k else False,
        k_batch=k_batch,
    )
    _run_compiled(
        executable,
        *(
            ptr_arg(kernel_output),
            ptr_arg(activations),
            ptr_arg(weight),
            ptr_arg(_empty(activations.device)),
            ptr_arg(weight_scale),
            ptr_arg(global_scale),
            ptr_arg(sorted_token_ids),
            ptr_arg(sorted_expert_ids),
            ptr_arg(weights),
            ptr_arg(num_valid_ids),
            activations.shape[0],
            inter_dim,
            model_dim,
            sorted_expert_ids.numel(),
            torch.cuda.current_stream(),
        ),
    )
    if partial_output is not None:
        silu_and_mul(
            output.view(-1, inter_dim),
            partial_output.view(-1, 2 * inter_dim),
        )
    return output


def nvfp4_moe_stage2(
    activations: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    *,
    topk: int,
    model_dim: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    output: torch.Tensor | None = None,
    sorted_weights: torch.Tensor | None = None,
    mode: Literal["atomic", "reduce"] = "atomic",
) -> torch.Tensor:
    """Run the BF16 × NVFP4 down projection."""
    _validate_tile(tile_m, tile_n, tile_k)
    if mode != "atomic":
        raise NotImplementedError(
            "vLLM's initial NVFP4 FlyDSL port supports atomic stage two"
        )
    inter_dim = activations.shape[-1]
    _validate_common(
        activations,
        weight,
        weight_scale,
        global_scale,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        expected_k=inter_dim,
    )
    if activations.ndim != 3 or activations.shape[1] != topk:
        raise ValueError("stage-two activations must have shape [M, topk, inter_dim]")
    if weight.shape[1] != model_dim or weight.shape[2] * 2 != inter_dim:
        raise ValueError(
            "stage-two NVFP4 weight shape does not match activations/model_dim"
        )
    if output is None:
        output = torch.zeros(
            (activations.shape[0], model_dim),
            dtype=torch.bfloat16,
            device=activations.device,
        )
    if output.shape != (activations.shape[0], model_dim):
        raise ValueError("stage-two output must have shape [M, model_dim]")
    if sorted_weights is None:
        sorted_weights = _empty(activations.device, torch.float32)

    executable = compile_moe_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=weight.shape[0],
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=sorted_weights.numel() > 0,
        group_size=16,
        out_dtype="bf16",
        accumulate=True,
    )
    _run_compiled(
        executable,
        *(
            ptr_arg(output),
            ptr_arg(activations),
            ptr_arg(weight),
            ptr_arg(_empty(activations.device)),
            ptr_arg(weight_scale),
            ptr_arg(global_scale),
            ptr_arg(sorted_token_ids),
            ptr_arg(sorted_expert_ids),
            ptr_arg(sorted_weights),
            ptr_arg(num_valid_ids),
            activations.shape[0],
            model_dim,
            inter_dim,
            sorted_expert_ids.numel(),
            torch.cuda.current_stream(),
        ),
    )
    return output
