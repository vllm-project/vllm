# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Helion kernels for fused batched matrix multiply + FP8 quantization.

Equivalent to the Triton kernels in vllm/kernels/triton/ops/bmm_fp8_quant.py,
but written using Helion's higher-level API for portability and maintainability.

Used by MLA's _v_up_proj to fuse the V up-projection BMM with the
post-attention FP8 quantization into a single kernel.

Two quantization modes:

1. Static per-tensor (bmm_fp8_quant_helion):
    output[b, n*V:(n+1)*V] = fp8_quant(input[n, b, :] @ weight[n, :, :], scale)

2. Dynamic per-group (bmm_fp8_group_quant_helion):
    group_scale[n, b] = max(abs(bmm_result[n, b, :])) / FP8_MAX
    output[b, n*V:(n+1)*V] = clamp(bmm_result / group_scale, -FP8_MAX, FP8_MAX)
"""

from typing import Any

import regex as re
import torch

from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "bmm_fp8_quant Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion.language as hl

from vllm.kernels.helion.register import register_kernel
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)

logger = init_logger(__name__)

# Platform-aware FP8 max (448.0 for e4m3fn on NVIDIA, 224.0 for e4m3fnuz on ROCm)
_, _FP8_MAX = get_fp8_min_max()


@register_kernel  # type: ignore[misc]
def bmm_fp8_quant_helion(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Fused batched matrix multiply + FP8 static quantization (Helion).

    Args:
        input: (N, B, L) input tensor in bf16/fp16
        weight: (N, L, V) weight tensor in bf16/fp16
        scale: scalar tensor - static quantization scale

    Returns:
        (B, N, V) tensor in FP8, viewable as (B, N*V)
    """
    N, B, _L = input.shape
    V = hl.specialize(weight.shape[2])

    # Output in (N, B, V) to match computation order (avoids in-kernel transpose)
    out = torch.empty(N, B, V, device=input.device, dtype=torch.float8_e4m3fn)

    for tile_n, tile_b in hl.tile([N, B]):
        # Batched matmul: contract over L dimension
        # input[tile_n, tile_b, :] shape: (tile_n, tile_b, L)
        # weight[tile_n, :, :]     shape: (tile_n, L, V)
        # result                   shape: (tile_n, tile_b, V)
        result = input[tile_n, tile_b, :] @ weight[tile_n, :, :]

        # Quantize to FP8: scale, clamp, cast
        result_f32 = result.to(torch.float32)
        scale_val = hl.load(scale, [0])
        result_scaled = (result_f32 * scale_val).clamp(-_FP8_MAX, _FP8_MAX)

        out[tile_n, tile_b, :] = result_scaled.to(out.dtype)

    # Transpose to (B, N, V) — .view(B, N*V) gives correct head-interleaved layout
    return out.permute(1, 0, 2).contiguous()


@bmm_fp8_quant_helion.register_input_generator  # type: ignore[misc]
def generate_bmm_fp8_quant_inputs() -> dict[str, tuple[Any, ...]]:
    """Generate inputs for autotuning across DeepSeek-V3/R1 MLA dimensions."""
    # DeepSeek-V3/R1 constants
    kv_lora_rank = 512
    v_head_dim = 128
    head_counts = [16, 64, 128]
    # Typical decode batch sizes
    batch_sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))

    inputs = {}
    for N in head_counts:
        for B in batch_sizes:
            input_tensor = torch.randn(
                N, B, kv_lora_rank, device="cuda", dtype=torch.bfloat16
            )
            weight_tensor = torch.randn(
                N, kv_lora_rank, v_head_dim, device="cuda", dtype=torch.bfloat16
            )
            scale_tensor = torch.tensor([0.01], device="cuda", dtype=torch.float32)

            config_key = f"heads_{N}_batch_{B}"
            inputs[config_key] = (input_tensor, weight_tensor, scale_tensor)

    return inputs


@bmm_fp8_quant_helion.register_config_picker  # type: ignore[misc]
def pick_bmm_fp8_quant_config(
    args: tuple[Any, ...], config_keys: list[str]
) -> str | None:
    """Pick the best pre-tuned config for given input shapes.

    Selection strategy:
      1. Find configs matching the closest num_heads.
      2. Among those, pick the smallest batch_size >= input's batch_size.
         Fall back to the largest if input exceeds all.
    """
    if not config_keys:
        return None

    input_tensor, _weight, _scale = args
    N = input_tensor.shape[0]  # num_heads
    B = input_tensor.shape[1]  # batch_size

    configs: dict[int, list[int]] = {}
    for key in config_keys:
        if key == "default":
            continue
        match = re.fullmatch(r"heads_(\d+)_batch_(\d+)", key)
        if not match:
            raise ValueError(
                f"Malformed config key '{key}', "
                f"expected format 'heads_{{int}}_batch_{{int}}'"
            )
        heads, batch = int(match.group(1)), int(match.group(2))
        configs.setdefault(heads, []).append(batch)

    if not configs:
        return "default" if "default" in config_keys else None

    # Find closest head count
    best_heads = min(configs, key=lambda h: abs(h - N))
    available_batches = sorted(configs[best_heads])
    # Pick smallest batch >= B, or fall back to largest
    best_batch = next((b for b in available_batches if b >= B), available_batches[-1])

    return f"heads_{best_heads}_batch_{best_batch}"


# ── Per-group (dynamic) FP8 quantization ────────────────────────────────────


@register_kernel  # type: ignore[misc]
def bmm_fp8_group_quant_helion(
    input: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused batched matrix multiply + per-group dynamic FP8 quantization.

    Each group is one head's V elements for one token. Scales are computed
    dynamically as abs_max / FP8_MAX per group.

    Args:
        input: (N, B, L) input tensor in bf16/fp16
        weight: (N, L, V) weight tensor in bf16/fp16

    Returns:
        out: (N, B, V) tensor in FP8 (caller permutes to (B, N, V))
        scales: (N, B) tensor in float32 (caller permutes to (B, N))
    """
    N, B, _L = input.shape
    V = hl.specialize(weight.shape[2])

    out = torch.empty(N, B, V, device=input.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(N, B, device=input.device, dtype=torch.float32)

    for tile_n, tile_b in hl.tile([N, B]):
        result = input[tile_n, tile_b, :] @ weight[tile_n, :, :]
        result_f32 = result.to(torch.float32)

        # Per-group: compute abs_max over V dim for each (n, b)
        # Use keepdim=True to keep 3D shape — avoids unsqueeze which
        # triggers a Helion codegen bug with 2D→3D broadcasting.
        abs_max = result_f32.abs().amax(dim=-1, keepdim=True)  # (tile_n, tile_b, 1)
        abs_max = abs_max.clamp(min=1e-12)
        inv_scale = _FP8_MAX / abs_max  # (tile_n, tile_b, 1) — broadcasts naturally

        result_scaled = (result_f32 * inv_scale).clamp(-_FP8_MAX, _FP8_MAX)

        out[tile_n, tile_b, :] = result_scaled.to(out.dtype)
        scales[tile_n, tile_b] = abs_max.squeeze(-1) / _FP8_MAX

    return out.permute(1, 0, 2).contiguous(), scales.permute(1, 0).contiguous()


@bmm_fp8_group_quant_helion.register_input_generator  # type: ignore[misc]
def generate_bmm_fp8_group_quant_inputs() -> dict[str, tuple[Any, ...]]:
    """Generate inputs for autotuning across DeepSeek-V3/R1 MLA dimensions."""
    kv_lora_rank = 512
    v_head_dim = 128
    head_counts = [16, 64, 128]
    batch_sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))

    inputs = {}
    for N in head_counts:
        for B in batch_sizes:
            input_tensor = torch.randn(
                N, B, kv_lora_rank, device="cuda", dtype=torch.bfloat16
            )
            weight_tensor = torch.randn(
                N, kv_lora_rank, v_head_dim, device="cuda", dtype=torch.bfloat16
            )

            config_key = f"heads_{N}_batch_{B}"
            inputs[config_key] = (input_tensor, weight_tensor)

    return inputs


@bmm_fp8_group_quant_helion.register_config_picker  # type: ignore[misc]
def pick_bmm_fp8_group_quant_config(
    args: tuple[Any, ...], config_keys: list[str]
) -> str | None:
    """Pick the best pre-tuned config for given input shapes."""
    if not config_keys:
        return None

    input_tensor, _weight = args
    N = input_tensor.shape[0]
    B = input_tensor.shape[1]

    configs: dict[int, list[int]] = {}
    for key in config_keys:
        if key == "default":
            continue
        match = re.fullmatch(r"heads_(\d+)_batch_(\d+)", key)
        if not match:
            raise ValueError(
                f"Malformed config key '{key}', "
                f"expected format 'heads_{{int}}_batch_{{int}}'"
            )
        heads, batch = int(match.group(1)), int(match.group(2))
        configs.setdefault(heads, []).append(batch)

    if not configs:
        return "default" if "default" in config_keys else None

    best_heads = min(configs, key=lambda h: abs(h - N))
    available_batches = sorted(configs[best_heads])
    best_batch = next((b for b in available_batches if b >= B), available_batches[-1])

    return f"heads_{best_heads}_batch_{best_batch}"
