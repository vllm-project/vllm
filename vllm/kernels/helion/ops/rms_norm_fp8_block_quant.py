# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import regex as re
import torch

from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "rms_norm_fp8_block_quant Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion.language as hl

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)


def generate_rms_norm_fp8_block_quant_inputs() -> dict[str, tuple[Any, ...]]:
    hidden_sizes = [2048, 4096, 8192, 14336]
    num_tokens_list = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))
    group_sizes = [128]

    inputs = {}
    for num_tokens in num_tokens_list:
        for hidden_size in hidden_sizes:
            for group_size in group_sizes:
                input_tensor = torch.randn(
                    num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
                )
                weight = torch.ones(
                    hidden_size, device="cuda", dtype=torch.bfloat16
                )
                config_key = (
                    f"hidden_{hidden_size}_numtokens_{num_tokens}"
                    f"_groupsize_{group_size}"
                )
                inputs[config_key] = (input_tensor, weight, group_size, 1e-6)

    return inputs


def pick_rms_norm_fp8_block_quant_config(
    args: tuple[Any, ...], config_keys: list[str]
) -> str | None:
    """Pick the best pre-tuned config for the given input shape.

    Selection strategy:
      1. Find the closest hidden_size + group_size combination.
      2. Among available num_tokens, pick the smallest >= input num_tokens.
         Fall back to the largest if input exceeds all available.

    Config keys must be "default" or follow the format
    "hidden_{int}_numtokens_{int}_groupsize_{int}".
    """
    if not config_keys:
        return None

    input_tensor, _weight, group_size, _eps = args
    input_2d = input_tensor.view(-1, input_tensor.shape[-1])
    num_tokens = input_2d.shape[0]
    hidden_size = input_2d.shape[1]

    configs: dict[tuple[int, int], list[int]] = {}
    for key in config_keys:
        if key == "default":
            continue
        match = re.fullmatch(
            r"hidden_(\d+)_numtokens_(\d+)_groupsize_(\d+)", key
        )
        if not match:
            raise ValueError(
                f"Malformed config key '{key}', "
                f"expected format "
                f"'hidden_{{int}}_numtokens_{{int}}_groupsize_{{int}}'"
            )
        h, nt, gs = int(match.group(1)), int(match.group(2)), int(match.group(3))
        configs.setdefault((h, gs), []).append(nt)

    if not configs:
        return "default" if "default" in config_keys else None

    best_key = min(
        configs,
        key=lambda k: (abs(k[0] - hidden_size), abs(k[1] - group_size)),
    )
    available_ntokens = sorted(configs[best_key])
    best_ntokens = next(
        (n for n in available_ntokens if n >= num_tokens), available_ntokens[-1]
    )

    return (
        f"hidden_{best_key[0]}_numtokens_{best_ntokens}_groupsize_{best_key[1]}"
    )


@register_kernel(
    config_picker=pick_rms_norm_fp8_block_quant_config,
    input_generator=generate_rms_norm_fp8_block_quant_inputs,
)
def rms_norm_fp8_block_quant(
    input: torch.Tensor,   # [..., hidden_size]
    weight: torch.Tensor,  # [hidden_size]
    group_size: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RMSNorm + fp8 block (group) quantization Helion kernel.

    Computes RMSNorm then dynamically quantizes the output into fp8 using
    per-group (block) scaling. Scales are computed dynamically inside the
    kernel — no pre-computed scale is needed.

    Args:
        input:      Input tensor of shape [..., hidden_size]
        weight:     RMSNorm weight of shape [hidden_size]
        group_size: Number of elements per quantization group (e.g. 128)
        epsilon:    Variance epsilon for numerical stability

    Returns:
        out:    fp8-quantized output, same shape as input
        scales: Per-group scales, shape [num_tokens, hidden_size // group_size]
    """
    input_2d = input.view(-1, input.shape[-1])
    num_tokens = input_2d.shape[0]
    hidden_size = hl.specialize(input_2d.shape[1])
    num_groups = hidden_size // group_size

    out = torch.empty(
        (num_tokens, hidden_size),
        device=input.device,
        dtype=torch.float8_e4m3fn,
    )
    scales = torch.empty(
        (num_tokens, num_groups),
        device=input.device,
        dtype=torch.float32,
    )

    fp8_max: float = 448.0

    for tile_m in hl.tile([num_tokens]):
        # Step 1: RMSNorm
        row_f32 = input_2d[tile_m, :].to(torch.float32)
        variance = (row_f32 * row_f32).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(variance + epsilon)
        normed = row_f32 * rms * weight.to(torch.float32)  # [tile_m, hidden_size]

        # Step 2: dynamic per-group fp8 quantization
        normed_groups = normed.reshape(-1, num_groups, group_size)

        amax = normed_groups.abs().amax(dim=-1)             # [tile_m, num_groups]
        group_scale = torch.clamp(amax / fp8_max, min=1e-12)

        scales[tile_m, :] = group_scale

        inv_scale = (1.0 / group_scale).unsqueeze(-1)       # [tile_m, num_groups, 1]
        quantized = (normed_groups * inv_scale).clamp(-fp8_max, fp8_max)
        out[tile_m, :] = quantized.reshape(-1, hidden_size).to(torch.float8_e4m3fn)

    return out.view(input.shape), scales


def rms_norm_fp8_block_quant_baseline(
    input: torch.Tensor,
    weight: torch.Tensor,
    group_size: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Baseline using the existing vLLM CUDA kernel for correctness comparison."""
    from vllm import _custom_ops as ops

    hidden_size = input.shape[-1]
    num_tokens = input.numel() // hidden_size
    num_groups = hidden_size // group_size

    out = torch.empty(input.shape, device=input.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (num_tokens, num_groups), device=input.device, dtype=torch.float32
    )
    ops.rms_norm_per_block_quant(
        out, input, weight, scales, epsilon, None, None, group_size, False
    )
    return out, scales