# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import regex as re
import torch

from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "fused_silu_mul_block_quant_fp8 Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion.language as hl

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)


def generate_silu_mul_block_quant_fp8_inputs() -> dict[str, tuple[Any, ...]]:
    intermediate_sizes = [2048, 2880, 4096, 8192, 11008, 14336]

    # Use the same num_tokens values as vLLM's default cudagraph capture sizes.
    # See vllm/config/vllm.py _set_cudagraph_sizes() for the canonical formula.
    num_tokens_list = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))

    inputs = {}
    for num_tokens in num_tokens_list:
        for intermediate_size in intermediate_sizes:
            input_tensor = torch.randn(
                num_tokens,
                2 * intermediate_size,
                device="cuda",
                dtype=torch.bfloat16,
            )

            config_key = f"intermediate_{intermediate_size}_numtokens_{num_tokens}"
            inputs[config_key] = input_tensor

    return inputs


def pick_silu_mul_block_quant_fp8_config(
    args: tuple[Any, ...], config_keys: list[str]
) -> str | None:
    """Pick the best pre-tuned config for the given input shape.

    Selection strategy:
      1. Find the closest intermediate_size among available configs
         (exact match preferred).
      2. Among the num_tokens values tuned for that intermediate_size, pick
         the smallest num_tokens >= the input's num_tokens. If the input is
         larger than all available num_tokens, fall back to the largest.

    Config keys must be "default" or follow the format
    "intermediate_{int}_numtokens_{int}".
    """
    if not config_keys:
        return None

    input_tensor: torch.Tensor = args[0]
    intermediate_size = input_tensor.shape[-1] // 2
    num_tokens = input_tensor.view(-1, input_tensor.shape[-1]).shape[0]
    configs: dict[int, list[int]] = {}
    for key in config_keys:
        if key == "default":
            continue
        match = re.fullmatch(r"intermediate_(\d+)_numtokens_(\d+)", key)
        if not match:
            raise ValueError(
                f"Malformed config key '{key}', "
                f"expected format 'intermediate_{{int}}_numtokens_{{int}}'"
            )
        isize_str, ntokens_str = match.groups()
        configs.setdefault(int(isize_str), []).append(int(ntokens_str))

    if not configs:
        return "default" if "default" in config_keys else None

    best_isize = min(configs, key=lambda s: abs(s - intermediate_size))
    available_ntokens = sorted(configs[best_isize])
    best_ntokens = next(
        (n for n in available_ntokens if n >= num_tokens), available_ntokens[-1]
    )

    return f"intermediate_{best_isize}_numtokens_{best_ntokens}"


@register_kernel(
    config_picker=pick_silu_mul_block_quant_fp8_config,
    input_generator=generate_silu_mul_block_quant_fp8_inputs,
)
def silu_mul_block_quant_fp8(
    input: torch.Tensor,
    scale_ub: torch.Tensor | None = None,
    is_scale_transposed: bool = False,
) -> torch.Tensor:
    """
    Helion kernel for blockwise quantized SiLU + mul fused kernel.
    """
    original_shape = input.shape
    two_d = hl.specialize(original_shape[-1])
    d = two_d // 2
    output_shape = original_shape[:-1] + (d,)

    input_2d = input.view(-1, original_shape[-1])
    m = input_2d.shape[0]

    # block sizes
    block_m = hl.register_block_size(m)
    block_d = hl.register_block_size(d)

    # scale_out tensor sizes
    scale_m = hl.cdiv(m, block_m)
    scale_d = hl.cdiv(d, block_d)

    out = torch.empty((m, d), device=input.device, dtype=torch.float8_e4m3fn)
    min_scaling_factor = 1 / (torch.finfo(torch.float8_e4m3fn).max * 512.0)

    if is_scale_transposed:
        scale_out = torch.empty(
            (scale_d, scale_m), device=input.device, dtype=torch.float32
        )
    else:
        scale_out = torch.empty(
            (scale_m, scale_d), device=input.device, dtype=torch.float32
        )

    input_part_a = input_2d[:, :d]
    input_part_b = input_2d[:, d:]

    for tile_m, tile_n in hl.tile([m, d], block_size=[block_m, block_d]):
        a_vals = input_part_a[tile_m, tile_n]
        silu_result = torch.nn.functional.silu(a_vals)
        b_vals = input_part_b[tile_m, tile_n]
        result = silu_result * b_vals
        result_f32 = result.to(torch.float32)

        abs_max = torch.max(torch.abs(result_f32))
        block_scale = abs_max / torch.finfo(torch.float8_e4m3fn).max

        if scale_ub is not None:
            block_scale = torch.min(block_scale, scale_ub)

        block_scale = torch.max(block_scale, min_scaling_factor)
        inv_block_scale = 1.0 / block_scale

        out[tile_m, tile_n] = (result_f32 * inv_block_scale).to(torch.float8_e4m3fn)

        if is_scale_transposed:
            scale_out[tile_n.begin // block_d, tile_m.begin // block_m] = (
                inv_block_scale
            )
        else:
            scale_out[tile_m.begin // block_m, tile_n.begin // block_d] = (
                inv_block_scale
            )

    return out.view(output_shape), scale_out


def silu_mul_block_quant_fp8_baseline(
    input: torch.Tensor,
    block_size: int,
    scale_ub: torch.Tensor | None = None,
    is_scale_transposed: bool = False,
) -> torch.Tensor:
    return torch.ops._C.silu_and_mul_per_block_quant(
        input, block_size, scale_ub, is_scale_transposed
    )
