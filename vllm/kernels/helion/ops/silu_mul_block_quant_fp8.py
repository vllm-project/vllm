# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import product
from typing import Any

import regex as re
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "silu_mul_block_quant_fp8 Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion
import helion.language as hl

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)


def _get_fp8_dtype() -> torch.dtype:
    return current_platform.fp8_dtype()


def generate_silu_mul_block_quant_fp8_inputs() -> dict[str, tuple[Any, ...]]:
    hidden_size_list = [2048, 4096, 8192, 16384]

    # Use the same num_tokens values as vLLM's default cudagraph capture sizes.
    # See vllm/config/vllm.py _set_cudagraph_sizes() for the canonical formula.
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    group_size_list = [64, 128]
    in_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    scale_dtype: torch.dtype = torch.float32

    inputs = {}
    for hidden_size, group_size, num_tokens in product(
        hidden_size_list, group_size_list, num_tokens_list
    ):
        input = torch.randn(num_tokens, hidden_size, device="cuda", dtype=in_dtype)
        out = torch.empty(
            (num_tokens, hidden_size // 2), device=input.device, dtype=out_dtype
        )

        scales = torch.empty(
            (num_tokens, hidden_size // (group_size * 2)),
            device=input.device,
            dtype=scale_dtype,
        )
        scale_ub = torch.mean(input.abs()).to(scale_dtype)

        config_key = (
            f"hidden_size_{hidden_size}_group_size_{group_size}_num_tokens_{num_tokens}"
        )
        inputs[config_key] = (input, out, scales, group_size, scale_ub)

    return inputs


def pick_silu_mul_block_quant_fp8_config(
    args: tuple[Any, ...], config_keys: list[str]
) -> str | None:
    """Pick the best pre-tuned config for the given input shape.

    Selection strategy:
      1. Find the closest hidden_size among available configs
         (exact match preferred).
      2. Find the closest group_size among available configs
         (exact match preferred).
      2. Among the num_tokens values tuned for that hidden_size and group_size, pick
         the smallest num_tokens >= the input's num_tokens. If the input is
         larger than all available num_tokens, fall back to the largest.

    Config keys must be "default" or follow the format
    "hidden_size_{int}_group_size_{int}_num_tokens_{int}".
    """
    if not config_keys:
        return None

    # input_tensor, _, _, group_size, _ = args
    input_tensor, group_size = args[0], args[3]
    hidden_size = input_tensor.shape[-1] // 2
    num_tokens = input_tensor.view(-1, input_tensor.shape[-1]).shape[0]
    configs: dict[int, dict[int, list[int]]] = {}
    for key in config_keys:
        if key == "default":
            continue
        match = re.fullmatch(
            r"hidden_size_(\d+)_group_size_(\d+)_num_tokens_(\d+)", key
        )
        if not match:
            raise ValueError(
                f"Malformed config key '{key}', "
                f"expected format 'hidden_size_{{int}}"
                f"_group_size_{{int}}_num_tokens_{{int}}'"
            )
        hidden_size_str, group_size_str, num_tokens_str = match.groups()
        configs.setdefault(int(hidden_size_str), {}).setdefault(
            int(group_size_str), []
        ).append(int(num_tokens_str))

    if not configs:
        return "default" if "default" in config_keys else None

    best_hidden_size = min(configs, key=lambda s: abs(s - hidden_size))
    best_group_size = min(configs[best_hidden_size], key=lambda s: abs(s - group_size))
    available_ntokens = sorted(configs[best_hidden_size][best_group_size])
    best_ntokens = next(
        (n for n in available_ntokens if n >= num_tokens), available_ntokens[-1]
    )

    return (
        f"hidden_size_{best_hidden_size}_group_size_"
        f"{best_group_size}_num_tokens_{best_ntokens}"
    )


@register_kernel(
    mutates_args=["out", "scales"],
    helion_settings=helion.Settings(
        autotune_baseline_atol=0.2,
        autotune_baseline_rtol=0.1,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
        # TODO search._autotune_metrics.num_accuracy_failures for tolerance strictness
    ),
    config_picker=pick_silu_mul_block_quant_fp8_config,
    input_generator=generate_silu_mul_block_quant_fp8_inputs,
)
def silu_mul_block_quant_fp8(
    input: torch.Tensor,  # [num_tokens, 2 * hidden_size]
    out: torch.Tensor,  # [num_tokens, hidden_size]
    scales: torch.Tensor,  # [num_tokens, hidden_size // block_size]
    block_size: int,
    scale_ub: torch.Tensor | None = None,
    is_scale_transposed: bool = False,  # dummy
) -> torch.Tensor:
    # This code assumes batch_dim and num_tokens are flattened
    group_size = hl.specialize(block_size)
    assert group_size == 64 or group_size == 128
    assert input.is_contiguous() and input.ndim == 2
    assert scales.is_contiguous() and scales.dtype == torch.float32
    fp8_dtype = _get_fp8_dtype()
    assert out.is_contiguous() and out.dtype == fp8_dtype
    assert out.shape[-1] == input.shape[-1] // 2

    if scale_ub is not None:
        assert scale_ub.dtype == torch.float32

    original_shape = input.shape
    two_d = hl.specialize(original_shape[-1])
    d = two_d // 2
    m = hl.specialize(original_shape[0])

    input_2d = input.view(-1, original_shape[-1])

    _, fp8_max = get_fp8_min_max()
    min_scaling_factor = 1 / (fp8_max * 512.0)

    for tile_m, tile_d in hl.tile([m, d], block_size=[1, group_size]):
        a_vals = hl.load(input_2d, [tile_m, tile_d])
        b_vals = hl.load(input_2d, [tile_m, tile_d + d])
        a_f32 = a_vals.to(torch.float32)
        b_f32 = b_vals.to(torch.float32)
        silu_result = torch.nn.functional.silu(a_f32)
        result_f32 = silu_result * b_f32
        abs_val = torch.abs(result_f32.reshape(-1))
        abs_max = torch.amax(abs_val)
        block_scale = abs_max / fp8_max

        if scale_ub is not None:
            ub_val = hl.load(scale_ub, index=[])
            block_scale = torch.clamp(block_scale, max=ub_val)

        block_scale = torch.clamp(block_scale, min=min_scaling_factor)
        inv_block_scale = 1.0 / block_scale

        out[tile_m, tile_d] = (result_f32 * inv_block_scale).to(out.dtype)

        scales[tile_m, tile_d.id] = block_scale

    return out, scales


def silu_mul_block_quant_fp8_baseline(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    block_size: int,
    scale_ub: torch.Tensor | None = None,
    is_scale_transposed: bool = False,
) -> torch.Tensor:
    return torch.ops._C.silu_and_mul_per_block_quant(
        out, input, scales, block_size, scale_ub, is_scale_transposed
    )
