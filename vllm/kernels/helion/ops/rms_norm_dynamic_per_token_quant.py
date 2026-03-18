# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import product
from typing import Any

import helion
import helion.language as hl
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
        "Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)


def _get_fp8_dtype() -> torch.dtype:
    return current_platform.fp8_dtype()


def _get_int8_min_max() -> tuple[int, int]:
    qtype_traits = torch.iinfo(torch.int8)
    return qtype_traits.min, qtype_traits.max


def _get_int8_min_scaling_factor() -> float:
    return torch.finfo(torch.float32).eps


def generate_inputs() -> dict[str, tuple[Any, ...]]:
    # TODO(xiaohongchen1991): it is difficult for kernel author to cover all
    # input property combination. Currently, dtypes are fixed. We need
    # optimization to bucket/skip some combinations
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    hidden_size_list = [2048, 4096, 8192]
    in_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    scale_dtype: torch.dtype = torch.float32
    inputs = {}

    for num_tokens, hidden_size in product(num_tokens_list, hidden_size_list):
        input = torch.randn(num_tokens, hidden_size, device="cuda", dtype=in_dtype)
        result = torch.empty(input.shape, device=input.device, dtype=out_dtype)
        scale = torch.empty((num_tokens, 1), device=input.device, dtype=scale_dtype)
        scale_ub = torch.mean(input).to(scale_dtype)
        residual = torch.randn_like(input)
        weight = torch.normal(
            mean=1.0,
            std=1.0,
            size=(hidden_size,),
            dtype=input.dtype,
            device=input.device,
        )
        epsilon = 1e-6

        config_key = f"hidden_size_{hidden_size}_num_tokens_{num_tokens}"
        inputs[config_key] = (result, input, weight, scale, epsilon, scale_ub, residual)

    return inputs


def pick_config(args: tuple[Any, ...], config_keys: list[str]) -> str | None:
    """Pick the best pre-tuned config for the given input shape.

    Selection strategy:
      1. Find the closest hidden_size among available configs
         (exact match preferred).
      2. Among the num_tokens values tuned for that hidden_size, pick
         the smallest num_tokens >= the input's num_tokens. If the input is
         larger than all available num_tokens, fall back to the largest.

    Config keys must be "default" or follow the format
    "hidden_size_{int}_num_tokens_{int}".
    """

    if not config_keys:
        return None

    _, input, *_ = args
    num_tokens, hidden_size = input.shape

    configs: dict[int, list[int]] = {}
    for key in config_keys:
        if key == "default":
            continue
        match = re.fullmatch(r"hidden_size_(\d+)_num_tokens_(\d+)", key)
        if not match:
            raise ValueError(
                f"Malformed config key '{key}', "
                f"expected format 'hidden_size_{{int}}_num_tokens_{{int}}'"
            )
        hidden_size_str, num_tokens_str = match.groups()
        configs.setdefault(int(hidden_size_str), []).append(int(num_tokens_str))

    if not configs:
        return "default" if "default" in config_keys else None

    best_hidden_size = min(configs, key=lambda s: abs(s - hidden_size))
    available_num_tokens = sorted(configs[best_hidden_size])
    best_num_tokens = next(
        (n for n in available_num_tokens if n >= num_tokens), available_num_tokens[-1]
    )

    return f"hidden_size_{best_hidden_size}_num_tokens_{best_num_tokens}"


@register_kernel(
    mutates_args=["result", "scale", "residual"],
    config_picker=pick_config,
    input_generator=generate_inputs,
    helion_settings=helion.Settings(
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ),
)  # type: ignore[misc]
def rms_norm_dynamic_per_token_quant(
    result: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]
    weight: torch.Tensor,  # [hidden_size]
    scale: torch.Tensor,  # [num_tokens, 1]
    epsilon: float,
    scale_ub: torch.Tensor | None = None,  # []
    residual: torch.Tensor | None = None,  # [num_tokens, hidden_size]
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)

    fp8_dtype = _get_fp8_dtype()
    assert result.dtype in [fp8_dtype, torch.int8]
    assert result.is_contiguous() and input.is_contiguous()

    if scale_ub is not None:
        assert result.dtype == fp8_dtype
        assert scale_ub.dtype == torch.float32

    assert input.dtype == weight.dtype
    assert scale.shape[0] == num_tokens
    assert scale.dtype == torch.float32

    if residual is not None:
        assert residual.dtype == input.dtype

    quant_dtype = result.dtype
    qtype_traits_min: int | float
    qtype_traits_max: int | float
    if quant_dtype == torch.int8:
        qtype_traits_min, qtype_traits_max = _get_int8_min_max()
        min_scaling_factor = _get_int8_min_scaling_factor()
    else:
        qtype_traits_min, qtype_traits_max = get_fp8_min_max()
        min_scaling_factor = 1.0 / (qtype_traits_max * 512.0)

    qtype_max = float(qtype_traits_max)

    for tile_m in hl.tile(num_tokens, block_size=1):
        rms = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(torch.float32)
            if residual is not None:
                x_blk = x_blk + residual[tile_m, tile_n]
            rms = rms + x_blk.pow(2).sum(dim=-1)

        rms = torch.rsqrt(rms * (1.0 / hidden_size) + epsilon)
        s_blk = hl.zeros([tile_m], dtype=torch.float32)

        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(torch.float32)
            if residual is not None:
                x_blk = x_blk + residual[tile_m, tile_n]
            x_blk = (x_blk * rms[:, None]).to(input.dtype) * weight[None, tile_n]
            tmp_blk = torch.amax(torch.abs(x_blk), dim=-1).to(torch.float32)
            s_blk = torch.maximum(s_blk, tmp_blk)

        if scale_ub is not None:
            scale_ub_s = hl.load(scale_ub, [])
            s_blk = s_blk.clamp(max=scale_ub_s)
        s_blk = s_blk * (1.0 / qtype_max)
        s_blk = s_blk.clamp(min=min_scaling_factor)
        scale[tile_m, 0] = s_blk

        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(torch.float32)
            if residual is not None:
                x_blk = x_blk + residual[tile_m, tile_n]
                residual[tile_m, tile_n] = x_blk.to(residual.dtype)
            x_blk = (x_blk * rms[:, None]).to(input.dtype) * weight[None, tile_n]
            if quant_dtype == torch.int8:
                s_inv_blk = 1.0 / s_blk[:, None]
                y_blk = x_blk * s_inv_blk
                y_blk = y_blk.round()
            else:
                y_blk = x_blk / s_blk[:, None]

            result[tile_m, tile_n] = y_blk.clamp(qtype_traits_min, qtype_traits_max).to(
                result.dtype
            )


def baseline(
    result: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]
    weight: torch.Tensor,  # [num_tokens]
    scale: torch.Tensor,  # [num_tokens, 1]
    epsilon: float,
    scale_ub: torch.Tensor | None = None,  # []
    residual: torch.Tensor | None = None,  # [num_tokens, hidden_size]
) -> None:
    torch.ops._C.rms_norm_dynamic_per_token_quant(
        result, input, weight, scale, epsilon, scale_ub, residual
    )
