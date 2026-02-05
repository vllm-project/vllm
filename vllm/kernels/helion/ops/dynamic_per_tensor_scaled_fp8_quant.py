# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import product
from typing import Any

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


def generate_inputs() -> dict[str, tuple[Any, ...]]:
    # TODO(xiaohongchen1991): it is difficult for kernel author to cover
    # all input property combination. Currently, dtypes are fixed. We need
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
        scale = torch.empty(1, device=input.device, dtype=scale_dtype)

        config_key = f"hidden_size_{hidden_size}_num_tokens_{num_tokens}"
        inputs[config_key] = (result, input, scale)

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

    _, input, _ = args
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
    mutates_args=["result", "scale"],
    config_picker=pick_config,
    input_generator=generate_inputs,
)  # type: ignore[misc]
def dynamic_per_tensor_scaled_fp8_quant(
    result: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]
    scale: torch.Tensor,  # [1]
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)

    assert result.shape == input.shape
    assert scale.shape[0] == 1
    assert scale.dtype == torch.float32
    assert input.stride()[-1] == 1
    assert result.stride()[-1] == 1

    _, fp8_max = get_fp8_min_max()

    scale.zero_()

    for tile_m, tile_n in hl.tile([num_tokens, hidden_size]):
        abs_value = torch.abs(input[tile_m, tile_n])
        # multiple reduction dimensions not supported yet
        max_value_per_row = torch.amax(abs_value, dim=1)
        max_value = torch.amax(max_value_per_row).to(dtype=torch.float32)
        hl.atomic_max(scale, [0], max_value / fp8_max)

    hl.barrier()

    for tile_m, tile_n in hl.tile([num_tokens, hidden_size]):
        inv_scale = (1.0 / scale[0]).to(dtype=torch.float32)
        x_blk = input[tile_m, tile_n].to(dtype=torch.float32)
        result[tile_m, tile_n] = (x_blk * inv_scale).to(result.dtype)


def baseline(
    result: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]
    scale: torch.Tensor,  # [1]
):
    torch.ops._C.dynamic_scaled_fp8_quant(result, input, scale)
