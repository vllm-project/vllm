# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import product
from typing import Any

import torch

from vllm.kernels.helion.case_key import CaseKey
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

import helion
import helion.language as hl

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)


def _get_fp8_dtype() -> torch.dtype:
    return current_platform.fp8_dtype()


def _get_int8_min_max() -> tuple[int, int]:
    qtype_traits = torch.iinfo(torch.int8)
    return qtype_traits.min, qtype_traits.max


def _get_int8_min_scaling_factor() -> float:
    return torch.finfo(torch.float32).eps


def generate_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    # TODO(xiaohongchen1991): it is difficult for kernel author to cover all input
    # property combination. Currently, dtypes are fixed. We need optimization to
    # bucket/skip some combinations
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    intermediate_size_list = [6144, 12288, 25600]
    in_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    scale_dtype: torch.dtype = torch.float32
    inputs = {}
    for num_tokens, intermediate_size in product(
        num_tokens_list, intermediate_size_list
    ):
        input = torch.randn(
            num_tokens, 2 * intermediate_size, device="cuda", dtype=in_dtype
        )
        result = torch.empty(
            num_tokens, intermediate_size, device=input.device, dtype=out_dtype
        )
        scale = torch.empty((num_tokens, 1), device=input.device, dtype=scale_dtype)
        scale_ub = torch.mean(input).to(scale_dtype)

        config_key = CaseKey(
            {"intermediate_size": intermediate_size, "num_tokens": num_tokens}
        )
        inputs[config_key] = (result, input, scale, scale_ub)

    return inputs


_pick_cache: dict[tuple[int, int], CaseKey | None] = {}


def pick_config(args: tuple[Any, ...], config_keys: list[CaseKey]) -> CaseKey | None:
    """Pick the best pre-tuned config for the given input shape.

    Selection strategy:
      1. Find the closest intermediate_size among available configs
         (exact match preferred).
      2. Among the num_tokens values tuned for that intermediate_size, pick
         the smallest num_tokens >= the input's num_tokens. If the input is
         larger than all available num_tokens, fall back to the largest.
    """

    if not config_keys:
        return None

    result, *_ = args
    num_tokens, intermediate_size = result.shape

    cache_key = (num_tokens, intermediate_size)
    cached = _pick_cache.get(cache_key)
    if cached is not None:
        return cached

    configs: dict[int, list[int]] = {}
    for key in config_keys:
        if key.is_default():
            continue
        configs.setdefault(key["intermediate_size"], []).append(key["num_tokens"])

    if not configs:
        return None

    best_intermediate_size = min(configs, key=lambda s: abs(s - intermediate_size))
    available_num_tokens = sorted(configs[best_intermediate_size])
    best_num_tokens = next(
        (n for n in available_num_tokens if n >= num_tokens), available_num_tokens[-1]
    )

    result = CaseKey(
        {"intermediate_size": best_intermediate_size, "num_tokens": best_num_tokens}
    )
    _pick_cache[cache_key] = result
    return result


def fake_impl(
    result: torch.Tensor,  # [num_tokens, intermediate_size]
    input: torch.Tensor,  # [num_tokens, 2 * intermediate_size]
    scale: torch.Tensor,  # [num_tokens, 1]
    scale_ub: torch.Tensor | None = None,  # scalar tensor
) -> None:
    return


def baseline(
    result: torch.Tensor,  # [num_tokens, intermediate_size]
    input: torch.Tensor,  # [num_tokens, 2 * intermediate_size]
    scale: torch.Tensor,  # [num_tokens, 1]
    scale_ub: torch.Tensor | None = None,  # scalar tensor
):
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.activation import SiluAndMul

    silu_and_mul_out = SiluAndMul.forward_native(input)

    intermediate_size = silu_and_mul_out.shape[1]
    # Overwrite first half of input in-place to match Helion kernel impl.
    # This is needed to pass Helion autotuning baseline check.
    input[:, :intermediate_size].copy_(silu_and_mul_out.to(input.dtype))

    out, scale_out = ops.scaled_fp8_quant(
        silu_and_mul_out, scale=None, scale_ub=scale_ub, use_per_token_if_dynamic=True
    )
    result.copy_(out)
    scale.copy_(scale_out)


@register_kernel(
    mutates_args=["result", "input", "scale"],
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        autotune_baseline_fn=baseline,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ),
)  # type: ignore[misc]
def silu_and_mul_dynamic_per_token_quant(
    result: torch.Tensor,  # [num_tokens, intermediate_size]
    input: torch.Tensor,  # [num_tokens, 2 * intermediate_size]
    scale: torch.Tensor,  # [num_tokens, 1]
    scale_ub: torch.Tensor | None = None,  # scalar tensor
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, two_intermediate_size = input.shape
    hl.specialize(two_intermediate_size)

    assert two_intermediate_size % 2 == 0
    intermediate_size = two_intermediate_size // 2

    assert result.shape[0] == num_tokens
    assert result.shape[1] == intermediate_size

    assert scale.shape[0] == num_tokens
    assert scale.dtype == torch.float32
    assert input.stride()[-1] == 1
    assert result.stride()[-1] == 1

    fp8_min, fp8_max = get_fp8_min_max()
    inv_fp8_max = 1.0 / fp8_max
    min_scaling_factor = inv_fp8_max / 512.0

    for tile_m in hl.tile(num_tokens, block_size=1):
        s_blk = hl.zeros([tile_m], dtype=torch.float32)

        for tile_n in hl.tile(intermediate_size):
            x_a_blk = input[tile_m, tile_n].to(torch.float32)
            x_b_blk = hl.load(
                input,
                [tile_m, tile_n.index + intermediate_size],
                extra_mask=((tile_n.index + intermediate_size) < two_intermediate_size)[
                    None, :
                ],
            ).to(torch.float32)
            x_blk = x_a_blk * torch.sigmoid(x_a_blk) * x_b_blk
            input[tile_m, tile_n] = x_blk.to(input.dtype)
            tmp_blk = torch.amax(torch.abs(x_blk), dim=-1)
            s_blk = torch.maximum(s_blk, tmp_blk)

        if scale_ub is not None:
            scale_ub_s = hl.load(scale_ub, [])
            s_blk = s_blk.clamp(max=scale_ub_s)
        s_blk = s_blk * inv_fp8_max
        s_blk = s_blk.clamp(min=min_scaling_factor)
        scale[tile_m, 0] = s_blk

        for tile_n in hl.tile(intermediate_size):
            x_blk = input[tile_m, tile_n].to(torch.float32)
            y_blk = x_blk / s_blk[:, None]

            result[tile_m, tile_n] = y_blk.clamp(fp8_min, fp8_max).to(result.dtype)
