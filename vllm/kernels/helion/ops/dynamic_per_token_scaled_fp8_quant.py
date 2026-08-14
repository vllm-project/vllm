# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import product
from typing import Any

import torch

from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.register import register_kernel
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

logger = init_logger(__name__)


def generate_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    # TODO(xiaohongchen1991): it is difficult for kernel author to cover all input
    # property combination. Currently, dtypes are fixed. We need optimization to
    # bucket/skip some combinations
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    hidden_size_list = [2048, 4096, 5120]
    in_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    scale_dtype: torch.dtype = torch.float32
    inputs = {}
    for num_tokens, hidden_size in product(num_tokens_list, hidden_size_list):
        input = torch.randn(num_tokens, hidden_size, device="cuda", dtype=in_dtype)
        result = torch.empty(input.shape, device=input.device, dtype=out_dtype)
        scale = torch.empty((num_tokens, 1), device=input.device, dtype=scale_dtype)
        # scale_ub clamps the per-token amax of |input|. Use a non-degenerate
        # upper bound (midway between the mean and max of |input|) so clamping is
        # partially active and the baseline comparison is meaningful.
        # torch.mean(input) ~= 0 for the zero-mean input would collapse every
        # scale to the floor and saturate the output.
        input_abs = input.to(torch.float32).abs()
        scale_ub = (0.5 * (input_abs.mean() + input_abs.amax())).to(scale_dtype)

        config_key = CaseKey({"hidden_size": hidden_size, "num_tokens": num_tokens})
        inputs[config_key] = (result, input, scale, scale_ub)

    return inputs


_pick_cache: dict[tuple[int, int], CaseKey | None] = {}


def pick_config(args: tuple[Any, ...], config_keys: list[CaseKey]) -> CaseKey | None:
    """Pick the best pre-tuned config for the given input shape.

    Selection strategy:
      1. Find the closest hidden_size among available configs
         (exact match preferred).
      2. Among the num_tokens values tuned for that hidden_size, pick
         the smallest num_tokens >= the input's num_tokens. If the input is
         larger than all available num_tokens, fall back to the largest.
    """

    if not config_keys:
        return None

    _, input, *_ = args
    num_tokens, hidden_size = input.shape

    cache_key = (num_tokens, hidden_size)
    cached = _pick_cache.get(cache_key)
    if cached is not None:
        return cached

    configs: dict[int, list[int]] = {}
    for key in config_keys:
        if key.is_default():
            continue
        configs.setdefault(key["hidden_size"], []).append(key["num_tokens"])

    if not configs:
        return None

    best_hidden_size = min(configs, key=lambda s: abs(s - hidden_size))
    available_num_tokens = sorted(configs[best_hidden_size])
    best_num_tokens = next(
        (n for n in available_num_tokens if n >= num_tokens), available_num_tokens[-1]
    )

    result = CaseKey({"hidden_size": best_hidden_size, "num_tokens": best_num_tokens})
    _pick_cache[cache_key] = result
    return result


def fake_impl(
    result: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]
    scale: torch.Tensor,  # [num_tokens, 1]
    scale_ub: torch.Tensor | None = None,  # scalar tensor
) -> None:
    return


def baseline(
    result: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]
    scale: torch.Tensor,  # [num_tokens, 1]
    scale_ub: torch.Tensor | None = None,  # scalar tensor
) -> None:
    fp8_min, fp8_max = get_fp8_min_max()
    min_scaling_factor = 1.0 / (fp8_max * 512.0)

    x = input.to(torch.float32)
    s = torch.amax(torch.abs(x), dim=-1, keepdim=True)
    if scale_ub is not None:
        s = s.clamp(max=scale_ub)
    s = (s * (1.0 / fp8_max)).clamp(min=min_scaling_factor)
    y = (x / s).clamp(fp8_min, fp8_max)

    scale.copy_(s)
    result.copy_(y.to(result.dtype))


# Overwrite autotune_baseline_atol and autotune_baseline_rtol
# if too many configs failed due to baseline check during autotuning
@register_kernel(
    mutates_args=["result", "scale"],
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        autotune_baseline_fn=baseline,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ),
)
def dynamic_per_token_scaled_fp8_quant(
    result: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]
    scale: torch.Tensor,  # [num_tokens, 1]
    scale_ub: torch.Tensor | None = None,  # scalar tensor
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)

    assert result.shape == input.shape
    assert scale.shape[0] == num_tokens
    assert scale.dtype == torch.float32
    assert input.stride()[-1] == 1
    assert result.stride()[-1] == 1

    fp8_min, fp8_max = get_fp8_min_max()
    min_scaling_factor = 1.0 / (fp8_max * 512.0)

    for tile_m in hl.tile(num_tokens, block_size=1):
        s_blk = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(dtype=torch.float32)
            tmp_blk = torch.amax(torch.abs(x_blk), dim=-1)
            s_blk = torch.maximum(s_blk, tmp_blk)

        if scale_ub is not None:
            scale_ub_s = hl.load(scale_ub, [])
            s_blk = s_blk.clamp(max=scale_ub_s)
        s_blk = s_blk * (1.0 / fp8_max)
        s_blk = s_blk.clamp(min=min_scaling_factor)
        scale[tile_m, 0] = s_blk

        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(torch.float32)
            y_blk = x_blk * (1.0 / s_blk[:, None])

            result[tile_m, tile_n] = y_blk.clamp(fp8_min, fp8_max).to(result.dtype)
