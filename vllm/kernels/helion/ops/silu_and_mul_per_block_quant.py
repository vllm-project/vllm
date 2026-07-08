# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import product
from typing import Any

import torch

from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.utils import (
    get_fp8_dtype,
    get_int8_min_max,
    get_int8_min_scaling_factor,
)
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


def generate_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    # TODO(xiaohongchen1991): it is difficult for kernel author to cover all input
    # property combination. Currently, dtypes are fixed. We need optimization to
    # bucket/skip some combinations
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    intermediate_size_list = [6144, 12288, 25600]

    in_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    scale_dtype: torch.dtype = torch.float32
    group_size_list = [128]
    inputs = {}
    for intermediate_size, group_size, num_tokens in product(
        intermediate_size_list, group_size_list, num_tokens_list
    ):
        input = torch.randn(
            num_tokens, 2 * intermediate_size, device="cuda", dtype=in_dtype
        )
        result = torch.empty(
            num_tokens, intermediate_size, device=input.device, dtype=out_dtype
        )
        scale = torch.empty(
            (num_tokens, intermediate_size // group_size),
            device=input.device,
            dtype=scale_dtype,
        )
        scale_ub = torch.mean(input).to(scale_dtype)

        config_key = CaseKey(
            {
                "intermediate_size": intermediate_size,
                "group_size": group_size,
                "num_tokens": num_tokens,
            }
        )
        inputs[config_key] = (result, input, scale, group_size, scale_ub, False)

    return inputs


_pick_cache: dict[tuple[int, int, int], CaseKey | None] = {}


def pick_config(args: tuple[Any, ...], config_keys: list[CaseKey]) -> CaseKey | None:
    """Pick the best pre-tuned config for the given input shape.

    Selection strategy:
      1. Find the closest intermediate_size among available configs
         (exact match preferred).
      2. Find the closest group_size among available configs
         (exact match preferred).
      3. Among the num_tokens values tuned for that intermediate_size and group_size,
         pick the smallest num_tokens >= the input's num_tokens. If the input is
         larger than all available num_tokens, fall back to the largest.
    """

    if not config_keys:
        return None

    result, _, _, group_size, *_ = args
    num_tokens, intermediate_size = result.shape

    cache_key = (num_tokens, group_size, intermediate_size)
    cached = _pick_cache.get(cache_key)
    if cached is not None:
        return cached

    configs: dict[int, dict[int, list[int]]] = {}
    for key in config_keys:
        if key.is_default():
            continue
        configs.setdefault(key["intermediate_size"], {}).setdefault(
            key["group_size"], []
        ).append(key["num_tokens"])

    if not configs:
        return None

    best_intermediate_size = min(configs, key=lambda s: abs(s - intermediate_size))
    best_group_size = min(
        configs[best_intermediate_size], key=lambda s: abs(s - group_size)
    )
    available_num_tokens = sorted(configs[best_intermediate_size][best_group_size])
    best_num_tokens = next(
        (n for n in available_num_tokens if n >= num_tokens), available_num_tokens[-1]
    )

    result = CaseKey(
        {
            "intermediate_size": best_intermediate_size,
            "group_size": best_group_size,
            "num_tokens": best_num_tokens,
        }
    )
    _pick_cache[cache_key] = result
    return result


def fake_impl(
    out: torch.Tensor,  # [num_tokens, intermediate_size]
    input: torch.Tensor,  # [num_tokens, 2 * intermediate_size]
    scales: torch.Tensor,  # [num_tokens, groups_per_row]
    group_size: int,
    scale_ub: torch.Tensor | None = None,  # scalar tensor
    is_scale_transposed: bool = False,
) -> None:
    return


def baseline(
    out: torch.Tensor,  # [num_tokens, intermediate_size]
    input: torch.Tensor,  # [num_tokens, 2 * intermediate_size]
    scales: torch.Tensor,  # [num_tokens, groups_per_row]
    group_size: int,
    scale_ub: torch.Tensor | None = None,  # scalar tensor
    is_scale_transposed: bool = False,
) -> None:
    torch.ops._C.silu_and_mul_per_block_quant(
        out, input, scales, group_size, scale_ub, is_scale_transposed
    )


@register_kernel(
    mutates_args=["out", "scales"],
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        autotune_baseline_fn=baseline,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ),
)  # type: ignore[misc]
def silu_and_mul_per_block_quant(
    out: torch.Tensor,  # [num_tokens, intermediate_size]
    input: torch.Tensor,  # [num_tokens, 2 * intermediate_size]
    scales: torch.Tensor,  # [num_tokens, groups_per_row]
    group_size: int,
    scale_ub: torch.Tensor | None = None,  # scalar tensor
    is_scale_transposed: bool = False,  # dummy
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, two_intermediate_size = input.shape
    hl.specialize(two_intermediate_size)

    assert two_intermediate_size % 2 == 0
    intermediate_size = two_intermediate_size // 2

    assert out.shape[0] == num_tokens
    assert out.shape[1] == intermediate_size
    fp8_dtype = get_fp8_dtype()
    assert out.dtype in [fp8_dtype, torch.int8]

    if scale_ub is not None:
        assert out.dtype == fp8_dtype
        assert scale_ub.dtype == torch.float32

    assert scales.ndim == 2 and scales.dtype == torch.float32

    assert scales.shape[0] == num_tokens
    groups_per_row = scales.shape[1]
    hl.specialize(groups_per_row)
    assert (
        intermediate_size % group_size == 0
        and intermediate_size // group_size == groups_per_row
    )

    assert group_size in [64, 128]
    hl.specialize(group_size)

    assert input.stride()[-1] == 1
    assert out.stride()[-1] == 1

    quant_dtype = out.dtype
    qtype_traits_min: int | float
    qtype_traits_max: int | float
    if quant_dtype == torch.int8:
        qtype_traits_min, qtype_traits_max = get_int8_min_max()
        min_scaling_factor = get_int8_min_scaling_factor()
    else:
        qtype_traits_min, qtype_traits_max = get_fp8_min_max()
        min_scaling_factor = 1.0 / (qtype_traits_max * 512.0)

    qtype_max = float(qtype_traits_max)

    input = input.view(num_tokens, -1, group_size)
    out = out.view(num_tokens, -1, group_size)

    for tile_m, tile_gn, tile_n in hl.tile(
        [num_tokens, groups_per_row, group_size], block_size=[1, None, group_size]
    ):
        x_a_blk = input[tile_m, tile_gn, tile_n].to(torch.float32)
        x_b_blk = hl.load(
            input,
            [tile_m, tile_gn.index + groups_per_row, tile_n],
            extra_mask=(tile_gn.index + groups_per_row < 2 * groups_per_row)[
                None, :, None
            ],
        ).to(torch.float32)
        x_blk = x_a_blk * torch.sigmoid(x_a_blk) * x_b_blk
        s_blk = torch.amax(torch.abs(x_blk), dim=-1).to(torch.float32)

        if scale_ub is not None:
            scale_ub_s = hl.load(scale_ub, [])
            s_blk = s_blk.clamp(max=scale_ub_s)
        s_blk = s_blk * (1.0 / qtype_max)
        s_blk = s_blk.clamp(min=min_scaling_factor)

        scales[tile_m, tile_gn] = s_blk
        if quant_dtype == torch.int8:
            y_blk = (x_blk * (1.0 / s_blk[:, :, None])).round()
        else:
            y_blk = x_blk / s_blk[:, :, None]

        out[tile_m, tile_gn, tile_n] = y_blk.clamp(
            qtype_traits_min, qtype_traits_max
        ).to(out.dtype)
