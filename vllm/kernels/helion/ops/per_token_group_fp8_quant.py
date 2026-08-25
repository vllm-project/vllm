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


def generate_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    # TODO(xiaohongchen1991): it is difficult for kernel author to cover all
    # input property combination. Currently, dtypes are fixed. We need
    # optimization to bucket/skip some combinations
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    hidden_size_list = [2048, 4096, 5120]
    group_size_list = [128]
    in_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    scale_dtype: torch.dtype = torch.float32

    use_ue8m0 = False
    column_major = False
    fp8_min, fp8_max = get_fp8_min_max()
    eps = 1e-10

    inputs = {}

    for hidden_size, group_size, num_tokens in product(
        hidden_size_list, group_size_list, num_tokens_list
    ):
        input = torch.randn(num_tokens, hidden_size, device="cuda", dtype=in_dtype)
        output_q = torch.empty(input.shape, device=input.device, dtype=out_dtype)
        output_s = torch.empty(
            (num_tokens, hidden_size // group_size),
            device=input.device,
            dtype=scale_dtype,
        )
        config_key = CaseKey(
            {
                "hidden_size": hidden_size,
                "group_size": group_size,
                "num_tokens": num_tokens,
            }
        )
        inputs[config_key] = (
            input,
            output_q,
            output_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            use_ue8m0,
            column_major,
            False,
        )

    return inputs


_pick_cache: dict[tuple[int, int, int], CaseKey | None] = {}


def pick_config(args: tuple[Any, ...], config_keys: list[CaseKey]) -> CaseKey | None:
    """Pick the best pre-tuned config for the given input shape.

    Selection strategy:
      1. Find the closest hidden_size among available configs
         (exact match preferred).
      2. Find the closest group_size among available configs
         (exact match preferred).
      3. Among the num_tokens values tuned for that hidden_size and group_size, pick
         the smallest num_tokens >= the input's num_tokens. If the input is
         larger than all available num_tokens, fall back to the largest.
    """

    if not config_keys:
        return None

    input, _, _, group_size, *_ = args
    num_tokens, hidden_size = input.shape

    cache_key = (num_tokens, group_size, hidden_size)
    cached = _pick_cache.get(cache_key)
    if cached is not None:
        return cached

    configs: dict[int, dict[int, list[int]]] = {}
    for key in config_keys:
        if key.is_default():
            continue
        configs.setdefault(key["hidden_size"], {}).setdefault(
            key["group_size"], []
        ).append(key["num_tokens"])

    if not configs:
        return None

    best_hidden_size = min(configs, key=lambda s: abs(s - hidden_size))
    best_group_size = min(configs[best_hidden_size], key=lambda s: abs(s - group_size))
    available_num_tokens = sorted(configs[best_hidden_size][best_group_size])
    best_num_tokens = next(
        (n for n in available_num_tokens if n >= num_tokens), available_num_tokens[-1]
    )

    result = CaseKey(
        {
            "hidden_size": best_hidden_size,
            "group_size": best_group_size,
            "num_tokens": best_num_tokens,
        }
    )
    _pick_cache[cache_key] = result
    return result


def fake_impl(
    input: torch.Tensor,  # [num_tokens, hidden_size]
    output_q: torch.Tensor,  # [num_tokens, hidden_size]
    output_s: torch.Tensor,  # [num_tokens, groups_per_row]
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    # Unused dummy args
    # Kept for consistency with existing kernel interface
    dummy_is_scale_transposed: bool = False,
    dummy_is_tma_aligned: bool = False,
) -> None:
    return


def baseline(
    input: torch.Tensor,  # [num_tokens, hidden_size]
    output_q: torch.Tensor,  # [num_tokens, hidden_size]
    output_s: torch.Tensor,  # [num_tokens, groups_per_row]
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    dummy_is_scale_transposed: bool = False,
    dummy_is_tma_aligned: bool = False,
) -> None:
    num_tokens, hidden_size = input.shape
    groups_per_row = hidden_size // group_size

    x = input.view(num_tokens, groups_per_row, group_size).to(torch.float32)
    s = torch.clamp(torch.amax(torch.abs(x), dim=-1), min=eps) / fp8_max
    if scale_ue8m0:
        s = torch.exp2(torch.ceil(torch.log2(s)))
    y = torch.clamp(x / s[:, :, None], fp8_min, fp8_max)

    output_s.copy_(s)
    output_q.copy_(y.view(num_tokens, hidden_size).to(output_q.dtype))


@register_kernel(
    mutates_args=["output_q", "output_s"],
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        autotune_baseline_fn=baseline,
    ),
)  # type: ignore[misc]
def per_token_group_fp8_quant(
    input: torch.Tensor,  # [num_tokens, hidden_size]
    output_q: torch.Tensor,  # [num_tokens, hidden_size]
    output_s: torch.Tensor,  # [num_tokens, groups_per_row]
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    # Unused dummy args
    # Kept for consistency with existing kernel interface
    dummy_is_scale_transposed: bool = False,
    dummy_is_tma_aligned: bool = False,
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)
    hl.specialize(group_size)

    groups_per_row = output_s.shape[1]
    hl.specialize(groups_per_row)
    assert hidden_size % group_size == 0 and hidden_size // group_size == groups_per_row
    assert output_s.ndim == 2 and output_s.dtype == torch.float32

    input = input.view(num_tokens, -1, group_size)
    output_q = output_q.view(num_tokens, -1, group_size)
    for tile_m, tile_gn, tile_n in hl.tile(
        [num_tokens, groups_per_row, group_size], block_size=[1, None, group_size]
    ):
        x_blk = input[tile_m, tile_gn, tile_n]
        y_s_blk = torch.clamp(torch.amax(torch.abs(x_blk), dim=-1), min=eps)
        y_s_blk = y_s_blk / fp8_max

        if scale_ue8m0:
            y_s_blk = torch.exp2(torch.ceil(torch.log2(y_s_blk)))

        y_q_blk = torch.clamp(x_blk / y_s_blk[:, :, None], fp8_min, fp8_max).to(
            output_q.dtype
        )

        output_s[tile_m, tile_gn] = y_s_blk
        output_q[tile_m, tile_gn, tile_n] = y_q_blk
