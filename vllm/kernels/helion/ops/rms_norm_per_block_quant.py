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
    # TODO(xiaohongchen1991): it is difficult for kernel author to cover all
    # input property combination. Currently, dtypes are fixed. We need
    # optimization to bucket/skip some combinations
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    hidden_size_list = [2048, 4096, 5120]
    group_size_list = [128]
    in_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    scale_dtype: torch.dtype = torch.float32
    inputs = {}

    for hidden_size, group_size, num_tokens in product(
        hidden_size_list, group_size_list, num_tokens_list
    ):
        input = torch.randn(num_tokens, hidden_size, device="cuda", dtype=in_dtype)
        result = torch.empty(input.shape, device=input.device, dtype=out_dtype)
        scale = torch.empty(
            (num_tokens, hidden_size // group_size),
            device=input.device,
            dtype=scale_dtype,
        )
        residual = torch.randn_like(input)
        weight = torch.normal(
            mean=1.0,
            std=1.0,
            size=(hidden_size,),
            dtype=input.dtype,
            device=input.device,
        )
        epsilon = 1e-6
        # scale_ub clamps the per-group amax of the RMS-normed, weighted output.
        # Use a non-degenerate upper bound (midway between the mean and max of
        # that magnitude) so clamping is partially active and the baseline
        # comparison is meaningful. torch.mean(input) ~= 0 for the zero-mean
        # input would collapse every scale to the floor and saturate the output.
        # Mirrors the reference normalization in baseline() below.
        x = input.to(torch.float32) + residual.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)
        x_norm_abs = ((x * rms).to(input.dtype) * weight).abs().to(torch.float32)
        scale_ub = (0.5 * (x_norm_abs.mean() + x_norm_abs.amax())).to(scale_dtype)

        config_key = CaseKey(
            {
                "hidden_size": hidden_size,
                "group_size": group_size,
                "num_tokens": num_tokens,
            }
        )
        inputs[config_key] = (
            result,
            input,
            weight,
            scale,
            epsilon,
            scale_ub,
            residual,
            group_size,
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
      2. Among the num_tokens values tuned for that hidden_size and group_size, pick
         the smallest num_tokens >= the input's num_tokens. If the input is
         larger than all available num_tokens, fall back to the largest.
    """

    if not config_keys:
        return None

    _, input, _, _, _, _, _, group_size, *_ = args
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
    result: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]
    weight: torch.Tensor,  # [hidden_size]
    scale: torch.Tensor,  # [num_tokens, groups_per_row]
    epsilon: float,
    scale_ub: torch.Tensor | None,  # []
    residual: torch.Tensor | None,  # [num_tokens, hidden_size]
    group_size: int,
    is_scale_transposed: bool,  # dummy
) -> None:
    return


def baseline(
    result: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]
    weight: torch.Tensor,  # [hidden_size]
    scale: torch.Tensor,  # [num_tokens, groups_per_row]
    epsilon: float,
    scale_ub: torch.Tensor | None,  # []
    residual: torch.Tensor | None,  # [num_tokens, hidden_size]
    group_size: int,
    is_scale_transposed: bool,
) -> None:
    num_tokens, hidden_size = input.shape
    groups_per_row = hidden_size // group_size
    quant_dtype = result.dtype
    qtype_min: int | float
    qtype_max: int | float

    if quant_dtype == torch.int8:
        qtype_min, qtype_max = get_int8_min_max()
        min_scaling_factor = get_int8_min_scaling_factor()
    else:
        qtype_min, qtype_max = get_fp8_min_max()
        min_scaling_factor = 1.0 / (qtype_max * 512.0)

    x = input.to(torch.float32)
    if residual is not None:
        x = x + residual
        residual.copy_(x.to(residual.dtype))

    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    x_norm = (x * rms).to(input.dtype) * weight
    x_grouped = x_norm.view(num_tokens, groups_per_row, group_size).to(torch.float32)

    s = torch.amax(torch.abs(x_grouped), dim=-1).to(torch.float32)
    if scale_ub is not None:
        s = s.clamp(max=scale_ub)
    s = (s * (1.0 / qtype_max)).clamp(min=min_scaling_factor)

    y = x_grouped / s[:, :, None]
    if quant_dtype == torch.int8:
        y = y.round()

    scale.copy_(s)
    result.copy_(
        y.clamp(qtype_min, qtype_max).view(num_tokens, hidden_size).to(result.dtype)
    )


@register_kernel(
    mutates_args=["result", "scale", "residual"],
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        autotune_baseline_fn=baseline,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ),
)  # type: ignore[misc]
def rms_norm_per_block_quant(
    result: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]
    weight: torch.Tensor,  # [hidden_size]
    scale: torch.Tensor,  # [num_tokens, groups_per_row]
    epsilon: float,
    scale_ub: torch.Tensor | None,  # []
    residual: torch.Tensor | None,  # [num_tokens, hidden_size]
    group_size: int,
    is_scale_transposed: bool,  # dummy
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)
    hl.specialize(group_size)

    groups_per_row = scale.shape[1]
    hl.specialize(groups_per_row)
    assert hidden_size % group_size == 0 and hidden_size // group_size == groups_per_row
    assert scale.shape[0] == num_tokens
    assert scale.dtype == torch.float32
    if scale.stride(1) > 1:
        assert is_scale_transposed

    fp8_dtype = get_fp8_dtype()
    assert result.dtype in [fp8_dtype, torch.int8]
    assert result.is_contiguous() and input.is_contiguous()

    if scale_ub is not None:
        assert result.dtype == fp8_dtype
        assert scale_ub.dtype == torch.float32

    assert input.dtype == weight.dtype

    if residual is not None:
        assert residual.dtype == input.dtype

    assert group_size in [64, 128]

    quant_dtype = result.dtype
    qtype_traits_min: int | float
    qtype_traits_max: int | float
    if quant_dtype == torch.int8:
        qtype_traits_min, qtype_traits_max = get_int8_min_max()
        min_scaling_factor = get_int8_min_scaling_factor()
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

        m_idx = tile_m.begin + hl.arange(tile_m.block_size)
        m_blk = m_idx[:, None, None]
        for tile_gn, tile_n in hl.tile(
            [groups_per_row, group_size], block_size=[None, group_size]
        ):
            gn_idx = tile_gn.index
            n_offset = tile_n.index
            n_idx = gn_idx[:, None] * group_size + n_offset[None, :]
            n_blk = n_idx[None, :, :]
            mask = (gn_idx < groups_per_row)[None, :, None]

            x_blk = hl.load(input, [m_blk, n_blk], extra_mask=mask).to(
                dtype=torch.float32
            )
            if residual is not None:
                r_blk = hl.load(residual, [m_blk, n_blk], extra_mask=mask)
                x_blk = x_blk + r_blk

            w_blk = hl.load(weight, [n_blk], extra_mask=mask)
            x_norm_blk = (x_blk * rms[:, None, None]).to(input.dtype) * w_blk
            s_blk = torch.amax(torch.abs(x_norm_blk), dim=-1).to(torch.float32)

            if scale_ub is not None:
                scale_ub_s = hl.load(scale_ub, [])
                s_blk = s_blk.clamp(max=scale_ub_s)

            s_blk = s_blk * (1.0 / qtype_max)
            s_blk = s_blk.clamp(min=min_scaling_factor)

            scale[tile_m, tile_gn] = s_blk

            if quant_dtype == torch.int8:
                y_blk = (x_norm_blk * (1.0 / s_blk[:, :, None])).round()
            else:
                y_blk = x_norm_blk / s_blk[:, :, None]

            y_blk = y_blk.clamp(qtype_traits_min, qtype_traits_max).to(result.dtype)
            hl.store(result, [m_blk, n_blk], y_blk, extra_mask=mask)

            if residual is not None:
                hl.store(
                    residual, [m_blk, n_blk], x_blk.to(residual.dtype), extra_mask=mask
                )
