# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import product
from typing import Any

import helion.language as hl
import regex as re
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    scaled_quantize,
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
    group_shape_list = [(-1, -1), (-1, 1), (1, -1)]
    in_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    scale_dtype: torch.dtype = torch.float32
    inputs = {}

    for hidden_size, group_shape, num_tokens in product(
        hidden_size_list, group_shape_list, num_tokens_list
    ):
        input = torch.randn(num_tokens, hidden_size, device="cuda", dtype=in_dtype)
        result = torch.empty(input.shape, device=input.device, dtype=out_dtype)
        _, scale = scaled_quantize(
            input,
            group_shape,
            out_dtype,
            compute_dtype=scale_dtype,
        )
        group_shape_m, group_shape_n = group_shape
        if scale.dim() == 0 or scale.numel() == 1:
            group_m = num_tokens
            group_n = hidden_size
            scale_2d = scale.reshape(1, 1)
        elif scale.dim() == 1:
            group_m = num_tokens if group_shape_m == -1 else int(group_shape_m)
            group_n = hidden_size if group_shape_n == -1 else int(group_shape_n)
            scale_2d = scale[None, :] if group_shape_m == -1 else scale[:, None]
        elif scale.dim() == 2:
            scale_2d = scale
            group_m = num_tokens if group_shape_m == -1 else int(group_shape_m)
            group_n = hidden_size if group_shape_n == -1 else int(group_shape_n)

        config_key = (
            f"hidden_size_{hidden_size}_"
            f"group_shape_m_{group_shape_m}_group_shape_n_{group_shape_n}_"
            f"num_tokens_{num_tokens}"
        )
        inputs[config_key] = (result, input, scale_2d, group_m, group_n)

    return inputs


def pick_config(args: tuple[Any, ...], config_keys: list[str]) -> str | None:
    """Pick the best pre-tuned config for the given input shape.

    Selection strategy:
      1. Find the closest hidden_size among available configs
         (exact match preferred).
      1. Find the closest (group_shape_m, group_shape_n) among available configs
         (exact match preferred).
      2. Among the num_tokens values tuned for that group shape, pick
         the smallest num_tokens >= the input's num_tokens. If the input is
         larger than all available num_tokens, fall back to the largest.

    Config keys must be "default" or follow the format
    "hidden_size_{int}_group_shape_m_{int}_group_shape_n_{int}_num_tokens_{int}".
    """

    if not config_keys:
        return None

    _, input, _, group_m, group_n = args
    num_tokens, hidden_size = input.shape
    group_shape_m = -1 if int(group_m) == num_tokens else int(group_m)
    group_shape_n = -1 if int(group_n) == hidden_size else int(group_n)

    configs: dict[int, dict[tuple[int, int], list[int]]] = {}
    for key in config_keys:
        if key == "default":
            continue
        match = re.fullmatch(
            r"hidden_size_(\d+)_group_shape_m_(-?\d+)_group_shape_n_(-?\d+)_num_tokens_(\d+)",
            key,
        )
        if not match:
            raise ValueError(
                f"Malformed config key '{key}', "
                "expected format "
                "'hidden_size_{int}_group_shape_m_{int}_group_shape_n_{int}_num_tokens_{int}'"
            )
        hidden_size_str, group_shape_m_str, group_shape_n_str, num_tokens_str = (
            match.groups()
        )
        configs.setdefault(int(hidden_size_str), {}).setdefault(
            (int(group_shape_m_str), int(group_shape_n_str)), []
        ).append(int(num_tokens_str))

    if not configs:
        return "default" if "default" in config_keys else None

    best_hidden_size = min(configs, key=lambda s: abs(s - hidden_size))
    group_configs = configs[best_hidden_size]

    if (group_shape_m, group_shape_n) in group_configs:
        best_group_shape = (group_shape_m, group_shape_n)
    else:
        best_group_shape = min(
            group_configs,
            key=lambda g: (abs(g[0] - group_shape_m), abs(g[1] - group_shape_n)),
        )
    available_num_tokens = sorted(group_configs[best_group_shape])
    best_num_tokens = next(
        (n for n in available_num_tokens if n >= num_tokens),
        available_num_tokens[-1],
    )

    return (
        f"hidden_size_{best_hidden_size}_"
        f"group_shape_m_{best_group_shape[0]}_group_shape_n_{best_group_shape[1]}_"
        f"num_tokens_{best_num_tokens}"
    )


@register_kernel(
    mutates_args=["result"],
    config_picker=pick_config,
    input_generator=generate_inputs,
)  # type: ignore[misc]
def static_scaled_fp8_quant(
    result: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    group_m: int,
    group_n: int,
) -> None:
    assert scale.dim() == 2
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2

    num_tokens, hidden_size = input.shape
    num_groups_m, num_groups_n = scale.shape
    hl.specialize(hidden_size)
    hl.specialize(group_n)
    hl.specialize(num_groups_n)

    for tile_gm, tile_gn, tile_m, tile_n in hl.tile(
        [num_groups_m, num_groups_n, group_m, group_n]
    ):
        gm_idx = tile_gm.index
        gn_idx = tile_gn.index

        # offset inside group
        m_offset = tile_m.index
        n_offset = tile_n.index

        # Global indices
        # m_idx: [tile_gm, tile_m]
        # n_idx: [tile_gn, tile_n]
        m_idx = gm_idx[:, None] * group_m + m_offset[None, :]
        n_idx = gn_idx[:, None] * group_n + n_offset[None, :]

        m_blk = m_idx[:, None, :, None]
        n_blk = n_idx[None, :, None, :]

        group_mask = (gm_idx < num_groups_m)[:, None] & (gn_idx < num_groups_n)[None, :]
        elem_mask = (
            (gm_idx < num_groups_m)[:, None, None, None]
            & (gn_idx < num_groups_n)[None, :, None, None]
            & (m_offset < group_m)[None, None, :, None]
            & (n_offset < group_n)[None, None, None, :]
        )

        # shape: [tile_gm, tile_gn]
        scale_blk = hl.load(
            scale,
            [gm_idx[:, None], gn_idx[None, :]],
            extra_mask=group_mask,
        )
        inv_scale_blk = (1.0 / scale_blk).to(dtype=torch.float32)

        # input tile shape:  [tile_gm, tile_gn, tile_m, tile_n]
        x_blk = hl.load(
            input,
            [m_blk, n_blk],
            extra_mask=elem_mask,
        ).to(torch.float32)

        # scale tile shape:  [tile_gm, tile_gn, 1, 1]
        y_blk = x_blk * inv_scale_blk[:, :, None, None]

        hl.store(
            result,
            [m_blk, n_blk],
            y_blk.to(result.dtype),
            extra_mask=elem_mask,
        )


# This is the real entrypoint for this kernel. Note that
# 1. Helion does not support this condition dispatch logic inside @helion.kernel
# decorator. So, the dispatch logic is implemented as a separate wrapper.
# 2. @register_kernel decorator will apply @helion.kernel to the fn eventually,
# so @register_kernel is applied for static_scaled_fp8_quant not this wrapper.
def static_scaled_fp8_quant_dispatch(
    result: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    group_shape: tuple[int, int] | None = None,
):
    assert input.stride(-1) == 1
    assert result.stride(-1) == 1

    num_tokens, hidden_size = input.shape

    if scale.dim() == 0 or scale.numel() == 1:
        # per tensor
        group_m = num_tokens
        group_n = hidden_size
        scale_2d = scale.reshape(1, 1)
    elif scale.dim() == 1:
        assert group_shape is not None
        group_shape_m, group_shape_n = group_shape
        assert group_shape_m == -1 or group_shape_n == -1
        group_m = num_tokens if group_shape_m == -1 else int(group_shape_m)
        group_n = hidden_size if group_shape_n == -1 else int(group_shape_n)
        scale_2d = scale[None, :] if group_shape_m == -1 else scale[:, None]
        inferred_group_m = num_tokens // scale_2d.size(0)
        inferred_group_n = hidden_size // scale_2d.size(1)
        assert group_m == inferred_group_m and group_n == inferred_group_n
    elif scale.dim() == 2:
        scale_2d = scale
        scale_size_0, scale_size_1 = scale.shape
        assert num_tokens % scale_size_0 == 0
        assert hidden_size % scale_size_1 == 0
        inferred_group_m = num_tokens // scale_size_0
        inferred_group_n = hidden_size // scale_size_1

        if group_shape is not None:
            group_shape_m, group_shape_n = group_shape
            group_m = num_tokens if group_shape_m == -1 else int(group_shape_m)
            group_n = hidden_size if group_shape_n == -1 else int(group_shape_n)
            assert group_m == inferred_group_m and group_n == inferred_group_n
        else:
            group_m, group_n = inferred_group_m, inferred_group_n

    static_scaled_fp8_quant(result, input, scale_2d, group_m, group_n)


def baseline(
    result: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    group_m: int,
    group_n: int,
) -> None:
    num_tokens, hidden_size = input.shape
    group_shape_m = -1 if int(group_m) == num_tokens else int(group_m)
    group_shape_n = -1 if int(group_n) == hidden_size else int(group_n)
    torch.ops._C.static_scaled_fp8_quant(
        result, input, scale, (group_shape_m, group_shape_n)
    )
