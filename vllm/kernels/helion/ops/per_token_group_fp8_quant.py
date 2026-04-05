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
    # TODO(xiaohongchen1991): it is difficult for kernel author to cover all
    # input property combination. Currently, dtypes are fixed. We need
    # optimization to bucket/skip some combinations
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    hidden_size_list = [2048, 4096, 6144, 8192, 12288]
    group_size_list = [64, 128]
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
        config_key = (
            f"hidden_size_{hidden_size}_group_size_{group_size}_num_tokens_{num_tokens}"
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
        )

    return inputs


def pick_config(args: tuple[Any, ...], config_keys: list[str]) -> str | None:
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

    input, _, _, group_size, *_ = args
    num_tokens, hidden_size = input.shape

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
    available_num_tokens = sorted(configs[best_hidden_size][best_group_size])
    best_num_tokens = next(
        (n for n in available_num_tokens if n >= num_tokens), available_num_tokens[-1]
    )

    return (
        f"hidden_size_{best_hidden_size}_group_size_"
        f"{best_group_size}_num_tokens_{best_num_tokens}"
    )


@register_kernel(
    mutates_args=["output_q", "output_s"],
    config_picker=pick_config,
    input_generator=generate_inputs,
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

    for tile_m, tile_gn in hl.tile([num_tokens, groups_per_row], block_size=[1, None]):
        m_idx = tile_m.begin + hl.arange(tile_m.block_size)
        gn_idx = tile_gn.index
        n_offset = hl.arange(group_size)
        n_idx = gn_idx[:, None] * group_size + n_offset[None, :]
        gn_mask = gn_idx < groups_per_row

        # shape: [tile_m, tile_gn, group_size]
        x_blk = hl.load(
            input,
            [m_idx[:, None, None], n_idx[None, :, :]],
            extra_mask=gn_mask[None, :, None],
        ).to(dtype=torch.float32)
        y_s_blk = torch.clamp(torch.amax(torch.abs(x_blk), dim=-1), min=eps)
        y_s_blk = y_s_blk / fp8_max

        if scale_ue8m0:
            y_s_blk = torch.exp2(torch.ceil(torch.log2(y_s_blk)))

        y_q_blk = torch.clamp(x_blk / y_s_blk[:, :, None], fp8_min, fp8_max).to(
            output_q.dtype
        )

        # output_s[tile_m, tile_gn] = y_s_blk
        hl.store(
            output_s,
            [m_idx[:, None], gn_idx[None, :]],
            y_s_blk,
            extra_mask=gn_mask[None, :],
        )
        hl.store(
            output_q,
            [m_idx[:, None, None], n_idx[None, :, :]],
            y_q_blk,
            extra_mask=gn_mask[None, :, None],
        )


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
    torch.ops._C.per_token_group_fp8_quant(
        input,
        output_q,
        output_s,
        group_size,
        eps,
        fp8_min,
        fp8_max,
        scale_ue8m0,
        dummy_is_scale_transposed,
        dummy_is_tma_aligned,
    )
