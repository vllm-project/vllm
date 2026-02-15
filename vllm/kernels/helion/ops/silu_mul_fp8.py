# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "silu_mul_fp8 Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion.language as hl

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)


@register_kernel  # type: ignore[misc]
def silu_mul_fp8(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    original_shape = input.shape
    two_d = hl.specialize(original_shape[-1])
    d = two_d // 2
    output_shape = original_shape[:-1] + (d,)

    input_2d = input.view(-1, original_shape[-1])
    m = input_2d.shape[0]

    # TODO(gmagogsfm): Support for more float8 subtypes (e4m3fnuz, e5m2) coming
    out = torch.empty((m, d), device=input.device, dtype=torch.float8_e4m3fn)

    input_part_a = input_2d[:, :d]
    input_part_b = input_2d[:, d:]

    assert scale.numel() == 1, "Scale must be a scalar Tensor"

    for tile_m, tile_n in hl.tile([m, d]):
        a_vals = input_part_a[tile_m, tile_n]
        silu_result = torch.nn.functional.silu(a_vals)
        b_vals = input_part_b[tile_m, tile_n]
        result = silu_result * b_vals
        result_f32 = result.to(torch.float32)
        scale_val = hl.load(scale, [0])
        inv_scale = 1.0 / scale_val
        result_scaled = result_f32 * inv_scale
        out[tile_m, tile_n] = result_scaled.to(out.dtype)

    return out.view(output_shape)


@silu_mul_fp8.register_config_picker  # type: ignore[misc]
def pick_silu_mul_fp8_config(
    args: tuple[Any, ...], config_keys: list[str]
) -> str | None:
    if not config_keys:
        return None

    input_tensor, scale = args
    intermediate_size = input_tensor.shape[-1] // 2

    # TODO(gmagosfm): Rerun autotuning to capture config for
    # other batch sizes.
    target_key = f"intermediate_{intermediate_size}_batchsize_256"
    if target_key in config_keys:
        return target_key

    intermediate_sizes = []
    for key in config_keys:
        if key.startswith("intermediate_") and "_batchsize_256" in key:
            try:
                size_str = key.split("_")[1]
                size = int(size_str)
                intermediate_sizes.append((abs(size - intermediate_size), key))
            except (ValueError, IndexError):
                continue

    if intermediate_sizes:
        _, best_key = min(intermediate_sizes)
        logger.debug(
            "No exact config for intermediate_size=%d, using closest match: %s",
            intermediate_size,
            best_key,
        )
        return best_key
    if "default" in config_keys:
        return "default"

    return None


def silu_mul_fp8_baseline(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    output_shape = input.shape[:-1] + (input.shape[-1] // 2,)
    out = torch.empty(output_shape, dtype=torch.float8_e4m3fn, device=input.device)
    torch.ops._C.silu_and_mul_quant(out, input, scale)
    return out
