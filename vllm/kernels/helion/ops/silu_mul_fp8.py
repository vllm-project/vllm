# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

import torch

from vllm.kernels.helion.case_key import CaseKey
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


def generate_silu_mul_fp8_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    intermediate_sizes = [2048, 2880, 4096, 8192, 11008, 14336]

    # Use the same num_tokens values as vLLM's default cudagraph capture sizes.
    # See vllm/config/vllm.py _set_cudagraph_sizes() for the canonical formula.
    num_tokens_list = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))

    inputs: dict[CaseKey, tuple[Any, ...]] = {}
    for num_tokens in num_tokens_list:
        for intermediate_size in intermediate_sizes:
            input_tensor = torch.randn(
                num_tokens,
                2 * intermediate_size,
                device="cuda",
                dtype=torch.bfloat16,
            )
            scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)

            key = CaseKey({"intermediate": intermediate_size, "numtokens": num_tokens})
            inputs[key] = (input_tensor, scale)

    return inputs


_pick_cache: dict[tuple[int, int], CaseKey | None] = {}


def pick_silu_mul_fp8_config(
    args: tuple[Any, ...], config_keys: list[CaseKey]
) -> CaseKey | None:
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

    input_tensor, _scale = args
    intermediate_size = int(input_tensor.shape[-1]) // 2
    num_tokens = int(input_tensor.view(-1, input_tensor.shape[-1]).shape[0])

    cache_key = (num_tokens, intermediate_size)
    cached = _pick_cache.get(cache_key)
    if cached is not None:
        return cached

    by_isize: dict[int, list[int]] = {}
    for k in config_keys:
        if k.is_default():
            continue
        by_isize.setdefault(k["intermediate"], []).append(k["numtokens"])

    if not by_isize:
        return None

    best_isize = min(by_isize, key=lambda s: abs(s - intermediate_size))
    available = sorted(by_isize[best_isize])
    best_ntokens = next((n for n in available if n >= num_tokens), available[-1])

    result = CaseKey({"intermediate": best_isize, "numtokens": best_ntokens})
    _pick_cache[cache_key] = result
    return result


@register_kernel(
    config_picker=pick_silu_mul_fp8_config,
    input_generator=generate_silu_mul_fp8_inputs,
)
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


def silu_mul_fp8_baseline(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    output_shape = input.shape[:-1] + (input.shape[-1] // 2,)
    out = torch.empty(output_shape, dtype=torch.float8_e4m3fn, device=input.device)
    torch.ops._C.silu_and_mul_quant(out, input, scale)
    return out
