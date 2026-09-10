# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic vocabulary noise for Uno, independent of the sampling RNG."""

import torch

_M1 = 0x3F58476D1CE4E5B9
_M2 = 0x14D049BB133111EB
_MASK = (1 << 62) - 1


def _mix(x: torch.Tensor) -> torch.Tensor:
    x = ((x ^ (x >> 30)) * _M1) & _MASK
    x = ((x ^ (x >> 27)) * _M2) & _MASK
    return x ^ (x >> 31)


def fill_uno_noise(
    input_ids: torch.Tensor,
    is_noise: torch.Tensor,
    req_seeds: torch.Tensor,
    step: int,
    low: int,
    high: int,
) -> None:
    """Fill noise rows in place with IDs in [low, high), preserving other rows.

    Args:
        input_ids: Flat input token buffer.
        is_noise: Boolean mask with the same shape as input_ids.
        req_seeds: Per-request seeds expanded to the input token shape.
        step: Decode step, independent of the sampler's random generator.
        low: Inclusive token range start.
        high: Exclusive token range end.
    """
    if low >= high:
        raise ValueError("Uno noise requires a nonempty token range")
    slot = torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.int64)
    # Mask Python arithmetic before converting it to a tensor for long decodes.
    step_term = (int(step) * 0x11B54A32D192ED0) & _MASK
    seeds = (req_seeds.to(torch.int64) * 0x1E3779B185EBCA8) & _MASK
    h = _mix((seeds + step_term + slot) & _MASK)
    noise = (h % (high - low)) + low
    input_ids.copy_(torch.where(is_noise, noise.to(input_ids.dtype), input_ids))
