# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch Gumbel-max sampling, bit-compatible with `gpu/sample/gumbel.py`.

The noise matches the Triton kernels: the same seed, position
and token draw the same uniform. uint32 arithmetic is done in int32, whose
multiplication wraps identically; right shifts are masked to be logical.
"""

import torch


def _i32(c: int) -> int:
    """A uint32 constant as the int32 with the same bits."""
    return c - (1 << 32) if c >= 1 << 31 else c


def _srl(x: torch.Tensor, shift: int) -> torch.Tensor:
    return (x >> shift) & ((1 << (32 - shift)) - 1)


def _rotl(x: torch.Tensor, shift: int) -> torch.Tensor:
    return (x << shift) | _srl(x, 32 - shift)


def _murmur3_mix(h: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    key = _rotl(key * _i32(0xCC9E2D51), 15) * _i32(0x1B873593)
    return _rotl(h ^ key, 13) * 5 + _i32(0xE6546B64)


def _murmur3_fmix32(h: torch.Tensor) -> torch.Tensor:
    h = (h ^ _srl(h, 16)) * _i32(0x85EBCA6B)
    h = (h ^ _srl(h, 13)) * _i32(0xC2B2AE35)
    return h ^ _srl(h, 16)


def _murmur3_hash32(
    seed: torch.Tensor, pos: torch.Tensor, offset: torch.Tensor, domain: int = 0
) -> torch.Tensor:
    """int32 hash bits; `seed` and `pos` are int64, `offset` is int32."""
    h = torch.full(seed.shape, _i32(domain), dtype=torch.int32)
    h = _murmur3_mix(h, seed.to(torch.int32))
    h = _murmur3_mix(h, (seed >> 32).to(torch.int32))
    h = _murmur3_mix(h, pos.to(torch.int32))
    h = _murmur3_mix(h, offset)
    return _murmur3_fmix32(h ^ 16)


def _log1p_neg_stable(value: torch.Tensor) -> torch.Tensor:
    polynomial = torch.full_like(value, 1.0 / 8.0)
    for coeff in (1 / 7, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1.0):
        polynomial = coeff + value * polynomial
    series = -value * polynomial
    direct = torch.log(torch.clamp_min(1.0 - value, 5.960464477539063e-08))
    return torch.where(value < 0.25, series, direct)


def gumbel_noise(
    seed: torch.Tensor, pos: torch.Tensor, token_ids: torch.Tensor, use_fp64: bool
) -> torch.Tensor:
    """Gumbel noise drawn by the Triton kernels for these tokens."""
    offset = token_ids.to(torch.int32)
    if use_fp64:
        lo = _murmur3_hash32(seed, pos, offset).to(torch.int64) & 0xFFFFFFFF
        hi = _murmur3_hash32(seed, pos, offset, domain=0x9E3779B9)
        random53 = ((hi.to(torch.int64) & 0xFFFFFFFF) << 21) | (lo >> 11)
        u = (random53.to(torch.float64) + 0.5) * 1.1102230246251565e-16
        u = torch.where(u == 1.0, u - 1.1102230246251565e-16, u)
        return -torch.log(-torch.log(u))
    random32 = _murmur3_hash32(seed, pos, offset)
    hi16 = _srl(random32, 16).to(torch.float32)
    lo16 = (random32 & 0xFFFF).to(torch.float32)
    u = hi16 * 1.52587890625e-05 + (lo16 + 0.5) * 2.3283064365386963e-10
    return -torch.log(-_log1p_neg_stable(u))


def _gumbel_argmax(
    logits: torch.Tensor,
    temp: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    apply_temperature: bool,
    use_fp64: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gumbel-max sample per row, or the plain argmax where `temp` is 0."""
    logits = logits.to(torch.float32)
    temp = temp.unsqueeze(1)
    is_sampled = temp != 0.0
    if apply_temperature:
        logits = torch.where(is_sampled, logits / temp, logits)
    if use_fp64:
        logits = logits.to(torch.float64)
    token_ids = torch.arange(logits.shape[1], dtype=torch.int32)
    noise = gumbel_noise(seed.unsqueeze(1), pos.unsqueeze(1), token_ids, use_fp64)
    logits = torch.where(is_sampled, logits + noise, logits)
    return logits.max(dim=1)


gumbel_argmax = torch.compile(_gumbel_argmax, dynamic=True, fullgraph=True)
