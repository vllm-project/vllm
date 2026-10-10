# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch implementations of the `gpu/sample/gumbel.py` Triton kernels.

The noise is bit-compatible with the Triton kernels: the same seed, position
and token draw the same uniform. uint32 arithmetic is done in int32, whose
multiplication wraps identically; right shifts are masked to be logical.
"""

import torch

_DRAFT_NOISE_SALT = 1 << 30


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


_gumbel_argmax_compiled = torch.compile(_gumbel_argmax, dynamic=True, fullgraph=True)


def apply_temperature(
    logits_ptr: torch.Tensor,
    logits_stride: int,
    expanded_idx_mapping_ptr: torch.Tensor,
    temperature_ptr: torch.Tensor,
    vocab_size: int,
    BLOCK_SIZE: int,
) -> None:
    num_tokens = logits_ptr.shape[0]
    temp = temperature_ptr[expanded_idx_mapping_ptr[:num_tokens].long()]
    temp = temp.to(torch.float32)
    # Temperatures 0 and 1 leave the logits untouched.
    temp = torch.where((temp == 0.0) | (temp == 1.0), 1.0, temp)
    logits = logits_ptr[:, :vocab_size]
    logits.copy_(logits.to(torch.float32) / temp.unsqueeze(1))


def gumbel_sample(
    local_argmax_ptr: torch.Tensor,
    local_argmax_stride: int,
    local_max_ptr: torch.Tensor,
    local_max_stride: int,
    logits_cache_ptr: torch.Tensor | None,
    logits_cache_stride_0: int,
    logits_cache_stride_1: int,
    logits_cache_col_ptr: torch.Tensor | None,
    logits_cache_source_ptr: torch.Tensor | None,
    logits_cache_source_stride: int,
    logits_ptr: torch.Tensor,
    logits_stride: int,
    expanded_idx_mapping_ptr: torch.Tensor,
    seeds_ptr: torch.Tensor,
    pos_ptr: torch.Tensor,
    temp_ptr: torch.Tensor,
    vocab_size: int,
    BLOCK_SIZE: int,
    IS_DRAFTING: bool,
    APPLY_TEMPERATURE: bool,
    USE_FP64: bool,
    PER_TOKEN_COL: bool,
) -> None:
    num_tokens = logits_ptr.shape[0]
    req = expanded_idx_mapping_ptr[:num_tokens].long()
    is_valid_req = req >= 0
    req = req.clamp_min(0)
    temp = torch.where(is_valid_req, temp_ptr[req].to(torch.float32), 0.0)
    seed = torch.where(is_valid_req, seeds_ptr[req].long(), 0)
    pos = pos_ptr[:num_tokens].long()
    if IS_DRAFTING:
        pos = pos + _DRAFT_NOISE_SALT

    if logits_cache_ptr is not None:
        assert logits_cache_source_ptr is not None
        assert logits_cache_col_ptr is not None
        # Cache the logits before temperature; the rejection sampler divides
        # by the same temperature on load.
        col = logits_cache_col_ptr.long().expand(num_tokens)
        valid = is_valid_req.nonzero().squeeze(1)
        source = logits_cache_source_ptr[valid, :vocab_size]
        logits_cache_ptr[req[valid], col[valid], :vocab_size] = source

    value, idx = _gumbel_argmax_compiled(
        logits_ptr[:, :vocab_size], temp, seed, pos, APPLY_TEMPERATURE, USE_FP64
    )
    # The caller reduces over the per-block partials; a single winning block
    # gives the same result.
    local_max_ptr.fill_(float("-inf"))
    local_max_ptr[:, 0] = value
    local_argmax_ptr[:, 0] = idx
