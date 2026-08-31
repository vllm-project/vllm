# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.watermarking.prfs.philox import (
    _CONTEXT_DOMAIN as _CONTEXT_DOMAIN_VALUE,
)
from vllm.v1.watermarking.prfs.philox import (
    _PHILOX_M0 as _PHILOX_M0_VALUE,
)
from vllm.v1.watermarking.prfs.philox import (
    _PHILOX_M1 as _PHILOX_M1_VALUE,
)
from vllm.v1.watermarking.prfs.philox import (
    _PHILOX_W0 as _PHILOX_W0_VALUE,
)
from vllm.v1.watermarking.prfs.philox import (
    _PHILOX_W1 as _PHILOX_W1_VALUE,
)
from vllm.v1.watermarking.prfs.philox import (
    _TOKEN_DOMAIN as _TOKEN_DOMAIN_VALUE,
)
from vllm.v1.watermarking.prfs.philox import (
    _UINT32_MASK as _UINT32_MASK_VALUE,
)

if HAS_TRITON:
    from triton.language import math as tl_math
else:
    tl_math = tl

_UINT32_MASK = tl.constexpr(_UINT32_MASK_VALUE) if HAS_TRITON else _UINT32_MASK_VALUE
_PHILOX_M0 = tl.constexpr(_PHILOX_M0_VALUE) if HAS_TRITON else _PHILOX_M0_VALUE
_PHILOX_M1 = tl.constexpr(_PHILOX_M1_VALUE) if HAS_TRITON else _PHILOX_M1_VALUE
_PHILOX_W0 = tl.constexpr(_PHILOX_W0_VALUE) if HAS_TRITON else _PHILOX_W0_VALUE
_PHILOX_W1 = tl.constexpr(_PHILOX_W1_VALUE) if HAS_TRITON else _PHILOX_W1_VALUE
_CONTEXT_DOMAIN = (
    tl.constexpr(_CONTEXT_DOMAIN_VALUE) if HAS_TRITON else _CONTEXT_DOMAIN_VALUE
)
_TOKEN_DOMAIN = tl.constexpr(_TOKEN_DOMAIN_VALUE) if HAS_TRITON else _TOKEN_DOMAIN_VALUE


@triton.jit
def _mulhilo32(multiplier: tl.constexpr, value):
    high = tl_math.umulhi(multiplier, value)
    low = tl.mul(multiplier, value, sanitize_overflow=False)
    return high, low


@triton.jit
def _philox4x32_10(counter_0, counter_1, counter_2, counter_3, key_0, key_1):
    for round_index in range(10):
        high_0, low_0 = _mulhilo32(_PHILOX_M0, counter_0)
        high_1, low_1 = _mulhilo32(_PHILOX_M1, counter_2)
        counter_0, counter_1, counter_2, counter_3 = (
            (high_1 ^ counter_1 ^ key_0) & _UINT32_MASK,
            low_1,
            (high_0 ^ counter_3 ^ key_1) & _UINT32_MASK,
            low_0,
        )
        if round_index != 9:
            key_0 = (key_0 + _PHILOX_W0) & _UINT32_MASK
            key_1 = (key_1 + _PHILOX_W1) & _UINT32_MASK
    return counter_0, counter_1, counter_2, counter_3


@triton.jit
def _uint32_to_uniform(value):
    mantissa = (value & _UINT32_MASK) >> 8
    scaled = (mantissa + 1).to(tl.float32) * 2**-24
    return (scaled.to(tl.uint32, bitcast=True) - 1).to(tl.float32, bitcast=True)


@triton.jit
def _gumbel_value(logits_ptr, output, mask):
    logits = tl.load(logits_ptr, mask=mask, other=float("-inf")).to(tl.float32)
    uniform = _uint32_to_uniform(output)
    return logits - tl.log(-tl.log(uniform))


@triton.jit
def _select_max(best_value, best_token, candidate_value, candidate):
    take_candidate = candidate_value > best_value
    return (
        tl.where(take_candidate, candidate_value, best_value),
        tl.where(take_candidate, candidate, best_token),
    )


@triton.jit
def _philox_gumbel_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    logits_ptr,
    logits_stride,
    contexts_ptr,
    context_stride,
    key_0_value,
    key_1_value,
    vocab_size,
    CONTEXT_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block_index = tl.program_id(1)
    groups = block_index * (BLOCK_SIZE // 4) + tl.arange(0, BLOCK_SIZE // 4)

    key_0 = key_0_value.to(tl.uint32)
    key_1 = key_1_value.to(tl.uint32)
    state_0 = tl.full((), _CONTEXT_DOMAIN, tl.uint32)
    state_1 = tl.full((), CONTEXT_WIDTH, tl.uint32)
    state_2 = tl.full((), 0, tl.uint32)
    state_3 = tl.full((), 0, tl.uint32)
    for offset in range(0, CONTEXT_WIDTH, 4):
        context_0 = tl.load(contexts_ptr + row * context_stride + offset).to(tl.uint32)
        if offset + 1 < CONTEXT_WIDTH:
            context_1 = tl.load(contexts_ptr + row * context_stride + offset + 1).to(
                tl.uint32
            )
        else:
            context_1 = tl.full((), _UINT32_MASK - 1, tl.uint32)
        if offset + 2 < CONTEXT_WIDTH:
            context_2 = tl.load(contexts_ptr + row * context_stride + offset + 2).to(
                tl.uint32
            )
        else:
            context_2 = tl.full((), _UINT32_MASK - 2, tl.uint32)
        if offset + 3 < CONTEXT_WIDTH:
            context_3 = tl.load(contexts_ptr + row * context_stride + offset + 3).to(
                tl.uint32
            )
        else:
            context_3 = tl.full((), _UINT32_MASK - 3, tl.uint32)
        state_0, state_1, state_2, state_3 = _philox4x32_10(
            (state_0 ^ context_0) & _UINT32_MASK,
            (state_1 ^ context_1) & _UINT32_MASK,
            (state_2 ^ context_2) & _UINT32_MASK,
            (state_3 ^ context_3) & _UINT32_MASK,
            (key_0 ^ offset) & _UINT32_MASK,
            key_1,
        )

    candidate_words = groups.to(tl.uint32)
    vector_zero = candidate_words * 0
    output_0, output_1, output_2, output_3 = _philox4x32_10(
        candidate_words & _UINT32_MASK,
        state_0 + vector_zero,
        state_1 + vector_zero,
        state_2 + vector_zero,
        ((key_0 ^ state_3) & _UINT32_MASK) + vector_zero,
        ((key_1 ^ _TOKEN_DOMAIN) & _UINT32_MASK) + vector_zero,
    )
    candidate_0 = groups * 4
    candidate_1 = candidate_0 + 1
    candidate_2 = candidate_0 + 2
    candidate_3 = candidate_0 + 3
    logits_row = logits_ptr + row * logits_stride
    value_0 = _gumbel_value(
        logits_row + candidate_0, output_0, candidate_0 < vocab_size
    )
    value_1 = _gumbel_value(
        logits_row + candidate_1, output_1, candidate_1 < vocab_size
    )
    value_2 = _gumbel_value(
        logits_row + candidate_2, output_2, candidate_2 < vocab_size
    )
    value_3 = _gumbel_value(
        logits_row + candidate_3, output_3, candidate_3 < vocab_size
    )
    best_value = value_0
    best_token = candidate_0
    best_value, best_token = _select_max(best_value, best_token, value_1, candidate_1)
    best_value, best_token = _select_max(best_value, best_token, value_2, candidate_2)
    best_value, best_token = _select_max(best_value, best_token, value_3, candidate_3)
    value = tl.max(best_value, axis=0)
    token_id = tl.min(tl.where(best_value == value, best_token, vocab_size), axis=0)
    tl.store(local_argmax_ptr + row * local_argmax_stride + block_index, token_id)
    tl.store(local_max_ptr + row * local_max_stride + block_index, value)


def philox_gumbel_sample(
    logits: torch.Tensor, contexts: torch.Tensor, key: int
) -> torch.Tensor:
    num_tokens, vocab_size = logits.shape
    block_size = 1024
    num_blocks = triton.cdiv(vocab_size, block_size)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=torch.float32)
    _philox_gumbel_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits,
        logits.stride(0),
        contexts,
        contexts.stride(0),
        key & _UINT32_MASK_VALUE,
        key >> 32,
        vocab_size,
        CONTEXT_WIDTH=contexts.shape[-1],
        BLOCK_SIZE=block_size,
    )
    max_block_index = local_max.argmax(dim=-1, keepdim=True)
    return local_argmax.gather(dim=-1, index=max_block_index).view(-1)
