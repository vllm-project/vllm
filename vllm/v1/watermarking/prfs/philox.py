# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Philox4x32-10 as defined by https://doi.org/10.1145/2063384.2063405."""

import torch

from vllm.v1.watermarking.prfs.base import WatermarkPRF, uint32_to_uniform

_UINT32_MASK = 2**32 - 1
_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85
_CONTEXT_DOMAIN = 0x574D4358
_TOKEN_DOMAIN = 0x574D544B


def _mulhilo32(
    multiplier: int, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    value_low = values & 0xFFFF
    value_high = values >> 16
    multiplier_low = multiplier & 0xFFFF
    multiplier_high = multiplier >> 16

    low_product = value_low * multiplier_low
    cross_product = value_high * multiplier_low + value_low * multiplier_high
    low_with_carry = low_product + ((cross_product & 0xFFFF) << 16)
    low = low_with_carry & _UINT32_MASK
    high = (
        value_high * multiplier_high + (cross_product >> 16) + (low_with_carry >> 32)
    ) & _UINT32_MASK
    return high, low


def _philox4x32_10(
    counter: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    key: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    counter_0, counter_1, counter_2, counter_3 = counter
    key_0, key_1 = key
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


class PhiloxPRF(WatermarkPRF):
    """Versioned Philox4x32-10 watermark PRF.

    Contexts are compressed in four-word blocks before a final Philox call
    combines the context state with each candidate token. This mapping and the
    uint32-to-float conversion are part of the compatibility contract.
    """

    version = "philox4x32-10-v1"

    def __init__(self, key: int) -> None:
        if not 0 <= key <= 2**64 - 1:
            raise ValueError("Philox keys must fit in 64 bits")
        self.key = key

    @property
    def max_context_width(self) -> int:
        return 16

    def uniform(self, contexts: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        contexts = contexts.to(torch.int64) & _UINT32_MASK
        prefix_shape = contexts.shape[:-1]
        device = contexts.device
        key_0 = torch.full(
            prefix_shape, self.key & _UINT32_MASK, dtype=torch.int64, device=device
        )
        key_1 = torch.full(
            prefix_shape, self.key >> 32, dtype=torch.int64, device=device
        )
        state = (
            torch.full(prefix_shape, _CONTEXT_DOMAIN, dtype=torch.int64, device=device),
            torch.full(
                prefix_shape,
                contexts.shape[-1],
                dtype=torch.int64,
                device=device,
            ),
            torch.zeros(prefix_shape, dtype=torch.int64, device=device),
            torch.zeros(prefix_shape, dtype=torch.int64, device=device),
        )

        for offset in range(0, contexts.shape[-1], 4):
            block = []
            for index in range(4):
                position = offset + index
                if position < contexts.shape[-1]:
                    block.append(contexts[..., position])
                else:
                    block.append(
                        torch.full(
                            prefix_shape,
                            _UINT32_MASK - index,
                            dtype=torch.int64,
                            device=device,
                        )
                    )
            counter = tuple(
                (state[index] ^ block[index]) & _UINT32_MASK for index in range(4)
            )
            state = _philox4x32_10(counter, ((key_0 ^ offset) & _UINT32_MASK, key_1))

        output_shape = torch.broadcast_shapes(prefix_shape + (1,), token_ids.shape)
        tokens = torch.broadcast_to(token_ids.to(device=device), output_shape)
        state = tuple(
            torch.broadcast_to(word.unsqueeze(-1), output_shape) for word in state
        )
        candidate_key_0 = torch.broadcast_to(key_0.unsqueeze(-1), output_shape)
        candidate_key_1 = torch.broadcast_to(key_1.unsqueeze(-1), output_shape)
        output = _philox4x32_10(
            (
                tokens.to(torch.int64) & _UINT32_MASK,
                state[0],
                state[1],
                state[2],
            ),
            (
                (candidate_key_0 ^ state[3]) & _UINT32_MASK,
                (candidate_key_1 ^ _TOKEN_DOMAIN) & _UINT32_MASK,
            ),
        )[0]
        return uint32_to_uniform(output)
