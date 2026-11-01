# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager reference oracle for Uno's fused draft-input preparation.

Serving uses ``prepare_uno_inputs_fused`` only; this module is the independent
eager implementation the tests compare it against, including the noise hash.
The hash includes the flat batch-row offset, so deterministic replay assumes a
fixed request row order as well as a fixed request seed and draft step.
"""

import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

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


def prepare_uno_inputs_reference(
    buffers: InputBuffers,
    slot_mapping: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    input_batch: InputBatch,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    seeds: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    k: int,
    max_model_len: int,
    noise_seed: int,
    noise_high: int,
    step: int,
) -> None:
    """Eager reference for the fused seed/noise preparation kernel.

    Temporary suffix KV starts after the last verified row. The target overwrites
    it on verification; neither request progress nor prefix ownership advances here.
    """
    n = input_batch.num_reqs
    count = n * k
    state_idx = input_batch.idx_mapping.long()
    valid_end = input_batch.query_start_loc[1 : n + 1] - num_rejected
    first_pos = input_batch.positions[valid_end.long() - 1] + 1
    offsets = torch.arange(k, dtype=torch.int64, device=buffers.device)
    positions = first_pos[:, None] + offsets
    seed_tokens = torch.where(
        num_sampled > 0,
        last_sampled.reshape(-1)[state_idx],
        next_prefill_tokens.reshape(-1, seeds.shape[0])[0, state_idx],
    )
    buffers.input_ids[:count].copy_(seed_tokens.repeat_interleave(k))
    is_noise = (offsets != 0).repeat(n)
    req_seeds = (seeds[state_idx] + noise_seed).repeat_interleave(k)
    fill_uno_noise(buffers.input_ids[:count], is_noise, req_seeds, step, 1, noise_high)
    # Keep unused, out-of-context rows in range for the model. Their KV writes
    # are suppressed below; native verification bounds usable candidates.
    buffers.positions[:count].copy_(positions.clamp(max=max_model_len - 1).flatten())
    buffers.seq_lens[:n].copy_((first_pos + k).clamp(max=max_model_len))
    buffers.seq_lens[n:].zero_()
    torch.arange(n + 1, out=buffers.query_start_loc[: n + 1])
    buffers.query_start_loc[: n + 1].mul_(k)
    buffers.query_start_loc[n + 1 :].fill_(count)
    buffers.input_ids[count:].zero_()
    buffers.positions[count:].zero_()
    sample_idx_mapping[:count].copy_(state_idx.repeat_interleave(k))
    sample_idx_mapping[count:].fill_(-1)

    block_numbers = positions // block_size
    block_ids = (
        block_table[:n]
        .gather(1, block_numbers.clamp(max=block_table.shape[1] - 1))
        .long()
    )
    valid = (
        (positions < max_model_len)
        & (block_numbers < block_table.shape[1])
        & (block_ids > 0)
    )
    slots = block_ids * block_size + positions % block_size
    slot_mapping[:count].copy_(torch.where(valid, slots, PAD_SLOT_ID).flatten())
    slot_mapping[count:].fill_(PAD_SLOT_ID)
