# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalState:
    m: torch.Tensor
    s: torch.Tensor
    w: torch.Tensor
    e: torch.Tensor
    clean_val: torch.Tensor
    clean_idx: torch.Tensor
    sample_val: torch.Tensor
    sample_idx: torch.Tensor
    normalizer: torch.Tensor


@dataclass(frozen=True)
class SamplerOutputs:
    entropy: torch.Tensor
    soft_embed: torch.Tensor
    clean_val: torch.Tensor
    clean_idx: torch.Tensor
    sample_val: torch.Tensor
    sample_idx: torch.Tensor


def _dense_reference(
    logits: torch.Tensor,
    embed_weight: torch.Tensor,
    gumbels: torch.Tensor,
    normalizer: torch.Tensor,
) -> SamplerOutputs:
    probs = logits.softmax(dim=-1)
    log_probs = logits.log_softmax(dim=-1)
    noisy = logits + gumbels
    clean_val, clean_idx = logits.max(dim=-1)
    sample_val, sample_idx = noisy.max(dim=-1)
    return SamplerOutputs(
        entropy=-(probs * log_probs).sum(dim=-1),
        soft_embed=torch.matmul(probs, embed_weight) * normalizer,
        clean_val=clean_val,
        clean_idx=clean_idx,
        sample_val=sample_val,
        sample_idx=sample_idx,
    )


def _local_state(
    local_logits: torch.Tensor,
    local_embed_weight: torch.Tensor,
    local_gumbels: torch.Tensor,
    *,
    vocab_start: int,
    local_vocab_width: int,
    normalizer: torch.Tensor,
) -> LocalState:
    real_logits = local_logits[:, :local_vocab_width]
    real_embed = local_embed_weight[:local_vocab_width]
    real_gumbels = local_gumbels[:, :local_vocab_width]

    m = real_logits.max(dim=-1).values
    exp_shifted = torch.exp(real_logits - m.unsqueeze(-1))
    clean_val, clean_local_idx = real_logits.max(dim=-1)
    sample_val, sample_local_idx = (real_logits + real_gumbels).max(dim=-1)
    return LocalState(
        m=m,
        s=exp_shifted.sum(dim=-1),
        w=(exp_shifted * real_logits).sum(dim=-1),
        e=torch.matmul(exp_shifted, real_embed),
        clean_val=clean_val,
        clean_idx=clean_local_idx + vocab_start,
        sample_val=sample_val,
        sample_idx=sample_local_idx + vocab_start,
        normalizer=normalizer,
    )


def _merge_tp_states(states: list[LocalState]) -> SamplerOutputs:
    m_parts = torch.stack([state.m for state in states])
    global_m = m_parts.max(dim=0).values
    scales = torch.exp(m_parts - global_m.unsqueeze(0))

    s = sum(state.s * scales[rank] for rank, state in enumerate(states))
    w = sum(state.w * scales[rank] for rank, state in enumerate(states))
    e = sum(
        state.e * scales[rank].unsqueeze(-1) for rank, state in enumerate(states)
    )
    entropy = torch.log(s) + global_m - w / s
    soft_embed = e / s.unsqueeze(-1) * states[0].normalizer

    clean_val, clean_idx = _reduce_argmax_pairs(
        [state.clean_val for state in states],
        [state.clean_idx for state in states],
    )
    sample_val, sample_idx = _reduce_argmax_pairs(
        [state.sample_val for state in states],
        [state.sample_idx for state in states],
    )
    return SamplerOutputs(
        entropy=entropy,
        soft_embed=soft_embed,
        clean_val=clean_val,
        clean_idx=clean_idx,
        sample_val=sample_val,
        sample_idx=sample_idx,
    )


def _reduce_argmax_pairs(
    values: list[torch.Tensor],
    indices: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    best_values = values[0].clone()
    best_indices = indices[0].clone()
    for value, index in zip(values[1:], indices[1:]):
        better = (value > best_values) | (
            (value == best_values) & (index < best_indices)
        )
        best_values = torch.where(better, value, best_values)
        best_indices = torch.where(better, index, best_indices)
    return best_values, best_indices


def _assert_outputs_close(actual: SamplerOutputs, expected: SamplerOutputs) -> None:
    torch.testing.assert_close(actual.entropy, expected.entropy)
    torch.testing.assert_close(actual.soft_embed, expected.soft_embed)
    torch.testing.assert_close(actual.clean_val, expected.clean_val)
    torch.testing.assert_close(actual.clean_idx, expected.clean_idx)
    torch.testing.assert_close(actual.sample_val, expected.sample_val)
    torch.testing.assert_close(actual.sample_idx, expected.sample_idx)


def _states_from_ranges(
    logits: torch.Tensor,
    embed_weight: torch.Tensor,
    gumbels: torch.Tensor,
    ranges: list[tuple[int, int]],
    normalizer: torch.Tensor,
) -> list[LocalState]:
    return [
        _local_state(
            logits[:, start:end],
            embed_weight[start:end],
            gumbels[:, start:end],
            vocab_start=start,
            local_vocab_width=end - start,
            normalizer=normalizer,
        )
        for start, end in ranges
    ]


def test_merge_tp4_state_matches_dense_reference() -> None:
    torch.manual_seed(0)
    rows, vocab_size, hidden_size = 5, 16, 6
    logits = torch.randn(rows, vocab_size, dtype=torch.float64)
    embed_weight = torch.randn(vocab_size, hidden_size, dtype=torch.float64)
    gumbels = torch.randn(rows, vocab_size, dtype=torch.float64)
    normalizer = torch.tensor(0.75, dtype=torch.float64)
    ranges = [(0, 4), (4, 8), (8, 12), (12, 16)]

    actual = _merge_tp_states(
        _states_from_ranges(logits, embed_weight, gumbels, ranges, normalizer)
    )

    expected = _dense_reference(logits, embed_weight, gumbels, normalizer)
    _assert_outputs_close(actual, expected)


def test_merge_handles_non_divisible_padded_shard_ranges() -> None:
    torch.manual_seed(1)
    rows, vocab_size, hidden_size = 4, 10, 5
    logits = torch.randn(rows, vocab_size, dtype=torch.float64)
    embed_weight = torch.randn(vocab_size, hidden_size, dtype=torch.float64)
    gumbels = torch.randn(rows, vocab_size, dtype=torch.float64)
    normalizer = torch.tensor(1.25, dtype=torch.float64)
    ranges = [(0, 3), (3, 6), (6, 8), (8, 10)]

    states: list[LocalState] = []
    for start, end in ranges:
        width = end - start
        local_logits = torch.full((rows, 3), 999.0, dtype=torch.float64)
        local_embed = torch.full((3, hidden_size), 999.0, dtype=torch.float64)
        local_gumbels = torch.full((rows, 3), 999.0, dtype=torch.float64)
        local_logits[:, :width] = logits[:, start:end]
        local_embed[:width] = embed_weight[start:end]
        local_gumbels[:, :width] = gumbels[:, start:end]
        states.append(
            _local_state(
                local_logits,
                local_embed,
                local_gumbels,
                vocab_start=start,
                local_vocab_width=width,
                normalizer=normalizer,
            )
        )

    actual = _merge_tp_states(states)

    expected = _dense_reference(logits, embed_weight, gumbels, normalizer)
    _assert_outputs_close(actual, expected)


def test_merge_argmax_ties_pick_smallest_global_token_id() -> None:
    logits = torch.tensor(
        [[0.0, 4.0, 3.0, 1.0, 2.0, 3.0, 4.0, -1.0]],
        dtype=torch.float64,
    )
    embed_weight = torch.arange(24, dtype=torch.float64).reshape(8, 3) / 10
    gumbels = torch.tensor(
        [[0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    normalizer = torch.tensor(0.5, dtype=torch.float64)
    ranges = [(0, 2), (2, 4), (4, 6), (6, 8)]

    actual = _merge_tp_states(
        _states_from_ranges(logits, embed_weight, gumbels, ranges, normalizer)
    )

    expected = _dense_reference(logits, embed_weight, gumbels, normalizer)
    _assert_outputs_close(actual, expected)
    torch.testing.assert_close(actual.clean_idx, torch.tensor([1]))
    torch.testing.assert_close(actual.sample_idx, torch.tensor([2]))
