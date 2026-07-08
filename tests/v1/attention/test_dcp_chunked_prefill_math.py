# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import random

import pytest


def _dcp_local_len_formula(length: int, world: int, rank: int, interleave: int) -> int:
    base = length // interleave // world * interleave
    remainder = length - base * world
    remainder = min(max(remainder - rank * interleave, 0), interleave)
    return base + remainder


def _dcp_local_len_reference(
    length: int, world: int, rank: int, interleave: int
) -> int:
    return sum(1 for pos in range(length) if (pos // interleave) % world == rank)


def _owner_and_local_offset(
    position: int, local_block_size: int, world: int, interleave: int
) -> tuple[int, int, int]:
    block_index = position // (local_block_size * world)
    block_offset = position % (local_block_size * world)
    owner = (block_offset // interleave) % world
    rounds = block_offset // (interleave * world)
    remainder = block_offset % interleave
    local_offset = rounds * interleave + remainder
    return owner, block_index, local_offset


def test_dcp_local_seq_len_formula_matches_position_counting():
    lengths = [
        0,
        1,
        2,
        3,
        4,
        5,
        15,
        16,
        47,
        48,
        127,
        128,
        129,
        4095,
        4096,
        4097,
        8191,
        8192,
        22860,
    ]

    for interleave in (1, 2, 4, 16):
        for length in lengths:
            local = [
                _dcp_local_len_formula(length, world=3, rank=rank, interleave=interleave)
                for rank in range(3)
            ]
            expected = [
                _dcp_local_len_reference(
                    length, world=3, rank=rank, interleave=interleave
                )
                for rank in range(3)
            ]
            assert local == expected
            assert sum(local) == length
            assert max(local, default=0) - min(local, default=0) <= interleave


def test_dcp_slot_mapping_matches_position_counting_across_128_chunks():
    world = 3
    interleave = 1
    local_block_size = 16
    prompt_len = 1024
    chunk_size = 128

    per_rank_seen: list[list[int]] = [[] for _ in range(world)]
    for chunk_start in range(0, prompt_len, chunk_size):
        chunk_end = min(prompt_len, chunk_start + chunk_size)
        for position in range(chunk_start, chunk_end):
            owner, block_index, local_offset = _owner_and_local_offset(
                position, local_block_size, world, interleave
            )
            per_rank_seen[owner].append(position)

            expected_owner = (position // interleave) % world
            expected_local_index = len(per_rank_seen[owner]) - 1
            assert owner == expected_owner
            assert block_index == expected_local_index // local_block_size
            assert local_offset == expected_local_index % local_block_size

    assert sorted(pos for rank_positions in per_rank_seen for pos in rank_positions) == list(
        range(prompt_len)
    )


def test_dcp_context_lengths_for_128_chunked_prefill_are_context_only():
    world = 3
    interleave = 1
    page_size = 16
    prompt_len = 1024
    chunk_size = 128

    for chunk_start in range(0, prompt_len, chunk_size):
        query_len = min(chunk_size, prompt_len - chunk_start)
        seq_len_before_localization = chunk_start + query_len
        context_len = seq_len_before_localization - query_len
        assert context_len == chunk_start

        for rank in range(world):
            local_context_len = _dcp_local_len_formula(
                context_len, world, rank, interleave
            )
            num_blocks = math.ceil(local_context_len / page_size)
            last_page_len = (
                page_size
                if local_context_len > 0 and local_context_len % page_size == 0
                else local_context_len % page_size
            )

            assert local_context_len == _dcp_local_len_reference(
                context_len, world, rank, interleave
            )
            assert num_blocks * page_size >= local_context_len
            if local_context_len == 0:
                assert num_blocks == 0
                assert last_page_len == 0


def _softmax_attention(q, keys, values):
    if not keys:
        return [0.0 for _ in values[0]], float("-inf")
    scores = [sum(a * b for a, b in zip(q, k)) for k in keys]
    max_score = max(scores)
    denom = sum(math.exp(score - max_score) for score in scores)
    lse = max_score + math.log(denom)
    probs = [math.exp(score - lse) for score in scores]
    out = [
        sum(prob * value[d] for prob, value in zip(probs, values))
        for d in range(len(values[0]))
    ]
    return out, lse


def _merge_lse_outputs(local_outputs, local_lses):
    merged_lse = math.log(sum(math.exp(lse) for lse in local_lses if math.isfinite(lse)))
    out_dim = len(local_outputs[0])
    merged = [0.0] * out_dim
    for out, lse in zip(local_outputs, local_lses):
        weight = math.exp(lse - merged_lse) if math.isfinite(lse) else 0.0
        for d in range(out_dim):
            merged[d] += weight * out[d]
    return merged, merged_lse


def test_dcp_lse_merge_matches_dense_attention():
    rng = random.Random(0)
    world = 3
    interleave = 1
    q = [rng.uniform(-0.5, 0.5) for _ in range(5)]
    keys = [[rng.uniform(-0.5, 0.5) for _ in range(5)] for _ in range(37)]
    values = [[rng.uniform(-0.5, 0.5) for _ in range(4)] for _ in range(37)]

    dense_out, dense_lse = _softmax_attention(q, keys, values)
    local_outputs = []
    local_lses = []
    for rank in range(world):
        owned = [
            pos for pos in range(len(keys)) if (pos // interleave) % world == rank
        ]
        out, lse = _softmax_attention(
            q, [keys[pos] for pos in owned], [values[pos] for pos in owned]
        )
        local_outputs.append(out)
        local_lses.append(lse)

    merged_out, merged_lse = _merge_lse_outputs(local_outputs, local_lses)
    assert merged_lse == pytest.approx(dense_lse, abs=1e-12)
    assert merged_out == pytest.approx(dense_out, abs=1e-12)
