# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the sample_recovered_and_bonus_tokens Triton kernel.

The kernel implements Gumbel-max sampling for speculative decoding:
  - For each draft token position: sample from max(0, target_probs - draft_probs)
  - For each bonus token (last per request): sample directly from target_probs
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.rejection_sample import (
    sample_recovered_and_bonus_tokens,
)

DEVICE = current_platform.device_type

# The Triton kernel processes the vocab in blocks of this size.
# Vocab sizes that are not multiples of this are the most likely to trigger
# the NaN bug fixed in this file (padding positions hit -inf - (-inf) = NaN).
KERNEL_BLOCK_SIZE = 1024


def make_inputs(
    num_draft_tokens_per_req: list[int],
    vocab_size: int,
    target_probs: torch.Tensor | None = None,
    draft_probs: torch.Tensor | None = None,
    seeds: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build the input tensors needed by sample_recovered_and_bonus_tokens.

    Each request contributes (num_draft + 1) token rows to target_probs:
    num_draft recovered-token rows followed by one bonus-token row.
    draft_probs has one row per draft token only (no bonus row).
    """
    num_reqs = len(num_draft_tokens_per_req)
    # Each request has num_draft draft positions + 1 bonus position.
    num_tokens = sum(n + 1 for n in num_draft_tokens_per_req)
    num_draft_tokens = sum(num_draft_tokens_per_req)

    # cu_num_logits[i+1] - cu_num_logits[i] = num_draft_tokens_per_req[i] + 1
    cu = [0]
    for n in num_draft_tokens_per_req:
        cu.append(cu[-1] + n + 1)
    cu_num_logits = torch.tensor(cu, dtype=torch.int32, device=DEVICE)

    # idx_mapping[token_idx] = req_idx for every token belonging to that req.
    idx_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE)
    for req_idx, n in enumerate(num_draft_tokens_per_req):
        start = cu[req_idx]
        end = cu[req_idx + 1]
        idx_mapping[start:end] = req_idx

    if seeds is None:
        seeds = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE)

    # pos is used as the RNG offset; distinct values give diverse random draws.
    pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

    if target_probs is None:
        target_probs = F.softmax(
            torch.rand(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE),
            dim=-1,
        )
    if draft_probs is None:
        if num_draft_tokens > 0:
            draft_probs = F.softmax(
                torch.rand(
                    num_draft_tokens, vocab_size, dtype=torch.float32, device=DEVICE
                ),
                dim=-1,
            )
        else:
            draft_probs = torch.empty(0, vocab_size, dtype=torch.float32, device=DEVICE)

    return target_probs, draft_probs, cu_num_logits, idx_mapping, seeds, pos


def peaked_probs(num_rows: int, vocab_size: int, token_ids: list[int]) -> torch.Tensor:
    """Return near-one-hot probability distributions.

    Row i has almost all probability mass on token_ids[i], making sampling
    effectively deterministic regardless of the Gumbel noise realisation.
    """
    assert len(token_ids) == num_rows
    probs = torch.full(
        (num_rows, vocab_size), 1e-10, dtype=torch.float32, device=DEVICE
    )
    for i, t in enumerate(token_ids):
        probs[i, t] = 1.0
    return probs / probs.sum(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Range / regression tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vocab_size",
    [
        100,
        1000,
        KERNEL_BLOCK_SIZE,  # aligned: last block is fully valid
        KERNEL_BLOCK_SIZE + 1,  # one valid token in last block
        KERNEL_BLOCK_SIZE * 2 - 1,  # one invalid token in last block
        128256,  # Llama-3 vocab; 128256 % 1024 == 256 (256 valid tokens in last block)
        128257,  # Llama-3 vocab + 1; 257 valid tokens in last block
    ],
)
@pytest.mark.parametrize(
    "num_draft_tokens_per_req",
    [
        [1],
        [3],
        [1, 2, 3],
    ],
)
def test_output_in_vocab_range(vocab_size: int, num_draft_tokens_per_req: list[int]):
    """All sampled token IDs must satisfy 0 <= id < vocab_size.

    This is a regression test for a NaN bug: when the vocab size is not a
    multiple of BLOCK_SIZE the out-of-bounds padding positions in the last
    block were loaded as -inf for both target_probs and draft_probs.  The
    subtraction -inf - (-inf) = NaN, which then propagated through tl.max
    and could select an out-of-vocab index.
    """
    args = make_inputs(num_draft_tokens_per_req, vocab_size)
    sampled = sample_recovered_and_bonus_tokens(*args)

    num_tokens = sum(n + 1 for n in num_draft_tokens_per_req)
    assert sampled.shape == (num_tokens,)
    assert (sampled >= 0).all(), f"Negative token IDs: {sampled}"
    assert (sampled < vocab_size).all(), (
        f"Out-of-vocab token IDs (vocab_size={vocab_size}): max={sampled.max().item()}"
    )


# ---------------------------------------------------------------------------
# Correctness: bonus vs. recovered token distinction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_draft_per_req", [0, 1, 3])
def test_bonus_token_samples_from_target_probs(num_draft_per_req: int):
    """The bonus token (last position per request) samples from target_probs.

    It must NOT be adjusted by draft_probs.  We verify this by concentrating
    the target mass on a known token at every bonus position and checking the
    output, regardless of what draft_probs contains.
    """
    vocab_size = 100
    num_reqs = 4
    num_draft_tokens_per_req = [num_draft_per_req] * num_reqs
    num_tokens = sum(n + 1 for n in num_draft_tokens_per_req)
    num_draft_tokens = sum(num_draft_tokens_per_req)

    cu = [0] + [
        sum(n + 1 for n in num_draft_tokens_per_req[: i + 1]) for i in range(num_reqs)
    ]

    # Assign a distinct expected token to the bonus position of each request.
    expected_bonus = list(range(num_reqs))  # token 0, 1, 2, 3
    target_token_ids = [0] * num_tokens
    for req_idx in range(num_reqs):
        bonus_pos = cu[req_idx + 1] - 1  # last position of the request
        target_token_ids[bonus_pos] = expected_bonus[req_idx]

    target_probs = peaked_probs(num_tokens, vocab_size, target_token_ids)
    # draft_probs is random; for bonus positions it should be ignored.
    draft_probs = (
        F.softmax(
            torch.rand(
                num_draft_tokens, vocab_size, dtype=torch.float32, device=DEVICE
            ),
            dim=-1,
        )
        if num_draft_tokens > 0
        else torch.empty(0, vocab_size, dtype=torch.float32, device=DEVICE)
    )

    _, _, cu_num_logits, idx_mapping, seeds, pos = make_inputs(
        num_draft_tokens_per_req, vocab_size
    )
    sampled = sample_recovered_and_bonus_tokens(
        target_probs, draft_probs, cu_num_logits, idx_mapping, seeds, pos
    )

    for req_idx in range(num_reqs):
        bonus_pos = cu[req_idx + 1] - 1
        got = sampled[bonus_pos].item()
        assert got == expected_bonus[req_idx], (
            f"Request {req_idx}: bonus token expected {expected_bonus[req_idx]}, "
            f"got {got}"
        )


def test_recovered_token_samples_from_target_minus_draft():
    """Recovered tokens sample from (target_probs - draft_probs).

    When target_probs ≈ one_hot(A) and draft_probs ≈ one_hot(B) with A ≠ B,
    the adjusted distribution has positive mass only at A, so the sampled
    token must be A.
    """
    vocab_size = 100
    num_draft_tokens_per_req = [3]
    num_tokens = 4  # 3 draft + 1 bonus
    num_draft_tokens = 3

    target_token = 10
    draft_token = 20  # different from target_token

    target_probs = peaked_probs(num_tokens, vocab_size, [target_token] * num_tokens)
    draft_probs = peaked_probs(
        num_draft_tokens, vocab_size, [draft_token] * num_draft_tokens
    )

    _, _, cu_num_logits, idx_mapping, seeds, pos = make_inputs(
        num_draft_tokens_per_req, vocab_size
    )
    sampled = sample_recovered_and_bonus_tokens(
        target_probs, draft_probs, cu_num_logits, idx_mapping, seeds, pos
    )

    # Recovered positions (0, 1, 2): adjusted mass is on target_token.
    for i in range(num_draft_tokens):
        got = sampled[i].item()
        assert got == target_token, (
            f"Recovered token at position {i}: expected {target_token}, got {got}"
        )
    # Bonus position (3): samples directly from target_probs → also target_token.
    assert sampled[3].item() == target_token


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_with_same_seeds():
    """Identical inputs and seeds always produce identical outputs."""
    vocab_size = 1000
    num_draft_tokens_per_req = [2, 3]
    args = make_inputs(num_draft_tokens_per_req, vocab_size)

    out1 = sample_recovered_and_bonus_tokens(*args)
    out2 = sample_recovered_and_bonus_tokens(*args)
    assert torch.equal(out1, out2)


def test_different_seeds_produce_different_outputs():
    """Different seeds produce different outputs (with overwhelming probability).

    Over a batch of 16 tokens drawn from a uniform distribution, the
    probability that two independent runs produce the same sequence is
    negligible.
    """
    vocab_size = 1000
    num_draft_tokens_per_req = [3, 3, 3, 3]
    num_reqs = len(num_draft_tokens_per_req)
    num_tokens = sum(n + 1 for n in num_draft_tokens_per_req)
    num_draft_tokens = sum(num_draft_tokens_per_req)

    # Shared probs; only the seeds differ.
    target_probs = F.softmax(
        torch.rand(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE), dim=-1
    )
    draft_probs = F.softmax(
        torch.rand(num_draft_tokens, vocab_size, dtype=torch.float32, device=DEVICE),
        dim=-1,
    )

    seeds_a = torch.zeros(num_reqs, dtype=torch.int64, device=DEVICE)
    seeds_b = torch.full((num_reqs,), 99999, dtype=torch.int64, device=DEVICE)

    _, _, cu_num_logits, idx_mapping, _, pos = make_inputs(
        num_draft_tokens_per_req, vocab_size
    )

    out_a = sample_recovered_and_bonus_tokens(
        target_probs, draft_probs, cu_num_logits, idx_mapping, seeds_a, pos
    )
    out_b = sample_recovered_and_bonus_tokens(
        target_probs, draft_probs, cu_num_logits, idx_mapping, seeds_b, pos
    )
    assert not torch.equal(out_a, out_b), (
        "Expected different seeds to produce different token sequences"
    )
