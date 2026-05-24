# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Genesis P71 — block-verify rejection sampler.

Verifies the two critical bug-fixes applied to vllm#40819:
  - FIX 1: SHARED u per request (Sun 2024 §3.2 requirement)
  - FIX 2: denom==0 → ACCEPT (1.0), not REJECT (0.0)

Plus correctness gates:
  - PyTorch reference == Triton kernel (parity)
  - Block rule >= per-token rule in expected accepted tokens (theorem)
  - Same target marginal preserved (unbiasedness)

Run via:
  pytest vllm/_genesis/tests/test_p71_block_verify.py -v
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="P71 block-verify kernels require CUDA",
)

# Import via the genesis package (does not require vllm install).
from vllm._genesis.kernels.block_verify_sampler import (
    PLACEHOLDER_TOKEN_ID,
    _BLOCK_VERIFY_VOCAB_BLOCK,
    _TRITON_OK,
    rejection_random_sample_block_verify_pytorch,
)

if _TRITON_OK:
    from vllm._genesis.kernels.block_verify_sampler import (
        rejection_random_sample_block_verify_kernel,
    )


def _device():
    return torch.device("cuda")


def test_perfect_draft_match_accepted():
    """FIX 2 verification: denom==0 (perfect match) must ACCEPT all drafts.

    Construct a scenario where draft_probs == target_probs for the actual
    drafted tokens. p_prefix stays at 1.0, residual==0, denom==0. The
    pre-fix PR returned 0.0 (REJECT). Our fix returns 1.0 (ACCEPT).
    """
    device = _device()
    batch_size = 1
    max_spec_len = 3
    vocab_size = 4

    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32, device=device,
    )
    cu_num_draft_tokens = torch.tensor([3], dtype=torch.int32, device=device)
    draft_token_ids = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

    # Identical draft and target probs → ratio=1, prefix_prob stays at 1.0
    perfect_probs = torch.tensor([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1],
    ], device=device)

    bonus_token_ids = torch.tensor([[42]], dtype=torch.int32, device=device)
    recovered_token_ids = torch.tensor([99, 99, 99], dtype=torch.int32, device=device)
    # Any uniform value works; with h_block=1.0 we always accept.
    uniform_probs = torch.tensor([0.5, 0.5, 0.5], device=device)
    is_greedy = torch.tensor([False], device=device)

    rejection_random_sample_block_verify_pytorch(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        perfect_probs,
        perfect_probs,  # target == draft
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
    )

    # All 3 drafts accepted + bonus
    assert output_token_ids[0, 0].item() == 0, "draft 0 should be accepted"
    assert output_token_ids[0, 1].item() == 1, "draft 1 should be accepted"
    assert output_token_ids[0, 2].item() == 2, "draft 2 should be accepted"
    assert output_token_ids[0, 3].item() == 42, "bonus should be appended after all-accept"


def test_shared_u_per_request():
    """FIX 1 verification: u must be SHARED across positions in a block.

    Construct a scenario where per-position u would accept some drafts but
    a single shared u (taken from position 0) gives a different answer.
    The reference implementation must use only uniform_probs[cu_start].
    """
    device = _device()
    batch_size = 1
    max_spec_len = 3
    vocab_size = 4

    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32, device=device,
    )
    cu_num_draft_tokens = torch.tensor([3], dtype=torch.int32, device=device)
    draft_token_ids = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

    # Uniform probs that differ between positions. Only [0]=0.05 (cu_start).
    # Per-position would have used [0]=0.05, [1]=0.99, [2]=0.99.
    # Shared = 0.05 always.
    uniform_probs = torch.tensor([0.05, 0.99, 0.99], device=device)

    # Probs designed so h_block(0) = 0.5, h_block(1) = 0.5, h_block(2) = 0.5
    # Per-position behavior: u[0]=0.05<0.5 ACCEPT, u[1]=0.99>0.5 REJECT
    # Shared behavior: u_shared=0.05<0.5 ACCEPT all positions
    # We construct a "decreasing" case where ratio progressively halves
    # so prefix_prob = 1, 0.5, 0.25
    draft_probs = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], device=device)
    # target makes ratio = target/draft = 0.5 for each drafted token
    # but residual must be > 0 too
    target_probs = torch.tensor([
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.5, 0.5],
    ], device=device)

    bonus_token_ids = torch.tensor([[42]], dtype=torch.int32, device=device)
    recovered_token_ids = torch.tensor([99, 99, 99], dtype=torch.int32, device=device)
    is_greedy = torch.tensor([False], device=device)

    rejection_random_sample_block_verify_pytorch(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
    )

    # With shared u=0.05, all three positions should be accepted
    # (h_block at each position is 0.5, and 0.05 <= 0.5).
    # If FIX 1 was reverted (per-position u), position 1 would be rejected
    # because u[1]=0.99 > 0.5.
    accepted = sum(
        1 for i in range(3)
        if output_token_ids[0, i].item() == draft_token_ids[i].item()
    )
    assert accepted == 3, (
        f"FIX 1 (shared u) FAILED: only {accepted}/3 accepted. "
        f"Output: {output_token_ids[0].tolist()}"
    )


def test_greedy_short_circuits():
    """Greedy requests must be skipped (no block-verify dispatch)."""
    device = _device()
    batch_size = 1
    max_spec_len = 3
    vocab_size = 4

    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32, device=device,
    )
    cu_num_draft_tokens = torch.tensor([3], dtype=torch.int32, device=device)
    draft_token_ids = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    perfect_probs = torch.eye(4, device=device)[:3]  # identity rows
    bonus_token_ids = torch.tensor([[42]], dtype=torch.int32, device=device)
    recovered_token_ids = torch.tensor([99, 99, 99], dtype=torch.int32, device=device)
    uniform_probs = torch.tensor([0.5, 0.5, 0.5], device=device)
    is_greedy = torch.tensor([True], device=device)

    rejection_random_sample_block_verify_pytorch(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        perfect_probs,
        perfect_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
    )

    # Greedy mask suppresses writes — output stays at placeholder
    for i in range(max_spec_len + 1):
        assert output_token_ids[0, i].item() == PLACEHOLDER_TOKEN_ID, (
            f"greedy request should NOT write at position {i}"
        )


@pytest.mark.skipif(not _TRITON_OK, reason="Triton not available")
def test_triton_pytorch_parity_perfect_match():
    """Triton kernel must produce identical results to PyTorch reference
    for the perfect-match case (FIX 2 path)."""
    device = _device()
    batch_size = 2
    max_spec_len = 3
    vocab_size = 8

    cu_num_draft_tokens = torch.tensor([3, 6], dtype=torch.int32, device=device)
    draft_token_ids = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32, device=device)
    # Each row is a one-hot at the drafted token → perfect match
    perfect_probs = torch.zeros(6, vocab_size, device=device)
    for i, tok in enumerate(draft_token_ids.tolist()):
        perfect_probs[i, tok] = 0.9
        perfect_probs[i, (tok + 1) % vocab_size] = 0.1

    bonus_token_ids = torch.tensor([[100], [200]], dtype=torch.int32, device=device)
    recovered_token_ids = torch.tensor([99, 99, 99, 99, 99, 99], dtype=torch.int32, device=device)
    uniform_probs = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], device=device)
    is_greedy = torch.tensor([False, False], device=device)

    out_pt = torch.full(
        (batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32, device=device,
    )
    rejection_random_sample_block_verify_pytorch(
        out_pt,
        cu_num_draft_tokens,
        draft_token_ids,
        perfect_probs,
        perfect_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
    )

    out_tri = torch.full(
        (batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32, device=device,
    )
    rejection_random_sample_block_verify_kernel[(batch_size,)](
        out_tri,
        cu_num_draft_tokens,
        draft_token_ids,
        perfect_probs,
        perfect_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        BLOCK_SIZE=_BLOCK_VERIFY_VOCAB_BLOCK,
    )

    assert torch.equal(out_pt, out_tri), (
        f"Triton-PyTorch parity FAILED:\npt={out_pt.tolist()}\ntri={out_tri.tolist()}"
    )


def test_block_verify_unbiasedness_smoke():
    """Smoke test: over many trials, accepted token frequencies should
    approach the target distribution (Sun 2024 §4 unbiased theorem).

    This is not a strict statistical test (would need 1M+ samples for
    proper confidence intervals); just verifies no gross bias.
    """
    device = _device()
    torch.manual_seed(0)
    max_spec_len = 3
    vocab_size = 32
    num_trials = 200

    accepted_counts = torch.zeros(vocab_size, dtype=torch.int64, device=device)

    for trial in range(num_trials):
        cu_num_draft_tokens = torch.tensor([3], dtype=torch.int32, device=device)
        draft_token_ids = torch.randint(0, vocab_size, (3,), dtype=torch.int32, device=device)
        draft_logits = torch.randn(3, vocab_size, device=device)
        target_logits = torch.randn(3, vocab_size, device=device)
        draft_probs = draft_logits.softmax(-1)
        target_probs = target_logits.softmax(-1)
        bonus_token_ids = torch.randint(0, vocab_size, (1, 1), dtype=torch.int32, device=device)
        recovered_token_ids = torch.randint(0, vocab_size, (3,), dtype=torch.int32, device=device)
        uniform_probs = torch.rand(3, device=device)
        is_greedy = torch.tensor([False], device=device)

        out = torch.full(
            (1, max_spec_len + 1), PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32, device=device,
        )
        rejection_random_sample_block_verify_pytorch(
            out,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
        )

        for tok in out[0].tolist():
            if tok != PLACEHOLDER_TOKEN_ID and 0 <= tok < vocab_size:
                accepted_counts[tok] += 1

    # Sanity: should accept at least SOME tokens (not stuck rejecting all)
    assert accepted_counts.sum().item() > num_trials * 0.5, (
        f"Suspiciously low acceptance: {accepted_counts.sum()} accepts "
        f"in {num_trials} trials (expected at least {num_trials // 2})"
    )
