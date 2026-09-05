# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Mirostat v2 sampling (state holder + SamplingParams validation)."""

import math

import pytest
import torch

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality
from vllm.v1.sample.mirostat_state import MirostatStateHolder


def _batch_update(added=(), removed=(), moved=(), batch_size=8) -> BatchUpdate:
    return BatchUpdate(
        batch_size=batch_size,
        removed=list(removed),
        added=list(added),
        moved=list(moved),
    )


def _mirostat_params(mode=2, tau=5.0, eta=0.1) -> SamplingParams:
    return SamplingParams(
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        mirostat_mode=mode,
        mirostat_tau=tau,
        mirostat_eta=eta,
    )


def _holder(max_num_reqs=8) -> MirostatStateHolder:
    return MirostatStateHolder(max_num_reqs=max_num_reqs, device="cpu")


# ---------------------------------------------------------------------------
# SamplingParams validation
# ---------------------------------------------------------------------------


def test_default_disabled_is_noop():
    sp = SamplingParams()
    assert sp.mirostat_mode == 0
    assert sp.mirostat_tau == 5.0
    assert sp.mirostat_eta == 0.1


def test_valid_mirostat_v2_params():
    sp = _mirostat_params()
    assert sp.mirostat_mode == 2


def test_invalid_mode_rejected():
    with pytest.raises(VLLMValidationError):
        SamplingParams(mirostat_mode=1, temperature=1.0)


def test_mirostat_requires_positive_temperature():
    with pytest.raises(VLLMValidationError):
        SamplingParams(mirostat_mode=2, temperature=0.0)


def test_mirostat_requires_topk_disabled():
    with pytest.raises(VLLMValidationError):
        SamplingParams(mirostat_mode=2, temperature=1.0, top_k=50)


def test_mirostat_requires_topp_disabled():
    with pytest.raises(VLLMValidationError):
        SamplingParams(mirostat_mode=2, temperature=1.0, top_p=0.9)


def test_mirostat_rejects_nonpositive_tau_eta():
    with pytest.raises(VLLMValidationError):
        SamplingParams(mirostat_mode=2, temperature=1.0, mirostat_tau=0.0)
    with pytest.raises(VLLMValidationError):
        SamplingParams(mirostat_mode=2, temperature=1.0, mirostat_eta=0.0)


# ---------------------------------------------------------------------------
# MirostatStateHolder batch bookkeeping
# ---------------------------------------------------------------------------


def test_sync_batch_add_remove():
    holder = _holder()
    assert not holder.has_tracked_requests()

    holder.sync_batch(_batch_update(added=[(0, _mirostat_params(tau=4.0), None, [])]))
    assert holder.has_tracked_requests()
    # mu initialized to 2*tau.
    assert holder.mu[0].item() == pytest.approx(8.0)
    assert holder.tau[0].item() == pytest.approx(4.0)

    holder.sync_batch(_batch_update(removed=[0]))
    assert not holder.has_tracked_requests()
    assert math.isinf(holder.mu[0].item())


def test_sync_batch_non_mirostat_not_tracked():
    holder = _holder()
    holder.sync_batch(
        _batch_update(added=[(0, SamplingParams(temperature=1.0), None, [])])
    )
    assert not holder.has_tracked_requests()
    assert math.isinf(holder.mu[0].item())


def test_sync_batch_move_and_swap():
    holder = _holder()
    holder.sync_batch(
        _batch_update(
            added=[
                (0, _mirostat_params(tau=3.0), None, []),
                (1, _mirostat_params(tau=6.0), None, []),
            ]
        )
    )
    # Swap 0 and 1.
    holder.sync_batch(_batch_update(moved=[(0, 1, MoveDirectionality.SWAP)]))
    assert holder.tau[0].item() == pytest.approx(6.0)
    assert holder.tau[1].item() == pytest.approx(3.0)
    assert holder.has_tracked_requests()

    # Unidirectional move 1 -> 2.
    holder.sync_batch(_batch_update(moved=[(1, 2, MoveDirectionality.UNIDIRECTIONAL)]))
    assert 1 not in holder._active
    assert math.isinf(holder.mu[1].item())
    assert holder.tau[2].item() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Truncation (apply_to_logits) and mu update
# ---------------------------------------------------------------------------


def test_apply_to_logits_truncates_high_surprise_but_keeps_argmax():
    holder = _holder()
    # Small mu -> aggressive truncation, on slot 0 only.
    holder._set_active(0, tau=5.0, eta=0.1)
    holder.mu[0] = 0.5

    logits = torch.tensor(
        [[10.0, 9.9, 1.0, -5.0], [1.0, 2.0, 3.0, 4.0]], dtype=torch.float32
    )
    original_row1 = logits[1].clone()
    out = holder.apply_to_logits(logits)

    # Argmax token of row 0 must survive.
    assert torch.isfinite(out[0, 0])
    # The very low-probability token should be truncated to -inf.
    assert out[0, 3] == float("-inf")
    # Untracked row is untouched.
    assert torch.equal(out[1], original_row1)


def test_apply_to_logits_noop_when_empty():
    holder = _holder()
    logits = torch.randn(3, 16)
    out = holder.apply_to_logits(logits.clone())
    assert torch.equal(out, logits)


def test_update_mu_moves_toward_target():
    holder = _holder()
    tau, eta = 5.0, 0.1
    holder._set_active(0, tau=tau, eta=eta)  # mu = 2*tau = 10

    # Uniform distribution over V tokens -> surprise = log2(V) for any token.
    vocab = 1024
    logits = torch.zeros(1, vocab, dtype=torch.float32)
    sampled = torch.tensor([0])

    expected_surprise = math.log2(vocab)  # 10.0 bits
    mu_before = holder.mu[0].item()
    holder.apply_to_logits(logits)
    holder.update_mu(sampled)
    mu_after = holder.mu[0].item()

    # surprise (10) > tau (5) -> mu should decrease by eta*(surprise - tau).
    assert mu_after == pytest.approx(mu_before - eta * (expected_surprise - tau))
    assert mu_after < mu_before


def test_update_mu_increases_when_confident():
    holder = _holder()
    tau, eta = 5.0, 0.1
    holder._set_active(0, tau=tau, eta=eta)  # mu = 2*tau = 10

    # Highly peaked distribution -> tiny surprise for the top token.
    logits = torch.full((1, 512), -50.0, dtype=torch.float32)
    logits[0, 0] = 50.0
    sampled = torch.tensor([0])

    mu_before = holder.mu[0].item()
    holder.apply_to_logits(logits)
    holder.update_mu(sampled)
    # surprise ~ 0 < tau -> mu increases (loosens truncation, escapes loops).
    assert holder.mu[0].item() > mu_before
