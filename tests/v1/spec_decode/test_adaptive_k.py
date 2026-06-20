# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Adaptive K scheduler cost model."""

from __future__ import annotations

from tests.v1.core.utils import create_scheduler


def _setup_adaptive_k(scheduler, alphas, prev_k=5):
    """Configure scheduler internals for adaptive K testing."""
    scheduler._enable_adaptive_k = True
    scheduler._per_position_ema = list(alphas)
    scheduler._adaptive_k_ema_alpha = 0.3
    scheduler._adaptive_k_c_draft = 0.05
    scheduler._adaptive_k_bs_penalty = 0.0
    scheduler._adaptive_k_min_tokens = 0
    scheduler._cooldown_steps = 4
    scheduler._alpha_prior = 0.75
    scheduler._adaptive_k_cooldown = 0
    scheduler._previous_adaptive_k = prev_k


def test_defaults_to_num_spec_tokens_when_disabled():
    s = create_scheduler(num_speculative_tokens=10)
    s._per_position_ema = None
    s._enable_adaptive_k = False
    assert s._compute_adaptive_k() == s.num_spec_tokens


def test_disabled_returns_max():
    s = create_scheduler(num_speculative_tokens=10)
    _setup_adaptive_k(s, [1.0] * 10)
    s._enable_adaptive_k = False
    assert s._compute_adaptive_k() == 10


def test_perfect_acceptance_selects_max_k():
    s = create_scheduler(num_speculative_tokens=10)
    _setup_adaptive_k(s, [1.0] * 10)
    assert s._compute_adaptive_k() == 10


def test_zero_acceptance_selects_k0():
    s = create_scheduler(num_speculative_tokens=10)
    _setup_adaptive_k(s, [0.01] * 10)
    result = s._compute_adaptive_k()
    assert result == 0


def test_cooldown_prevents_change():
    s = create_scheduler(num_speculative_tokens=10)
    _setup_adaptive_k(s, [0.9] * 10, prev_k=5)
    s._adaptive_k_cooldown = 3
    assert s._compute_adaptive_k() == 5


def test_k0_selected_when_utility_below_one():
    s = create_scheduler(num_speculative_tokens=10)
    _setup_adaptive_k(s, [0.2] * 10, prev_k=5)
    s._adaptive_k_c_draft = 0.3  # high cost → utility < 1
    assert s._compute_adaptive_k() == 0


def test_ema_not_initialized_uses_prior():
    s = create_scheduler(num_speculative_tokens=5)
    _setup_adaptive_k(s, [])
    s._per_position_ema = None
    s._alpha_prior = 0.75
    assert s._compute_adaptive_k() >= 1


def test_intermediate_k_for_declining_alphas():
    s = create_scheduler(num_speculative_tokens=10)
    _setup_adaptive_k(s, [0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01])
    k = s._compute_adaptive_k()
    assert 1 <= k <= 10
