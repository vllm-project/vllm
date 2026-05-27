# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for Quest backend tests.

The vLLM v1 attention selector caches its lookup. Every test that exercises
backend selection must run with a clean cache and a clean env.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_attention_backend_env(monkeypatch):
    """Always start each Quest test with VLLM_ATTENTION_BACKEND unset.

    Tests that need the env set should set it explicitly inside the test body.
    """
    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)


@pytest.fixture(autouse=True)
def _clear_selector_cache():
    """Drop any previously cached selector decisions."""
    try:
        from vllm.v1.attention.selector import _cached_get_attn_backend
    except ImportError:
        yield
        return
    _cached_get_attn_backend.cache_clear()
    yield
    _cached_get_attn_backend.cache_clear()


@pytest.fixture
def quest_env(monkeypatch):
    """Set VLLM_ATTENTION_BACKEND to point at our backend class."""
    monkeypatch.setenv(
        "VLLM_ATTENTION_BACKEND",
        "vllm.v1.attention.backends.quest.backend.QuestSparseOffloadBackend",
    )
    try:
        from vllm.v1.attention.selector import _cached_get_attn_backend
        _cached_get_attn_backend.cache_clear()
    except ImportError:
        pass
    yield
