# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The prefill sparse-indexer logits budget: the env var wins whenever it is
set; unset, the default is 64 MiB on integrated (unified-memory) GPUs and
512 MiB everywhere else. The env var is exercised through the environment
only (``vllm.envs`` resolves it lazily), never by assigning module attributes."""

import pytest

from vllm.v1.attention import sparse_indexer_budget as budget

MIB = 1024 * 1024
ENV = "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"


@pytest.fixture
def integrated(monkeypatch):
    monkeypatch.setattr(budget.current_platform, "is_integrated_gpu", lambda: True)


@pytest.fixture
def discrete(monkeypatch):
    monkeypatch.setattr(budget.current_platform, "is_integrated_gpu", lambda: False)


def test_unset_on_integrated_gpu_is_64mib(monkeypatch, integrated):
    monkeypatch.delenv(ENV, raising=False)
    assert budget.sparse_indexer_max_logits_bytes() == 64 * MIB


def test_unset_on_discrete_gpu_is_512mib(monkeypatch, discrete):
    monkeypatch.delenv(ENV, raising=False)
    assert budget.sparse_indexer_max_logits_bytes() == 512 * MIB


def test_explicit_env_wins_on_integrated_gpu(monkeypatch, integrated):
    monkeypatch.setenv(ENV, "256")
    assert budget.sparse_indexer_max_logits_bytes() == 256 * MIB


def test_explicit_default_value_still_wins_on_integrated_gpu(monkeypatch, integrated):
    # Setting the env var to the documented default is an explicit choice.
    monkeypatch.setenv(ENV, "512")
    assert budget.sparse_indexer_max_logits_bytes() == 512 * MIB
