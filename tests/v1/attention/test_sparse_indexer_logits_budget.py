# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sparse-indexer logits budget: the env var wins whenever it is set; when
it is unset the default is 64 MiB on integrated (unified-memory) CUDA devices
and 512 MiB everywhere else."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.mla import indexer as indexer_mod

MIB = 1024 * 1024


@pytest.fixture
def unset_env(monkeypatch):
    monkeypatch.delenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", raising=False)
    monkeypatch.setattr(indexer_mod.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 512)


def _fake_cuda(monkeypatch, is_integrated: bool):
    monkeypatch.setattr(indexer_mod.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _dev: SimpleNamespace(is_integrated=is_integrated),
    )


def test_unset_on_integrated_cuda_is_64mib(monkeypatch, unset_env):
    _fake_cuda(monkeypatch, is_integrated=True)
    assert indexer_mod.sparse_indexer_max_logits_bytes() == 64 * MIB


def test_unset_on_discrete_cuda_is_512mib(monkeypatch, unset_env):
    _fake_cuda(monkeypatch, is_integrated=False)
    assert indexer_mod.sparse_indexer_max_logits_bytes() == 512 * MIB


def test_explicit_env_wins_on_integrated_cuda(monkeypatch, unset_env):
    _fake_cuda(monkeypatch, is_integrated=True)
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "256")
    monkeypatch.setattr(indexer_mod.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 256)
    assert indexer_mod.sparse_indexer_max_logits_bytes() == 256 * MIB


def test_unset_on_non_cuda_is_512mib(monkeypatch, unset_env):
    monkeypatch.setattr(indexer_mod.current_platform, "is_cuda", lambda: False)
    assert indexer_mod.sparse_indexer_max_logits_bytes() == 512 * MIB
