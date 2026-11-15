# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Workers must agree on enable_extensible_kv_cache before branching."""

import pytest
from torch.distributed import ReduceOp

from vllm.v1.worker import gpu_worker


class _FakeGroup:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class _FakeWorldGroup:
    def __init__(self, size: int) -> None:
        self.cpu_group = _FakeGroup(size)


def _patch_world(monkeypatch, size: int, reductions: list[int]):
    monkeypatch.setattr(gpu_worker, "get_world_group", lambda: _FakeWorldGroup(size))
    results = iter(reductions)
    seen: list[ReduceOp] = []

    def fake_all_reduce(group, value, op=ReduceOp.MIN):
        seen.append(op)
        return next(results)

    monkeypatch.setattr(gpu_worker, "all_reduce_int", fake_all_reduce)
    return seen


def test_single_rank_returns_without_reducing(monkeypatch):
    seen = _patch_world(monkeypatch, size=1, reductions=[])
    gpu_worker._require_extensible_kv_cache_agreement(True)
    assert seen == []


def test_agreement_passes(monkeypatch):
    seen = _patch_world(monkeypatch, size=4, reductions=[1, 1])
    gpu_worker._require_extensible_kv_cache_agreement(True)
    assert seen == [ReduceOp.MIN, ReduceOp.MAX]


def test_disabled_everywhere_passes(monkeypatch):
    _patch_world(monkeypatch, size=4, reductions=[0, 0])
    gpu_worker._require_extensible_kv_cache_agreement(False)


def test_disagreement_raises_with_both_values(monkeypatch):
    _patch_world(monkeypatch, size=4, reductions=[0, 1])
    with pytest.raises(RuntimeError, match="differs across workers"):
        gpu_worker._require_extensible_kv_cache_agreement(True)
