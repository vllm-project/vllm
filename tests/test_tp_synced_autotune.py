# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for VLLM_TRITON_SYNC_AUTOTUNE_TP (CPU-only, no kernel launch)."""

from types import SimpleNamespace

import pytest

from vllm.triton_utils import HAS_TRITON, triton

if not HAS_TRITON:
    pytest.skip("triton not available", allow_module_level=True)

from vllm.triton_utils import tp_synced_autotune  # noqa: E402


@pytest.fixture
def patched_bench(monkeypatch: pytest.MonkeyPatch):
    Autotuner = triton.runtime.autotuner.Autotuner
    original = Autotuner._bench
    monkeypatch.setattr(tp_synced_autotune, "_installed", False)
    monkeypatch.setattr(Autotuner, "_bench", lambda self, *a, **k: self.local_timing)
    tp_synced_autotune.install()
    yield Autotuner
    Autotuner._bench = original


def test_outside_tp_keeps_local_timing(patched_bench, monkeypatch):
    monkeypatch.setattr(tp_synced_autotune, "_sync_group", lambda: None)
    fake = SimpleNamespace(local_timing=[3.0, 2.0, 4.0])
    assert patched_bench._bench(fake, config=None) == [3.0, 2.0, 4.0]


def test_inside_tp_takes_rank0_timing(patched_bench, monkeypatch):
    import torch.distributed as dist

    group = SimpleNamespace(ranks=[0, 1], cpu_group=object(), world_size=2)
    monkeypatch.setattr(tp_synced_autotune, "_sync_group", lambda: group)
    sent = []

    def fake_broadcast(box, src, group):
        sent.append((list(box), src))
        box[0] = [1.0, 1.0, 1.0]  # what rank 0 measured

    monkeypatch.setattr(dist, "broadcast_object_list", fake_broadcast)
    fake = SimpleNamespace(local_timing=[9.0, 9.0, 9.0])
    assert patched_bench._bench(fake, config=None) == [1.0, 1.0, 1.0]
    assert sent == [([[9.0, 9.0, 9.0]], 0)]


def test_install_is_idempotent(patched_bench):
    first = patched_bench._bench
    tp_synced_autotune.install()
    assert patched_bench._bench is first
