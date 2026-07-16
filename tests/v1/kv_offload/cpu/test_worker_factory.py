# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType
from unittest.mock import MagicMock

from vllm.v1.kv_offload.cpu import worker_factory


def test_create_cpu_offloading_worker_uses_ascend_worker(monkeypatch):
    created = object()

    class FakeAscendWorker:
        def __new__(cls, **kwargs):
            assert kwargs["block_size_factor"] == 8
            assert kwargs["num_cpu_blocks"] == 16
            return created

    module = ModuleType("vllm.v1.kv_offload.cpu.npu_worker")
    module.AscendCPUOffloadingWorker = FakeAscendWorker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(worker_factory, "is_ascend_platform", lambda: True)

    worker = worker_factory.create_cpu_offloading_worker(
        kv_caches=MagicMock(),
        block_size_factor=8,
        num_cpu_blocks=16,
    )

    assert worker is created


def test_create_cpu_offloading_worker_uses_default_worker(monkeypatch):
    created = object()
    monkeypatch.setattr(worker_factory, "is_ascend_platform", lambda: False)
    monkeypatch.setattr(worker_factory, "CPUOffloadingWorker", lambda **kwargs: created)

    worker = worker_factory.create_cpu_offloading_worker(
        kv_caches=MagicMock(),
        block_size_factor=1,
        num_cpu_blocks=4,
    )

    assert worker is created
