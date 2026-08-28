# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any, cast

import torch

import vllm.v1.worker.gpu.attn_utils as attn_utils
import vllm.v1.worker.gpu_model_runner as gpu_model_runner
from vllm.v1.worker.gpu_worker import Worker


class _AllocationScope(AbstractContextManager):
    def __init__(self) -> None:
        self.active = False

    def __enter__(self):
        assert not self.active
        self.active = True
        return self

    def __exit__(self, *args: Any) -> None:
        assert self.active
        self.active = False


def test_mrv2_kv_pool_only_wraps_backing_allocation(monkeypatch) -> None:
    scope = _AllocationScope()
    kv_caches = {"layer": torch.empty(0)}

    def allocate(*args, **kwargs):
        assert scope.active
        return kv_caches

    def bind(*args, **kwargs):
        assert not scope.active

    monkeypatch.setattr(attn_utils, "allocate_kv_cache", allocate)
    monkeypatch.setattr(attn_utils, "bind_kv_cache", bind)
    monkeypatch.setattr(attn_utils, "get_shared_kv_cache_layers", lambda config: {})

    config = SimpleNamespace(
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: None),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="test")),
    )
    result = attn_utils.init_kv_cache(
        [],
        {},
        object(),
        torch.device("cpu"),
        [],
        config,
        kv_cache_allocation_context=scope,
    )

    assert result is kv_caches
    assert not scope.active


def test_mrv1_kv_pool_only_wraps_backing_allocation(monkeypatch) -> None:
    scope = _AllocationScope()
    kv_caches = {"layer": torch.empty(0)}

    def allocate(*args, **kwargs):
        assert scope.active
        return kv_caches

    def bind(*args, **kwargs):
        assert not scope.active

    monkeypatch.setattr(gpu_model_runner, "allocate_kv_cache", allocate)
    monkeypatch.setattr(gpu_model_runner, "bind_kv_cache", bind)

    runner = SimpleNamespace(
        device=torch.device("cpu"),
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: None),
        shared_kv_cache_layers={},
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="test")),
        compilation_config=SimpleNamespace(static_forward_context={}),
        kv_caches=[],
    )
    result = gpu_model_runner.GPUModelRunner.initialize_kv_cache_tensors(
        runner,
        object(),
        [],
        kv_cache_allocation_context=scope,
    )

    assert result is kv_caches
    assert not scope.active


def test_kv_wake_does_not_run_model_runner_recovery() -> None:
    model = torch.nn.Module()
    model.register_buffer("_k_scale", torch.tensor(0.5))
    model.register_buffer("_v_scale", torch.tensor(0.25))

    class Runner:
        def __init__(self) -> None:
            self.model = model
            self.layout_tensors = tuple(torch.tensor([i]) for i in range(5))
            self.recovery_calls = 0

        def post_kv_cache_wake_up(self) -> None:
            self.recovery_calls += 1
            self.model.get_buffer("_k_scale").fill_(1.0)
            self.model.get_buffer("_v_scale").fill_(1.0)
            self.layout_tensors = tuple(torch.tensor([i]) for i in range(5))

    runner = Runner()
    worker = cast(
        Worker,
        SimpleNamespace(
            _get_sleep_mode_backend=lambda: SimpleNamespace(resume=lambda tags: None),
            _sleep_saved_buffers={},
            _sleep_saved_draft_buffers={},
            model_runner=runner,
            synchronize_device=lambda: None,
        ),
    )
    layout_tensors = runner.layout_tensors
    layout_ptrs = tuple(t.data_ptr() for t in layout_tensors)

    Worker.wake_up(worker, tags=["kv_cache"])

    assert runner.recovery_calls == 0
    assert model.get_buffer("_k_scale").item() == 0.5
    assert model.get_buffer("_v_scale").item() == 0.25
    assert all(
        actual is expected
        for actual, expected in zip(runner.layout_tensors, layout_tensors, strict=True)
    )
    assert tuple(t.data_ptr() for t in runner.layout_tensors) == layout_ptrs
