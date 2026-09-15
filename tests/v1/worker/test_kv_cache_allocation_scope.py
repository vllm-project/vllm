# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import vllm.v1.hisparse.binding as hisparse_binding
import vllm.v1.worker.gpu.attn_utils as attn_utils
import vllm.v1.worker.gpu_model_runner as gpu_model_runner
from vllm.v1.hisparse.layout import _build_hisparse_kv_cache_tensors
from vllm.v1.hisparse.runtime import HiSparseCacheHandle, HiSparseRuntime
from vllm.v1.kv_cache_interface import (
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheGroupSpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.gpu_worker import Worker


@pytest.mark.cpu_test
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.uint8])
def test_hisparse_layer_cache_writes_are_independent(dtype):
    """Layers packed into one allocation must retain distinct resident/hot KV."""
    num_blocks, block_size, row_width = 3, 2, 8
    names = ["layer.0", "layer.1"]
    resident_names = [name + ".hisparse_resident" for name in names]
    hot_names = [name + ".hisparse_hot" for name in names]
    page_size = block_size * row_width * dtype.itemsize
    groups = [
        KVCacheGroupSpec(
            resident_names,
            HiSparseResidentSpec(block_size=block_size, page_size=page_size),
        ),
        KVCacheGroupSpec(
            hot_names,
            HiSparseHotSpec(
                block_size=block_size, page_size=page_size, blocks_per_request=1
            ),
        ),
    ]
    # Reserve a leading page to exercise both the tensor and layer offsets.
    block_stride = 3 * page_size
    raw = torch.zeros(num_blocks * block_stride, dtype=torch.int8)
    tensors = _build_hisparse_kv_cache_tensors(
        groups, num_blocks, raw.numel(), KVCacheLayout.BLNHC, block_stride
    )
    for tensor in tensors:
        tensor.offset = page_size
    host = torch.zeros(2, num_blocks, block_size, row_width, dtype=dtype)
    handles = []
    for _ in names:
        # Binding requires no CUDA replacement state.
        runtime = object.__new__(HiSparseRuntime)
        runtime.kv_dtype, runtime.row_width = dtype, row_width
        handles.append(HiSparseCacheHandle(runtime))
    kv_caches = dict(zip(names, host))
    kv_caches.update((name, raw) for name in resident_names + hot_names)
    hisparse_binding.bind_hisparse_kv_caches(
        forward_context={
            name: SimpleNamespace(hisparse_cache=handle)
            for name, handle in zip(names, handles)
        },
        kv_cache_config=SimpleNamespace(
            kv_cache_tensors=tensors,
            kv_cache_groups=groups,
            num_blocks=num_blocks,
            host_group_ids=[0],
        ),
        kv_caches=kv_caches,
        block_tables=SimpleNamespace(
            input_block_tables=[torch.zeros(1, 1, dtype=torch.int32)] * 2,
            slot_mappings=[torch.zeros(1, dtype=torch.int64)] * 2,
        ),
        host_pool=SimpleNamespace(registered=host, shared_region=None),
    )
    for i, handle in enumerate(handles):
        handle.view.cache[1].fill_(i + 1)
        handle.runtime.hot.cache[2].fill_(i + 3)
    for i, handle in enumerate(handles):
        assert torch.all(handle.view.cache[1] == i + 1)
        assert torch.all(handle.runtime.hot.cache[2] == i + 3)
    assert not raw.view(num_blocks, block_stride)[:, :page_size].count_nonzero()


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


@pytest.mark.parametrize("hisparse", [False, True])
def test_mrv2_kv_pool_only_wraps_backing_allocation(monkeypatch, hisparse) -> None:
    scope = _AllocationScope()
    kv_caches = {"layer": torch.empty(0)}

    def allocate(*args, **kwargs):
        assert scope.active
        return kv_caches

    def bind(*args, **kwargs):
        assert not scope.active

    monkeypatch.setattr(attn_utils, "allocate_kv_cache", allocate)
    monkeypatch.setattr(attn_utils, "bind_kv_cache_to_layers", bind)
    monkeypatch.setattr(attn_utils, "get_shared_kv_cache_layers", lambda config: {})

    hisparse_bindings = []
    if hisparse:
        host_pool = SimpleNamespace()

        def bind_hisparse(**kwargs):
            hisparse_bindings.append(kwargs)
            assert scope.active
            assert kwargs["kv_caches"] is kv_caches
            assert kwargs["host_pool"] is host_pool
            return [
                SimpleNamespace(
                    view=SimpleNamespace(cache=torch.empty(1, 1, 1), block_size=1),
                    runtime=SimpleNamespace(),
                )
            ]

        monkeypatch.setattr(
            hisparse_binding,
            "HiSparseHostPool",
            lambda vllm_config, kv_cache_config: host_pool,
        )
        monkeypatch.setattr(hisparse_binding, "allocate_hisparse_kv_caches", allocate)
        monkeypatch.setattr(hisparse_binding, "bind_hisparse_kv_caches", bind_hisparse)

    config = SimpleNamespace(
        attention_config=SimpleNamespace(
            hisparse_config=object() if hisparse else None
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=1),
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: None),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="test")),
    )
    result = attn_utils.init_kv_cache(
        {},
        SimpleNamespace(kv_cache_groups=[]),
        torch.device("cpu"),
        [],
        config,
        kv_cache_allocation_context=scope,
        block_tables=SimpleNamespace(),
    )

    assert len(hisparse_bindings) == int(hisparse)
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
        SimpleNamespace(kv_cache_groups=[]),
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
            sleep_mode_backend=SimpleNamespace(resume=lambda tags: None),
            _sleep_saved_buffers={},
            _sleep_saved_draft_buffers={},
            model_runner=runner,
            synchronize_device=lambda: None,
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(enable_nccl_comm_suspend=False)
            ),
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
