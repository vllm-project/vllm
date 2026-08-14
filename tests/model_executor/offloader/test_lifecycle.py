# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch.nn as nn

from vllm.compilation import cuda_graph as compilation_cuda_graph
from vllm.compilation import wrapper as compilation_wrapper
from vllm.model_executor.offloader import base as offloader_base
from vllm.model_executor.offloader import prefetch as prefetch_module
from vllm.model_executor.offloader.base import NoopOffloader
from vllm.model_executor.offloader.prefetch import (
    PrefetchOffloader,
    PrefetchTransferStats,
)
from vllm.v1.worker.gpu import cudagraph_utils as v1_cudagraph_utils
from vllm.v1.worker.gpu_worker import _reset_offloader_after_weight_wake


@dataclass(frozen=True)
class _FakeRuntimeUnit:
    unit_idx: int
    slot_idx: int = 0


class _FakeRuntime:
    def __init__(self):
        self.reset_count = 0
        self.begin_prefetch_calls = []
        self.mark_prefetch_started_calls = []
        self.units = (_FakeRuntimeUnit(0), _FakeRuntimeUnit(1))

    def reset(self) -> None:
        self.reset_count += 1

    def initial_prefetches(self):
        return self.units

    def begin_prefetch(self, unit_idx: int):
        self.begin_prefetch_calls.append(unit_idx)
        return None

    def mark_prefetch_started(self, unit_idx: int, *, in_capture: bool) -> None:
        self.mark_prefetch_started_calls.append((unit_idx, in_capture))


class _FakeModuleOffloader:
    def __init__(self, in_capture: bool):
        self.in_capture = in_capture
        self.reset_count = 0
        self.start_count = 0

    def reset_runtime_tracking(self) -> None:
        self.reset_count += 1

    def start_onload_to_static(
        self,
        *,
        allow_paced_chunking: bool = False,
    ) -> bool:
        self.start_count += 1
        return self.in_capture


class _RecordingOffloader:
    def __init__(self):
        self.reset_count = 0

    def reset_runtime_state(self) -> None:
        self.reset_count += 1


class _FakePostInitModuleOffloader:
    def __init__(
        self,
        *,
        offloaded_bytes: int,
        direct_buffer_bytes: int,
        uses_slab_buffers: bool,
        uses_storage_group_fallback: bool,
    ):
        self.offloaded_bytes = offloaded_bytes
        self.direct_buffer_bytes = direct_buffer_bytes
        self.uses_slab_buffers = uses_slab_buffers
        self.uses_storage_group_fallback = uses_storage_group_fallback
        self.storage_group_infos = (object(),) if uses_storage_group_fallback else ()
        self.assigned_slots: list[int] = []
        self.synced = False
        self.post_inited = False
        self.device = "cuda"

    def sync_cpu_storage(self) -> None:
        self.synced = True

    def get_param_infos(self):
        return [object()]

    def assign_buffer_slot(
        self, buffer_pool, storage_group_pool, slot_idx: int
    ) -> None:
        self.assigned_slots.append(slot_idx)

    def post_init(self) -> None:
        self.post_inited = True


class _ToyRoutedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


class _ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class _ToyStreamingMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)
        self.experts = _ToyRoutedExperts()
        self.shared_expert = nn.Linear(4, 4, bias=False)


class _ToyStreamingLayer(nn.Module):
    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx
        self.self_attn = _ToyAttention()
        self.mlp = _ToyStreamingMlp()
        self.self_attn.idx = idx
        self.mlp.idx = idx
        self.mlp.experts.idx = idx


def _make_streaming_prefetch_offloader(
    *,
    group_size: int,
    num_in_group: int,
    selectors: set[str],
) -> PrefetchOffloader:
    offloader = PrefetchOffloader.__new__(PrefetchOffloader)
    offloader.group_size = group_size
    offloader.num_in_group = num_in_group
    offloader.prefetch_step = 1
    offloader.offload_params = set()
    offloader.offload_selectors = selectors
    offloader.mode = "cpu"
    offloader.copy_stream = object()
    offloader.tail_copy_scheduler = SimpleNamespace()
    offloader.module_offloaders = []
    offloader.runtime = None
    offloader.transfer_stats = PrefetchTransferStats()
    offloader._hook_module_forward = lambda unit_idx, module: None
    return offloader


def _record_module_offloads(monkeypatch):
    offload_calls = []

    class FakeModuleOffloader:
        def __init__(self, *, module, whitelist_param_names=(), **kwargs):
            self.layer_idx = kwargs.get("layer_idx")
            offload_calls.append(
                SimpleNamespace(
                    module=module,
                    module_idx=getattr(module, "idx", None),
                    param_names=tuple(whitelist_param_names),
                    layer_idx=self.layer_idx,
                )
            )

    monkeypatch.setattr(prefetch_module, "_ModuleOffloader", FakeModuleOffloader)
    return offload_calls


def test_prefetch_wrap_modules_offloads_streaming_during_generation(monkeypatch):
    events: list[str] = []

    class FakeModuleOffloader:
        def __init__(self, *, module, **kwargs):
            events.append(f"offload-{module.idx}")

    monkeypatch.setattr(prefetch_module, "_ModuleOffloader", FakeModuleOffloader)
    offloader = _make_streaming_prefetch_offloader(
        group_size=1,
        num_in_group=1,
        selectors={"routed_experts"},
    )

    def modules():
        for idx in range(2):
            events.append(f"construct-{idx}")
            yield _ToyStreamingLayer(idx)

    offloader.wrap_modules(modules())

    assert events == ["construct-0", "offload-0", "construct-1", "offload-1"]


def test_prefetch_wrap_modules_preserves_group_and_selector_semantics(monkeypatch):
    offload_calls = _record_module_offloads(monkeypatch)
    offloader = _make_streaming_prefetch_offloader(
        group_size=2,
        num_in_group=1,
        selectors={"routed_experts"},
    )
    layers = [_ToyStreamingLayer(idx) for idx in range(4)]

    modules = offloader.wrap_modules(iter(layers))

    assert [module.idx for module in modules] == [0, 1, 2, 3]
    assert [(call.module_idx, call.layer_idx) for call in offload_calls] == [
        (1, 0),
        (3, 1),
    ]
    assert [call.module for call in offload_calls] == [
        layers[1].mlp.experts,
        layers[3].mlp.experts,
    ]
    assert all(
        call.param_names
        == (
            "gate_up_proj.weight",
            "down_proj.weight",
        )
        for call in offload_calls
    )


def test_prefetch_wrap_modules_retargets_routed_expert_unit(monkeypatch):
    offload_calls = _record_module_offloads(monkeypatch)
    offloader = _make_streaming_prefetch_offloader(
        group_size=1,
        num_in_group=1,
        selectors={"routed_experts"},
    )
    hooked: list[tuple[int, nn.Module]] = []
    offloader._hook_module_forward = lambda unit_idx, module: hooked.append(
        (unit_idx, module)
    )
    layer = _ToyStreamingLayer(0)

    offloader.wrap_modules(iter([layer]))

    assert len(offload_calls) == 1
    assert offload_calls[0].module is layer.mlp.experts
    assert offload_calls[0].param_names == (
        "gate_up_proj.weight",
        "down_proj.weight",
    )
    assert hooked == [(0, layer.mlp.experts)]


@pytest.mark.parametrize(
    ("selector", "target_module_name", "expected_param_names"),
    [
        (
            "attention",
            "self_attn",
            ("qkv_proj.weight", "o_proj.weight"),
        ),
        (
            "dense_mlp",
            "mlp",
            ("gate_up_proj.weight", "down_proj.weight"),
        ),
        (
            "shared_experts",
            "mlp.shared_expert",
            ("weight",),
        ),
    ],
)
def test_prefetch_wrap_modules_retargets_single_selector_unit(
    monkeypatch,
    selector: str,
    target_module_name: str,
    expected_param_names: tuple[str, ...],
):
    offload_calls = _record_module_offloads(monkeypatch)
    offloader = _make_streaming_prefetch_offloader(
        group_size=1,
        num_in_group=1,
        selectors={selector},
    )
    hooked: list[tuple[int, nn.Module]] = []
    offloader._hook_module_forward = lambda unit_idx, module: hooked.append(
        (unit_idx, module)
    )
    layer = _ToyStreamingLayer(0)
    target_module = layer.get_submodule(target_module_name)

    offloader.wrap_modules(iter([layer]))

    assert len(offload_calls) == 1
    assert offload_calls[0].module is target_module
    assert offload_calls[0].param_names == expected_param_names
    assert hooked == [(0, target_module)]


def test_prefetch_wrap_modules_keeps_mixed_selection_on_layer(monkeypatch):
    offload_calls = _record_module_offloads(monkeypatch)
    offloader = _make_streaming_prefetch_offloader(
        group_size=1,
        num_in_group=1,
        selectors={"routed_experts"},
    )
    offloader.offload_params = {"shared_expert"}
    layer = _ToyStreamingLayer(0)

    offloader.wrap_modules(iter([layer]))

    assert len(offload_calls) == 1
    assert offload_calls[0].module is layer
    assert offload_calls[0].param_names == (
        "mlp.experts.gate_up_proj.weight",
        "mlp.experts.down_proj.weight",
        "mlp.shared_expert.weight",
    )


def test_prefetch_wrap_modules_keeps_mixed_selectors_on_layer(monkeypatch):
    offload_calls = _record_module_offloads(monkeypatch)
    offloader = _make_streaming_prefetch_offloader(
        group_size=1,
        num_in_group=1,
        selectors={"attention", "dense_mlp"},
    )
    layer = _ToyStreamingLayer(0)

    offloader.wrap_modules(iter([layer]))

    assert len(offload_calls) == 1
    assert offload_calls[0].module is layer
    assert offload_calls[0].param_names == (
        "self_attn.qkv_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_up_proj.weight",
        "mlp.down_proj.weight",
    )


def test_prefetch_wrap_modules_assigns_runtime_unit_indices_to_hooks(monkeypatch):
    offload_calls = _record_module_offloads(monkeypatch)
    offloader = _make_streaming_prefetch_offloader(
        group_size=2,
        num_in_group=1,
        selectors={"routed_experts"},
    )
    layers = [_ToyStreamingLayer(idx) for idx in range(4)]
    hooked: list[tuple[int, nn.Module]] = []
    offloader._hook_module_forward = lambda unit_idx, module: hooked.append(
        (unit_idx, module)
    )

    offloader.wrap_modules(iter(layers))

    offloaded_layer_indices = [call.module_idx for call in offload_calls]
    assert offloaded_layer_indices == [1, 3]
    assert [unit.unit_idx for unit in offloader.runtime.units] == [0, 1]
    module_layer_indices = [
        module_offloader.layer_idx for module_offloader in offloader.module_offloaders
    ]
    assert module_layer_indices == [0, 1]
    assert hooked == [(0, layers[1].mlp.experts), (1, layers[3].mlp.experts)]


def test_base_offloader_reports_no_static_runtime_buffers_by_default():
    assert NoopOffloader().static_runtime_buffer_bytes == 0


def test_prefetch_offloader_post_init_records_static_runtime_buffers(monkeypatch):
    class FakeStaticBufferPool:
        total_bytes = 4096

        def __init__(self, *, module_param_infos, slot_capacity, device):
            pass

    class FakeStorageGroupBufferPool:
        total_bytes = 2048

        def __init__(self, *, module_storage_group_infos, slot_capacity, device):
            pass

    monkeypatch.setattr(prefetch_module, "StaticBufferPool", FakeStaticBufferPool)
    monkeypatch.setattr(
        prefetch_module, "StorageGroupBufferPool", FakeStorageGroupBufferPool
    )

    first = _FakePostInitModuleOffloader(
        offloaded_bytes=100,
        direct_buffer_bytes=512,
        uses_slab_buffers=True,
        uses_storage_group_fallback=True,
    )
    second = _FakePostInitModuleOffloader(
        offloaded_bytes=200,
        direct_buffer_bytes=256,
        uses_slab_buffers=False,
        uses_storage_group_fallback=False,
    )
    offloader = PrefetchOffloader.__new__(PrefetchOffloader)
    offloader.module_offloaders = [first, second]
    offloader.runtime = SimpleNamespace(
        units=(_FakeRuntimeUnit(0, slot_idx=3), _FakeRuntimeUnit(1, slot_idx=4))
    )
    offloader.group_size = 2
    offloader.num_in_group = 1
    offloader.prefetch_step = 5
    offloader.mode = "cpu"
    offloader.total_offloaded_bytes = 0
    offloader._start_initial_prefetches = lambda: None

    offloader.post_init()

    assert offloader.static_runtime_buffer_bytes == 4096 + 2048 + 512 + 256
    assert offloader.total_offloaded_bytes == 300
    assert first.synced and second.synced
    assert first.post_inited and second.post_inited
    assert first.assigned_slots == [3]
    assert second.assigned_slots == [4]


def test_prefetch_offloader_post_init_keeps_zero_buffers_without_modules():
    offloader = PrefetchOffloader.__new__(PrefetchOffloader)
    offloader.module_offloaders = []

    offloader.post_init()

    assert offloader.static_runtime_buffer_bytes == 0


def test_prefetch_offloader_reset_runtime_state_restarts_initial_prefetches():
    runtime = _FakeRuntime()
    module_offloaders = [
        _FakeModuleOffloader(in_capture=False),
        _FakeModuleOffloader(in_capture=True),
    ]
    offloader = PrefetchOffloader.__new__(PrefetchOffloader)
    offloader.runtime = runtime
    offloader.module_offloaders = module_offloaders
    offloader.transfer_stats = PrefetchTransferStats(h2d_bytes=128, copy_count=2)
    sync_calls = []
    offloader.sync_prev_onload = lambda: sync_calls.append("sync")

    offloader.reset_runtime_state()

    assert sync_calls == ["sync"]
    assert runtime.reset_count == 1
    assert offloader.transfer_stats.snapshot()["h2d_bytes"] == 0
    assert [module.reset_count for module in module_offloaders] == [1, 1]
    assert [module.start_count for module in module_offloaders] == [1, 1]
    assert runtime.begin_prefetch_calls == [0, 1]
    assert runtime.mark_prefetch_started_calls == [(0, False), (1, True)]


@pytest.mark.parametrize(
    ("tags", "expected_reset_count"),
    [
        (None, 1),
        (["weights"], 1),
        (["kv_cache"], 0),
    ],
)
def test_weight_wake_resets_prefetch_runtime_state_for_weight_tags(
    monkeypatch,
    tags: list[str] | None,
    expected_reset_count: int,
):
    offloader = _RecordingOffloader()
    monkeypatch.setattr(offloader_base, "get_offloader", lambda: offloader)

    _reset_offloader_after_weight_wake(tags)

    assert offloader.reset_count == expected_reset_count


@pytest.mark.parametrize(
    ("module", "reset_fn"),
    [
        (
            compilation_cuda_graph,
            compilation_cuda_graph._reset_offloader_for_cudagraph_capture,
        ),
        (
            v1_cudagraph_utils,
            v1_cudagraph_utils._reset_offloader_for_cudagraph_capture,
        ),
        (
            offloader_base,
            compilation_wrapper._reset_offloader_for_recompile,
        ),
    ],
)
def test_compile_boundaries_reset_prefetch_runtime_state(
    monkeypatch,
    module,
    reset_fn,
):
    offloader = _RecordingOffloader()
    monkeypatch.setattr(module, "get_offloader", lambda: offloader)

    reset_fn()

    assert offloader.reset_count == 1


def test_model_reload_replaces_global_offloader_instance(monkeypatch):
    previous = offloader_base.get_offloader()
    first = _RecordingOffloader()
    second = _RecordingOffloader()

    offloader_base.set_offloader(first)
    assert offloader_base.get_offloader() is first

    offloader_base.set_offloader(second)
    assert offloader_base.get_offloader() is second
    assert offloader_base.get_offloader() is not first

    offloader_base.set_offloader(previous)
