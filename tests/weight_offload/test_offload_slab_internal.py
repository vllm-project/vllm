# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from vllm.model_executor.offloader.prefetch import (
    ParamInfo,
    PrefetchTransferStats,
    _ModuleOffloader,
)
from vllm.model_executor.offloader.prefetch_runtime_buffers import (
    RuntimeBufferPlan,
    StaticBufferPool,
    StorageGroupBufferPool,
    build_runtime_buffer_plan,
    build_storage_group_infos,
    view_storage_group_tensor,
)
from vllm.model_executor.offloader.slab import (
    build_slab_layout,
    storage_size_in_bytes,
    view_slab_tensor,
)


class _FakeCudaStream:
    def __init__(self):
        self.waited_events = []
        self.recorded_events = []

    def wait_event(self, event):
        self.waited_events.append(event)

    def record_event(self, event):
        self.recorded_events.append(event)


class _FakeCudaEvent:
    def __init__(self, *args, **kwargs):
        self.recorded_streams = []

    def record(self, stream=None):
        self.recorded_streams.append(stream)


class _FakeParamOffloader:
    def __init__(self):
        self.freshness_checks = 0
        self.sync_marks = 0

    def ensure_cpu_master_freshness(self):
        self.freshness_checks += 1

    def mark_cpu_master_synced(self):
        self.sync_marks += 1


def test_slab_layout_tracks_offsets_and_dtype_alignment():
    tensors = [
        ("a", torch.empty(4, dtype=torch.float16)),
        ("b", torch.empty(3, dtype=torch.float32)),
    ]

    layout = build_slab_layout(tensors, alignment_bytes=16)

    assert layout.specs[0].offset_bytes == 0
    assert layout.specs[1].offset_bytes == 16
    assert layout.specs[0].dtype == torch.float16
    assert layout.specs[1].dtype == torch.float32
    assert layout.total_bytes == 28


def test_slab_layout_preserves_shape_and_stride_in_views():
    tensor = torch.empty_strided((2, 3), (4, 1), dtype=torch.float16)
    layout = build_slab_layout([("experts", tensor)])
    slab = torch.arange(layout.total_bytes, dtype=torch.uint8)

    view = view_slab_tensor(slab, layout.specs[0])

    assert tuple(view.shape) == (2, 3)
    assert tuple(view.stride()) == (4, 1)
    assert view.dtype == torch.float16


def test_slab_layout_uses_storage_span_not_numel_for_strided_tensors():
    tensor = torch.empty_strided((2, 3), (4, 1), dtype=torch.float16)

    assert tensor.numel() * tensor.element_size() == 12
    assert (
        storage_size_in_bytes(
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
        )
        == 14
    )


def test_static_buffer_pool_reuses_one_slab_per_module_layout():
    module_infos = [
        [
            ParamInfo("a", (4,), (1,), torch.float16),
            ParamInfo("b", (2,), (1,), torch.float32),
        ],
        [
            ParamInfo("a", (4,), (1,), torch.float16),
            ParamInfo("b", (2,), (1,), torch.float32),
        ],
    ]

    pool = StaticBufferPool(
        module_param_infos=module_infos,
        slot_capacity=2,
        device=torch.device("cpu"),
    )

    _, first_slab, first_views = pool.get_slab_assignment(module_infos[0], slot_idx=0)
    _, second_slab, second_views = pool.get_slab_assignment(module_infos[1], slot_idx=0)

    assert pool.total_bytes == 48
    assert first_slab.data_ptr() == second_slab.data_ptr()
    assert first_views["a"].data_ptr() == second_views["a"].data_ptr()
    assert first_views["b"].data_ptr() == second_views["b"].data_ptr()


def test_static_buffer_pool_returns_stride_preserving_views():
    module_infos = [
        [
            ParamInfo("packed", (2, 3), (4, 1), torch.float16),
        ]
    ]

    pool = StaticBufferPool(
        module_param_infos=module_infos,
        slot_capacity=1,
        device=torch.device("cpu"),
    )

    _, _, views = pool.get_slab_assignment(module_infos[0], slot_idx=0)

    assert tuple(views["packed"].shape) == (2, 3)
    assert tuple(views["packed"].stride()) == (4, 1)
    assert views["packed"].dtype == torch.float16


def test_runtime_buffer_plan_keeps_noncontiguous_zero_offset_tensor_packable():
    packed = torch.empty_strided((2, 3), (4, 1), dtype=torch.float16)

    plan = build_runtime_buffer_plan([("packed", packed)])

    assert isinstance(plan, RuntimeBufferPlan)
    assert plan.slab_param_names == ("packed",)
    assert plan.storage_group_infos == ()
    assert plan.fallback_reasons == ()


def test_runtime_buffer_plan_supports_mixed_slab_and_storage_group_paths():
    dense = torch.empty(4, dtype=torch.float16)
    base = torch.arange(8, dtype=torch.float32)
    view = base[2:6]

    plan = build_runtime_buffer_plan([("dense", dense), ("base", base), ("view", view)])

    assert plan.slab_param_names == ("dense",)
    assert len(plan.storage_group_infos) == 1
    assert tuple(spec.name for spec in plan.storage_group_infos[0].view_specs) == (
        "base",
        "view",
    )
    assert plan.fallback_reasons == ("base, view share underlying storage",)


def test_runtime_buffer_plan_routes_non_strided_quant_tensor_to_direct_fallback():
    dense = torch.empty(4, dtype=torch.float16)
    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([1, 2], dtype=torch.int32)
    packed_sparse = torch.sparse_coo_tensor(indices, values, (2, 2))

    plan = build_runtime_buffer_plan(
        [("dense", dense), ("packed_sparse", packed_sparse)]
    )

    assert plan.slab_param_names == ("dense",)
    assert plan.storage_group_infos == ()
    assert plan.direct_param_names == ("packed_sparse",)
    assert plan.fallback_reasons == (
        "packed_sparse uses non-strided layout torch.sparse_coo",
    )


def test_storage_group_infos_preserve_aliasing_views():
    base = torch.arange(8, dtype=torch.float32)
    view = base[2:6]

    groups = build_storage_group_infos(
        [
            ("base", base),
            ("view", view),
        ]
    )

    assert len(groups) == 1
    group = groups[0]
    assert group.storage_numel == 8
    assert group.dtype == torch.float32
    assert tuple(spec.name for spec in group.view_specs) == ("base", "view")
    assert tuple(spec.storage_offset for spec in group.view_specs) == (0, 2)

    runtime_storage = torch.arange(group.storage_numel, dtype=group.dtype)
    runtime_base = view_storage_group_tensor(runtime_storage, group.view_specs[0])
    runtime_view = view_storage_group_tensor(runtime_storage, group.view_specs[1])

    assert torch.equal(runtime_base, torch.arange(8, dtype=torch.float32))
    assert torch.equal(runtime_view, torch.arange(2, 6, dtype=torch.float32))

    runtime_view.add_(10)

    assert torch.equal(runtime_storage, torch.tensor([0, 1, 12, 13, 14, 15, 6, 7]))


def test_storage_group_buffer_pool_reuses_one_assignment_per_layout():
    base_a = torch.arange(8, dtype=torch.float32)
    view_a = base_a[2:6]
    base_b = torch.arange(8, dtype=torch.float32)
    view_b = base_b[2:6]

    module_infos = [
        build_storage_group_infos([("base", base_a), ("view", view_a)]),
        build_storage_group_infos([("base", base_b), ("view", view_b)]),
    ]

    pool = StorageGroupBufferPool(
        module_storage_group_infos=module_infos,
        slot_capacity=2,
        device=torch.device("cpu"),
    )

    first_buffers, first_views = pool.get_assignment(module_infos[0], slot_idx=0)
    second_buffers, second_views = pool.get_assignment(module_infos[1], slot_idx=0)

    assert pool.total_bytes == 64
    assert first_buffers[0].data_ptr() == second_buffers[0].data_ptr()
    assert first_views["base"].data_ptr() == second_views["base"].data_ptr()
    assert first_views["view"].data_ptr() == second_views["view"].data_ptr()


def test_storage_group_buffer_pool_views_preserve_aliasing():
    base = torch.arange(8, dtype=torch.float32)
    view = base[2:6]
    module_info = build_storage_group_infos([("base", base), ("view", view)])

    pool = StorageGroupBufferPool(
        module_storage_group_infos=[module_info],
        slot_capacity=1,
        device=torch.device("cpu"),
    )

    buffers, views = pool.get_assignment(module_info, slot_idx=0)

    assert len(buffers) == 1
    buffers[0].zero_()
    views["view"].add_(10)

    assert torch.equal(buffers[0], torch.tensor([0, 0, 10, 10, 10, 10, 0, 0]))
    assert torch.equal(views["base"], torch.tensor([0, 0, 10, 10, 10, 10, 0, 0]))


def test_module_onload_copies_mixed_slab_and_storage_group_buffers(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_onload.should_pin_memory",
        lambda: False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_onload.torch.cuda.Event",
        _FakeCudaEvent,
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_onload.torch.cuda.current_stream",
        lambda: _FakeCudaStream(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_onload.torch.cuda.is_current_stream_capturing",
        lambda: False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_onload.torch.cuda.stream",
        lambda stream: nullcontext(),
    )

    base = torch.arange(8, dtype=torch.float32)
    view = base[2:6]
    storage_group = build_storage_group_infos([("base", base), ("view", view)])[0]
    gpu_storage_group = torch.zeros(
        storage_group.storage_numel,
        dtype=storage_group.dtype,
    )

    offloader = _ModuleOffloader.__new__(_ModuleOffloader)
    offloader.layer_idx = 0
    offloader._buffer_slot_idx = 0
    offloader._tail_copy_scheduler = SimpleNamespace(submit=lambda *a, **kw: None)
    offloader._slab_param_names = ("dense",)
    offloader._storage_group_infos = (storage_group,)
    offloader._storage_group_buffers = [gpu_storage_group]
    offloader._direct_param_names = ()
    offloader._direct_buffers = {}
    offloader._buffer_pool = object()
    offloader._cpu_slab = torch.arange(8, dtype=torch.uint8)
    offloader._gpu_slab = torch.zeros_like(offloader._cpu_slab)
    offloader._use_slab_copy = True
    offloader.copy_stream = _FakeCudaStream()
    offloader.transfer_stats = PrefetchTransferStats()
    offloader._copy_done_event = _FakeCudaEvent()
    offloader._event_valid_for_eager = False
    offloader._copy_thread_error = None
    offloader._copy_done_event_recorded = SimpleNamespace(
        wait=lambda: None, clear=lambda: None, set=lambda: None
    )
    offloader._param_offloaders = {
        "dense": _FakeParamOffloader(),
        "base": _FakeParamOffloader(),
        "view": _FakeParamOffloader(),
    }

    assert offloader.start_onload_to_static() is False

    assert torch.equal(offloader._gpu_slab, offloader._cpu_slab)
    assert torch.equal(gpu_storage_group, base)
