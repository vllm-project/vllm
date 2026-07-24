# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import vllm.config.parallel as parallel_config_module
import vllm.distributed.eplb.eplb_communicator as communicator_module
import vllm.distributed.eplb.platform_backend as platform_backend_module
from vllm.config.parallel import EPLBConfig, ParallelConfig
from vllm.distributed.eplb.eplb_communicator import (
    EplbCommunicator,
    create_eplb_communicator,
)
from vllm.distributed.eplb.eplb_utils import CrossThreadDeviceEvent
from vllm.distributed.eplb.platform_backend import (
    EplbDeviceEvent,
    EplbDeviceRuntime,
    EplbPlatformBackend,
    get_eplb_platform_backend,
    resolve_eplb_platform_backend_cls,
)
from vllm.distributed.eplb.rebalance_execute import (
    move_from_buffer,
    move_to_buffer,
)
from vllm.distributed.eplb.weight_utils import (
    EplbExpertWeight,
    empty_eplb_weight_like,
    get_eplb_expert_tensor,
    validate_eplb_weight,
)
from vllm.platforms.interface import Platform, PlatformEnum


class FakeEvent:
    def __init__(self, events: list[str], enable_timing: bool = False) -> None:
        self.events = events
        self.enable_timing = enable_timing

    def record(self, stream: Any | None = None) -> None:
        self.events.append("record")

    def wait(self, stream: Any | None = None) -> None:
        self.events.append("wait")

    def synchronize(self) -> None:
        self.events.append("synchronize")

    def elapsed_time(self, end_event: EplbDeviceEvent) -> float:
        return 1.0


class FakeDeviceRuntime(EplbDeviceRuntime):
    def __init__(self) -> None:
        self.events: list[str] = []

    def get_device_index(self, device: torch.device) -> int:
        return device.index or 0

    def set_device(self, device_index: int) -> None:
        self.events.append(f"set_device:{device_index}")

    def create_stream(self, device_index: int) -> object:
        self.events.append(f"create_stream:{device_index}")
        return object()

    def stream_context(self, stream: Any):
        self.events.append("stream_context")
        return nullcontext()

    def create_event(self, enable_timing: bool = False) -> EplbDeviceEvent:
        self.events.append(f"create_event:{enable_timing}")
        return FakeEvent(self.events, enable_timing)

    def synchronize(self, stream: Any | None = None) -> None:
        self.events.append("runtime_synchronize")


class FakeCommunicator(EplbCommunicator):
    def __init__(self) -> None:
        self.sends: list[tuple[list[torch.Tensor], int, int]] = []
        self.recvs: list[tuple[list[torch.Tensor], int, int]] = []
        self.execute_count = 0

    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,
    ) -> None:
        self.sends.append((tensors, dst_rank, expert_id))

    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,
    ) -> None:
        self.recvs.append((tensors, src_rank, expert_id))

    def execute(self) -> None:
        self.execute_count += 1


class FakeEplbBackend(EplbPlatformBackend):
    instance_count = 0
    communicator_count = 0

    def __init__(self) -> None:
        type(self).instance_count += 1
        self._runtime = FakeDeviceRuntime()

    @classmethod
    def resolve_communicator(cls, parallel_config: ParallelConfig) -> str:
        return "platform"

    @classmethod
    def validate_config(cls, parallel_config: ParallelConfig) -> None:
        if parallel_config.eplb_config.use_async and parallel_config.enable_elastic_ep:
            raise ValueError("Fake Backend does not support async elastic EPLB.")

    def map_and_record(
        self,
        topk_ids: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        expert_load_view: torch.Tensor,
        record_enabled: torch.Tensor,
        num_unpadded_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        del logical_replica_count, num_unpadded_tokens
        valid = topk_ids >= 0
        safe_ids = topk_ids.clamp(min=0)
        mapped = logical_to_physical_map[safe_ids, 0]
        mapped = torch.where(valid, mapped, -1)
        if bool(record_enabled):
            expert_load_view.scatter_add_(
                0,
                mapped[valid].flatten(),
                torch.ones_like(mapped[valid].flatten(), dtype=expert_load_view.dtype),
            )
        return mapped

    def create_communicator(
        self,
        group_coordinator,
        expert_weights,
        expert_buffer,
    ) -> EplbCommunicator:
        del group_coordinator, expert_weights, expert_buffer
        type(self).communicator_count += 1
        return FakeCommunicator()

    @property
    def device_runtime(self) -> EplbDeviceRuntime:
        return self._runtime


class FakePlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = "fake"
    device_type = "fake"

    @classmethod
    def get_eplb_backend_cls(cls) -> str | None:
        return f"{__name__}.FakeEplbBackend"


class NoEplbPlatform(FakePlatform):
    device_name = "no-eplb"

    @classmethod
    def get_eplb_backend_cls(cls) -> None:
        return None


class InvalidEplbPlatform(FakePlatform):
    device_name = "invalid-eplb"

    @classmethod
    def get_eplb_backend_cls(cls) -> str:
        return "builtins.object"


class MissingEplbPlatform(FakePlatform):
    device_name = "missing-eplb"

    @classmethod
    def get_eplb_backend_cls(cls) -> str:
        return "missing.eplb.Backend"


@pytest.fixture(autouse=True)
def clear_backend_caches():
    platform_backend_module._resolve_eplb_backend_cls.cache_clear()
    platform_backend_module._create_eplb_backend.cache_clear()
    FakeEplbBackend.instance_count = 0
    FakeEplbBackend.communicator_count = 0
    yield
    platform_backend_module._resolve_eplb_backend_cls.cache_clear()
    platform_backend_module._create_eplb_backend.cache_clear()


def patch_platform(monkeypatch: pytest.MonkeyPatch, platform: Platform) -> None:
    monkeypatch.setattr(platform_backend_module, "current_platform", platform)
    monkeypatch.setattr(parallel_config_module, "current_platform", platform)


def test_backend_resolution_is_process_cached(monkeypatch: pytest.MonkeyPatch):
    patch_platform(monkeypatch, FakePlatform())

    assert resolve_eplb_platform_backend_cls() is FakeEplbBackend
    first = get_eplb_platform_backend()
    second = get_eplb_platform_backend()

    assert first is second
    assert FakeEplbBackend.instance_count == 1


def test_parallel_config_uses_platform_default(monkeypatch: pytest.MonkeyPatch):
    patch_platform(monkeypatch, FakePlatform())

    config = ParallelConfig(
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        enable_eplb=True,
        eplb_config=EPLBConfig(use_async=False),
        distributed_executor_backend="mp",
    )

    assert config.eplb_config.communicator == "platform"


def test_parallel_config_delegates_combination_validation(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_platform(monkeypatch, FakePlatform())

    with pytest.raises(ValueError, match="does not support async elastic EPLB"):
        ParallelConfig(
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            enable_eplb=True,
            enable_elastic_ep=True,
            eplb_config=EPLBConfig(use_async=True),
            distributed_executor_backend="external_launcher",
        )


def test_parallel_config_rejects_missing_backend(monkeypatch: pytest.MonkeyPatch):
    patch_platform(monkeypatch, NoEplbPlatform())

    with pytest.raises(ValueError, match="does not provide an EPLB Platform Backend"):
        ParallelConfig(
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            enable_eplb=True,
            distributed_executor_backend="mp",
        )


def test_backend_resolution_rejects_invalid_type(monkeypatch: pytest.MonkeyPatch):
    patch_platform(monkeypatch, InvalidEplbPlatform())

    with pytest.raises(TypeError, match="must subclass EplbPlatformBackend"):
        resolve_eplb_platform_backend_cls()


def test_backend_resolution_preserves_import_error(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_platform(monkeypatch, MissingEplbPlatform())

    with pytest.raises(
        RuntimeError, match="Failed to load EPLB Platform Backend"
    ) as exc:
        resolve_eplb_platform_backend_cls()
    assert exc.value.__cause__ is not None


def test_platform_communicator_delegates_to_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_platform(monkeypatch, FakePlatform())
    backend = get_eplb_platform_backend()
    weights = [[torch.arange(6).reshape(2, 3)]]
    buffers = [torch.empty_like(weights[0][0])]

    communicator = create_eplb_communicator(
        group_coordinator=SimpleNamespace(),
        backend="platform",
        expert_weights=weights,
        expert_buffer=buffers,
        platform_backend=backend,
    )

    assert isinstance(communicator, FakeCommunicator)
    assert FakeEplbBackend.communicator_count == 1


def test_nixl_rejects_per_expert_sequences_early(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(communicator_module, "has_nixl", lambda: True)
    weights = [[[torch.empty(2, device="meta"), torch.empty(2, device="meta")]]]
    buffers = [[torch.empty(2, device="meta"), torch.empty(2, device="meta")]]

    with pytest.raises(ValueError, match="requires aggregated Tensor weights"):
        create_eplb_communicator(
            group_coordinator=SimpleNamespace(
                cpu_group=object(),
                device_group=object(),
            ),
            backend="nixl",
            expert_weights=weights,
            expert_buffer=buffers,
        )


def test_fake_backend_maps_and_records_load():
    backend = FakeEplbBackend()
    topk_ids = torch.tensor([[0, 1], [2, -1]])
    logical_to_physical = torch.tensor([[2], [0], [1]])
    replica_count = torch.ones(3, dtype=torch.long)
    load = torch.zeros(3, dtype=torch.int32)

    mapped = backend.map_and_record(
        topk_ids,
        logical_to_physical,
        replica_count,
        load,
        torch.tensor(True),
        None,
    )

    torch.testing.assert_close(mapped, torch.tensor([[2, 0], [1, -1]]))
    torch.testing.assert_close(load, torch.ones(3, dtype=torch.int32))


@pytest.mark.parametrize("as_sequence", [False, True])
def test_weight_buffers_preserve_structure_and_storage(as_sequence: bool):
    if as_sequence:
        weight: EplbExpertWeight = [torch.arange(3), torch.arange(3, 6)]
    else:
        weight = torch.arange(6).reshape(2, 3)

    validate_eplb_weight(weight, local_num_experts=2)
    buffer = empty_eplb_weight_like(weight)

    assert isinstance(buffer, torch.Tensor) == isinstance(weight, torch.Tensor)
    for expert_id in range(2):
        src = get_eplb_expert_tensor(weight, expert_id)
        dst = get_eplb_expert_tensor(buffer, expert_id)
        assert src.shape == dst.shape
        assert src.dtype == dst.dtype
        assert src.device == dst.device
        assert src.untyped_storage().data_ptr() != dst.untyped_storage().data_ptr()


@pytest.mark.parametrize(
    "weight, error",
    [
        ([], "must not be empty"),
        ([torch.empty(2)], "unexpected number"),
        ([torch.empty(2), "not-a-tensor"], "must be a torch.Tensor"),
        (
            [torch.empty(2, dtype=torch.float32), torch.empty(2, dtype=torch.float64)],
            "one dtype",
        ),
        ([torch.empty(2), torch.empty(3)], "one shape"),
    ],
)
def test_invalid_weight_sequences_fail_early(weight, error):
    with pytest.raises((TypeError, ValueError), match=error):
        validate_eplb_weight(weight, local_num_experts=2)


def test_weight_sequence_requires_independent_storage():
    shared = torch.empty(2, 3)
    with pytest.raises(ValueError, match="independent tensor storage"):
        validate_eplb_weight([shared[0], shared[1]], local_num_experts=2)


@pytest.mark.parametrize("as_sequence", [False, True])
def test_local_rearrange_supports_weight_sequences(as_sequence: bool):
    if as_sequence:
        weights = [[torch.tensor([1, 2]), torch.tensor([3, 4])]]
    else:
        weights = [torch.tensor([[1, 2], [3, 4]])]
    buffers = [empty_eplb_weight_like(weights[0])]
    communicator = FakeCommunicator()
    old_indices = np.array([0, 1])
    new_indices = np.array([1, 0])
    container_id = id(weights[0])
    tensor_ids = [id(tensor) for tensor in weights[0]] if as_sequence else []

    metadata = move_to_buffer(
        num_local_experts=2,
        old_indices=old_indices,
        new_indices=new_indices,
        expert_weights=weights,
        expert_weights_buffers=buffers,
        transfer_stream=None,
        ep_rank=0,
        communicator=communicator,
    )
    move_from_buffer(
        expert_weights=weights,
        expert_weights_buffers=buffers,
        transfer_metadata=metadata,
        new_indices=new_indices,
        ep_rank=0,
    )

    torch.testing.assert_close(
        get_eplb_expert_tensor(weights[0], 0), torch.tensor([3, 4])
    )
    torch.testing.assert_close(
        get_eplb_expert_tensor(weights[0], 1), torch.tensor([1, 2])
    )
    assert id(weights[0]) == container_id
    if as_sequence:
        assert [id(tensor) for tensor in weights[0]] == tensor_ids


def test_cross_thread_event_waits_for_record():
    calls: list[str] = []
    event = CrossThreadDeviceEvent(lambda: FakeEvent(calls))
    returned = threading.Event()

    def wait_for_record() -> None:
        event.wait()
        returned.set()

    thread = threading.Thread(target=wait_for_record)
    thread.start()
    time.sleep(0.01)
    assert not returned.is_set()

    event.record()
    thread.join(timeout=1)

    assert returned.is_set()
    assert calls == ["record", "wait"]


def test_fake_device_runtime_operations_are_platform_owned():
    runtime = FakeDeviceRuntime()

    runtime.set_device(3)
    stream = runtime.create_stream(3)
    with runtime.stream_context(stream):
        event = runtime.create_event(enable_timing=True)
        event.record(stream)
        event.wait(stream)
    runtime.synchronize(stream)

    assert runtime.events == [
        "set_device:3",
        "create_stream:3",
        "stream_context",
        "create_event:True",
        "record",
        "wait",
        "runtime_synchronize",
    ]
