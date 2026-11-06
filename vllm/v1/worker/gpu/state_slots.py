# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded session views and the state-write completion contract."""

from collections.abc import Mapping
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    PersistentStateSpec,
    StateCompletion,
    StateSlotHandle,
)
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    KVConnectorOutput,
    ModelRunnerOutput,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class StateSlotViews:
    """Expose only the current generation's bounded slot to a model."""

    def __init__(self, spec: PersistentStateSpec, tensor: torch.Tensor):
        if (
            tensor.dtype not in (torch.int8, torch.uint8)
            or tensor.ndim != 1
            or not tensor.is_contiguous()
            or tensor.numel() % spec.page_size_bytes
        ):
            raise ValueError("Invalid persistent state backing storage")
        self.spec = spec
        self.capacity = tensor.numel() // spec.page_size_bytes
        self._tensor = tensor
        self._views: dict[str, torch.Tensor] = {}
        for descriptor in spec.descriptors:
            if (
                tensor.data_ptr() + descriptor.offset_bytes
            ) % descriptor.alignment_bytes:
                raise ValueError("Persistent state backing storage is misaligned")
            stride = 1
            strides = []
            for dimension in reversed(descriptor.shape):
                strides.append(stride)
                stride *= dimension
            self._views[descriptor.name] = torch.as_strided(
                tensor[descriptor.offset_bytes :].view(descriptor.dtype),
                size=(self.capacity, *descriptor.shape),
                stride=(
                    spec.page_size_bytes // descriptor.element_size_bytes,
                    *reversed(strides),
                ),
            )
        self._epoch: str | None = None
        self._owners: dict[int, tuple[str, StateSlotHandle]] = {}

    def bind(self, request_id: str, handle: StateSlotHandle) -> None:
        if not 0 < handle.slot < self.capacity or handle.generation <= 0:
            raise ValueError("Invalid state slot handle")
        if self._epoch is not None and handle.engine_epoch != self._epoch:
            raise ValueError("Stale state engine epoch")
        previous = self._owners.get(handle.slot)
        if previous is not None:
            if previous == (request_id, handle):
                return
            if handle.generation <= previous[1].generation:
                raise ValueError("Stale state slot generation")
        start = handle.slot * self.spec.page_size_bytes
        self._tensor[start : start + self.spec.page_size_bytes].zero_()
        for descriptor in self.spec.descriptors:
            target = self._views[descriptor.name][handle.slot]
            initialized = descriptor.initializer(
                descriptor.shape, dtype=descriptor.dtype, device=target.device
            )
            if initialized.shape != descriptor.shape:
                raise ValueError("Persistent state initializer returned wrong shape")
            target.copy_(initialized)
        self._epoch = handle.engine_epoch
        self._owners[handle.slot] = (request_id, handle)

    def validate_handle(self, request_id: str, handle: StateSlotHandle) -> None:
        if self._owners.get(handle.slot) != (request_id, handle):
            raise ValueError("Stale state slot handle")

    def view(self, request_id: str, handle: StateSlotHandle) -> dict[str, torch.Tensor]:
        self.validate_handle(request_id, handle)
        return {name: tensor[handle.slot] for name, tensor in self._views.items()}


@dataclass(frozen=True)
class StateSlotRows:
    """Step-scoped resident pools for trusted model execution.

    Consumers must validate again before commit and write only the bound slots
    on the runner stream or a registered writing stream. Retained tensors are
    not revocable capabilities.
    """

    _bindings: "StateSlotBindings"
    _step_id: object
    layer: str
    step_seq: int
    request_ids: tuple[str, ...]
    handles: tuple[StateSlotHandle, ...]

    @property
    def slot_indices_cpu(self) -> tuple[int, ...]:
        return tuple(handle.slot for handle in self.handles)

    def validate_current(self) -> None:
        slots = self._bindings
        completion = slots._completion
        if (
            completion is None
            or slots._cleaning_up
            or slots._step_id is not self._step_id
            or completion.step_seq != self.step_seq
        ):
            raise ValueError("Stale state step access")
        slots._validate_rows(self.layer, self.request_ids, self.handles)

    @property
    def spec(self) -> PersistentStateSpec:
        self.validate_current()
        return self._bindings.views[self.layer].spec

    @property
    def component_pools(self) -> Mapping[str, torch.Tensor]:
        self.validate_current()
        return MappingProxyType(self._bindings.views[self.layer]._views)

    @property
    def views(self) -> Mapping[str, torch.Tensor]:
        return self.component_pools


class StateSlotBindings:
    def __init__(
        self, groups: list[KVCacheGroupSpec], tensors: dict[str, torch.Tensor]
    ):
        self.views = {
            group.layer_names[0]: StateSlotViews(
                group.kv_cache_spec, tensors[group.layer_names[0]]
            )
            for group in groups
            if isinstance(group.kv_cache_spec, PersistentStateSpec)
        }
        self._streams: list[torch.cuda.Stream] = []
        self._completion: StateCompletion | None = None
        self._main_stream: torch.cuda.Stream | None = None
        self._cleaning_up = False
        self._step_id: object | None = None

    def _validate_rows(
        self,
        layer: str,
        request_ids: tuple[str, ...],
        handles: tuple[StateSlotHandle, ...],
    ) -> None:
        if self._completion is None or self._cleaning_up:
            raise ValueError("No live state step is active")
        if layer not in self.views:
            raise ValueError("Unknown state group")
        if (
            not request_ids
            or len(request_ids) != len(handles)
            or len(set(request_ids)) != len(request_ids)
            or len({handle.slot for handle in handles}) != len(handles)
        ):
            raise ValueError("State rows must have distinct requests and slots")
        views = self.views[layer]
        for request_id, handle in zip(request_ids, handles):
            if (
                not 0 < handle.slot < views.capacity
                or handle.generation <= 0
                or self._completion.handles.get(request_id, {}).get(layer) != handle
            ):
                raise ValueError("Stale or invalid state row handle/order")
            views.validate_handle(request_id, handle)

    def bind_rows(
        self,
        layer: str,
        ordered_request_ids: tuple[str, ...],
        expected_handles: tuple[StateSlotHandle, ...],
    ) -> StateSlotRows:
        """Validate full handles before exposing resident component pools."""
        request_ids, handles = tuple(ordered_request_ids), tuple(expected_handles)
        self._validate_rows(layer, request_ids, handles)
        assert self._completion is not None and self._step_id is not None
        return StateSlotRows(
            self, self._step_id, layer, self._completion.step_seq, request_ids, handles
        )

    def register_writing_stream(self, stream: torch.cuda.Stream) -> None:
        """Register a stream before enqueueing writes outside the runner stream."""
        if stream not in self._streams:
            self._streams.append(stream)
            if self._main_stream is not None:
                stream.wait_stream(self._main_stream)

    def begin_step(
        self, scheduled: "SchedulerOutput", main_stream: torch.cuda.Stream
    ) -> None:
        for stream in self._streams:
            main_stream.wait_stream(stream)
        cleanup = scheduled.state_cleanup_handles or {}
        live = scheduled.state_slot_handles or {}
        for request_id, handles in cleanup.items():
            if handles.keys() != self.views.keys():
                raise ValueError("Cleanup bindings do not match the state groups")
            for name, handle in handles.items():
                self.views[name].view(request_id, handle)
        for request_id, handles in live.items():
            if handles.keys() != self.views.keys():
                raise ValueError(
                    "State bindings do not match the declared state groups"
                )
            for name, handle in handles.items():
                self.views[name].bind(request_id, handle)
        for stream in self._streams:
            stream.wait_stream(main_stream)
        self._main_stream = main_stream
        self._step_id = object()
        self._completion = StateCompletion(
            scheduled.state_step_seq,
            deepcopy(live),
            deepcopy(cleanup),
        )
        self._cleaning_up = True

    def activate_requests(self) -> None:
        self._cleaning_up = False

    def get_handle(self, request_id: str, layer_name: str) -> StateSlotHandle:
        """Resolve a handle only while its request is scheduled on this worker."""
        if self._completion is None:
            raise ValueError("No state step is active")
        if self._cleaning_up:
            return self._completion.cleanup_handles[request_id][layer_name]
        return self._completion.handles[request_id][layer_name]

    def fail_requests(self, request_ids: set[str]) -> None:
        """Report model failures with the same fence and handle identity as writes."""
        if self._completion is None or self._cleaning_up:
            raise ValueError("No live state step is active")
        if not request_ids <= self._completion.handles.keys():
            raise ValueError("Failed request is outside the active state step")
        self._completion = replace(
            self._completion,
            failed_req_ids=self._completion.failed_req_ids.union(request_ids),
        )

    def finish_step(
        self,
        output: AsyncModelRunnerOutput | ModelRunnerOutput,
        main_stream: torch.cuda.Stream,
    ) -> "StateCompletionOutput":
        if self._completion is None:
            raise RuntimeError("State completion requires a scheduled step")
        for stream in self._streams:
            main_stream.wait_stream(stream)
        event = torch.cuda.Event(blocking=True)
        event.record(main_stream)
        result = StateCompletionOutput(output, event, self._completion)
        self._completion = None
        self._step_id = None
        self._main_stream = None
        return result


class StateCompletionOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        output: AsyncModelRunnerOutput | ModelRunnerOutput,
        event: torch.cuda.Event,
        completion: StateCompletion,
    ):
        self.output = output
        self.event = event
        self.completion = completion

    def get_output(self) -> ModelRunnerOutput:
        self.event.synchronize()
        output = (
            self.output.get_output()
            if isinstance(self.output, AsyncModelRunnerOutput)
            else self.output
        )
        output = copy(output)
        output.state_completion = self.completion
        return output

    def get_kv_connector_output(self) -> KVConnectorOutput | None:
        if isinstance(self.output, AsyncModelRunnerOutput):
            return self.output.get_kv_connector_output()
        return self.output.kv_connector_output
