# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform extension interfaces for expert parallel load balancing."""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Protocol

import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.config import ParallelConfig
    from vllm.distributed.parallel_state import GroupCoordinator

    from .eplb_communicator import EplbCommunicator
    from .weight_utils import EplbExpertWeight, EplbLayerWeights


class EplbDeviceEvent(Protocol):
    """Device event operations required by EPLB."""

    def record(self, stream: Any | None = None) -> None: ...

    def wait(self, stream: Any | None = None) -> None: ...

    def synchronize(self) -> None: ...

    def elapsed_time(self, end_event: EplbDeviceEvent) -> float: ...


class EplbDeviceRuntime(ABC):
    """Device operations used by synchronous and asynchronous EPLB."""

    @abstractmethod
    def get_device_index(self, device: torch.device) -> int:
        pass

    @abstractmethod
    def set_device(self, device_index: int) -> None:
        pass

    @abstractmethod
    def create_stream(self, device_index: int) -> Any:
        pass

    @abstractmethod
    def stream_context(self, stream: Any) -> AbstractContextManager[Any]:
        pass

    @abstractmethod
    def create_event(self, enable_timing: bool = False) -> EplbDeviceEvent:
        pass

    @abstractmethod
    def synchronize(self, stream: Any | None = None) -> None:
        pass


EplbMapAndRecord = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ],
    torch.Tensor,
]


class EplbPlatformBackend(ABC):
    """Hardware-specific operations used by the upstream EPLB state machine."""

    @classmethod
    @abstractmethod
    def resolve_communicator(cls, parallel_config: ParallelConfig) -> str:
        pass

    @classmethod
    @abstractmethod
    def validate_config(cls, parallel_config: ParallelConfig) -> None:
        pass

    @abstractmethod
    def map_and_record(
        self,
        topk_ids: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        expert_load_view: torch.Tensor,
        record_enabled: torch.Tensor,
        num_unpadded_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def create_communicator(
        self,
        group_coordinator: GroupCoordinator,
        expert_weights: Sequence[EplbLayerWeights],
        expert_buffer: Sequence[EplbExpertWeight],
    ) -> EplbCommunicator:
        pass

    @property
    @abstractmethod
    def device_runtime(self) -> EplbDeviceRuntime:
        pass


def _platform_name() -> str:
    return getattr(current_platform, "device_name", type(current_platform).__name__)


@functools.cache
def _resolve_eplb_backend_cls(
    qualname: str,
    platform_name: str,
) -> type[EplbPlatformBackend]:
    try:
        backend_cls = resolve_obj_by_qualname(qualname)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load EPLB Platform Backend {qualname!r} for "
            f"Platform {platform_name!r}."
        ) from exc
    if not isinstance(backend_cls, type) or not issubclass(
        backend_cls, EplbPlatformBackend
    ):
        raise TypeError(
            f"EPLB backend {qualname!r} for Platform {platform_name!r} must "
            "subclass EplbPlatformBackend."
        )
    return backend_cls


def resolve_eplb_platform_backend_cls() -> type[EplbPlatformBackend] | None:
    """Resolve and cache the current Platform's EPLB Backend class."""
    qualname = current_platform.get_eplb_backend_cls()
    if qualname is None:
        return None
    return _resolve_eplb_backend_cls(qualname, _platform_name())


@functools.cache
def _create_eplb_backend(
    qualname: str,
    platform_name: str,
) -> EplbPlatformBackend:
    return _resolve_eplb_backend_cls(qualname, platform_name)()


def get_eplb_platform_backend() -> EplbPlatformBackend:
    """Return the process-cached EPLB Backend for the current Platform."""
    qualname = current_platform.get_eplb_backend_cls()
    if qualname is None:
        raise RuntimeError(
            f"Platform {_platform_name()!r} does not provide an EPLB Backend."
        )
    return _create_eplb_backend(qualname, _platform_name())
