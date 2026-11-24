# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Abstractions for a platform runtime used by the CPU offload worker.

A runtime supplies everything that is platform-specific: capabilities, host
registration, capture relaxation, and the configuration of the copy mechanism
(see `engines/`). It does not implement copies itself. The factory that picks
a runtime by platform lives in `factory.py`.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from vllm.v1.kv_offload.cpu.engines.base import CopyEngine


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    """What a platform runtime can do for the CPU offload worker."""

    # Whether this platform can run the CPU offload worker at all.
    cpu_offload: bool = False
    # Whether the worker may use the shared mmap region for host memory.
    shared_region: bool = False
    # Whether host setup may be deferred and run while the engine serves.
    async_init: bool = False


class DeviceRuntime(ABC):
    """Platform-specific services needed by the CPU offload worker."""

    capabilities: ClassVar[RuntimeCapabilities] = RuntimeCapabilities()

    @abstractmethod
    def create_copy_engine(
        self,
        *,
        device_to_cpu: bool,
        host_memory_is_pinned: bool,
        page_sizes: Sequence[int],
    ) -> CopyEngine:
        """Create the copy engine for one transfer direction."""

    def relax_stream_capture(self) -> None:  # noqa: B027
        """Allow host registration while another thread captures a graph."""

    def register_host(self, ptr: int, num_bytes: int) -> bool:
        """Register a host range for direct device access.

        Returns True when the range was registered and later needs
        `unregister_host`; registration is a bandwidth optimization, so a
        False return only means the range stays pageable.
        """
        return False

    def unregister_host(self, ptr: int) -> None:  # noqa: B027
        """Release a range previously registered by `register_host`."""
