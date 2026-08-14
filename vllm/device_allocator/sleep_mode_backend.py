# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable sleep-mode backends (RFC #34303).

vLLM's sleep/wake-up today is hard-wired to ``CuMemAllocator``: the GPU worker
calls ``allocator.sleep(...)`` / ``allocator.wake_up(...)`` directly. RFC #34303
proposes additional mechanisms for freeing and restoring GPU state - CUDA
process checkpoint, CRIU, durable snapshot/restore - that share the *dispatch*
(``/sleep`` endpoint -> engine -> executor -> worker) but differ in *mechanism*
and in which resources they preserve (NCCL communicators, compiled kernels,
CUDA graphs, survival across process restart).

This module introduces a thin backend abstraction so those mechanisms can be
selected by name without changing the public API. The default ``cumem`` backend
wraps today's ``CuMemAllocator`` path 1:1, so existing users see no behavior
change. The factory mirrors ``KVConnectorFactory`` and lets third-party
backends register through a ``vllm.general_plugins`` entry point at import time.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig

logger = init_logger(__name__)

SleepModeState = Literal["RUNNING", "SUSPENDED", "RESUMING"]


class SleepModeBackend(ABC):
    """Interface for a mechanism that frees and restores GPU state.

    A backend owns the *mechanism* of suspend/resume. The dispatch path
    (``/sleep`` endpoint -> engine -> executor -> worker) is shared across all
    backends and lives outside this class.

    Capability flags are ``@classmethod`` so callers (executor, ``/health``,
    AUTO selection) can introspect a backend without instantiating it, matching
    the capability-flag convention used by attention backends.
    """

    def __init__(self) -> None:
        self._state: SleepModeState = "RUNNING"

    @abstractmethod
    def suspend(self, level: int = 1) -> None:
        """Free GPU state.

        ``level`` follows existing sleep-mode semantics: level 1 offloads
        weights to host RAM (restorable in-process); level 2 discards weights
        (reloaded from the model source on resume).
        """
        raise NotImplementedError

    @abstractmethod
    def resume(self, tags: list[str] | None = None) -> None:
        """Restore previously-suspended GPU state.

        ``tags`` optionally limits which tagged allocations are restored
        (e.g. ``["weights"]`` or ``["kv_cache"]``).
        """
        raise NotImplementedError

    def state(self) -> SleepModeState:
        """Current lifecycle state. Lets ``/health`` distinguish a healthy-idle
        (suspended) engine from a healthy-serving one (see RFC #34303)."""
        return self._state

    # -- Capability introspection (no instance required) --

    @classmethod
    def is_supported(cls) -> bool:
        """Whether this backend can run on the current platform/driver."""
        return True

    @classmethod
    def preserves_communicators(cls) -> bool:
        """If False, collective communicators (e.g. NCCL) are destroyed by
        ``suspend`` and the executor must re-initialize them on ``resume``."""
        return False

    @classmethod
    def preserves_compiled_artifacts(cls) -> bool:
        """If True, torch.compile / JIT kernels survive suspend/resume and need
        not be recompiled on resume."""
        return False

    @classmethod
    def preserves_graphs_with_communicators(cls) -> bool:
        """If True, CUDA graphs containing collective communicators (e.g. NCCL)
        stay valid after resume. False when communicators are rebuilt (embedded
        comm handles go stale)."""
        return False

    @classmethod
    def supports_durable_storage(cls) -> bool:
        """If True, suspended state can be persisted beyond the process
        lifetime (disk or object storage) and restored in a new process."""
        return False


class CuMemBackend(SleepModeBackend):
    """Default backend.

    Wraps the platform sleep-mode allocator exactly as the GPU worker did
    before this abstraction existed, so behavior is identical to vLLM's current
    sleep/wake-up. ``get_mem_allocator_instance()`` resolves to
    ``CuMemAllocator`` on CUDA and ``XpuMemAllocator`` on XPU; suspend offloads
    per-allocation between GPU and host, with NCCL buffers left untouched (they
    are allocated outside the allocator pool).
    """

    def suspend(self, level: int = 1) -> None:
        from vllm.device_allocator import get_mem_allocator_instance

        self._state = "SUSPENDED"
        allocator = get_mem_allocator_instance()
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())

    def resume(self, tags: list[str] | None = None) -> None:
        from vllm.device_allocator import get_mem_allocator_instance

        self._state = "RESUMING"
        allocator = get_mem_allocator_instance()
        allocator.wake_up(tags)
        self._state = "RUNNING"

    @classmethod
    def preserves_communicators(cls) -> bool:
        # Communicator buffers (e.g. NCCL) live outside CuMemAllocator's pool, so
        # an allocator-level sleep leaves them intact (no reinit needed on resume).
        return True


class SleepModeBackendFactory:
    """Registry and resolver for sleep-mode backends.

    Mirrors ``KVConnectorFactory``: lazy module/class registration and a
    built-in registry populated at import time. Third-party backends register
    the same way from a ``vllm.general_plugins`` entry point.
    """

    _registry: dict[str, Callable[[], type[SleepModeBackend]]] = {}

    @classmethod
    def register_backend(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a backend with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Sleep-mode backend '{name}' is already registered.")

        def loader() -> type[SleepModeBackend]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def get_backend_class(cls, name: str) -> type[SleepModeBackend]:
        """Resolve a registered backend class by name."""
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "<none>"
            raise ValueError(
                f"Unsupported sleep-mode backend '{name}'. "
                f"Registered backends: {available}."
            )
        return cls._registry[name]()

    @classmethod
    def create_backend(cls, model_config: ModelConfig) -> SleepModeBackend:
        """Instantiate the backend selected by ``model_config``."""
        name = model_config.sleep_mode_backend
        backend_cls = cls.get_backend_class(name)
        if not backend_cls.is_supported():
            raise ValueError(
                f"Sleep-mode backend '{name}' is not supported on this platform."
            )
        logger.info("Using sleep-mode backend: %s", name)
        return backend_cls()


# Register built-in backends here. Registration is lazy: only the module for the
# selected backend is imported. Third-party backends (CUDA checkpoint, CRIU,
# durable snapshot) register the same way through a vllm.general_plugins entry
# point, without changes to vLLM core.
SleepModeBackendFactory.register_backend(
    "cumem",
    "vllm.device_allocator.sleep_mode_backend",
    "CuMemBackend",
)
