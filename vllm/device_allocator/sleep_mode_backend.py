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
    def suspend(self, level: int = 1, tags: tuple[str, ...] | None = None) -> None:
        """Free GPU state.

        ``level`` follows existing sleep-mode semantics: level 1 offloads
        weights to host RAM (restorable in-process); level 2 discards weights
        (reloaded from the model source on resume).

        ``tags`` is an optional, backend-specific override. Backends that do
        not implement ``supports_selective_offload`` ignore the argument; the
        signature is shared so the GPU worker can dispatch the same kwarg to
        any backend without conditional plumbing.
        """
        raise NotImplementedError

    @abstractmethod
    def resume(self, tags: list[str] | None = None) -> None:
        """Restore previously-suspended GPU state.

        ``tags`` optionally limits which tagged allocations are restored
        (e.g. ``["weights"]`` or ``["kv_cache"]``).
        """
        raise NotImplementedError

    def suspended_tags(self) -> tuple[str, ...] | None:
        """Tags actually offloaded by the most recent ``suspend()`` call.

        Returns ``None`` when the backend is RUNNING (nothing offloaded) or
        when the backend does not track per-tag suspend state. The GPU worker
        uses this to gate the ``post_kv_cache_wake_up`` re-init path so it
        does not zero live KV-cache pages that were never suspended.

        Default implementation returns ``None`` (no tracking). Backends that
        support selective offload should override.
        """
        return None

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
    def preserves_nccl(cls) -> bool:
        """If False, NCCL communicators are destroyed by ``suspend`` and the
        executor must re-initialize them on ``resume``."""
        return False

    @classmethod
    def preserves_compiled_artifacts(cls) -> bool:
        """If True, torch.compile / JIT kernels survive suspend/resume and need
        not be recompiled on resume."""
        return False

    @classmethod
    def preserves_graphs_with_nccl(cls) -> bool:
        """If True, CUDA graphs containing NCCL collectives stay valid after
        resume. False when NCCL is rebuilt (embedded comm handles go stale)."""
        return False

    @classmethod
    def supports_durable_storage(cls) -> bool:
        """If True, suspended state can be persisted beyond the process
        lifetime (disk or object storage) and restored in a new process."""
        return False

    @classmethod
    def supports_selective_offload(cls) -> bool:
        """If True, ``suspend`` accepts an explicit ``tags`` argument and
        ``resume`` honors a matching ``tags`` argument, letting the caller
        manage weights / KV cache / compiled artifacts on independent
        lifecycles. False (default) means the backend offloads and restores
        everything together, controlled only by ``level``."""
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

    def suspend(self, level: int = 1, tags: tuple[str, ...] | None = None) -> None:
        # ``tags`` is accepted to match the SleepModeBackend signature but
        # ignored: CuMemBackend has no selective-offload semantics (per
        # ``supports_selective_offload`` which stays False). The tag set is
        # derived from ``level`` exactly as the pre-abstraction worker did.
        from vllm.device_allocator import get_mem_allocator_instance

        del tags  # silence linters; behavior intentional - see docstring
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
    def preserves_nccl(cls) -> bool:
        # NCCL buffers live outside CuMemAllocator's pool, so an allocator-level
        # sleep leaves the communicators intact (no reinit needed on resume).
        return True


class CuMemTagBackend(CuMemBackend):
    """CuMemBackend with tag-based selective offload.

    Where ``CuMemBackend`` always offloads exactly the tags implied by
    ``level`` (``("weights",)`` at level 1, ``()`` at level 2), this backend
    lets the caller specify on each call - or fix at construction - exactly
    which tags to offload on suspend and which to restore on resume. That
    enables multi-model in-process scenarios where weights, KV cache, and
    compiled artifacts have independent lifecycles (e.g. swap weights while
    keeping the KV cache hot for a different tenant on the same GPU).

    Motivated by the field report on RFC #34303 from the Alibaba Cloud ACK
    team (`comment 4689082496
    <https://github.com/vllm-project/vllm/issues/34303#issuecomment-4689082496>`_):
    same-architecture model switch in ~1-2s on production hardware by
    reusing CUDA graphs and rebuilding only the weight pages.

    Implementation is a thin wrapper around
    ``CuMemAllocator.sleep(offload_tags=...)`` / ``.wake_up(tags=...)``.
    With its defaults this backend is behavior-identical to ``CuMemBackend``;
    the extension is the *option* to override on a per-call basis.

    Safety: this backend tracks which tags were offloaded by the most recent
    ``suspend()`` call (see :py:meth:`suspended_tags`) so the GPU worker can
    avoid running ``post_kv_cache_wake_up`` against pages that were never
    suspended, and so ``resume()`` can refuse selective wake-ups that would
    silently corrupt live state.
    """

    #: Default tag set offloaded by ``suspend(level=1)`` when no explicit
    #: tags are passed. Matches ``CuMemBackend`` so the no-argument path is
    #: a no-op compared to the existing backend.
    DEFAULT_SUSPEND_TAGS_L1: tuple[str, ...] = ("weights",)
    #: Default tag set offloaded by ``suspend(level=2)``. Empty matches
    #: ``CuMemBackend`` (level 2 discards weights, so nothing is offloaded
    #: to host RAM - they get reloaded from the model source on resume).
    DEFAULT_SUSPEND_TAGS_L2: tuple[str, ...] = ()

    def __init__(self, suspend_tags: tuple[str, ...] | None = None) -> None:
        super().__init__()
        # Optional construction-time override. When set, takes precedence over
        # the level-based defaults but is still overridable per-call. Public
        # so callers (and tests) can introspect the effective tag set without
        # touching internals.
        #
        # Normalize to ``tuple`` (preserving the ``None`` sentinel) so config
        # surfaces that deliver lists - CLI JSON, YAML-via-
        # ``sleep_mode_backend_options`` - round-trip into a hashable,
        # immutable value with the same return-type contract as
        # ``effective_suspend_tags()``. matteso1's review nit.
        self.suspend_tags: tuple[str, ...] | None = (
            tuple(suspend_tags) if suspend_tags is not None else None
        )
        # Tags actually offloaded by the most recent ``suspend()`` call, plus
        # the level used. ``None`` means "currently RUNNING / nothing
        # suspended". Used by ``resume()`` to refuse selective wake-ups of
        # tags that were never suspended (which on the existing GPU worker
        # path would zero live KV-cache pages via ``post_kv_cache_wake_up``),
        # and by the worker to gate that same re-init call.
        self._suspended_tags: tuple[str, ...] | None = None
        self._suspended_level: int | None = None

    def effective_suspend_tags(self, level: int = 1) -> tuple[str, ...]:
        """Return the tags that ``suspend(level=...)`` would offload right now.

        Public accessor so tests and callers can validate behavior without
        invoking the GPU path. Mirrors the resolution precedence in
        ``suspend()`` but with no per-call override (which is by definition
        only known at the call site).
        """
        if self.suspend_tags is not None:
            return self.suspend_tags
        return (
            self.DEFAULT_SUSPEND_TAGS_L1
            if level == 1
            else self.DEFAULT_SUSPEND_TAGS_L2
        )

    def suspended_tags(self) -> tuple[str, ...] | None:
        """Tags actually offloaded by the most recent ``suspend()`` call.

        Returns ``None`` when the backend is RUNNING or when ``suspend()`` has
        not been called yet. Lets the GPU worker decide whether
        ``post_kv_cache_wake_up`` is safe to call on the next ``wake_up``.
        """
        return self._suspended_tags

    def suspend(
        self,
        level: int = 1,
        tags: tuple[str, ...] | None = None,
    ) -> None:
        """Free GPU state for the selected ``tags``.

        Tag resolution precedence (most to least specific):
          1. Explicit ``tags=`` keyword argument on this call.
          2. Construction-time ``suspend_tags`` (if set).
          3. Level-based default (``DEFAULT_SUSPEND_TAGS_L1`` or ``...L2``).
        """
        from vllm.device_allocator import get_mem_allocator_instance

        if tags is not None:
            effective_tags = tuple(tags)
        else:
            effective_tags = self.effective_suspend_tags(level)
        allocator = get_mem_allocator_instance()
        # Drive the allocator FIRST. Only commit suspend bookkeeping
        # (state, ``_suspended_tags``, ``_suspended_level``) after the
        # allocator call returns successfully. If ``allocator.sleep`` raises,
        # the backend stays RUNNING with no suspended tags - matching the
        # actual GPU state. The previous ordering wrote bookkeeping first
        # and left the backend in a phantom SUSPENDED state on allocator
        # failure, which the executor would then try to ``resume`` from -
        # masking the real failure with a confusing wake-up error.
        allocator.sleep(offload_tags=effective_tags)
        self._suspended_tags = effective_tags
        self._suspended_level = level
        self._state = "SUSPENDED"

    def resume(self, tags: list[str] | None = None) -> None:
        """Restore previously-suspended GPU state.

        Selective-suspend safety: if ``tags`` is non-empty, each tag must
        appear in the set of tags actually offloaded by the most recent
        ``suspend()`` call. This guards against the
        ``suspend(tags=("weights",))`` -> ``wake_up(tags=("kv_cache",))``
        pattern, which would otherwise traverse a code path in
        ``gpu_worker.wake_up`` that re-initializes the KV cache - zeroing
        live GPU pages that were never offloaded.

        When ``tags is None``, fall back to whatever tags were suspended so
        the existing single-instance dispatch path (which calls
        ``wake_up(tags=None)`` for "wake everything") still works, without
        widening the wake set beyond what was actually offloaded.

        L2-suspended weights: the allocator-level ``wake_up`` call below
        cannot reload weights that were discarded by ``level=2`` suspend -
        that reload must come from ``worker.load_model()`` /
        ``model_runner.reload_weights()``, which is the executor's
        responsibility, not this backend's. To avoid silently returning a
        "RUNNING" backend with garbage weight pages, raise loudly when the
        caller tries to wake L2-suspended weights through this path. The
        intent is the L2-wake corruption mode observed on AWQ models (HTTP
        200 healthy, garbage tokens) fails fast at the API boundary instead
        of in the user's downstream traffic.
        """
        from vllm.device_allocator import get_mem_allocator_instance

        # Compute the effective wake set FIRST so the allocator call uses the
        # same clamped tag set we just validated. Passing the original
        # ``tags`` (potentially ``None``) to ``allocator.wake_up`` would
        # widen the wake beyond what was actually suspended - on a backend
        # whose pool may also hold pages suspended by another caller (or by
        # a prior selective suspend in this process), an unclamped
        # ``wake_up(None)`` zeroes those live pages too. Pre-compute
        # ``effective_wake_tags`` from the clamped ``requested_set`` and
        # pass that down in every branch.
        effective_wake_tags: tuple[str, ...] | None = None
        if self._suspended_tags is not None:
            suspended_set = set(self._suspended_tags)
            requested_set = set(tags) if tags else suspended_set
            spurious = sorted(requested_set - suspended_set)
            if spurious:
                # Leave state unchanged - bookkeeping is intact for retry,
                # and the backend stays SUSPENDED rather than RESUMING with
                # no actual allocator transition under way.
                raise ValueError(
                    f"resume(tags={sorted(requested_set)}) requested tags "
                    f"{spurious} that were not suspended (suspended tags: "
                    f"{sorted(suspended_set)}). Waking a non-suspended tag "
                    "would corrupt live GPU state through the "
                    "post_kv_cache_wake_up path in gpu_worker.wake_up; "
                    "pass tags= matching the suspend() call."
                )
            if (
                self._suspended_level == 2
                and "weights" in requested_set
            ):
                raise RuntimeError(
                    "resume() cannot restore weights that were discarded by "
                    "suspend(level=2): the allocator pages were freed and "
                    "must be reloaded from disk by "
                    "worker.load_model() / model_runner.reload_weights() "
                    "before this backend can mark them RUNNING. This "
                    "selective-offload backend does not own the reload "
                    "step; the caller (engine/executor) must invoke the "
                    "reload path before resuming weights. See RFC #34303 "
                    "for the level-2 lifecycle contract."
                )
            # Snap ``effective_wake_tags`` to the validated ``requested_set``,
            # preserving the suspend-time order so allocator-side iteration
            # stays deterministic.
            effective_wake_tags = tuple(
                t for t in self._suspended_tags if t in requested_set
            )
        else:
            # No tracked suspend state (e.g. resume() called without a
            # preceding suspend, or from the abstract path on a freshly-
            # constructed backend). Pass tags through verbatim - any caller
            # who explicitly asked for selective wake without a tracked
            # suspend is responsible for the consequences.
            effective_wake_tags = tuple(tags) if tags else None

        self._state = "RESUMING"
        allocator = get_mem_allocator_instance()
        try:
            allocator.wake_up(effective_wake_tags)
        except BaseException:
            # Allocator failed mid-wake. Don't leave the backend stuck in
            # RESUMING - revert to SUSPENDED so the executor sees a state
            # that matches the actual GPU situation (pages still offloaded)
            # and can decide whether to retry or surface the error.
            self._state = "SUSPENDED"
            raise
        # Clear suspend bookkeeping only for tags that were actually woken.
        if self._suspended_tags is not None:
            if effective_wake_tags:
                woken = set(effective_wake_tags)
                remaining = tuple(
                    t for t in self._suspended_tags if t not in woken
                )
                self._suspended_tags = remaining if remaining else None
                if self._suspended_tags is None:
                    self._suspended_level = None
            else:
                self._suspended_tags = None
                self._suspended_level = None
        self._state = "RUNNING"

    @classmethod
    def supports_selective_offload(cls) -> bool:
        # Selective per-tag offload is the whole point of this backend.
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
    def create_backend(cls, model_config: "ModelConfig") -> SleepModeBackend:
        """Instantiate the backend selected by ``model_config``.

        Backend-specific kwargs come from
        ``model_config.sleep_mode_backend_options`` and are ``**``-unpacked
        into the concrete backend's ``__init__``. Validation of those kwargs
        is the backend's responsibility; ``ModelConfig`` does not enforce a
        schema so that adding a new backend with new options does not require
        a config-side change.
        """
        name = model_config.sleep_mode_backend
        backend_cls = cls.get_backend_class(name)
        if not backend_cls.is_supported():
            raise ValueError(
                f"Sleep-mode backend '{name}' is not supported on this platform."
            )
        options = getattr(model_config, "sleep_mode_backend_options", {}) or {}
        logger.info(
            "Using sleep-mode backend: %s (options=%s)",
            name,
            options if options else "{}",
        )
        return backend_cls(**options)

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a registered backend.

        Intended for tests of plugin-author backends that want to isolate
        registry state between cases without mutating the private
        ``_registry`` dict directly. Idempotent: unregistering a name that is
        not currently registered is a no-op. Not intended for production code.
        """
        cls._registry.pop(name, None)


# Register built-in backends here. Registration is lazy: only the module for the
# selected backend is imported. Third-party backends (CUDA checkpoint, CRIU,
# durable snapshot) register the same way through a vllm.general_plugins entry
# point, without changes to vLLM core.
SleepModeBackendFactory.register_backend(
    "cumem",
    "vllm.device_allocator.sleep_mode_backend",
    "CuMemBackend",
)
SleepModeBackendFactory.register_backend(
    "cumem_tag",
    "vllm.device_allocator.sleep_mode_backend",
    "CuMemTagBackend",
)
