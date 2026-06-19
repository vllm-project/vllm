# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-agnostic core of the routed-experts secondary-tier sidecar.

A ``RoutedExpertsSecondaryStore`` persists / restores offloaded-block rows for
one secondary tier; ``RoutedExpertsStoreFactory`` maps a tier ``type`` to a
builder so the scheduler never hard-codes an implementation, and
``RoutedExpertsStoreContext`` carries the inputs a builder needs. Concrete
backends live in sibling modules (e.g. ``fs``); the KV-tier event bridge lives
in ``observer``.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec


class RoutedExpertsSecondaryStore(ABC):
    """Persist / restore offloaded-block rows for a secondary tier.

    The routed-experts offload buffer lives in the CPU primary tier alongside
    KV. When KV blocks cascade to / promote from a secondary tier, the matching
    offloaded-block rows must follow so routing survives CPU eviction exactly
    as the KV bytes do. Implementations key each row by its ``OffloadKey``
    (block hash + group idx), mirroring the KV ``FileMapper`` layout.

    Rows are ``(factor, block_size, num_layers, top_k)`` arrays of the
    manager's expert-id dtype. ``persist`` may be asynchronous (the built-in
    ``fs`` backend enqueues writes onto a thread pool and returns, keeping the
    scheduler thread off the disk-IO critical path); ``restore`` is
    synchronous but must observe writes issued by a prior ``persist`` for the
    same keys (read-after-write), regardless of whether the write has reached
    its backing store yet.
    """

    @abstractmethod
    def persist(self, keys: Sequence[bytes], rows: np.ndarray) -> None:
        """Write one row per key. ``rows[i]`` corresponds to ``keys[i]``."""

    @abstractmethod
    def restore(self, keys: Sequence[bytes]) -> np.ndarray | None:
        """Read rows for keys, stacked in order.

        Returns ``None`` if any key is missing (the caller then leaves the
        offloaded-block rows untouched — the connector would not have issued a
        load for an absent block).
        """

    def prefetch(self, keys: Sequence[bytes]) -> None:  # noqa: B027
        """Hint that ``restore(keys)`` is coming soon; warm a read-ahead cache.

        Called when the matching KV blocks begin promoting (secondary ->
        primary), so a backend can overlap its read with the KV-byte transfer
        and serve the subsequent ``restore`` from memory instead of blocking
        the scheduler on disk. Default no-op: ``restore`` stays correct without
        it (it just reads synchronously). Idempotent and best-effort.
        """


class RoutedExpertsStoreContext(NamedTuple):
    """Inputs a ``RoutedExpertsSecondaryStore`` builder may need.

    Passed to every builder registered with ``RoutedExpertsStoreFactory`` so
    a backend (filesystem, object store, Mooncake, ...) can construct its
    store without the scheduler hard-coding any one implementation. The
    context is backend-agnostic: each backend reads whatever it needs (a
    ``root_dir``, an endpoint, credentials, ...) from ``tier_config``.

    Args:
        tier_config: The secondary-tier config dict from
            ``kv_connector_extra_config['secondary_tiers'][i]`` (includes
            ``type`` plus any backend-specific keys, e.g. ``root_dir`` for
            ``fs`` or endpoint / namespace for a remote store).
        offloading_spec: The resolved ``CPUOffloadingSpec`` (subclassed by
            ``TieringOffloadingSpec``); exposes ``block_size_factor``,
            ``vllm_config``, ``extra_config``, etc.
        row_shape: Offloaded-block row shape ``(factor, block_size, layers,
            top_k)``.
        dtype: Expert-id dtype of the offloaded-block rows.
    """

    tier_config: dict
    offloading_spec: OffloadingSpec
    row_shape: tuple[int, ...]
    dtype: np.dtype


class RoutedExpertsStoreFactory:
    """Registry mapping a secondary-tier ``type`` to a routing-store builder.

    Mirrors ``vllm.v1.kv_offload.tiering.factory.SecondaryTierFactory``: a
    backend registers a builder under the same ``type`` string its KV tier
    uses (e.g. ``"fs"``, ``"mooncake"``), and the scheduler looks it up
    instead of hard-coding any implementation. Registration is lazy (module
    path + factory name), so a backend's heavy imports load only when its tier
    is configured, and out-of-tree backends register without importing this
    module.

    To add a backend, implement the two-method ``RoutedExpertsSecondaryStore``
    contract (``persist`` / ``restore``, keyed by ``OffloadKey``) and register
    a builder::

        # my_pkg/mooncake_store.py
        class MooncakeRoutedExpertsStore(RoutedExpertsSecondaryStore):
            def __init__(self, ctx):
                self._store = open_mooncake(ctx.tier_config)  # by-key put/get
                self._shape, self._dtype = ctx.row_shape, ctx.dtype

            def persist(self, keys, rows):
                for k, row in zip(keys, rows):
                    self._store.put(k.hex(), row.tobytes())

            def restore(self, keys):
                bufs = [self._store.get(k.hex()) for k in keys]
                if any(not b for b in bufs):
                    return None
                return np.stack(
                    [np.frombuffer(b, self._dtype).reshape(self._shape) for b in bufs]
                )


        def build_store(ctx):
            return MooncakeRoutedExpertsStore(ctx)


        RoutedExpertsStoreFactory.register_store(
            "mooncake", "my_pkg.mooncake_store", "build_store"
        )

    A tier ``type`` with no registered builder just gets no routing sidecar (a
    warning is logged); KV still tiers normally.
    """

    _registry: dict[str, tuple[str, str]] = {}

    @classmethod
    def register_store(
        cls, tier_type: str, module_path: str, factory_name: str
    ) -> None:
        """Register a store-builder factory for a secondary-tier ``type``.

        Args:
            tier_type: Tier type string (must match the KV secondary tier's
                ``type``, e.g. ``"fs"``, ``"mooncake"``).
            module_path: Import path of the module holding the builder.
            factory_name: Name of the builder callable within that module;
                it takes a ``RoutedExpertsStoreContext`` and returns a
                ``RoutedExpertsSecondaryStore``.

        Raises:
            ValueError: If ``tier_type`` is already registered.
        """
        if tier_type in cls._registry:
            raise ValueError(
                f"Routed-experts store for tier '{tier_type}' is already registered."
            )
        cls._registry[tier_type] = (module_path, factory_name)

    @classmethod
    def is_registered(cls, tier_type: str) -> bool:
        return tier_type in cls._registry

    @classmethod
    def create(
        cls, tier_type: str, ctx: RoutedExpertsStoreContext
    ) -> RoutedExpertsSecondaryStore | None:
        """Build the store for ``tier_type``, or None if no builder is known."""
        entry = cls._registry.get(tier_type)
        if entry is None:
            return None
        module_path, factory_name = entry
        module = importlib.import_module(module_path)
        builder = getattr(module, factory_name)
        return builder(ctx)
