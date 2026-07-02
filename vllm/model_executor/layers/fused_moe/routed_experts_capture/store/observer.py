# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bridge KV-tier cascade / promotion events into a secondary routing store.

``RoutedExpertsBlockLifecycleObserver`` plugs the routed-experts offload buffer
into the generic ``BlockLifecycleObserver`` hook (``kv_offload/base.py``): on
cascade it persists the affected offloaded-block rows to a
``RoutedExpertsSecondaryStore``, on promotion it restores them, so routing
follows the KV cache through every tier (GPU <-> CPU <-> disk/object/...).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from vllm.model_executor.layers.fused_moe.routed_experts_capture.store.base import (
    RoutedExpertsSecondaryStore,
)
from vllm.v1.kv_offload.base import BlockLifecycleObserver

logger = logging.getLogger(__name__)


class _OffloadBuffer(Protocol):
    """The subset of RoutedExpertsManager the observer needs.

    Declared structurally so this module does not import the manager at
    runtime (the observer only reads/writes whole offloaded-block rows).
    """

    def read_cpu_blocks(self, cpu_block_ids: np.ndarray) -> np.ndarray: ...

    def write_cpu_blocks(self, cpu_block_ids: np.ndarray, rows: np.ndarray) -> None: ...


class RoutedExpertsBlockLifecycleObserver(BlockLifecycleObserver):
    """Mirror cascade / promotion events into a secondary routing store.

    Registered on a ``TieringOffloadingManager``. On cascade (CPU primary ->
    secondary) it persists the offloaded-block rows of the affected CPU
    blocks; on promotion (secondary -> CPU primary) it restores them, so the
    routed-experts offload buffer's lifecycle matches the KV cache's across
    all tiers (GPU <-> CPU <-> disk/object).
    """

    def __init__(
        self,
        manager: _OffloadBuffer,
        store: RoutedExpertsSecondaryStore,
    ) -> None:
        self._manager = manager
        self._store = store
        # Cumulative counters (blocks), for observability. The disk round-trip
        # is otherwise invisible; these let tests/ops confirm the routing
        # offload buffer actually followed KV through the secondary tier.
        self.cascaded_blocks = 0
        self.promoted_blocks = 0

    def on_blocks_cascaded(
        self, keys: Sequence[bytes], cpu_block_ids: np.ndarray
    ) -> None:
        if len(keys) == 0:
            return
        rows = self._manager.read_cpu_blocks(np.asarray(cpu_block_ids))
        self._store.persist(keys, rows)
        self.cascaded_blocks += len(keys)
        logger.debug(
            "routed-experts offload: cascaded %d block(s) to secondary (total=%d)",
            len(keys),
            self.cascaded_blocks,
        )

    def on_blocks_promotion_started(
        self, keys: Sequence[bytes], cpu_block_ids: np.ndarray
    ) -> None:
        # KV bytes just started loading secondary -> primary; warm the routing
        # read-ahead cache in parallel so the matching ``on_blocks_promoted``
        # restore (after the KV load completes) serves from memory.
        if len(keys) == 0:
            return
        self._store.prefetch(keys)

    def on_blocks_promoted(
        self, keys: Sequence[bytes], cpu_block_ids: np.ndarray
    ) -> None:
        if len(keys) == 0:
            return
        rows = self._store.restore(keys)
        if rows is None:
            # KV-present-but-routing-absent: the connector promoted these blocks,
            # so their routing rows MUST exist. Fail closed rather than leave
            # stale offload-buffer rows.
            raise RuntimeError(
                f"routed-experts sidecar missing for {len(keys)} promoted "
                "block(s); KV was promoted but its routing rows are absent"
            )
        self._manager.write_cpu_blocks(np.asarray(cpu_block_ids), rows)
        self.promoted_blocks += len(keys)
        logger.debug(
            "routed-experts offload: promoted %d block(s) from secondary (total=%d)",
            len(keys),
            self.promoted_blocks,
        )

    def shutdown(self) -> None:
        """Flush the secondary store's pending writes, if it has any."""
        store_shutdown = getattr(self._store, "shutdown", None)
        if callable(store_shutdown):
            store_shutdown()
