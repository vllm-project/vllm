# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter that bridges ``unified_comm`` into vLLM's ``GroupCoordinator``.

Implements the adapter pattern: a :class:`GroupCoordinator` may
optionally delegate its collective operations to a ``CollectiveGroup``
from this package while keeping its public interface unchanged.

The adapter is activated only when the environment variable
``UNIFIED_COMM_ENABLED=1`` is set; otherwise it is never built and the
default code path is used.

Design principles:
  - Non-intrusive: the ``GroupCoordinator`` base class and any vLLM
    upstream files are not modified beyond the small hook used to
    obtain the adapter lazily.
  - Safe fallback: any exception during construction or execution
    falls back to the default code path.
  - Lazy initialization: the unified_comm modules are only imported
    when the adapter is actually requested.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)


def is_unified_comm_enabled() -> bool:
    """Return True if the unified communication layer is enabled."""
    val = os.environ.get("UNIFIED_COMM_ENABLED", "0").lower()
    return val in ("1", "true", "yes")


class UnifiedCommAdapter:
    """Adapter that bridges a unified_comm ``CollectiveGroup`` to vLLM's
    ``GroupCoordinator``-compatible interface.

    Responsibilities:
      - Hold a ``CollectiveGroup`` instance on behalf of a
        ``GroupCoordinator``.
      - Translate vLLM's collective calls (``all_reduce``,
        ``all_gather``, ``reduce_scatter``, ...) into
        ``CollectiveGroup`` calls.
      - Reconcile interface differences (``dim`` argument handling,
        ``ReduceOp`` conversions, etc.).
      - Return ``None`` from any operation on failure so the caller
        can fall back to the default code path.

    Usage::

        adapter = UnifiedCommAdapter.try_create(ranks, rank, device)
        if adapter is not None:
            result = adapter.all_reduce(tensor)
            if result is not None:
                return result  # success
        # otherwise fall back to the default code path
    """

    def __init__(self, collective_group: Any):
        self._group = collective_group
        # Bind the group to any registered TransferPlane(s).
        self._initialize_transfer_planes()

    @classmethod
    def try_create(
        cls,
        ranks: list[int],
        local_rank: int,
        device: torch.device,
        existing_device_group: Any = None,
        existing_cpu_group: Any = None,
    ) -> UnifiedCommAdapter | None:
        """Try to construct an adapter.

        Returns ``None`` when unified_comm is not enabled or when
        construction fails for any reason; the caller is expected to
        fall back to the default code path.

        Args:
            ranks: global ranks belonging to this group.
            local_rank: this process' global rank.
            device: device to bind the communication to.
            existing_device_group: pre-existing device ``ProcessGroup``
                (when reusing the one created by ``GroupCoordinator``).
            existing_cpu_group: pre-existing CPU ``ProcessGroup``.
        """
        if not is_unified_comm_enabled():
            return None

        try:
            from vllm.distributed.unified_comm.collective import (
                CollectiveGroup,
            )

            group = CollectiveGroup.create(
                ranks=ranks,
                local_rank=local_rank,
                device=device,
                existing_device_group=existing_device_group,
                existing_cpu_group=existing_cpu_group,
            )
            logger.info(
                "[UnifiedCommAdapter] Created for ranks=%s, "
                "local_rank=%s, device=%s, backend=%s",
                ranks,
                local_rank,
                device,
                group.backend_name,
            )
            return cls(group)
        except Exception as e:
            logger.warning(
                "[UnifiedCommAdapter] Failed to create CollectiveGroup, "
                "falling back to original path: %s",
                e,
            )
            return None

    # ----------------------------------------------------------
    # Collective-op adapters
    # ----------------------------------------------------------

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor | None:
        """Adapter for AllReduce; returns ``None`` on failure."""
        try:
            from vllm.distributed.unified_comm.backend import ReduceOp

            return self._group.all_reduce(input_, op=ReduceOp.SUM)
        except Exception as e:
            logger.warning("[UnifiedCommAdapter] all_reduce failed: %s", e)
            return None

    def all_gather(self, input_: torch.Tensor, dim: int = 0) -> torch.Tensor | None:
        """Adapter for AllGather; returns ``None`` on failure.

        Handles the ``dim`` argument by transposing to dim-0 around the
        underlying call when ``dim != 0``.
        """
        try:
            if dim == 0:
                return self._group.all_gather(input_)
            else:
                perm = list(range(input_.dim()))
                perm[0], perm[dim] = perm[dim], perm[0]
                transposed = input_.permute(perm).contiguous()
                gathered = self._group.all_gather(transposed)
                return gathered.permute(perm).contiguous()
        except Exception as e:
            logger.warning("[UnifiedCommAdapter] all_gather failed: %s", e)
            return None

    def reduce_scatter(self, input_: torch.Tensor, dim: int = 0) -> torch.Tensor | None:
        """Adapter for ReduceScatter; returns ``None`` on failure."""
        try:
            from vllm.distributed.unified_comm.backend import ReduceOp

            if dim == 0:
                return self._group.reduce_scatter(input_, op=ReduceOp.SUM)
            else:
                perm = list(range(input_.dim()))
                perm[0], perm[dim] = perm[dim], perm[0]
                transposed = input_.permute(perm).contiguous()
                scattered = self._group.reduce_scatter(transposed, op=ReduceOp.SUM)
                return scattered.permute(perm).contiguous()
        except Exception as e:
            logger.warning(
                "[UnifiedCommAdapter] reduce_scatter failed: %s",
                e,
            )
            return None

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> torch.Tensor | None:
        """Adapter for Broadcast; returns ``None`` on failure."""
        try:
            return self._group.broadcast(input_, src=src)
        except Exception as e:
            logger.warning("[UnifiedCommAdapter] broadcast failed: %s", e)
            return None

    def send(self, input_: torch.Tensor, dst: int) -> bool | None:
        """Adapter for point-to-point send.

        Returns ``True`` on success, ``None`` on failure (caller should
        fall back).
        """
        try:
            self._group.send(input_, dst)
            return True
        except Exception as e:
            logger.warning("[UnifiedCommAdapter] send failed: %s", e)
            return None

    def recv(self, input_: torch.Tensor, src: int) -> torch.Tensor | None:
        """Adapter for point-to-point recv; returns ``None`` on failure."""
        try:
            return self._group.recv(input_, src)
        except Exception as e:
            logger.warning("[UnifiedCommAdapter] recv failed: %s", e)
            return None

    def all_to_all(
        self,
        input_: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = -1,
        scatter_sizes: list[int] | None = None,
        gather_sizes: list[int] | None = None,
    ) -> torch.Tensor | None:
        """Adapter for All-to-All; returns ``None`` on failure."""
        try:
            return self._group.all_to_all(
                input_, scatter_dim, gather_dim, scatter_sizes, gather_sizes
            )
        except Exception as e:
            logger.warning("[UnifiedCommAdapter] all_to_all failed: %s", e)
            return None

    def barrier(self) -> bool | None:
        """Adapter for a synchronization barrier.

        Returns ``True`` on success, ``None`` on failure.
        """
        try:
            self._group.barrier()
            return True
        except Exception as e:
            logger.warning("[UnifiedCommAdapter] barrier failed: %s", e)
            return None

    # ----------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------

    def _initialize_transfer_planes(self) -> None:
        """Bind the underlying ``CollectiveGroup`` to any registered
        ``TransferPlane`` instances."""
        try:
            from vllm.distributed.unified_comm.transfer_plane import (
                TransferPlaneRegistry,
            )

            registry = TransferPlaneRegistry()
            for transfer_type in registry.list_planes():
                plane = registry.get(transfer_type)
                if not getattr(plane, "_initialized", False):
                    plane.initialize(self._group)
                    logger.debug(
                        "[UnifiedCommAdapter] Initialized TransferPlane: %s",
                        transfer_type.name,
                    )
        except Exception as e:
            logger.debug(
                "[UnifiedCommAdapter] TransferPlane initialization skipped: %s",
                e,
            )

    def destroy(self) -> None:
        """Release the underlying ``CollectiveGroup``."""
        try:
            if self._group is not None:
                self._group.destroy()
                self._group = None
        except Exception as e:
            logger.warning("[UnifiedCommAdapter] destroy failed: %s", e)

    @property
    def strategy_name(self) -> str:
        """Name of the currently active strategy, or ``"none"``."""
        try:
            if self._group and self._group._strategy:
                return self._group._strategy.name()
        except Exception:
            pass
        return "none"

    def __repr__(self) -> str:
        if self._group:
            return f"UnifiedCommAdapter(group={self._group})"
        return "UnifiedCommAdapter(destroyed)"
