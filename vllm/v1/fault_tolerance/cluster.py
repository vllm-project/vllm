# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ClusterCoordinator — protocol over the multi-engine manager.

Wraps whatever multi-engine manager the deployment uses (Ray actor manager
or the multiprocessing equivalent) so cluster-scope recovery plans can be
written backend-agnostically.

Cluster-scope plans (``scale_down``, ``abort``, future ``scale_up``) call
methods on this protocol rather than reaching into Ray-specific code.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.engine.core_client import MPClient

logger = init_logger(__name__)


@runtime_checkable
class ClusterCoordinator(Protocol):
    """Protocol over the multi-engine manager."""

    def alive_engine_indices(self) -> list[int]:
        """Engine indices currently considered alive."""
        ...

    def scale_down(self, dead_index: int) -> None:
        """Drop ``dead_index`` from the topology and rebuild communicators."""
        ...

    def abort_inflight_on(self, dead_index: int) -> int:
        """Fail every in-flight request assigned to ``dead_index``.

        Returns the number of requests aborted (or ``-1`` if the count is
        unavailable).
        """
        ...


class RayClusterCoordinator:
    """Coordinator backed by an ``MPClient`` running on Ray DP backend.

    Delegates to existing helpers on the client so we don't re-implement
    elastic EP — we just expose them through the supervisor-friendly
    protocol.
    """

    def __init__(self, client: MPClient):
        self._client = client

    def alive_engine_indices(self) -> list[int]:
        # The client tracks live engines via core_engines (ordered by current
        # list index, not original dp_rank). Translate to ints.
        return list(range(len(self._client.core_engines)))

    def scale_down(self, dead_index: int) -> None:
        """Run the existing fault-triggered scale-down on the client's loop.

        The client already has ``_fault_triggered_scale_down`` (an async
        method that orchestrates fail-inflight + scale_down_elastic_ep).
        We schedule it on the client's event loop and wait for it.
        """
        fn = getattr(self._client, "_fault_triggered_scale_down", None)
        if fn is None:
            raise RuntimeError(
                "client does not expose _fault_triggered_scale_down; the "
                "deployment is not configured for fault-tolerant scale-down."
            )

        loop = getattr(self._client, "_ft_event_loop", None)
        if loop is None or loop.is_closed():
            raise RuntimeError("client's _ft_event_loop is unavailable")

        # Schedule on the client's loop and block until the coroutine is done.
        future = asyncio.run_coroutine_threadsafe(fn(dead_index), loop)
        future.result()

    def abort_inflight_on(self, dead_index: int) -> int:
        """Fail in-flight requests assigned to engine ``dead_index``.

        Resolves dp_rank → engine identity, then calls the existing
        ``_fail_inflight_requests_for_engine`` helper. Returns the count
        of dropped request IDs.
        """
        fail_fn = getattr(self._client, "_fail_inflight_requests_for_engine", None)
        if fail_fn is None:
            logger.warning(
                "client does not expose _fail_inflight_requests_for_engine; "
                "abort cannot reach in-flight requests."
            )
            return -1

        # The client uses a 2-byte little-endian identity for engines.
        engine_identity = dead_index.to_bytes(2, "little")
        dead_req_ids = fail_fn(engine_identity)
        count = len(dead_req_ids) if dead_req_ids else 0

        # If a higher-level abort callback is registered, propagate so the
        # OutputProcessor sends FinishReason.ABORT to clients (LoRA cleanup,
        # pooling handling, etc.). See `register_ft_abort_callback`.
        cb = getattr(self._client, "_ft_abort_callback", None)
        if cb is not None and dead_req_ids:
            try:
                cb(dead_req_ids)
            except Exception as e:
                logger.warning("ft_abort_callback raised: %s", e)

        return count


def get_cluster_coordinator(client: Any) -> ClusterCoordinator | None:
    """Resolve a coordinator from an engine client.

    Returns ``None`` if the client isn't an MPClient with the elastic-EP
    helpers, in which case cluster-scope plans aren't applicable.
    """
    if client is None:
        return None
    if not hasattr(client, "_fault_triggered_scale_down"):
        return None
    return RayClusterCoordinator(client)
