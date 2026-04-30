# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ClusterCoordinator — protocol over the multi-engine manager.

Wraps whatever multi-engine manager the deployment uses (Ray actor manager
or the multiprocessing equivalent) so cluster-scope recovery plans can be
written backend-agnostically. Today only ``RayClusterCoordinator`` is
provided; an MP equivalent is straightforward to add.

Cluster-scope plans (``scale_down``, future ``scale_up``) call methods on
this protocol rather than reaching into Ray-specific code.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vllm.logger import init_logger

logger = init_logger(__name__)


@runtime_checkable
class ClusterCoordinator(Protocol):
    """Protocol over the multi-engine manager.

    Cluster-scope recovery plans (e.g., ``scale_down``) take a coordinator
    and call these methods. Implementations wrap whatever the deployment
    uses (Ray ``CoreEngineActorManager``, multiprocessing client, etc.).
    """

    def alive_engine_indices(self) -> list[int]:
        """Engine indices that are currently considered alive."""
        ...

    def call_engines(
        self,
        indices: list[int],
        method: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> dict[int, Any]:
        """Invoke a method on the given engine indices and aggregate results."""
        ...

    def abort_inflight_on(self, dead_index: int) -> None:
        """Abort in-flight requests assigned to a dead engine."""
        ...


class RayClusterCoordinator:
    """Ray-backed coordinator.

    Wraps an existing ``CoreEngineActorManager`` (or equivalent) without
    re-implementing any of the elastic EP machinery. Cluster-scope plans
    use this to talk to surviving engines.
    """

    def __init__(self, actor_manager: Any):
        self._actor_manager = actor_manager

    def alive_engine_indices(self) -> list[int]:
        # The actor manager is expected to expose the alive set; fall back
        # to "all configured engines" if it doesn't.
        fn = getattr(self._actor_manager, "alive_engine_indices", None)
        if fn is None:
            indices = getattr(self._actor_manager, "engine_indices", None)
            if indices is None:
                return []
            return list(indices)
        return list(fn())

    def call_engines(
        self,
        indices: list[int],
        method: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> dict[int, Any]:
        kwargs = kwargs or {}
        results: dict[int, Any] = {}
        for idx in indices:
            try:
                actor = self._actor_manager.get_actor(idx)
            except Exception as e:
                results[idx] = e
                continue
            fn = getattr(actor, method, None)
            if fn is None:
                results[idx] = AttributeError(f"engine {idx} has no method '{method}'")
                continue
            try:
                # Ray actor methods return ObjectRefs; the caller is
                # responsible for `ray.get` on them. We collect the refs.
                results[idx] = fn(*args, **kwargs)
            except Exception as e:
                results[idx] = e
        return results

    def abort_inflight_on(self, dead_index: int) -> None:
        fn = getattr(self._actor_manager, "abort_inflight_on", None)
        if fn is None:
            logger.warning(
                "actor_manager has no abort_inflight_on; skipping abort for engine %d",
                dead_index,
            )
            return
        try:
            fn(dead_index)
        except Exception as e:
            logger.warning("abort_inflight_on(%d) failed: %s", dead_index, e)


def get_cluster_coordinator(client: Any) -> ClusterCoordinator | None:
    """Resolve a coordinator from an engine client.

    Returns ``None`` if the client doesn't expose a cluster manager — in
    which case cluster-scope plans aren't applicable to this deployment.
    """
    actor_manager = getattr(client, "core_engine_actor_manager", None)
    if actor_manager is None:
        actor_manager = getattr(client, "actor_manager", None)
    if actor_manager is None:
        return None
    return RayClusterCoordinator(actor_manager)
