# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fault-tolerance coordination for the EngineCore client."""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgspec

from vllm.logger import init_logger
from vllm.v1.fault_tolerance.engine_core_sentinel import FT_UTILITY_METHOD
from vllm.v1.fault_tolerance.utils import (
    FaultToleranceRequest,
    FaultToleranceResult,
)

if TYPE_CHECKING:
    from vllm.v1.engine.core_client import AsyncMPClient

logger = init_logger(__name__)


@dataclass
class _ScaleDownRecovery:
    request_id: str
    removed_ranks: frozenset[int]
    result: FaultToleranceResult
    frontend_synced: bool = False


class FrontendSentinel:
    """Coordinate engine recovery and frontend topology updates for one client."""

    def __init__(self, client: "AsyncMPClient") -> None:
        self.client = client
        self.lock = asyncio.Lock()
        self._frontend_sync_status: dict[str, Any] | None = None
        self._scale_down_recovery: _ScaleDownRecovery | None = None
        self.reset_after_scale_reconfiguration(
            client.vllm_config.parallel_config.data_parallel_size
        )

    def reset_after_scale_reconfiguration(self, new_data_parallel_size: int) -> None:
        self.active_dp_ranks = list(range(new_data_parallel_size))
        self.baseline_dp_size = new_data_parallel_size
        self._scale_down_recovery = None

    def validate_scale_up(self, new_data_parallel_size: int) -> None:
        parallel_config = self.client.vllm_config.parallel_config
        old_data_parallel_size = parallel_config.data_parallel_size
        assert (
            new_data_parallel_size <= old_data_parallel_size
            or old_data_parallel_size >= self.baseline_dp_size
            or parallel_config.all2all_backend == "nixl_ep"
        )

        if old_data_parallel_size < new_data_parallel_size < self.baseline_dp_size:
            raise ValueError(
                "Partial scale-up after fault recovery is not supported; "
                f"restore at least {self.baseline_dp_size} DP ranks."
            )

    async def handle_fault(
        self, fault_tolerance_request: FaultToleranceRequest
    ) -> FaultToleranceResult:
        """Recover the engine and synchronize its frontend topology."""
        if fault_tolerance_request.instruction != "scale_down":
            result = await self._dispatch(fault_tolerance_request)
            cached = self._scale_down_recovery
            if result.success and (cached is None or cached.frontend_synced):
                self._frontend_sync_status = None
            return result
        async with self.lock:
            self._frontend_sync_status = {
                "ft_state": "recovering",
                "last_ft_request_id": fault_tolerance_request.request_id,
            }
            try:
                result = await self._handle_scale_down(fault_tolerance_request)
            except Exception as exc:
                self._frontend_sync_status.update(
                    ft_state="failed", ft_error=str(exc) or type(exc).__name__
                )
                raise
            if result.success:
                self._frontend_sync_status = None
            else:
                self._frontend_sync_status.update(
                    ft_state="failed", ft_error=result.reason
                )
            return result

    async def _handle_scale_down(
        self, request: FaultToleranceRequest
    ) -> FaultToleranceResult:
        """Recover EngineCore once and synchronize the frontend topology.

        The caller holds `self.lock`. A retry of the same request resumes
        frontend synchronization after a successful EngineCore recovery.
        """
        parallel = self.client.vllm_config.parallel_config
        removed_ranks = request.params.get("removed_dp_ranks")
        if (
            not isinstance(removed_ranks, list)
            or not removed_ranks
            or any(type(rank) is not int for rank in removed_ranks)
        ):
            raise ValueError(
                "'removed_dp_ranks' must be a non-empty list of integer rank IDs."
            )
        removed = frozenset(removed_ranks)
        recovery = self._scale_down_recovery
        if recovery is not None:
            same_request = (
                recovery.request_id == request.request_id
                and recovery.removed_ranks == removed
            )
            if not recovery.frontend_synced and not same_request:
                raise ValueError(
                    "Frontend migration is pending; retry the same scale_down "
                    "request_id and removed_dp_ranks."
                )
            if same_request and recovery.frontend_synced:
                return recovery.result
            if not same_request:
                recovery = None

        migrate_store = (
            parallel.enable_elastic_ep
            and parallel.data_parallel_external_lb
            and parallel._data_parallel_master_rank in removed
        )
        if migrate_store:
            self._validate_coord_store_migration(request)
        if recovery is None:
            result = await self._dispatch(request)
            if not result.success:
                return result
            recovery = _ScaleDownRecovery(request.request_id, removed, result)
            self._scale_down_recovery = recovery

        await self._sync_frontend_after_scale_down(request, removed, migrate_store)
        recovery.frontend_synced = True
        return recovery.result

    @staticmethod
    def _validate_coord_store_migration(request: FaultToleranceRequest) -> None:
        """Check migration inputs before EngineCore changes its topology."""
        params = request.params
        port = params.get("coord_store_port")
        if (
            not params.get("dp_master_ip")
            or type(port) is not int
            or not 1 <= port <= 65535
            or port == params.get("dp_store_port")
            or not request.request_id
        ):
            raise ValueError(
                "Removing the external Elastic EP master requires dp_master_ip, "
                "a shared coord_store_port distinct from dp_store_port, and "
                "a non-empty request_id."
            )

    async def _sync_frontend_after_scale_down(
        self,
        request: FaultToleranceRequest,
        removed: frozenset[int],
        migrate_store: bool,
    ) -> None:
        """Migrate the store if needed, then publish the surviving DP topology."""
        parallel = self.client.vllm_config.parallel_config
        active = [rank for rank in self.active_dp_ranks if rank not in removed]
        if migrate_store:
            params = request.params
            await self.client.recover_elastic_ep_coord_store(
                params["dp_master_ip"],
                params["coord_store_port"],
                active.index(parallel.data_parallel_rank),
                len(active),
                request.request_id,
            )
            parallel._data_parallel_master_rank = active[0]
        if parallel.enable_eplb:
            num_logical = self.client.vllm_config.model_config.get_num_experts()
            num_local_experts = (
                num_logical + parallel.eplb_config.num_redundant_experts
            ) // parallel.data_parallel_size
            parallel.eplb_config.num_redundant_experts = (
                num_local_experts * len(active) - num_logical
            )
        parallel.data_parallel_size = len(active)
        self.active_dp_ranks = active
        logger.info("[FT] Frontend topology synchronized: active_dp_ranks=%s", active)

    async def _dispatch(
        self, ft_request: FaultToleranceRequest
    ) -> FaultToleranceResult:
        status = self.client._engine_status[self.client.engine_ranks_managed[0]]

        # Reject a second recovery while one is in flight
        if status.get("ft_state") == "recovering":
            return FaultToleranceResult(
                request_id=ft_request.request_id,
                success=False,
                reason="another recovery is in progress",
            )

        status["ft_state"] = "recovering"
        status["last_ft_request_id"] = ft_request.request_id
        status.pop("ft_error", None)

        try:
            res = await self.client.call_utility_async(FT_UTILITY_METHOD, ft_request)
            result = msgspec.convert(res, FaultToleranceResult)
        except Exception as e:
            result = FaultToleranceResult(ft_request.request_id, False, str(e))

        if not result.success:
            # Status pushes replace the cached entry; re-read before recording.
            status = self.client._engine_status[self.client.engine_ranks_managed[0]]
            status["ft_state"] = "failed"
            status["last_ft_request_id"] = result.request_id
            status["ft_error"] = result.reason

        return result

    def get_status(self) -> dict:
        status: dict[str, Any] = {
            "schema_version": 1,
            "total_engines": len(self.client.engine_ranks_managed),
            "engines": list(self.client._engine_status.values()),
        }
        if self._frontend_sync_status is not None:
            # EngineCore can be healthy before the frontend store migration ends.
            status["engines"] = [
                {**engine, **self._frontend_sync_status} for engine in status["engines"]
            ]
        return status
