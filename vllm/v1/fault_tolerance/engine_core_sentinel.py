# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngineCoreSentinel and fault_tolerant_wrapper for the engine core."""

import json
import threading
from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import msgspec
from torch.distributed import TCPStore

from vllm.config import set_current_vllm_config
from vllm.distributed import (
    stateless_destroy_torch_distributed_process_group,
    stateless_init_torch_distributed_process_group,
)
from vllm.distributed.utils import get_cpu_distributed_timeout_or_none
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_port
from vllm.v1.engine import (
    FT_STATUS_CALL_ID,
    EngineCoreOutputs,
    EngineStatusType,
    UtilityOutput,
)
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest, FaultToleranceResult
from vllm.v1.request import RequestStatus
from vllm.v1.serial_utils import UtilityResult, run_method

if TYPE_CHECKING:
    from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc

logger = init_logger(__name__)

FT_UTILITY_METHOD = "handle_fault_tolerance"


class EngineCoreSentinel:
    """Manages fault tolerance state for a single engine core."""

    def __init__(self, engine: "DPEngineCoreProc", parallel_config):
        self.engine = engine
        self.engine_index = engine.engine_index
        self.parallel_config = parallel_config
        ft_config = parallel_config.fault_tolerance_config
        self.engine_recovery_timeout_sec = ft_config.engine_recovery_timeout_sec
        self.auto_recovery = ft_config.auto_recovery

        self.resumed = threading.Event()
        self.resumed.set()
        self.status_type = EngineStatusType.HEALTHY
        self.fault_info: str | None = None
        self._dp_reinit_epoch = 0
        self._initial_dp_size = parallel_config.data_parallel_size
        self._dead_dp_ranks: set[int] = set()
        # Guards against concurrent recovery: auto-recovery runs on the
        # busy-loop thread, external commands on the input-sockets thread.
        self._recovery_lock = threading.Lock()

    def handle_command(self, client_idx: int, call_id: int, ft_args: dict):
        """Dispatch an FT command by instruction name."""
        ft_request = FaultToleranceRequest(**ft_args)
        reject_reason: str | None = None
        acquired = False
        if self.status_type != EngineStatusType.UNHEALTHY:
            reject_reason = f"status is {self.status_type.name}"
        elif not (acquired := self._recovery_lock.acquire(blocking=False)):
            reject_reason = "recovery already in progress"
        if reject_reason is not None:
            reason = (
                f"[FT] Rejecting {ft_request.instruction} on engine "
                f"{self.engine_index}: {reject_reason}"
            )
            logger.warning(reason)
            result = FaultToleranceResult(
                request_id=ft_request.request_id,
                success=False,
                reason=reason,
            )
        else:
            try:
                result = run_method(self, ft_request.instruction, (ft_request,), {})
            except Exception as e:
                logger.exception("[FT] Instruction '%s' failed", ft_request.instruction)
                result = FaultToleranceResult(
                    request_id=ft_request.request_id, success=False, reason=str(e)
                )
            finally:
                if acquired:
                    self._recovery_lock.release()

        uo = UtilityOutput(call_id)
        uo.result = UtilityResult(msgspec.structs.asdict(result))
        self.engine.output_queue.put_nowait(
            (client_idx, EngineCoreOutputs(utility_output=uo))
        )

    def on_fault(self, exc: Exception):
        """Called by the wrapper when the busy loop raises an exception."""
        self.resumed.clear()
        logger.warning(
            "[FT] Busy loop raised %s. Waiting for recovery.", type(exc).__name__
        )

        engine = self.engine
        aborted = engine.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
        engine._send_abort_outputs(aborted)
        if engine.batch_queue is not None:
            engine.batch_queue.clear()
        if (
            hasattr(engine.model_executor, "is_failed")
            and engine.model_executor.is_failed
        ):
            self.status_type = EngineStatusType.DEAD
        else:
            self.status_type = EngineStatusType.UNHEALTHY
        self.fault_info = f"{type(exc).__name__}"
        logger.info(
            "[FT] Engine %d status -> %s:",
            self.engine_index,
            self.status_type.name,
            exc_info=exc,
        )

        with self._recovery_lock:
            self._push_status()
            if self.auto_recovery and self.status_type == EngineStatusType.UNHEALTHY:
                try:
                    self.auto_recover()
                except Exception:
                    logger.exception("[FT] Auto-recovery failed")

    def _push_status(self):
        """Push current health to the client so it can refresh its cache."""
        payload = {"id": self.engine_index, "status": self.status_type.name.lower()}
        if self.status_type == EngineStatusType.UNHEALTHY:
            payload["fault_info"] = self.fault_info
            try:
                payload["mask"] = self._query_mask()
            except Exception:
                logger.warning("[FT] Failed to query mask for status push")
        outputs = EngineCoreOutputs(
            utility_output=UtilityOutput(
                call_id=FT_STATUS_CALL_ID,
                result=UtilityResult(payload),
            )
        )
        outputs.engine_index = self.engine_index
        self.engine.output_queue.put_nowait((0, outputs))

    def _query_mask(self) -> list[int]:
        """Union of all workers' all2all masks.

        A rank is excluded if any worker suspects it.
        """
        ft_request = FaultToleranceRequest(instruction="query_mask", params={})
        results = self.engine.model_executor.collective_rpc(
            "handle_ft_command", args=(ft_request,)
        )
        return [max(bits) for bits in zip(*(r["mask"] for r in results))]

    def _dead_dp_ranks_from_mask(self, mask: list[int]) -> set[int]:
        """Dead DP ranks (original coordinates) from an EP all2all mask."""
        tp_size = self.parallel_config.tensor_parallel_size
        return {r // tp_size for r, v in enumerate(mask) if v}

    def _exchange_masks(self, my_mask: list[int]) -> list[int] | None:
        """The lowest alive rank unions all engines' masks via dp_store and
        publishes it back.

        Returns None on store timeout so the caller fails closed.
        """
        parallel_config = self.engine.vllm_config.parallel_config
        dp_rank = parallel_config.data_parallel_rank
        tp_size = parallel_config.tensor_parallel_size
        store = self.engine.dp_store
        epoch = self._dp_reinit_epoch
        final_key = f"ft_final_mask_{epoch}"

        store.set(f"ft_mask_{epoch}_{dp_rank}", json.dumps(my_mask).encode())
        alive = sorted(set(range(self._initial_dp_size)) - self._dead_dp_ranks)
        if dp_rank != alive[0]:
            try:
                return json.loads(store.get(final_key).decode())
            except RuntimeError:
                return None

        union_mask = list(my_mask)
        for rank in range(self._initial_dp_size):
            if rank == dp_rank or rank in self._dead_dp_ranks:
                continue
            ep_range = range(rank * tp_size, (rank + 1) * tp_size)
            if all(union_mask[i] for i in ep_range):
                continue  # presumed dead: it will never write its mask
            try:
                other = json.loads(store.get(f"ft_mask_{epoch}_{rank}").decode())
            except RuntimeError:
                return None
            union_mask = [max(a, b) for a, b in zip(union_mask, other)]
        store.set(final_key, json.dumps(union_mask).encode())
        return union_mask

    def auto_recover(self):
        """Auto-recover based on the cluster-wide all2all mask."""
        mask = self._exchange_masks(self._query_mask())
        if mask is None:
            logger.warning(
                "[FT] Auto-recovery aborted: mask exchange failed; "
                "waiting for external command"
            )
            return

        dead_dp_ranks = self._dead_dp_ranks_from_mask(mask) | self._dead_dp_ranks
        if not dead_dp_ranks - self._dead_dp_ranks:
            logger.info("[FT] Auto-recovery: no newly dead ranks, retrying")
            ft_request = FaultToleranceRequest(instruction="retry", params={})
            self.retry(ft_request)
            return

        if self.parallel_config.data_parallel_rank in dead_dp_ranks:
            logger.warning(
                "[FT] Auto-recovery aborted: this rank is masked as dead "
                "by the cluster; waiting for external command"
            )
            return

        dead = sorted(dead_dp_ranks)
        logger.info("[FT] Auto-recovery: scale_down, dead_dp_ranks=%s", dead)
        ft_request = FaultToleranceRequest(
            instruction="scale_down",
            params={"removed_dp_ranks": dead},
        )
        self.scale_down(ft_request)

    def retry(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        # Workers replay masks for the cumulative dead set.
        ft_request.params.setdefault("dead_dp_ranks", sorted(self._dead_dp_ranks))
        return self._reinit_dp_and_dispatch_command(ft_request)

    def scale_down(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        parallel_config = self.engine.vllm_config.parallel_config

        removed_set = set(ft_request.params["removed_dp_ranks"])
        my_rank = parallel_config.data_parallel_rank
        newly_dead = removed_set - self._dead_dp_ranks
        if (
            not removed_set
            or not removed_set <= set(range(self._initial_dp_size))
            or my_rank in removed_set
            or set(range(self._initial_dp_size)) <= self._dead_dp_ranks | newly_dead
            or not newly_dead
        ):
            raise ValueError(
                f"Invalid removed_dp_ranks {sorted(removed_set)} for engine "
                f"{self.engine_index} (dp_rank={my_rank}, "
                f"dead_dp_ranks={sorted(self._dead_dp_ranks)})"
            )

        new_dead = self._dead_dp_ranks | newly_dead
        ft_request.params["dead_dp_ranks"] = sorted(new_dead)
        new_alive = sorted(set(range(self._initial_dp_size)) - new_dead)

        master_ip = parallel_config.data_parallel_master_ip
        # Rank 0 hosts the TCPStore master; rebuild if it was just removed.
        if 0 in newly_dead:
            dp_store_port = ft_request.params.get("dp_store_port")
            new_master_ip = ft_request.params.get("dp_master_ip")
            if dp_store_port is None or new_master_ip is None:
                raise ValueError(
                    "dp_store_port and dp_master_ip required when rank 0 is removed"
                )
            master_ip = new_master_ip
            self._rebuild_dp_store(
                master_ip,
                dp_store_port,
                is_master=(my_rank == new_alive[0]),
                num_clients=len(new_alive),
            )

        result = self._reinit_dp_and_dispatch_command(
            ft_request, master_ip=master_ip, dead_ranks=new_dead
        )
        # Commit the dead set only after the reinit succeeded.
        self._dead_dp_ranks = new_dead
        logger.info(
            "[FT] Engine %d scale_down complete: removed %s, "
            "cumulative dead_dp_ranks=%s",
            self.engine_index,
            sorted(newly_dead),
            sorted(new_dead),
        )
        return result

    def _rebuild_dp_store(
        self,
        host: str,
        port: int,
        is_master: bool,
        num_clients: int,
    ) -> None:
        """Rebuild dp_store when the old master (rank 0) was removed."""
        timeout = get_cpu_distributed_timeout_or_none()
        if timeout is None:
            timeout = timedelta(seconds=self.engine_recovery_timeout_sec)
        self.engine.dp_store = TCPStore(
            host,
            port,
            num_clients,
            is_master=is_master,
            timeout=timeout,
        )

    def _reinit_dp_and_dispatch_command(
        self,
        ft_request: FaultToleranceRequest,
        master_ip: str | None = None,
        dead_ranks: set[int] | None = None,
    ) -> FaultToleranceResult:
        """Reinit the DP group, commit the master IP, dispatch to workers."""
        engine = self.engine
        parallel_config = engine.vllm_config.parallel_config
        params = ft_request.params
        if master_ip is None:
            master_ip = parallel_config.data_parallel_master_ip
        dead = self._dead_dp_ranks if dead_ranks is None else dead_ranks
        # The rebuilt gloo group contains only alive members; its internal
        # rank/size are dense over sorted(alive), while parallel_config keeps
        # the frozen original values.
        alive = sorted(set(range(self._initial_dp_size)) - dead)
        dense_rank = alive.index(parallel_config.data_parallel_rank)
        recovery_round = ft_request.request_id or str(self._dp_reinit_epoch)

        with set_current_vllm_config(engine.vllm_config):
            params.update(
                self._reinit_engine_groups(
                    master_ip, dense_rank, len(alive), recovery_round
                )
            )
        params["dp_master_ip"] = master_ip
        params["dp_group_rank"] = dense_rank
        params["dp_group_size"] = len(alive)
        # Commit the master IP only after the group reinit succeeded, so a
        # failed recovery leaves a consistent state that can be retried.
        parallel_config.data_parallel_master_ip = master_ip

        if hasattr(engine, "step_counter"):
            engine.step_counter = 0

        engine.model_executor.collective_rpc("handle_ft_command", args=(ft_request,))

        self.status_type = EngineStatusType.HEALTHY
        logger.info("[FT] Engine %d status -> HEALTHY", self.engine_index)
        self.resumed.set()
        self._push_status()
        return FaultToleranceResult(request_id=ft_request.request_id, success=True)

    def _reinit_engine_groups(
        self, master_ip: str, dense_rank: int, dense_size: int, recovery_round: str
    ) -> dict:
        """Reinit the engine DP group and coordinate fresh ports for the
        worker DP/EP/EPLB groups. Returns worker params."""
        engine = self.engine
        if not hasattr(engine, "dp_group") or not hasattr(engine, "dp_store"):
            return {}

        parallel_config = engine.vllm_config.parallel_config
        self._dp_reinit_epoch += 1

        worker_params: dict[str, Any] = {}

        worker_params["new_stateless_dp_group_ports"] = self._coordinate_ports(
            "ft_worker_dp_ports",
            dense_rank,
            recovery_round,
            count=parallel_config.world_size,
        )
        worker_params["new_ep_group_port"] = self._coordinate_ports(
            "ft_worker_ep_port", dense_rank, recovery_round
        )[0]
        worker_params["new_eplb_group_port"] = self._coordinate_ports(
            "ft_worker_eplb_port", dense_rank, recovery_round
        )[0]
        if parallel_config.tensor_parallel_size > 1:
            # The TP group is engine-local; no store coordination needed.
            worker_params["new_tp_group_port"] = get_open_port()

        engine_port = self._coordinate_ports(
            "ft_engine_dp_port", dense_rank, recovery_round
        )[0]
        stateless_destroy_torch_distributed_process_group(engine.dp_group)
        engine.dp_group, engine.dp_store = (
            stateless_init_torch_distributed_process_group(
                master_ip,
                engine_port,
                dense_rank,
                dense_size,
                backend="gloo",
                return_store=True,
            )
        )
        return worker_params

    def _coordinate_ports(
        self,
        key_prefix: str,
        dense_rank: int,
        recovery_round: str,
        count: int = 1,
    ) -> list[int]:
        """Dense rank 0 picks fresh ports and publishes them via dp_store;
        other ranks block-read them."""
        key = f"{key_prefix}_{recovery_round}"
        store = self.engine.dp_store
        if dense_rank == 0:
            ports = [get_open_port() for _ in range(count)]
            store.set(key, json.dumps(ports).encode())
        else:
            ports = json.loads(store.get(key).decode())
        return ports


def fault_tolerant_wrapper(busy_loop_func: Callable):
    """Wrap the busy loop to catch faults and delegate recovery."""

    def run_with_fault_tolerance(self: "EngineCoreProc"):
        while True:
            try:
                busy_loop_func(self)
            except SystemExit:
                raise
            except Exception as exc:
                if not self.enable_fault_tolerance:
                    raise
                self.ft_sentinel.on_fault(exc)
                recovered = self.ft_sentinel.resumed.wait(
                    timeout=self.ft_sentinel.engine_recovery_timeout_sec
                )
                if recovered:
                    continue
                logger.error(
                    "[FT] No recovery within %ds timeout.",
                    self.ft_sentinel.engine_recovery_timeout_sec,
                )
                raise

    return run_with_fault_tolerance
