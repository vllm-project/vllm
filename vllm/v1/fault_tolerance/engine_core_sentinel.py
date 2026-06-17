# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngineCoreSentinel and fault_tolerant_wrapper for the engine core."""

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.config import set_current_vllm_config
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.distributed.utils import stateless_init_torch_distributed_process_group
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_port
from vllm.v1.engine import EngineCoreOutputs, EngineStatusType, UtilityOutput
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest
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

        self.resumed = threading.Event()
        self.resumed.set()
        self.status_type = EngineStatusType.HEALTHY
        self._dp_reinit_epoch = 0

    def handle_command(self, client_idx: int, call_id: int, ft_args: dict):
        """Dispatch an FT command by instruction name and enqueue result."""
        ft_request = FaultToleranceRequest(**ft_args)
        try:
            result = run_method(self, ft_request.instruction, (ft_request,), {})
        except Exception as e:
            logger.exception("[FT] Instruction '%s' failed", ft_request.instruction)
            result = {
                "request_id": ft_request.request_id,
                "success": False,
                "reason": str(e),
            }

        uo = UtilityOutput(call_id)
        uo.result = UtilityResult(result)
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

        self.status_type = EngineStatusType.UNHEALTHY
        logger.info(
            "[FT] Engine %d status -> UNHEALTHY:", self.engine_index, exc_info=exc
        )

    def status(self, ft_request: FaultToleranceRequest) -> dict:
        return {
            "request_id": ft_request.request_id,
            "success": True,
            "engine_id": self.engine_index,
            "status": self.status_type.name.lower(),
        }

    def retry(self, ft_request: FaultToleranceRequest) -> dict:
        engine = self.engine
        executor = engine.model_executor

        with set_current_vllm_config(engine.vllm_config):
            ft_request.params.update(self._reinit_dp_group())
        if hasattr(engine, "step_counter"):
            engine.step_counter = 0

        executor.collective_rpc("handle_ft_command", args=(ft_request,))

        self.status_type = EngineStatusType.HEALTHY
        logger.info("[FT] Engine %d status -> HEALTHY", self.engine_index)
        self.resumed.set()
        return {"request_id": ft_request.request_id, "success": True}

    def scale_down(self, ft_request: FaultToleranceRequest) -> dict:
        engine = self.engine
        parallel_config = engine.vllm_config.parallel_config
        removed_dp_ranks = ft_request.params["removed_dp_ranks"]

        old_dp_size = parallel_config.data_parallel_size
        old_dp_rank = parallel_config.data_parallel_rank
        new_dp_size = old_dp_size - len(removed_dp_ranks)
        removed_set = set(removed_dp_ranks)

        # Densify: map old sparse ranks to new contiguous ranks.
        surviving = [r for r in range(old_dp_size) if r not in removed_set]
        new_dp_rank = surviving.index(old_dp_rank)

        parallel_config.data_parallel_size = new_dp_size
        parallel_config.data_parallel_rank = new_dp_rank
        engine.dp_rank = new_dp_rank
        engine.dp_size = new_dp_size
        self.engine_index = new_dp_rank

        # Rank 0 hosts the TCPStore master; rebuild if it was removed.
        if 0 in removed_set and hasattr(engine, "dp_store"):
            dp_store_port = ft_request.params.get("dp_store_port")
            new_master_ip = ft_request.params.get("dp_master_ip")
            if dp_store_port is None or new_master_ip is None:
                raise ValueError(
                    "dp_store_port and dp_master_ip required when rank 0 is removed "
                )
            parallel_config.data_parallel_master_ip = new_master_ip
            self._rebuild_dp_store(
                parallel_config.data_parallel_master_ip,
                dp_store_port,
                new_dp_rank,
                new_dp_size,
            )

        with set_current_vllm_config(engine.vllm_config):
            reinit_result = self._reinit_dp_group(
                new_dp_size=new_dp_size,
                new_dp_rank=new_dp_rank,
            )

        if hasattr(engine, "step_counter"):
            engine.step_counter = 0

        ft_request.params.update(
            {
                "new_dp_size": new_dp_size,
                "new_dp_rank": new_dp_rank,
            }
        )
        ft_request.params.update(reinit_result)

        engine.model_executor.collective_rpc("handle_ft_command", args=(ft_request,))

        self.status_type = EngineStatusType.HEALTHY
        logger.info(
            "[FT] Engine %d scale_down complete: dp_size %d->%d, "
            "dp_rank %d->%d, removed %s",
            self.engine_index,
            old_dp_size,
            new_dp_size,
            old_dp_rank,
            new_dp_rank,
            removed_dp_ranks,
        )
        self.resumed.set()
        return {"request_id": ft_request.request_id, "success": True}

    def _rebuild_dp_store(
        self,
        host: str,
        port: int,
        dp_rank: int,
        dp_size: int,
    ) -> None:
        """Rebuild dp_store when the old master (rank 0) is dead."""
        from datetime import timedelta

        from torch.distributed import TCPStore

        self.engine.dp_store = TCPStore(
            host,
            port,
            dp_size,
            is_master=(dp_rank == 0),
            timeout=timedelta(seconds=self.engine_recovery_timeout_sec),
        )

    def _reinit_dp_group(
        self,
        new_dp_size: int | None = None,
        new_dp_rank: int | None = None,
    ) -> dict:
        """Reinit DP process group. Returns worker params."""
        engine = self.engine
        if not hasattr(engine, "dp_group") or not hasattr(engine, "dp_store"):
            return {}

        parallel_config = engine.vllm_config.parallel_config
        dp_rank = (
            new_dp_rank
            if new_dp_rank is not None
            else (parallel_config.data_parallel_rank)
        )
        dp_size = (
            new_dp_size
            if new_dp_size is not None
            else (parallel_config.data_parallel_size)
        )
        dp_master_ip = parallel_config.data_parallel_master_ip

        worker_port = self._coordinate_port("ft_worker_dp_port")
        engine_port = self._coordinate_port("ft_engine_dp_port")
        self._dp_reinit_epoch += 1

        stateless_destroy_torch_distributed_process_group(engine.dp_group)
        engine.dp_group, engine.dp_store = (
            stateless_init_torch_distributed_process_group(
                dp_master_ip,
                engine_port,
                dp_rank,
                dp_size,
                backend="gloo",
                return_store=True,
            )
        )
        return {"new_stateless_dp_group_port": worker_port}

    def _coordinate_port(self, key_prefix: str) -> int:
        """Rank 0 picks a fresh port, publishes via dp_store;
        others block-read it."""
        key = f"{key_prefix}_{self._dp_reinit_epoch}"
        dp_rank = self.engine.vllm_config.parallel_config.data_parallel_rank
        if dp_rank == 0:
            port = get_open_port()
            self.engine.dp_store.set(key, str(port).encode())
        else:
            port = int(self.engine.dp_store.get(key).decode())
        return port


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
