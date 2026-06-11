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
    from vllm.v1.engine.core import EngineCoreProc

logger = init_logger(__name__)

FT_UTILITY_METHOD = "handle_fault_tolerance"


class EngineCoreSentinel:
    """Manages fault tolerance state for a single engine core."""

    def __init__(self, engine: "EngineCoreProc", parallel_config):
        self.engine = engine
        self.engine_index = engine.engine_index
        self.parallel_config = parallel_config
        ft_config = parallel_config.fault_tolerance_config
        self.engine_recovery_timeout_sec = ft_config.engine_recovery_timeout_sec

        self.resumed = threading.Event()
        self.resumed.set()
        self.status_type = EngineStatusType.HEALTHY
        self._dp_reinit_epoch = 0

    # ------------------------------------------------------------------
    # Command dispatch (called from process_input_sockets thread)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Fault handling (called by wrapper, runs in busy-loop thread)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Instruction handlers (method name == instruction string)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Recovery helpers
    # ------------------------------------------------------------------

    def _reinit_dp_group(self) -> dict:
        """Reinit DP process group if in DP mode. Returns worker params."""
        engine = self.engine
        if not hasattr(engine, "dp_group") or not hasattr(engine, "dp_store"):
            return {}

        parallel_config = engine.vllm_config.parallel_config
        worker_key = f"ft_worker_dp_port_{self._dp_reinit_epoch}"
        engine_key = f"ft_engine_dp_port_{self._dp_reinit_epoch}"
        self._dp_reinit_epoch += 1

        if parallel_config.data_parallel_rank == 0:
            worker_port = get_open_port()
            engine_port = get_open_port()
            engine.dp_store.set(worker_key, str(worker_port).encode())
            engine.dp_store.set(engine_key, str(engine_port).encode())
        else:
            worker_port = int(engine.dp_store.get(worker_key).decode())
            engine_port = int(engine.dp_store.get(engine_key).decode())

        stateless_destroy_torch_distributed_process_group(engine.dp_group)
        engine.dp_group, engine.dp_store = (
            stateless_init_torch_distributed_process_group(
                parallel_config.data_parallel_master_ip,
                engine_port,
                parallel_config.data_parallel_rank,
                parallel_config.data_parallel_size,
                backend="gloo",
                return_store=True,
            )
        )
        return {"new_stateless_dp_group_port": worker_port}


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
