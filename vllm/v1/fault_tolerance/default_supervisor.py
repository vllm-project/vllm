# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DefaultFaultSupervisor — the supervisor vLLM ships out of the box.

One instance per engine process. Owns:

* the per-engine FaultStatus state machine,
* a non-blocking publish path (queue + daemon thread → FaultStateBus),
* a small interrupt PUB socket scoped to operations that genuinely need an
  aux thread (e.g., `ncclCommAbort` on a stuck worker — see §9.4 / §9.11
  of the architecture doc),
* dispatch from `/fault_tolerance/apply` to registered RecoveryPlans.

The supervisor is registered by name (``"default"``) so orchestrators can
override it with their own implementation via ``@register_fault_supervisor``.
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.fault_tolerance.errors import FaultToleranceError
from vllm.v1.fault_tolerance.fault_state_bus import FaultStateBus
from vllm.v1.fault_tolerance.registry import (
    get_plan,
    register_fault_supervisor,
)
from vllm.v1.fault_tolerance.types import (
    Disposition,
    EngineLocalRecoveryPlan,
    FaultInfo,
    FaultSignal,
    FaultStatus,
    FaultToleranceHooks,
    FaultToleranceRequest,
    FaultToleranceResult,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)


_PUBLISH_QUEUE_MAX = 1024
_PUBLISH_LOOP_TIMEOUT_SEC = 1.0


@register_fault_supervisor("default")
class DefaultFaultSupervisor(FaultToleranceHooks):
    """Reference supervisor implementation."""

    def __init__(self, vllm_config: VllmConfig, engine: Any | None = None):
        self.vllm_config = vllm_config
        self.engine = engine
        self.engine_index = (
            getattr(engine, "engine_index", 0) if engine is not None else 0
        )

        self.bus = FaultStateBus(vllm_config)

        self._state: FaultStatus = FaultStatus.HEALTHY
        self._engine_status: dict[int, FaultStatus] = {self.engine_index: self._state}
        self._state_lock = threading.Lock()
        self._pause_cv = threading.Condition(self._state_lock)

        self._queue: queue.Queue[FaultInfo] = queue.Queue(maxsize=_PUBLISH_QUEUE_MAX)
        self._shutdown = False
        self._publish_thread = threading.Thread(
            target=self._publish_loop,
            daemon=True,
            name=f"DefaultFaultSupervisorPublishThread_{self.engine_index}",
        )
        self._publish_thread.start()

        # Small interrupt PUB socket (§9.4 / §9.11). Used ONLY for operations
        # that need an aux thread on the worker side (e.g.
        # ncclCommAbort). Common pause / retry actions go via
        # `executor.collective_rpc` and don't touch this socket.
        self._interrupt_pub = self._init_interrupt_pub()

    def _init_interrupt_pub(self):
        try:
            import zmq

            from vllm.utils.network_utils import make_zmq_socket
        except ImportError:
            return None
        ft_cfg = getattr(self.vllm_config, "fault_tolerance_config", None)
        addr = getattr(ft_cfg, "interrupt_addr", None) if ft_cfg else None
        if addr is None:
            return None
        try:
            return make_zmq_socket(
                ctx=zmq.Context.instance(),
                path=addr,
                socket_type=zmq.PUB,
                bind=True,
            )
        except Exception as e:
            logger.warning("Failed to bind interrupt PUB at %s: %s", addr, e)
            return None

    # ---- engine-thread hooks (must not block) ------------------------------

    def before_step(self, engine: Any) -> None:
        with self._state_lock:
            while self._state is FaultStatus.PAUSED and not self._shutdown:
                self._pause_cv.wait()

    def after_step(self, engine: Any, output: ModelRunnerOutput | None) -> None:
        if output is None:
            return
        signal: FaultSignal | None = getattr(output, "fault_signal", None)
        if signal is None:
            return
        info = FaultInfo.from_signal(self.engine_index, signal)
        self._set_status(info.status)
        self._enqueue(info)

    def on_step_error(self, engine: Any, exc: BaseException) -> Disposition:
        # Typed FT exceptions carry a structured signal; fall back to
        # FaultInfo.from_exception for everything else.
        if isinstance(exc, FaultToleranceError) and exc.signal is not None:
            info = FaultInfo.from_signal(self.engine_index, exc.signal)
        else:
            info = FaultInfo.from_exception(self.engine_index, exc)
        self._set_status(info.status)
        self._enqueue(info)
        # Default policy: pause and let the orchestrator decide what to do.
        # Orchestrators that want propagate-on-first-error can register a
        # different supervisor.
        return Disposition.PAUSE_LOOP

    # ---- public surface for HTTP / engine ----------------------------------

    def apply(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        """Dispatch an instruction to its registered plan."""
        try:
            plan = get_plan(ft_request.instruction)
        except KeyError as e:
            return FaultToleranceResult(
                success=False, reason=str(e), request_id=ft_request.request_id
            )
        if not plan.is_applicable(self.vllm_config):
            return FaultToleranceResult(
                success=False,
                reason=(
                    f"Plan '{ft_request.instruction}' is not applicable "
                    "to this vllm_config."
                ),
                request_id=ft_request.request_id,
            )

        if isinstance(plan, EngineLocalRecoveryPlan):
            executor = getattr(self.engine, "executor", None) if self.engine else None
            if executor is None:
                return FaultToleranceResult(
                    success=False,
                    reason="No executor on engine; supervisor cannot run plan.",
                    request_id=ft_request.request_id,
                )
            try:
                result = plan.execute(executor, ft_request.params)
            except Exception as e:
                logger.exception("Plan %s failed: %s", ft_request.instruction, e)
                return FaultToleranceResult(
                    success=False,
                    reason=f"plan raised: {e}",
                    request_id=ft_request.request_id,
                )
            if result.request_id is None:
                result.request_id = ft_request.request_id
            return result

        # Cluster plans are not dispatched per-engine; the API layer routes
        # them through the ClusterCoordinator instead.
        return FaultToleranceResult(
            success=False,
            reason=(
                f"Plan '{ft_request.instruction}' is a ClusterRecoveryPlan; "
                "must be dispatched at the API layer with a ClusterCoordinator."
            ),
            request_id=ft_request.request_id,
        )

    def get_status(self) -> dict[int, FaultStatus]:
        with self._state_lock:
            return dict(self._engine_status)

    def run_retry_plan(self, engine: Any) -> FaultToleranceResult:
        self._set_status(FaultStatus.RECOVERING)
        result = self.apply(FaultToleranceRequest(instruction="retry", params={}))
        if result.success:
            self.resume()
        return result

    def resume(self) -> None:
        """Move from PAUSED/RECOVERING back to HEALTHY and wake before_step."""
        with self._state_lock:
            self._state = FaultStatus.HEALTHY
            self._engine_status[self.engine_index] = FaultStatus.HEALTHY
            self._pause_cv.notify_all()
        self._enqueue(
            FaultInfo(engine_index=self.engine_index, status=FaultStatus.HEALTHY)
        )

    def wait_until_resumed(self) -> None:
        with self._state_lock:
            while self._state is FaultStatus.PAUSED and not self._shutdown:
                self._pause_cv.wait()

    def send_interrupt(self, target: int | str, command: str) -> None:
        """Fire a one-way interrupt to a worker's aux thread (§9.11).

        ``target`` is either a worker local rank or the literal string ``"all"``.
        ``command`` is one of the interrupt commands the worker's
        `_ft_interrupt_loop` understands (today: ``"abort_communicator"``).
        """
        if self._interrupt_pub is None:
            logger.warning(
                "send_interrupt called but interrupt PUB socket is not bound"
            )
            return
        topic = f"worker_{target}" if target != "all" else "all"
        try:
            self._interrupt_pub.send_multipart(
                [topic.encode("utf-8"), command.encode("utf-8")]
            )
        except Exception as e:
            logger.warning("send_interrupt failed: %s", e)

    # ---- internal ----------------------------------------------------------

    def _set_status(self, status: FaultStatus) -> None:
        with self._state_lock:
            self._state = status
            self._engine_status[self.engine_index] = status

    def _enqueue(self, info: FaultInfo) -> None:
        try:
            self._queue.put_nowait(info)
        except queue.Full:
            logger.warning(
                "fault publish queue full; dropping %s for engine %d",
                info.kind or info.status.name,
                info.engine_index,
            )

    def _publish_loop(self) -> None:
        while not self._shutdown:
            try:
                info = self._queue.get(timeout=_PUBLISH_LOOP_TIMEOUT_SEC)
            except queue.Empty:
                # Natural extension point for a future heartbeat / liveness
                # probe (see vLLM FT design §10.2 Mode B).
                continue
            try:
                self.bus.publish(info)
            except Exception as e:
                logger.warning("Bus publish failed: %s", e)

    def shutdown(self) -> None:
        from contextlib import suppress

        self._shutdown = True
        with self._state_lock:
            self._pause_cv.notify_all()
        with suppress(Exception):
            self._publish_thread.join(timeout=2.0)
        self.bus.close()
        if self._interrupt_pub is not None:
            with suppress(Exception):
                self._interrupt_pub.close(linger=0)
            self._interrupt_pub = None
