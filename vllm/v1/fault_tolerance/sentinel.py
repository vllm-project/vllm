# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import traceback
from abc import abstractmethod

import zmq

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.fault_tolerance.utils import (
    FaultToleranceRequest,
    FaultToleranceResult,
)
from vllm.v1.serial_utils import run_method

logger = init_logger(__name__)


class BaseSentinel:
    """
    Core functionalities of the sentinel covered:
    - Fault listening
    - Fault tolerance instruction reception
    - Fault tolerance instruction execution
    - Upstream and downstream communication
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        sentinel_tag: str | None,
        identity: bytes,
    ):
        self.sentinel_dead = False
        self.ctx = zmq.Context()
        self.sentinel_tag = sentinel_tag
        self.logger = self._make_logger()
        self.vllm_config = vllm_config
        self.ft_config = vllm_config.fault_tolerance_config
        self.identity = identity

    def _make_logger(self):
        def log(msg, *args, level="info", **kwargs):
            """msg: log message"""
            prefix = self.sentinel_name
            getattr(logger, level)(prefix + msg, *args, **kwargs)

        return log

    @property
    def sentinel_name(self) -> str:
        if self.sentinel_tag is None:
            return f"[{self.__class__.__name__}] "
        return f"[{self.__class__.__name__}_{self.sentinel_tag}] "

    @abstractmethod
    def run(self) -> None:
        """
        Run continuously to listen for control commands or error signals;
        on receipt, execute the command or report the result.
        """
        raise NotImplementedError

    def _execute_cmd(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        method = ft_request.instruction
        self.logger("Executing command: %s", ft_request, level="info")
        try:
            success: bool = run_method(self, method, args=(), kwargs=ft_request.params)
            self.logger("Command (%s) succeeded: %s", method, success, level="info")
            reason = None
        except Exception as e:
            self.logger("Error executing ft request: %s", ft_request, level="error")
            success = False
            reason = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return FaultToleranceResult(
            success=success, request_id=ft_request.request_id, reason=reason
        )

    @abstractmethod
    def pause(self, timeout: int = 1, **kwargs) -> bool:
        """
        Pause the vLLM instance to enter fault-tolerance mode.
        This method should be called when a fault is detected. It pauses the
        execution, allowing the system to wait for fault-tolerance instructions
        (e.g., retry, scale-down, or other control commands).
        """
        raise NotImplementedError

    @abstractmethod
    def retry(self, timeout: int = 1, **kwargs) -> bool:
        """
        Retry execution after a transient recoverable fault.
        """
        raise NotImplementedError

    def shutdown(self):
        self.ctx.term()
        self.sentinel_dead = True
