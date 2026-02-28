# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
import uuid
from abc import abstractmethod

import msgspec.msgpack
import zmq

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import close_sockets, make_zmq_socket
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
        upstream_cmd_addr: str | None,
        downstream_cmd_addr: str | None,
        sentinel_identity: bytes | None,
        sentinel_tag: str | None,
    ):
        self.sentinel_dead = False
        self.ctx = zmq.Context()
        self.sentinel_tag = sentinel_tag
        self.logger = self._make_logger()
        self.vllm_config = vllm_config
        self.ft_config = vllm_config.fault_tolerance_config
        self.upstream_cmd_socket = None
        self.downstream_cmd_socket = None
        if upstream_cmd_addr is not None:
            assert sentinel_identity is not None
            self.upstream_cmd_socket = make_zmq_socket(
                self.ctx,
                upstream_cmd_addr,
                zmq.DEALER,
                bind=False,
                identity=sentinel_identity,
            )
            self.upstream_cmd_poller = zmq.Poller()
            self.upstream_cmd_poller.register(self.upstream_cmd_socket, zmq.POLLIN)
        if downstream_cmd_addr is not None:
            self.downstream_cmd_socket = make_zmq_socket(
                ctx=self.ctx,
                path=downstream_cmd_addr,
                socket_type=zmq.ROUTER,
                bind=True,
            )
            self.downstream_cmd_poller = zmq.Poller()
            self.downstream_cmd_poller.register(self.downstream_cmd_socket, zmq.POLLIN)

    def _make_logger(self):
        def log(msg, *args, level="info", **kwargs):
            """
            msg: log message
            """
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
        The run() method is launched as a separate thread when a Sentinel
        instance is created.

        This background thread typically runs persistently to ensure real-time
        detection of errors and timely reception of fault tolerance instructions
        from upstream sentinels.
        """
        raise NotImplementedError

    def poll_and_execute_upstream_cmd(self):
        """
        Receive and execute a command from upstream sentinel and send back
        the execution result.
        """
        assert self.upstream_cmd_socket is not None
        try:
            # Polls the upstream command socket
            events = self.upstream_cmd_poller.poll(timeout=100)
            if events:
                _, ft_request_bytes = self.upstream_cmd_socket.recv_multipart()
                ft_request = msgspec.msgpack.decode(
                    ft_request_bytes, type=FaultToleranceRequest
                )
                ft_result = self._execute_cmd(ft_request)
                msg_bytes = msgspec.msgpack.encode(ft_result)
                self.upstream_cmd_socket.send_multipart([b"", msg_bytes])
        except zmq.ZMQError:
            self.logger(
                "Socket closed, terminating %s", self.sentinel_name, level="info"
            )
            self.sentinel_dead = True

    def _execute_cmd(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        method = ft_request.instruction
        self.logger("Executing command: %s", ft_request, level="info")
        try:
            success: bool = run_method(self, method, args=(), kwargs=ft_request.params)
            self.logger("Command (%s) succeeded: %s", method, success, level="info")
            reason = None
        except Exception as e:
            self.logger(
                "Error executing ft request: %s",
                ft_request,
                level="error",
            )
            success = False
            reason = f"{type(e).__name__}: {e}"
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

    def _execute_command_on_downstreams(
        self,
        method_name: str,
        target_downstream_sentinels: set[bytes],
        timeout: int = 5,
        **kwargs,
    ) -> tuple[bool, dict[bytes, FaultToleranceResult]]:
        """
        Broadcast a command to downstream sentinels and collect responses.
        """
        assert self.downstream_cmd_socket is not None
        # Create fault tolerance request
        kwargs["timeout"] = timeout
        ft_request = FaultToleranceRequest(
            request_id=str(uuid.uuid4()),
            instruction=method_name,
            params=kwargs,
        )
        # Broadcast the instruction
        msg_bytes = msgspec.msgpack.encode(ft_request)
        for identity in target_downstream_sentinels:
            self.downstream_cmd_socket.send_multipart([identity, b"", msg_bytes])
        # Wait for responses
        responses = self._wait_for_execution_result(
            target_downstream_sentinels,
            timeout,
            ft_request,
        )
        # check the execution results
        for sentinel_identity in target_downstream_sentinels:
            response = responses.get(sentinel_identity)
            if response is None:
                self.logger(
                    'Downstream sentinels timed out on "%s".',
                    method_name,
                    level="error",
                )
                return False, responses
            elif not response.success:
                self.logger(
                    'Downstream sentinels failed to "%s" (reason: %s)',
                    method_name,
                    response.reason or "unknown",
                    level="error",
                )
                return False, responses

        return True, responses

    def _wait_for_execution_result(
        self,
        target_identities: set[bytes] | list[bytes],
        timeout: int,
        ft_request: "FaultToleranceRequest",
    ) -> dict[bytes, "FaultToleranceResult"]:
        """Collect responses for the given request.
        Returns partial results on timeout or error.
        """
        assert self.downstream_cmd_socket is not None
        deadline = time.monotonic() + timeout
        responses: dict[bytes, FaultToleranceResult] = {}
        pending = set(target_identities)
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                events = self.downstream_cmd_poller.poll(int(remaining * 1000))
                if not events:
                    break
                identity, _, payload = self.downstream_cmd_socket.recv_multipart()
                result = msgspec.msgpack.decode(payload, type=FaultToleranceResult)
                # Ignore unrelated responses
                if result.request_id != ft_request.request_id:
                    logger.debug("Discarding outdated response: %s", result)
                    continue

                responses[identity] = result
                pending.discard(identity)
            except Exception as e:
                logger.error("Error while processing engine response: %s", e)
                break

        return responses

    def shutdown(self):
        close_sockets([self.upstream_cmd_socket, self.downstream_cmd_socket])
        self.ctx.term()
        self.sentinel_dead = True
