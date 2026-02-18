# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from abc import abstractmethod

import msgspec.msgpack
import zmq

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket, recv_router_dealer_message
from vllm.v1.engine import FaultToleranceRequest, FaultToleranceResult
from vllm.v1.serial_utils import run_method

logger = init_logger(__name__)
# Polling timeout in milliseconds for non-blocking message reception
POLL_TIMEOUT_MS = 100


class BaseSentinel:
    """
    Abstract and constrain the core functionalities of the Sentinel.

    Core functionalities covered:
    - Fault listening
    - Fault tolerance instruction reception
    - Fault tolerance instruction execution
    - Upstream and downstream communication

    This class serves as the base abstraction for all LLM-related Sentinel
    implementations, enforcing standardized fault tolerance behavior across
    the system.
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
        if upstream_cmd_addr is not None:
            assert sentinel_identity is not None
            self.upstream_cmd_socket = make_zmq_socket(
                self.ctx,
                upstream_cmd_addr,
                zmq.DEALER,
                bind=False,
                identity=sentinel_identity,
            )
        if downstream_cmd_addr is not None:
            self.downstream_cmd_socket = make_zmq_socket(
                ctx=self.ctx,
                path=downstream_cmd_addr,
                socket_type=zmq.ROUTER,
                bind=True,
            )

    def _make_logger(self):
        def log(msg, *args, level="info", **kwargs):
            """
            level: "info", "warning", "error", "debug"
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

    def receive_upstream_cmd(self) -> tuple[bool, FaultToleranceRequest | None]:
        """
        This method polls the upstream command socket and attempts to receive
        a fault tolerance request.
        """
        try:
            has_msg, _, ft_request_bytes = recv_router_dealer_message(
                self.upstream_cmd_socket,
                use_poller=True,
                poll_timeout=POLL_TIMEOUT_MS,
                return_bytes=True,
            )
        except zmq.ZMQError:
            self.logger(
                "Socket closed, terminating %s", self.sentinel_name, level="info"
            )
            self.sentinel_dead = True
            return False, None
        if has_msg:
            ft_request = msgspec.msgpack.decode(
                ft_request_bytes, type=FaultToleranceRequest
            )
        else:
            ft_request = None
        return has_msg, ft_request

    def receive_and_execute_upstream_cmd(self):
        """
        Receive and execute a command from upstream sentinel and send back
        the execution result.
        """
        has_msg, ft_request = self.receive_upstream_cmd()
        if has_msg:
            assert ft_request is not None
            ft_result = self._execute_cmd(ft_request)
            self._send_execution_result(ft_result)

    def _execute_cmd(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        """
        Execute a fault tolerance command on the current object.

        Args:
            ft_request (FaultToleranceRequest): The fault tolerance request containing
                the instruction (method name), request_id (unique ID), and params
                (method parameters).

        Returns:
            FaultToleranceResult: The result of the command execution containing:
                - success (bool): True if the method ran without error; False otherwise.
                - request_id (str): Unique ID of the command.
                - reason (str | None): Error message if execution failed.
        """
        method, method_uuid, method_params = (
            ft_request.instruction,
            ft_request.request_id,
            ft_request.params,
        )
        self.logger("Executing command: %s", ft_request, level="info")
        try:
            method_kwargs = method_params or {}
            success: bool = run_method(self, method, args=(), kwargs=method_kwargs)
            self.logger("Command (%s) succeeded: %s", method, success, level="info")
            reason = None
        except Exception as e:
            self.logger(
                "Error executing method %s: %s, %s",
                method,
                type(e).__name__,
                e,
                level="error",
            )
            success = False
            reason = f"{type(e).__name__}: {e}"
        return FaultToleranceResult(
            success=success, request_id=method_uuid, reason=reason
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

    def _send_execution_result(self, ft_result: FaultToleranceResult):
        """
        Send the result of executing a command back to the upstream.
        """
        assert hasattr(self, "upstream_cmd_socket"), "upstream_cmd_socket is required"
        msg_bytes = msgspec.msgpack.encode(ft_result)
        self.upstream_cmd_socket.send_multipart([b"", msg_bytes])

    def _broadcast_command_to_downstream(
        self,
        method_name: str,
        target_downstream_sentinels: set[bytes],
        timeout: int = 5,
        **kwargs,
    ) -> tuple[bool, dict[bytes, FaultToleranceResult]]:
        """
        Broadcast a command to downstream sentinels and collect responses.
        """
        import uuid

        from vllm.v1.engine import FaultToleranceRequest

        # Create fault tolerance request
        request_id = str(uuid.uuid4())
        kwargs["timeout"] = timeout
        ft_request = FaultToleranceRequest(
            request_id=request_id,
            instruction=method_name,
            params=kwargs,
        )

        # Broadcast the instruction
        msg_bytes = msgspec.msgpack.encode(ft_request)
        for identity in target_downstream_sentinels:
            self.downstream_cmd_socket.send_multipart([identity, b"", msg_bytes])

        # Wait for responses
        responses = self.wait_for_execution_result(
            target_downstream_sentinels,
            timeout,
            ft_request,
        )
        # check the execution results
        all_success = True
        for sentinel_identity in target_downstream_sentinels:
            response = responses.get(sentinel_identity)
            if response is None:
                self.logger(
                    'Downstream sentinels timed out on "%s".',
                    method_name,
                    level="error",
                )
                all_success = False
                break
            elif not response.success:
                self.logger(
                    'Downstream sentinels failed to "%s" (reason: %s)',
                    method_name,
                    response.reason or "unknown",
                    level="error",
                )

                all_success = False
                break

        return all_success, responses

    def wait_for_execution_result(
        self,
        target_identities: set[bytes] | list[bytes],
        timeout: int,
        ft_request: "FaultToleranceRequest",
    ) -> dict[bytes, "FaultToleranceResult"]:
        """Wait for responses to a previously broadcast instruction and return
        collected responses.

        Args:
            target_identities: Identities expected to respond (bytes).
            timeout: Maximum wait time in seconds.
            ft_request: FaultToleranceRequest object to match responses.

        Returns:
            Mapping from identity (bytes) to the parsed FaultToleranceResult.
            Partial results are returned on timeout or error.
        """
        start = time.monotonic()
        responses: dict[bytes, FaultToleranceResult] = {}

        target_identities = set(target_identities)

        while target_identities:
            remaining = timeout - (time.monotonic() - start)
            if remaining <= 0:
                logger.debug(
                    "Timeout while waiting for responses for instruction %s "
                    "from identities: %s",
                    ft_request.instruction,
                    target_identities,
                )
                # Return partial results collected so far
                return responses

            try:
                has_msg, identity, response_bytes = recv_router_dealer_message(
                    self.downstream_cmd_socket,
                    use_poller=True,
                    poll_timeout=int(remaining * 1000),
                    return_bytes=True,
                )

                # Skip if no message was received during this polling period
                if not has_msg:
                    continue

                assert identity is not None
                assert response_bytes is not None
                ft_result = msgspec.msgpack.decode(
                    response_bytes, type=FaultToleranceResult
                )
                recv_uuid = ft_result.request_id

                # Ignore outdated or unrelated messages
                if recv_uuid != ft_request.request_id:
                    logger.debug(
                        "Discarding outdated response: expected request_id=%s, got %s",
                        ft_request.request_id,
                        recv_uuid,
                    )
                    continue

                # Record this endpoint's response
                responses[identity] = ft_result
                target_identities.discard(identity)

            except Exception as e:
                logger.error("Error while processing engine response: %s", e)
                # Return partial results even on exception to avoid data loss
                return responses

        return responses

    def shutdown(self):
        if (
            hasattr(self, "upstream_cmd_socket")
            and self.upstream_cmd_socket is not None
        ):
            self.upstream_cmd_socket.close(linger=0)
        if (
            hasattr(self, "downstream_cmd_socket")
            and self.downstream_cmd_socket is not None
        ):
            self.downstream_cmd_socket.close(linger=0)
        if self.ctx is not None:
            self.ctx.term()
        self.sentinel_dead = True
