# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from abc import abstractmethod

import zmq

from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket, recv_router_dealer_message
from vllm.v1.engine.utils import broadcast_instruction, wait_for_instruction_result
from vllm.v1.serial_utils import (
    deserialize_method_call,
    run_method,
)

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
        upstream_cmd_addr: str | None,
        downstream_cmd_addr: str | None,
        dealer_identity: bytes | None,
        sentinel_index: str | None,
    ):
        self.sentinel_dead = False
        self.ctx = zmq.Context()
        self.sentinel_index = sentinel_index
        self.sentinel_name = (
            f"[{self.__class__.__name__}]"
            if sentinel_index is None
            else f"[{self.__class__.__name__}_{sentinel_index}]"
        )
        self.logger = self._make_logger(self.sentinel_name)
        if upstream_cmd_addr is not None:
            assert dealer_identity is not None
            self.upstream_cmd_socket = make_zmq_socket(
                self.ctx,
                upstream_cmd_addr,
                zmq.DEALER,
                bind=False,
                identity=dealer_identity,
            )
        if downstream_cmd_addr is not None:
            self.downstream_cmd_socket = make_zmq_socket(
                ctx=self.ctx,
                path=downstream_cmd_addr,
                socket_type=zmq.ROUTER,
                bind=True,
            )

    def _make_logger(self, prefix):
        def log(msg, *args, level="info", **kwargs):
            """
            level: "info", "warning", "error", "debug"
            msg: log message
            """
            getattr(logger, level)(prefix + msg, *args, **kwargs)

        return log

    @abstractmethod
    def run(self) -> None:
        """
        The run() method is typically launched as a separate thread when a Sentinel
        instance is created, and is used for continuous error monitoring and instruction
        reception.

        This background thread runs persistently to ensure real-time detection of errors
        and timely reception of fault tolerance instructions from upstream components
        (e.g., EngineCoreSentinel).
        """
        raise NotImplementedError

    def receive_upstream_cmd(self) -> tuple[bool, str | None]:
        """Receive commands from upstream_cmd_socket and execute them, with optional
         custom command and result transmission control.

        Core function: Listen to and read command content from the upstream command
         socket (upstream_cmd_socket), then execute the corresponding command logic.
        Additionally, it supports directly executing a passed command string (bypassing
         socket reception) and allows controlling whether to send the execution result
          back to the upstream.

        Returns:
            tuple[str, str|None]:
            - has_msg (bool): Whether there has message been received.
            - cmd_str (str): Serialized command string.
        """
        try:
            has_msg, _, cmd_str = recv_router_dealer_message(
                self.upstream_cmd_socket,
                use_poller=True,
                poll_timeout=POLL_TIMEOUT_MS,
            )
        except zmq.ZMQError:
            self.logger(
                "Socket closed, terminating %s", self.sentinel_name, level="info"
            )
            return False, None
        return has_msg, cmd_str

    def fault_listener(self) -> bool:
        raise NotImplementedError

    def _execute_cmd(self, cmd_str: str) -> tuple[bool, str, str | None]:
        """
        Execute a command received from upstream_cmd_socket.

        Args:
        cmd_str (str): JSON string representing a serialized method call.

        Returns:
            tuple[bool, str, str | None]:
            - success (bool): execution status of method call.
            - method_uuid (str): The UUID identifying the method call.
            - reason (str | None): reason for executing method call when failed.
        """
        method, method_uuid, method_params = deserialize_method_call(cmd_str)
        self.logger("Executing command: %s", method, level="info")
        try:
            success: bool = run_method(self, method, args=(), kwargs=method_params)
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
        return success, method_uuid, reason

    @abstractmethod
    def pause(self, timeout: int = 1, soft_pause: bool = True) -> bool:
        raise NotImplementedError

    @abstractmethod
    def retry(self, timeout: int = 1, new_stateless_dp_group_port: int = 8000) -> bool:
        raise NotImplementedError

    def _send_execution_result(
        self, success: bool, method_uuid: str, reason: str | None
    ):
        msg = {
            "sentinel_index": self.sentinel_index,
            "success": success,
            "method_uuid": method_uuid,
        }
        if not success and reason is not None:
            msg["reason"] = reason
        msg_bytes = json.dumps(msg).encode("utf-8")
        self.upstream_cmd_socket.send_multipart([b"", msg_bytes])

    def _broadcast_command_to_downstream(
        self,
        method_name,
        target_downstream_sentinels,
        response_timeout: int = 5,
        **kwargs,
    ):
        method_uuid = broadcast_instruction(
            self.downstream_cmd_socket,
            target_downstream_sentinels,
            method_name,
            **kwargs,
        )

        downstream_sentinel_responses = wait_for_instruction_result(
            self.downstream_cmd_socket,
            target_downstream_sentinels,
            method_name,
            response_timeout,
            method_uuid,
        )

        # check the execution results
        all_success = True
        for sentinel_identity in target_downstream_sentinels:
            response = downstream_sentinel_responses.get(sentinel_identity)

            if response is None:
                self.logger(
                    '%s did not respond to command "%s" within timeout.',
                    self.sentinel_name,
                    method_name,
                    level="info",
                )
                all_success = False
            elif not response.get("success", False):
                self.logger(
                    '%s failed to execute command "%s" (reason: %s)',
                    self.sentinel_name,
                    method_name,
                    response.get("reason", "unknown"),
                    level="error",
                )
                all_success = False

        return all_success, downstream_sentinel_responses

    def shutdown(self):
        if (
            hasattr(self, "upstream_cmd_socket")
            and self.upstream_cmd_socket is not None
        ):
            self.upstream_cmd_socket.close()
        if (
            hasattr(self, "downstream_cmd_socket")
            and self.downstream_cmd_socket is not None
        ):
            self.downstream_cmd_socket.close()
        if self.ctx is not None:
            self.ctx.term()
        self.sentinel_dead = True
