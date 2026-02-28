# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

import msgspec

from vllm.config import FaultToleranceConfig
from vllm.v1.utils import get_engine_client_zmq_addr


class FaultInfo(msgspec.Struct):
    type: str
    message: str
    engine_id: str
    timestamp: str | None = None
    additional_info: dict | None = None

    def __post_init__(self):
        # If no exit time is specified, the current timestamp will be used by default.

        local_time = time.localtime(time.time())
        if self.timestamp is None:
            self.timestamp = time.strftime("%H:%M:%S", local_time)

    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        engine_id: str | int,
        additional_info: dict | None = None,
    ) -> "FaultInfo":
        """Create FaultInfo from an exception."""
        return cls(
            type=type(exception).__name__,
            message=str(exception),
            engine_id=str(engine_id),
            additional_info=additional_info or {},
        )


class FaultToleranceResult(msgspec.Struct):
    """
    Result of applying fault tolerance instructions.
    """

    request_id: str
    success: bool
    reason: str | None = None


class FaultToleranceRequest(msgspec.Struct):
    """
    Request for fault tolerance instructions, used in the fault tolerance protocol.
    """

    request_id: str
    instruction: str
    params: dict[str, Any]


@dataclass
class FaultToleranceZmqAddresses:
    # ZMQ fault_state_pub_socket address of client sentinel
    fault_state_pub_socket_addr: str
    # ZMQ client_sentinel_request socket address of client sentinel
    client_sentinel_request_addr: str
    # ZMQ engine_core_sentinel_cmd socket address of engine_core sentinel
    engine_core_sentinel_cmd_addr: str
    # ZMQ engine_fault socket address of EngineCoreSentinel
    engine_fault_socket_addr: str
    # Identities of engine core DEALER sockets, keyed by engine index.
    # These identities are used by the ClientSentinel (ROUTER) to route
    # messages to the corresponding engine core.
    engine_core_sentinel_identities: dict[int, bytes]

    @classmethod
    def build(cls, host, local_engines_only, dp_size, ft_config: FaultToleranceConfig):
        engine_fault_socket_addr = get_engine_client_zmq_addr(
            local_only=False,
            host=host,
            port=ft_config.internal_fault_report_port,
        )
        client_sentinel_request_addr = get_engine_client_zmq_addr(
            local_only=True, host=host
        )
        engine_core_sentinel_cmd_addr = get_engine_client_zmq_addr(
            local_only=local_engines_only, host=host
        )
        identity_group = [str(uuid.uuid4()).encode("utf8") for _ in range(dp_size)]
        engine_core_sentinel_identities = {
            rank: identity for rank, identity in enumerate(identity_group)
        }
        fault_state_pub_socket_addr = get_engine_client_zmq_addr(
            local_only=False,
            host=host,
            port=ft_config.external_fault_notify_port,
        )
        return cls(
            fault_state_pub_socket_addr=fault_state_pub_socket_addr,
            client_sentinel_request_addr=client_sentinel_request_addr,
            engine_core_sentinel_cmd_addr=engine_core_sentinel_cmd_addr,
            engine_fault_socket_addr=engine_fault_socket_addr,
            engine_core_sentinel_identities=engine_core_sentinel_identities,
        )

    def to_str(self) -> str:
        payload = {
            "fault_state_pub_socket_addr": self.fault_state_pub_socket_addr,
            "client_sentinel_request_addr": self.client_sentinel_request_addr,
            "engine_core_sentinel_cmd_addr": self.engine_core_sentinel_cmd_addr,
            "engine_fault_socket_addr": self.engine_fault_socket_addr,
            "engine_core_sentinel_identities": {
                str(rank): identity.hex()
                for rank, identity in self.engine_core_sentinel_identities.items()
            },
        }
        return json.dumps(payload, separators=(",", ":"))

    @classmethod
    def from_str(cls, s: str) -> "FaultToleranceZmqAddresses":
        payload = json.loads(s)

        identities = {
            int(rank): bytes.fromhex(identity_hex)
            for rank, identity_hex in payload["engine_core_sentinel_identities"].items()
        }

        return cls(
            fault_state_pub_socket_addr=payload["fault_state_pub_socket_addr"],
            client_sentinel_request_addr=payload["client_sentinel_request_addr"],
            engine_core_sentinel_cmd_addr=payload["engine_core_sentinel_cmd_addr"],
            engine_fault_socket_addr=payload["engine_fault_socket_addr"],
            engine_core_sentinel_identities=identities,
        )
