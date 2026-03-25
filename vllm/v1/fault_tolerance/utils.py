# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import json
import time
import uuid
from dataclasses import dataclass

import msgspec
import zmq

from vllm.config import FaultToleranceConfig
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.engine import EngineStatusType
from vllm.v1.utils import get_engine_client_zmq_addr


class FaultInfo(msgspec.Struct):
    type: str
    message: str
    engine_id: str
    engine_status: EngineStatusType
    timestamp: str | None = None
    additional_info: dict | None = None

    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        engine_id: str | int,
        engine_status: EngineStatusType,
        additional_info: dict | None = None,
    ) -> "FaultInfo":
        """Create FaultInfo from an exception."""
        local_time = time.localtime(time.time())
        return cls(
            type=type(exception).__name__,
            message=str(exception),
            engine_id=str(engine_id),
            engine_status=engine_status,
            timestamp=time.strftime("%H:%M:%S", local_time),
            additional_info=additional_info or {},
        )


@dataclass
class FaultToleranceZmqAddresses:
    # ZMQ fault_state_pub_socket address of client sentinel
    fault_state_pub_socket_addr: str
    # ZMQ engine_fault socket address of EngineCoreSentinel
    engine_fault_socket_addr: str
    # Identities of engine core DEALER sockets, keyed by engine index.
    # These identities are used by the ClientSentinel (ROUTER) to route
    # messages to the corresponding engine core.
    engine_core_sentinel_identities: dict[int, bytes]

    @classmethod
    def build(cls, host: str, dp_size: int, ft_config: FaultToleranceConfig):
        engine_fault_socket_addr = get_engine_client_zmq_addr(
            local_only=False,
            host=host,
            port=ft_config.internal_fault_report_port,
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
            engine_fault_socket_addr=engine_fault_socket_addr,
            engine_core_sentinel_identities=engine_core_sentinel_identities,
        )

    def to_str(self) -> str:
        payload = {
            "fault_state_pub_socket_addr": self.fault_state_pub_socket_addr,
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
            engine_fault_socket_addr=payload["engine_fault_socket_addr"],
            engine_core_sentinel_identities=identities,
        )


def make_engine_down_report_socket(vllm_config):
    zmq_ctx = zmq.Context()
    zmq_addr = get_engine_client_zmq_addr(
        local_only=False,
        host=vllm_config.parallel_config.data_parallel_master_ip,
        port=vllm_config.fault_tolerance_config.internal_fault_report_port,
    )
    engine_down_socket = make_zmq_socket(
        ctx=zmq_ctx,
        path=zmq_addr,
        socket_type=zmq.DEALER,
        bind=False,
        identity=str(uuid.uuid4()).encode("utf8"),
    )
    return zmq_ctx, engine_down_socket


def notify_engine_down(engine_down_socket, engine_id):
    fault_info = FaultInfo(
        type="EngineDeadError",
        message="Engine died unexpectedly.",
        engine_id=str(engine_id),
        engine_status=EngineStatusType.DEAD,
    )
    with contextlib.suppress(zmq.ZMQError):
        engine_down_socket.send_multipart([b"", msgspec.msgpack.encode(fault_info)])
