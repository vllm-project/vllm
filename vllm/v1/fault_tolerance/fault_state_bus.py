# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FaultStateBus — thin pub/sub surface for fault events.

Wraps a ZMQ PUB socket (the wire format established by vllm-project/vllm#34833)
plus an in-process callback list so subscribers in the same process can
consume the same events without setting up their own SUB socket.

External subscribers (Dynamo, monitoring tools) receive via ZMQ PUB exactly
as they did with #34833. In-process subscribers (e.g., the DP routing
policy) receive via the callback list.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.fault_tolerance.types import FaultInfo

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


# Topic byte-string used on the ZMQ PUB topic frame. Matches the constant
# the existing fault-reporting work has used (preserve external compat).
FAULT_STATE_PUB_TOPIC = b"vllm_fault"


class FaultStateBus:
    """ZMQ PUB + in-process subscribe.

    The PUB socket is bound at construction time; subscribers can attach via
    ``subscribe(cb)`` to receive events synchronously when ``publish`` is
    called. The PUB delivery is fire-and-forget (PUB sockets never block on
    a slow subscriber).
    """

    def __init__(self, vllm_config: VllmConfig):
        self._vllm_config = vllm_config
        self._subscribers: list[Callable[[FaultInfo], None]] = []
        self._sub_lock = threading.Lock()
        self._pub_socket = None
        self._closed = False
        self._init_pub_socket()

    def _init_pub_socket(self) -> None:
        """Create the ZMQ PUB socket lazily so the bus is usable in tests
        even when ZMQ isn't available."""
        try:
            import zmq

            from vllm.utils.network_utils import make_zmq_socket
        except ImportError:
            logger.debug("zmq unavailable; FaultStateBus running in-process only")
            return

        ft_cfg = getattr(self._vllm_config, "fault_tolerance_config", None)
        port = getattr(ft_cfg, "external_fault_notify_port", None) if ft_cfg else None
        if port is None:
            # No port configured; skip socket creation.
            return
        host = getattr(ft_cfg, "external_fault_notify_host", "0.0.0.0")
        path = f"tcp://{host}:{port}"
        try:
            self._pub_socket = make_zmq_socket(
                ctx=zmq.Context.instance(),
                path=path,
                socket_type=zmq.PUB,
                bind=True,
            )
        except Exception as e:
            logger.warning("Failed to bind FaultStateBus PUB at %s: %s", path, e)
            self._pub_socket = None

    # ---- subscribe surface (in-process) ------------------------------------

    def subscribe(self, cb: Callable[[FaultInfo], None]) -> None:
        """Register an in-process callback. Callback is invoked synchronously
        on the publishing thread; keep it fast."""
        with self._sub_lock:
            self._subscribers.append(cb)

    def unsubscribe(self, cb: Callable[[FaultInfo], None]) -> None:
        from contextlib import suppress

        with self._sub_lock, suppress(ValueError):
            self._subscribers.remove(cb)

    # ---- publish surface ---------------------------------------------------

    def publish(self, info: FaultInfo) -> None:
        """Fan out an event. PUB delivery is best-effort; callback delivery
        is synchronous."""
        if self._closed:
            return
        # Out-of-process subscribers (Dynamo, monitoring, etc.).
        if self._pub_socket is not None:
            try:
                self._pub_socket.send_multipart(
                    [FAULT_STATE_PUB_TOPIC, _encode_fault_info(info)]
                )
            except Exception as e:
                logger.debug("FaultStateBus PUB send failed: %s", e)
        # In-process subscribers.
        with self._sub_lock:
            cbs = list(self._subscribers)
        for cb in cbs:
            try:
                cb(info)
            except Exception as e:
                logger.warning("FaultStateBus subscriber raised: %s", e)

    # ---- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        from contextlib import suppress

        if self._closed:
            return
        self._closed = True
        if self._pub_socket is not None:
            with suppress(Exception):
                self._pub_socket.close(linger=0)
            self._pub_socket = None


def _encode_fault_info(info: FaultInfo) -> bytes:
    """Encode a FaultInfo for the wire.

    Uses msgpack via msgspec when available (matching the existing
    fault-reporting wire format); falls back to JSON otherwise.
    """
    payload = {
        "schema_version": 1,
        "engine_index": info.engine_index,
        "status": info.status.name.lower(),
        "kind": info.kind,
        "detail": info.detail
        if isinstance(info.detail, (str, type(None)))
        else str(info.detail),
    }
    try:
        import msgspec.msgpack

        return msgspec.msgpack.encode(payload)
    except ImportError:
        import json

        return json.dumps(payload).encode("utf-8")
