# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZMQ control plane for Mooncake encoder-cache reservations and events.

Tensor bytes never travel through this module.  It coordinates destination
reservations, completion/cancellation, TP-shard discovery, and readiness
notifications while :mod:`transfer` owns the Mooncake data plane.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import torch
import zmq

from vllm.logger import init_logger

logger = init_logger(__name__)

_MAX_PENDING_EVENTS = 4096
_RESERVATION_REAP_INTERVAL_SECONDS = 1


ControlRequest = dict[str, Any]
ControlResponse = dict[str, Any]


class ControlClient:
    """Send control requests through reusable, thread-local REQ sockets."""

    def __init__(self, timeout_ms: int) -> None:
        self._context = zmq.Context()
        self._timeout_ms = timeout_ms
        self._local = threading.local()
        self._topologies: dict[str, list[str]] = {}
        self._closed = False

    def _sockets(self) -> dict[str, zmq.Socket]:
        sockets = getattr(self._local, "sockets", None)
        if sockets is None:
            sockets = {}
            self._local.sockets = sockets
        return sockets

    def _discard(self, addr: str) -> None:
        socket = self._sockets().pop(addr, None)
        if socket is not None:
            socket.close(linger=0)

    def _exchange(self, addr: str, payload: ControlRequest) -> ControlResponse:
        sockets = self._sockets()
        socket = sockets.get(addr)
        if socket is None:
            socket = self._context.socket(zmq.REQ)
            socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(addr)
            sockets[addr] = socket
        try:
            socket.send_json(payload)
            response = socket.recv_json()
        except Exception:
            self._discard(addr)
            raise
        assert isinstance(response, dict)
        return response

    def request(self, addr: str, payload: ControlRequest) -> Any:
        response = self._exchange(addr, payload)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "EC control request failed"))
        return response.get("result")

    def discover_shards(self, base_addr: str) -> list[str] | None:
        if base_addr in self._topologies:
            return self._topologies[base_addr]
        try:
            reply = self.request(base_addr, {"op": "peers"})
            ports = reply.get("ports") if isinstance(reply, dict) else None
            if not isinstance(ports, list) or not ports:
                raise ValueError("invalid or empty peer list")
            prefix = base_addr.rsplit(":", 1)[0]
            shards = [f"{prefix}:{int(port)}" for port in ports]
        except Exception:
            logger.warning(
                "EC Mooncake consumer at %s did not report its shards",
                base_addr,
                exc_info=True,
            )
            return None
        self._topologies[base_addr] = shards
        return shards

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._context.destroy(linger=0)


def make_cancel_request(
    transfer_id: str,
    reservation_id: str = "",
    *,
    abandon: bool = False,
    refresh: bool = False,
) -> ControlRequest:
    request: ControlRequest = {
        "op": "cancel",
        "transfer_id": transfer_id,
        "reservation_id": reservation_id,
    }
    if abandon:
        request["abandon"] = True
    if refresh:
        request["refresh"] = True
    return request


class EventInbox:
    """Receive Consumer readiness events without blocking the Scheduler."""

    def __init__(self, client: ControlClient) -> None:
        self._client = client
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._closed = False
        self.shard_count = 1

    def _connect(self, base_addr: str) -> None:
        if self._socket is not None:
            return
        shards = self._client.discover_shards(base_addr)
        if shards is None:
            return
        endpoints = []
        for addr in shards:
            try:
                event_port = self._client.request(addr, {"op": "event_port"})
                address, _ = addr.rsplit(":", 1)
                endpoints.append(f"{address}:{int(event_port)}")
            except Exception:
                logger.warning(
                    "EC Mooncake could not subscribe to the event channel of "
                    "consumer shard %s; retrying the complete topology later.",
                    addr,
                )
                return
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        for endpoint in endpoints:
            socket.connect(endpoint)
        self._context = context
        self._socket = socket
        self.shard_count = len(shards)

    def drain(self, base_addr: str) -> list[dict[str, Any]]:
        self._connect(base_addr)
        if self._socket is None:
            return []
        events = []
        while True:
            try:
                events.append(self._socket.recv_json(flags=zmq.DONTWAIT))
            except zmq.Again:
                return events

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._socket is not None:
            self._socket.close(linger=0)
        if self._context is not None:
            self._context.term()


class ConsumerControlServer:
    """Expose Consumer reservations and readiness events over ZMQ.

    One server runs on every receiving TP rank.  The REP channel handles
    reservation operations, while a PUSH channel publishes newly ready items
    to the Scheduler.
    """

    def __init__(
        self,
        host: str,
        port: int,
        reserve: Callable[[dict[str, Any]], dict[str, Any]],
        status: Callable[[str], dict[str, Any] | None],
        complete: Callable[[str, str], tuple[bool, bool]],
        cancel: Callable[[str, str, bool, bool], bool],
        reap: Callable[[], int],
        peer_ports: list[int] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.peer_ports = peer_ports or [port]
        self._device = device
        self.event_port: int | None = None
        self._reserve = reserve
        self._status = status
        self._complete = complete
        self._cancel = cancel
        self._reap = reap
        self._stop = threading.Event()
        self._started = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: Exception | None = None

    def start(self) -> None:
        def loop() -> None:
            if self._device is not None and self._device.type == "cuda":
                torch.accelerator.set_device_index(self._device.index or 0)
            context = zmq.Context()
            socket = context.socket(zmq.REP)
            event_socket = context.socket(zmq.PUSH)
            pending_events: deque[dict[str, Any]] = deque()

            def queue_event(event: dict[str, Any]) -> None:
                event["shard"] = self.port
                if len(pending_events) >= _MAX_PENDING_EVENTS:
                    pending_events.popleft()
                pending_events.append(event)

            def queue_ready(transfer_id: str) -> None:
                status = self._status(transfer_id)
                if status is not None:
                    queue_event({"transfer_id": transfer_id, **status})

            last_reap_at = time.monotonic()
            socket.setsockopt(zmq.RCVTIMEO, 100)
            try:
                socket.bind(f"tcp://{self.host}:{self.port}")
                self.event_port = event_socket.bind_to_random_port(f"tcp://{self.host}")
            except Exception as e:
                self._startup_error = e
                self._started.set()
                socket.close(linger=0)
                event_socket.close(linger=0)
                context.term()
                return
            self._started.set()
            try:
                while not self._stop.is_set():
                    while pending_events:
                        try:
                            event_socket.send_json(
                                pending_events[0], flags=zmq.DONTWAIT
                            )
                        except zmq.Again:
                            break
                        pending_events.popleft()
                    now = time.monotonic()
                    if now - last_reap_at >= _RESERVATION_REAP_INTERVAL_SECONDS:
                        self._reap()
                        last_reap_at = now
                    try:
                        request = socket.recv_json()
                    except zmq.Again:
                        continue
                    try:
                        op = request.get("op")
                        result: Any = None
                        if op == "reserve":
                            result = self._reserve(request)
                            if result.get("ready"):
                                queue_ready(str(request["transfer_id"]))
                        elif op == "status":
                            result = self._status(str(request["transfer_id"]))
                        elif op == "event_port":
                            result = self.event_port
                        elif op == "peers":
                            result = {"ports": self.peer_ports}
                        elif op in ("complete", "complete_batch"):
                            items = (
                                request["items"]
                                if op == "complete_batch"
                                else [request]
                            )
                            completions = []
                            for item in items:
                                transfer_id = str(item["transfer_id"])
                                accepted, became_ready = self._complete(
                                    transfer_id, str(item["reservation_id"])
                                )
                                completions.append(
                                    {
                                        "completed": accepted,
                                        "became_ready": became_ready,
                                    }
                                )
                                if not became_ready:
                                    continue
                                queue_ready(transfer_id)
                            result = (
                                {"items": completions}
                                if op == "complete_batch"
                                else completions[0]
                            )
                        elif op == "cancel":
                            result = {
                                "cancelled": self._cancel(
                                    str(request["transfer_id"]),
                                    str(request.get("reservation_id", "")),
                                    bool(request.get("abandon", False)),
                                    bool(request.get("refresh", False)),
                                )
                            }
                        else:
                            raise ValueError(f"unknown control op: {op!r}")
                        socket.send_json({"ok": True, "result": result})
                    except Exception as e:
                        socket.send_json({"ok": False, "error": str(e)})
            finally:
                socket.close(linger=0)
                event_socket.close(linger=0)
                context.term()

        self._thread = threading.Thread(
            target=loop, name="ec-mooncake-control", daemon=True
        )
        self._thread.start()
        if not self._started.wait(timeout=5):
            raise RuntimeError("EC Mooncake control channel failed to start")
        if self._startup_error is not None:
            raise RuntimeError("EC Mooncake control channel failed to bind") from (
                self._startup_error
            )
        logger.info(
            "EC Mooncake control channel listening on tcp://%s:%d (events tcp://%s:%d)",
            self.host,
            self.port,
            self.host,
            self.event_port,
        )

    def close(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
