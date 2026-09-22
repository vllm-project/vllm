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
from concurrent.futures import Executor, Future
from typing import Any

import torch
import zmq

from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path

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
            socket.setsockopt(zmq.IPV6, 1)
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
    """Receive Consumer readiness events without blocking the Scheduler.

    Subscribing needs one blocking control request per shard, so it runs on
    `executor` when one is given: an unreachable Consumer must cost the
    Scheduler nothing, not one control timeout per drain.
    """

    def __init__(self, client: ControlClient, executor: Executor | None = None) -> None:
        self._client = client
        self._executor = executor
        self._discovery: Future[list[str] | None] | None = None
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._closed = False
        self.shard_count = 1

    def _endpoints(self, base_addr: str) -> list[str] | None:
        """Ask every shard where it publishes readiness events.

        Blocking: called on `self._executor` unless there is none.
        """
        shards = self._client.discover_shards(base_addr)
        if shards is None:
            return None
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
                return None
        return endpoints

    def _connect(self, base_addr: str) -> None:
        if self._socket is not None or self._closed:
            return
        if self._executor is None:
            self._install(self._endpoints(base_addr))
            return
        if self._discovery is None:
            self._discovery = self._executor.submit(self._endpoints, base_addr)
            return
        if not self._discovery.done():
            return
        discovery, self._discovery = self._discovery, None
        try:
            self._install(discovery.result())
        except Exception:
            logger.warning(
                "EC Mooncake event-channel discovery for %s failed; retrying.",
                base_addr,
                exc_info=True,
            )

    def _install(self, endpoints: list[str] | None) -> None:
        """Adopt a discovered topology.

        Runs on the caller's thread so the socket is only ever touched there.
        """
        if not endpoints or self._closed:
            return
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.setsockopt(zmq.IPV6, 1)
        for endpoint in endpoints:
            socket.connect(endpoint)
        self._context = context
        self._socket = socket
        self.shard_count = len(endpoints)

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
            except zmq.ZMQError:
                logger.warning("EC Mooncake event channel failed", exc_info=True)
                return events
            except (ValueError, UnicodeError):
                # The event channel is a plain PULL socket: an undecodable
                # frame must cost one frame, not the engine.
                logger.warning(
                    "Discarding an undecodable EC Mooncake event.", exc_info=True
                )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._discovery is not None:
            self._discovery.cancel()
            self._discovery = None
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
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
        drain_ready: Callable[[], list[str]] = lambda: [],
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
        self._drain_ready = drain_ready
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
            socket.setsockopt(zmq.IPV6, 1)
            event_socket.setsockopt(zmq.IPV6, 1)
            pending_events: deque[dict[str, Any]] = deque()

            def queue_event(event: dict[str, Any]) -> None:
                event["shard"] = self.port
                if len(pending_events) >= _MAX_PENDING_EVENTS:
                    dropped = pending_events.popleft()
                    logger.warning(
                        "EC Mooncake event backlog full on port %d; dropping "
                        "readiness for transfer_id=%s",
                        self.port,
                        dropped.get("transfer_id"),
                    )
                pending_events.append(event)

            def queue_ready(transfer_id: str) -> None:
                status = self._status(transfer_id)
                if status is not None:
                    queue_event({"transfer_id": transfer_id, **status})

            last_reap_at = time.monotonic()
            socket.setsockopt(zmq.RCVTIMEO, 100)
            try:
                socket.bind(make_zmq_path("tcp", self.host, self.port))
                event_socket.bind(make_zmq_path("tcp", self.host, 0))
                self.event_port = int(
                    event_socket.getsockopt(zmq.LAST_ENDPOINT)
                    .decode()
                    .rsplit(":", 1)[1]
                )
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
                    except Exception:
                        # The frame arrived but did not decode. REP still owes a
                        # reply, so answer before returning to the loop.
                        logger.exception(
                            "EC Mooncake control channel on port %d received an "
                            "undecodable request",
                            self.port,
                        )
                        socket.send_json(
                            {"ok": False, "error": "malformed control request"}
                        )
                        continue
                    try:
                        op = request.get("op")
                        result: Any = None
                        if op in ("reserve", "reserve_batch"):
                            items = (
                                request["items"] if op == "reserve_batch" else [request]
                            )
                            results = []
                            for item in items:
                                try:
                                    reserved = self._reserve(item)
                                    results.append({"ok": True, "result": reserved})
                                    if reserved.get("ready"):
                                        queue_ready(str(item["transfer_id"]))
                                except Exception as exc:
                                    results.append({"ok": False, "error": str(exc)})
                            if op == "reserve_batch":
                                result = {"items": results}
                            elif not results[0]["ok"]:
                                raise RuntimeError(results[0]["error"])
                            else:
                                result = results[0]["result"]
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
                                for ready_id in self._drain_ready():
                                    queue_ready(ready_id)
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
            except Exception:
                # `finally` closes the sockets and the thread ends, so without
                # this every later reserve against this shard would surface
                # only as a control timeout with nothing to attribute it to.
                logger.exception(
                    "EC Mooncake control channel on port %d stopped serving",
                    self.port,
                )
                raise
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
