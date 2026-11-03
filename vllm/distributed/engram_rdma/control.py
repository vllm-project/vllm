# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small setup/drain control channel for a shared Engram owner.

This protocol carries lifecycle control only. It is not on the lookup/data path.
The caller must wait for CUDA/NIC work to complete before releasing its lease;
these messages cannot prove GPU or network completion.
"""

from __future__ import annotations

import json
import secrets
import socket
import socketserver
import threading
from contextlib import AbstractContextManager, suppress
from types import MappingProxyType

MAX_FRAME = 64 * 1024


class ControlError(RuntimeError):
    """A lifecycle request was rejected or could not be completed."""


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _read_frame(sock):
    data = bytearray()
    while len(data) <= MAX_FRAME:
        chunk = sock.recv(min(4096, MAX_FRAME + 1 - len(data)))
        if not chunk:
            break
        pos = chunk.find(b"\n")
        data.extend(chunk if pos < 0 else chunk[:pos])
        if pos >= 0:
            if len(data) > MAX_FRAME:
                raise ValueError("frame too large")
            return bytes(data)
    if len(data) > MAX_FRAME:
        raise ValueError("frame too large")
    raise ValueError("unterminated frame")


def _send(sock, obj):
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(body) > MAX_FRAME:
        raise ValueError("frame too large")
    sock.sendall(body)


class _ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class OwnerServer(AbstractContextManager):
    """Serve attach/release/drain for one owner generation.

    Metadata is copied and made recursively immutable at construction. `close`
    first enters draining and raises while any lease remains; it never revokes
    leases or claims that device work has completed.
    """

    def __init__(self, metadata, host="127.0.0.1", port=0):
        if not isinstance(metadata, dict) or not isinstance(
            metadata.get("checkpoint_id"), str
        ):
            raise ValueError("metadata must include string checkpoint_id")
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > MAX_FRAME - 256:
            raise ValueError("metadata too large for control frame")
        self._metadata_json = encoded
        self.metadata = _freeze(json.loads(encoded))
        self.generation = secrets.token_urlsafe(24)
        self._lock = threading.RLock()
        self._leases = set()
        self._draining = False
        self._closed = False
        owner = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                try:
                    request = json.loads(_read_frame(self.request).decode("utf-8"))
                    if not isinstance(request, dict):
                        raise ValueError("request must be an object")
                    result = owner._dispatch(request)
                    response = {"ok": True, **result}
                except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    response = {"ok": False, "error": f"invalid request: {exc}"}
                except ControlError as exc:
                    response = {"ok": False, "error": str(exc)}
                with suppress(OSError, ValueError):
                    _send(self.request, response)

        self._server = _ThreadingServer((host, port), Handler)
        self._thread = None

    @property
    def address(self):
        return self._server.server_address

    @property
    def lease_count(self):
        with self._lock:
            return len(self._leases)

    @property
    def draining(self):
        with self._lock:
            return self._draining

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._server.serve_forever, daemon=True
            )
            self._thread.start()
        return self

    def _check_generation(self, request):
        if request.get("generation") != self.generation:
            raise ControlError("stale or wrong generation")

    def _dispatch(self, request):
        op = request.get("op")
        with self._lock:
            if self._closed:
                raise ControlError("owner is closed")
            self._check_generation(request)
            if op == "status":
                return {
                    "generation": self.generation,
                    "leases": len(self._leases),
                    "draining": self._draining,
                }
            if op == "attach":
                if request.get("checkpoint_id") != self.metadata["checkpoint_id"]:
                    raise ControlError("checkpoint identity mismatch")
                if self._draining:
                    raise ControlError("owner is draining")
                lease = secrets.token_urlsafe(24)
                self._leases.add(lease)
                return {
                    "metadata": json.loads(self._metadata_json),
                    "generation": self.generation,
                    "lease": lease,
                }
            if op == "release":
                lease = request.get("lease")
                if not isinstance(lease, str) or lease not in self._leases:
                    raise ControlError("unknown or already released lease")
                self._leases.remove(lease)
                return {"released": True}
            if op == "drain":
                self._draining = True
                return {"draining": True, "leases": len(self._leases)}
            raise ValueError("unknown operation")

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._draining = True
            if self._leases:
                raise ControlError(f"cannot close: {len(self._leases)} lease(s) remain")
            self._closed = True
        if self._thread is not None:
            self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.close()


class OwnerClient:
    """A client pinned to an expected checkpoint identity and generation."""

    def __init__(self, host, port, checkpoint_id, generation, timeout=5.0):
        self.host, self.port = host, port
        self.checkpoint_id, self.generation = checkpoint_id, generation
        self.timeout = timeout

    def _request(self, op, **fields):
        request = {"op": op, "generation": self.generation, **fields}
        with socket.create_connection((self.host, self.port), self.timeout) as sock:
            sock.settimeout(self.timeout)
            _send(sock, request)
            response = json.loads(_read_frame(sock).decode("utf-8"))
        if not isinstance(response, dict) or not response.get("ok"):
            raise ControlError(
                response.get("error", "invalid server response")
                if isinstance(response, dict)
                else "invalid server response"
            )
        return response

    def attach(self):
        response = self._request("attach", checkpoint_id=self.checkpoint_id)
        if (
            response.get("generation") != self.generation
            or response.get("metadata", {}).get("checkpoint_id") != self.checkpoint_id
        ):
            raise ControlError("server identity validation failed")
        return Lease(self, response["lease"], response["metadata"])

    def begin_drain(self):
        return self._request("drain")

    def status(self):
        return self._request("status")


class Lease(AbstractContextManager):
    """Explicit reader lease; release only after caller has drained device work."""

    def __init__(self, client, token, metadata):
        self.client, self.token = client, token
        self.metadata = _freeze(metadata)
        self._released = False

    def release(self):
        if self._released:
            raise ControlError("lease already released")
        # Caller obligation: synchronize/drain all CUDA/NIC operations that can
        # access the owner table before invoking release().
        self.client._request("release", lease=self.token)
        self._released = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
