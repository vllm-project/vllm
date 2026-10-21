# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Daemon-to-daemon seeding over an authenticated JSON channel.

A :class:`Seedable` is one kind of state a daemon can seed to or from a peer
daemon: warmup caches today, weight shards (manifests moved over CUDA IPC or
RDMA) later. The flow is shared — the target names what it wants by kind and
kind-specific key fields, the source resolves the key to a payload, and the
target validates and installs it:

    request:  {"cmd": "get_seed", "kind": "flashinfer_autotune",
               <kind-specific key fields>..., "token": "..."}
    response: {"status": "ok", "payload": <kind-specific>}
              {"status": "miss"} | {"status": "error", "message": "..."}

Payloads are JSON values. Large binary state (weights) must not be inlined:
the payload then describes the transfer (e.g. a storage manifest) and the
kind's ``install`` drives the data plane itself.

:class:`SeedTransport` is the communication layer: one authenticated listener
per daemon serving the daemon's seedables, plus the target-side request round
trip. The wire format is length-prefixed JSON (protocol.send_json/recv_json);
the connection is unencrypted beyond the shared token, so keep the listener
on a trusted network.
"""

import contextlib
import hmac
import socket
import threading
import time
from collections.abc import Iterable
from typing import Any, Protocol

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.protocol import (
    recv_json,
    send_json,
)

logger = init_logger(__name__)

GET_SEED_CMD = "get_seed"

# Requests carry a command and small key fields; payloads flow in responses.
MAX_REMOTE_REQUEST_SIZE = 64 << 10
_REQUEST_TIMEOUT_S = 30.0
# An unauthenticated peer gets a whole-message deadline, not a per-read
# timeout: per-read timeouts let a peer trickle bytes forever and stall the
# serial accept loop. Authenticated handlers (e.g. weight manifests) may be
# slow, so the budget relaxes once the token checks out.
_REMOTE_REQUEST_TIMEOUT_S = 10.0
_AUTHENTICATED_TIMEOUT_S = 300.0


class Seedable(Protocol):
    """One kind of state a daemon can seed to or from a peer daemon."""

    kind: str

    def request_params(self, ctx: Any) -> dict[str, Any]:
        """Target side: the key fields identifying what this rank needs."""
        ...

    def validate_request(self, request: Any) -> dict[str, Any]:
        """Source side: validate the wire key, return it normalized.

        Raise ValueError on any malformed field; the request must be rejected
        before touching local state.
        """
        ...

    def resolve(self, params: dict[str, Any]) -> Any | None:
        """Source side: produce the payload named by ``params``, None on miss."""
        ...

    def validate_payload(self, payload: Any) -> Any:
        """Target side: validate the fetched payload, or raise ValueError."""
        ...

    def install(self, ctx: Any, payload: Any) -> None:
        """Target side: install a validated payload."""
        ...


class SeedTransport:
    """The daemon's D2D seeding endpoint: serves and fetches seedables."""

    def __init__(
        self,
        seedables: Iterable[Seedable],
        listen: tuple[str, int] | None = None,
        source: tuple[str, int] | None = None,
        token: str | None = None,
    ):
        if (listen is not None or source is not None) and not token:
            raise ValueError("A daemon-to-daemon listener/source requires a token")
        self._seedables = list(seedables)
        self._listen = listen
        self._source = source
        self._token = token
        self._server: socket.socket | None = None

    def adopt(self, seedable: Seedable, ctx: Any) -> None:
        """Best-effort: fetch and install ``seedable`` from the source daemon.

        Every failure mode (unreachable, refused, missing, incompatible)
        degrades to the local path with a log line.
        """
        response = self.request(
            {
                "cmd": GET_SEED_CMD,
                "kind": seedable.kind,
                **seedable.request_params(ctx),
            },
            label=f"{seedable.kind} seed",
        )
        if response is None:
            return
        status = response.get("status")
        if status == "miss":
            logger.info(
                "Source has no %s seed for this rank; using the local path",
                seedable.kind,
            )
            return
        if status != "ok":
            logger.warning(
                "Source refused the %s seed request: %r; using the local path",
                seedable.kind,
                response,
            )
            return
        try:
            seedable.install(ctx, seedable.validate_payload(response.get("payload")))
        except Exception as e:
            logger.warning(
                "Rejected the remote %s seed; using the local path: %s",
                seedable.kind,
                e,
            )
            return
        logger.info("Adopted remote %s seed", seedable.kind)

    def handle_request(self, request: Any) -> dict[str, Any]:
        """Source side: answer one authenticated ``get_seed`` request."""
        if not isinstance(request, dict) or request.get("cmd") != GET_SEED_CMD:
            return {"status": "error", "message": "unsupported seed command"}
        kind_name = request.get("kind")
        seedable = next((s for s in self._seedables if s.kind == kind_name), None)
        if seedable is None:
            return {"status": "error", "message": "unsupported seed kind"}
        try:
            payload = seedable.resolve(seedable.validate_request(request))
        except (OSError, ValueError) as e:
            return {"status": "error", "message": str(e)}
        if payload is None:
            return {"status": "miss"}
        return {"status": "ok", "payload": payload}

    def request(self, payload: dict[str, Any], *, label: str) -> dict[str, Any] | None:
        """Send one authenticated request to the source daemon.

        Returns the response, or None with a log line on any transport
        failure; the caller falls back to its local path then.
        """
        if self._source is None:
            return None
        try:
            with socket.create_connection(
                self._source, timeout=_REQUEST_TIMEOUT_S
            ) as sock:
                send_json(sock, {**payload, "token": self._token})
                response = recv_json(sock)
        except (OSError, ConnectionError, ValueError) as e:
            logger.warning(
                "Cannot fetch %s from %s over D2D; falling back locally: %s",
                label,
                self._source,
                e,
            )
            return None
        if not isinstance(response, dict):
            logger.warning(
                "Source %s sent a malformed response for %s; falling back locally",
                self._source,
                label,
            )
            return None
        return response

    def start_listener(self) -> tuple[str, int] | None:
        """Start serving this rank's seedables to peer daemons.

        Returns the bound address, or None when no listener is configured.
        """
        if self._listen is None:
            return None
        family = socket.AF_INET6 if ":" in self._listen[0] else socket.AF_INET
        self._server = socket.create_server(self._listen, family=family)
        threading.Thread(
            target=self._serve,
            name="vllm-weight-cache-d2d",
            daemon=True,
        ).start()
        host, port = self._server.getsockname()[:2]
        return host, port

    def close(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None

    def _serve(self) -> None:
        assert self._server is not None
        while True:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            with conn:
                try:
                    request = recv_json(
                        conn,
                        max_size=MAX_REMOTE_REQUEST_SIZE,
                        deadline=time.monotonic() + _REMOTE_REQUEST_TIMEOUT_S,
                    )
                    presented = (
                        request.get("token") if isinstance(request, dict) else None
                    )
                    if (
                        not self._token
                        or not isinstance(presented, str)
                        or not hmac.compare_digest(presented, self._token)
                    ):
                        send_json(
                            conn,
                            {"status": "error", "message": "Unauthorized request"},
                        )
                        continue
                    # Authenticated handlers (e.g. weight manifests) may be slow.
                    conn.settimeout(_AUTHENTICATED_TIMEOUT_S)
                    send_json(conn, self.handle_request(request))
                except (OSError, ConnectionError, ValueError) as e:
                    with contextlib.suppress(OSError):
                        send_json(conn, {"status": "error", "message": str(e)})
