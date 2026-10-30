# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Daemon-to-daemon seeding over an authenticated JSON channel.

A :class:`Seedable` is one kind of state a daemon can seed to or from a
peer daemon; kinds are bidirectional (a warmup cache is served and
adopted by the same object). The target names what it wants by kind and
key fields; the source resolves the key to a payload; the target validates
and installs it:

    request:  {"kind": "flashinfer_autotune", <key fields>..., "token": "..."}
    response: {"status": "ok", "payload": <kind-specific>}
              {"status": "miss"} | {"status": "error", "message": "..."}

Payloads are JSON values on the remote plane. Large binary state (weights)
must not be inlined: the payload then describes the transfer (a storage
manifest plus a mover's seed) and the fetching side drives the data plane
itself.

A same-host source is addressed by its Unix socket path instead and speaks
the local pickle protocol (owner-verified, no token), which is what lets the
``peer_ipc`` weight mover ship real CUDA IPC handles.

The connection is unencrypted beyond the shared token; keep the listener on a
trusted network.
"""

import contextlib
import hmac
import socket
import threading
import time
from collections.abc import Iterable
from typing import Any, NamedTuple, Protocol

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.protocol import (
    WeightCacheUnavailableError,
    recv_json,
    recv_msg,
    send_json,
    send_msg,
    verify_socket_owner,
)

logger = init_logger(__name__)

# Requests carry a kind and small key fields; payloads flow in responses.
_MAX_REMOTE_REQUEST_SIZE = 64 << 10
_REQUEST_TIMEOUT_S = 30.0
# An unauthenticated peer gets a whole-message deadline, not a per-read
# timeout: per-read timeouts let a peer trickle bytes forever and stall the
# serial accept loop. Authenticated handlers (e.g. weight manifests) may be
# slow, so the budget relaxes once the token checks out.
_REMOTE_REQUEST_TIMEOUT_S = 10.0
_AUTHENTICATED_TIMEOUT_S = 300.0


class Seedable(Protocol):
    """One kind of state that can flow between peer daemons.

    A kind is bidirectional: the same seedable answers peers (``resolve``)
    and fetches from them (``request_params``/``validate_payload``).
    """

    kind: str

    def resolve(self, request: dict[str, Any]) -> Any | None:
        """Source side: produce the payload named by ``request``.

        Validate the wire key first — raise an error on any malformed field
        (``ValueError`` for bad values, ``KeyError`` for missing ones), before
        touching local state. The transport turns it into an error response.
        Return None on a miss.
        """
        ...

    def validate_remote(self, request: dict[str, Any]) -> None:
        """Refuse to serve ``request`` on the remote JSON plane.

        Raise ValueError to refuse the request; the default serves remote
        requests the same as local ones.
        """
        return None

    def request_params(self, ctx: Any) -> dict[str, Any]:
        """Target side: the key fields identifying what this rank needs."""
        ...

    def validate_payload(self, payload: Any) -> Any:
        """Target side: validate the fetched payload, or raise ValueError."""
        ...

    def install(self, ctx: Any, payload: Any) -> None:
        """Target side: install a validated payload."""
        return None


class SeedConfig(NamedTuple):
    """One daemon's D2D seeding configuration: its listener, its source
    replica, and the shared token authenticating both.

    A source that is a plain string is the source daemon's Unix socket path
    (same-host, owner-verified, no token); a ``(host, port)`` tuple dials the
    remote JSON control plane, which requires a token, as does any listener.
    """

    listen: tuple[str, int] | None = None
    source: str | tuple[str, int] | None = None
    token: str | None = None


class SeedTransport:
    """The daemon's D2D seeding endpoint: serves and fetches seedables."""

    def __init__(
        self,
        seedables: Iterable[Seedable] = (),
        seed_config: SeedConfig | None = None,
    ):
        seed_config = seed_config or SeedConfig()
        listen, source, token = seed_config
        remote_source = source is not None and not isinstance(source, str)
        if (listen is not None or remote_source) and not token:
            raise ValueError("A daemon-to-daemon listener/source requires a token")
        self._seedables = list(seedables)
        self._seed_config = seed_config
        self._server: socket.socket | None = None

    def fetch(self, seedable: Seedable, ctx: Any) -> Any | None:
        """Fetch and validate a seed payload from the source daemon.

        Returns the validated payload, or None on a miss (the source has no
        such seed). Raises ``RuntimeError`` on any other failure — the entry
        point for callers with no local path to degrade to.
        """
        response = self.request(
            {"kind": seedable.kind, **seedable.request_params(ctx)},
        )
        if response is None:
            raise RuntimeError(
                f"Could not fetch the {seedable.kind} seed from "
                f"{self._seed_config.source}; the transport logged the reason"
            )
        status = response.get("status")
        if status == "miss":
            return None
        if status != "ok":
            raise RuntimeError(
                f"Source daemon refused the {seedable.kind} seed request "
                f"(status {status}): {response.get('message', response)}"
            )
        return seedable.validate_payload(response.get("payload"))

    @property
    def source(self) -> str | tuple[str, int] | None:
        """The configured source daemon address, if any."""
        return self._seed_config.source

    def adopt(self, seedable: Seedable, ctx: Any) -> None:
        """Best-effort: fetch and install, or fall back to the local path."""
        try:
            payload = self.fetch(seedable, ctx)
        except Exception as e:
            logger.warning(
                "No remote %s seed adopted; using the local path: %s",
                seedable.kind,
                e,
            )
            return
        if payload is None:
            logger.info(
                "No remote %s seed adopted (miss); using the local path",
                seedable.kind,
            )
            return
        try:
            seedable.install(ctx, payload)
        except Exception as e:
            logger.warning(
                "Rejected the remote %s seed; using the local path: %s",
                seedable.kind,
                e,
            )
            return
        logger.info("Adopted remote %s seed", seedable.kind)

    def handle_request(self, request: Any, *, remote: bool = False) -> dict[str, Any]:
        """Source side: answer one authenticated seed request.

        ``remote`` marks a request from the untrusted JSON control plane; a
        seedable may then restrict what it serves (e.g. refuse a node-local
        mover) through ``Seedable.validate_remote``.
        """
        kind_name = request.get("kind") if isinstance(request, dict) else None
        seedable = next((s for s in self._seedables if s.kind == kind_name), None)
        if seedable is None:
            return {"status": "error", "message": "unsupported seed kind"}
        try:
            if remote:
                seedable.validate_remote(request)
            payload = seedable.resolve(request)
        except (OSError, KeyError, TypeError, ValueError) as e:
            return {"status": "error", "message": str(e)}
        if payload is None:
            return {"status": "miss"}
        return {"status": "ok", "payload": payload}

    def request(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Send one authenticated request to the source daemon.

        Returns the response, or None with a log line on any transport
        failure; the caller falls back to its local path then.
        """
        source = self._seed_config.source
        if source is None:
            return None
        label = payload.get("kind", "payload")
        try:
            if isinstance(source, str):
                response = self._request_local(source, payload)
            else:
                with socket.create_connection(
                    source, timeout=_REQUEST_TIMEOUT_S
                ) as sock:
                    send_json(sock, {**payload, "token": self._seed_config.token})
                    response = recv_json(sock)
        except (OSError, ValueError, WeightCacheUnavailableError) as e:
            logger.warning(
                "Cannot fetch %s from %s over D2D; falling back locally: %s",
                label,
                source,
                e,
            )
            return None
        if not isinstance(response, dict):
            logger.warning(
                "Source %s sent a malformed response for %s; falling back locally",
                source,
                label,
            )
            return None
        return response

    def _request_local(self, path: str, payload: dict[str, Any]) -> Any:
        """Fetch from a same-host source over its Unix socket.

        The socket is owner-verified rather than token-authenticated, and it
        speaks the local pickle protocol, which is what carries ``peer_ipc``
        CUDA handles. The daemon routes the ``seed`` command to
        :meth:`handle_request`.
        """
        verify_socket_owner(path, strict_perms=False)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(_REQUEST_TIMEOUT_S)
            sock.connect(path)
            # Authenticated by ownership: seed handlers may be slow.
            sock.settimeout(_AUTHENTICATED_TIMEOUT_S)
            send_msg(sock, {"cmd": "seed", **payload})
            return recv_msg(sock)

    def start_listener(self) -> tuple[str, int] | None:
        """Start serving this rank's seedables to peer daemons.

        Returns the bound address, or None when no listener is configured.
        """
        listen = self._seed_config.listen
        if listen is None:
            return None
        family = socket.AF_INET6 if ":" in listen[0] else socket.AF_INET
        self._server = socket.create_server(listen, family=family)
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
        # Bound once: close() nils self._server out from under the accept loop.
        server = self._server
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            with conn:
                try:
                    request = recv_json(
                        conn,
                        max_size=_MAX_REMOTE_REQUEST_SIZE,
                        deadline=time.monotonic() + _REMOTE_REQUEST_TIMEOUT_S,
                    )
                    presented = (
                        request.get("token") if isinstance(request, dict) else None
                    )
                    if (
                        not self._seed_config.token
                        or not isinstance(presented, str)
                        or not hmac.compare_digest(presented, self._seed_config.token)
                    ):
                        send_json(
                            conn,
                            {"status": "error", "message": "Unauthorized request"},
                        )
                        continue
                    # Authenticated handlers (e.g. weight manifests) may be slow.
                    conn.settimeout(_AUTHENTICATED_TIMEOUT_S)
                    send_json(conn, self.handle_request(request, remote=True))
                except (OSError, ValueError) as e:
                    with contextlib.suppress(OSError):
                        send_json(conn, {"status": "error", "message": str(e)})
                except Exception as e:
                    # A handler bug or a payload json.dumps cannot serialize
                    # must answer the peer and leave the listener serving.
                    logger.exception("Error handling seed peer; continuing")
                    with contextlib.suppress(OSError):
                        send_json(conn, {"status": "error", "message": str(e)})
