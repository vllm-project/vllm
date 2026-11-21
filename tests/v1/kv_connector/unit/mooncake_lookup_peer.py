# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A scripted stand-in for `LookupKeyServer` in Mooncake lookup client tests."""

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

import zmq

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import (
    protocol,
    worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import (  # noqa: E501
    MooncakeLookupResult,
)


@dataclass
class PeerRequest:
    identity: bytes
    request_id: bytes
    budget_ms: int
    msg_type: bytes
    payload: list[bytes]


Reply = tuple[bytes, bytes]


def hit(hit_length: int) -> Reply:
    result = MooncakeLookupResult(hit_length)
    return protocol.RESP_OK, protocol.encode_lookup_response(result)


class LookupPeer:
    """ROUTER socket bound where `LookupKeyClient` connects.

    Records every request. `responder` answers a request as it arrives;
    returning None (or no responder) leaves it for the test to answer later
    with `reply`, in any order.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        responder: Callable[[PeerRequest], Reply | None] | None = None,
    ):
        self.responder = responder
        self.requests: list[PeerRequest] = []
        self._cond = threading.Condition()
        self._replies: queue.Queue[tuple[PeerRequest, bytes, bytes]] = queue.Queue()
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.ROUTER)
        self._sock.bind(worker.get_zmq_rpc_path_lookup(vllm_config))
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # The socket stays on this thread; replies from tests arrive by queue.
        while self._running:
            if self._sock.poll(5):
                frames = self._sock.recv_multipart()
                request = PeerRequest(
                    identity=frames[0],
                    request_id=frames[1],
                    budget_ms=int.from_bytes(frames[2], "big"),
                    msg_type=frames[3],
                    payload=frames[4:],
                )
                with self._cond:
                    self.requests.append(request)
                    self._cond.notify_all()
                responder = self.responder
                answer = responder(request) if responder is not None else None
                if answer is not None:
                    self._send(request, *answer)
            while True:
                try:
                    self._send(*self._replies.get_nowait())
                except queue.Empty:
                    break

    def _send(self, request: PeerRequest, status: bytes, payload: bytes) -> None:
        self._sock.send_multipart(
            [request.identity, request.request_id, status, payload]
        )

    def reply(self, request: PeerRequest, status: bytes, payload: bytes = b"") -> None:
        self._replies.put((request, status, payload))

    def reply_hit(self, request: PeerRequest, hit_length: int) -> None:
        self.reply(request, *hit(hit_length))

    def wait_for_requests(self, count: int, timeout: float = 5.0) -> list[PeerRequest]:
        with self._cond:
            assert self._cond.wait_for(lambda: len(self.requests) >= count, timeout), (
                f"expected {count} requests, got {len(self.requests)}"
            )
            return list(self.requests)

    def close(self) -> None:
        self._running = False
        self._thread.join(5)
        self._sock.close(linger=0)
        self._ctx.term()
