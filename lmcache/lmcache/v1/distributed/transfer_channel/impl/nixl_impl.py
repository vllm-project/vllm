# SPDX-License-Identifier: Apache-2.0
"""Nixl-backed implementation of the transfer channel abstraction."""

# Standard
from typing import TYPE_CHECKING, Optional, Union
import importlib
import math
import threading
import uuid

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.transfer_channel.abstract import (
    TransferChannelClient,
    TransferChannelContext,
    TransferChannelServer,
)
from lmcache.v1.distributed.transfer_channel.api import (
    TransferChannelAddress,
    TransferChannelReadResult,
)
from lmcache.v1.distributed.transfer_channel.factory import (
    register_transfer_channel_factory,
)
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError

if TYPE_CHECKING:
    # Third Party
    from nixl._api import nixl_agent, nixl_agent_config, nixl_prepped_dlist_handle

logger = init_logger(__name__)

# Timeout for each blocking recv during the client handshake. Without it a
# misconfigured/unreachable server url would block the connecting thread
# forever. Hard-coded for now.
_HANDSHAKE_TIMEOUT_MS = 60_000


############################################################
# Helper functions
############################################################
def _load_nixl() -> tuple["nixl_agent", "nixl_agent_config"]:
    """Import the nixl Python bindings, tolerating the cuXX-suffixed packages."""
    last_err: Optional[Exception] = None
    for modname in ("nixl._api", "nixl_cu12._api", "nixl_cu13._api"):
        try:
            mod = importlib.import_module(modname)
            return mod.nixl_agent, mod.nixl_agent_config
        except ImportError as err:  # noqa: PERF203
            last_err = err
    raise RuntimeError(
        "NIXL is not available (tried nixl._api, nixl_cu12._api, nixl_cu13._api)"
    ) from last_err


def _parse_url(url: str) -> tuple[str, int]:
    """Parse a ``host:port`` (optionally ``tcp://host:port``) into (host, port)."""
    stripped = url.split("://", 1)[-1]
    host, _, port = stripped.rpartition(":")
    if not host or not port:
        raise ValueError(f"Invalid transfer channel url: {url!r} (expected host:port)")
    return host, int(port)


############################################################
# Handshake messages (msgspec, tagged union)
############################################################
class HandshakeMsgBase(msgspec.Struct, tag=True):
    pass


class InitReq(HandshakeMsgBase):
    agent_name: str
    agent_meta: bytes


class InitResp(HandshakeMsgBase):
    agent_name: str
    agent_meta: bytes


class MemRegReq(HandshakeMsgBase):
    sender_agent_name: str
    sender_advertise_url: str
    xfer_descs: bytes


class MemRegResp(HandshakeMsgBase):
    xfer_descs: bytes


HandshakeMsg = Union[InitReq, InitResp, MemRegReq, MemRegResp]


############################################################
# Client
############################################################
class NixlTransferChannelClient(TransferChannelClient):
    def __init__(
        self,
        context: "NixlTransferChannelContext",
        remote_agent_name: str,
        remote_dlist_handle: "nixl_prepped_dlist_handle",
    ):
        self._ctx = context
        self._remote_agent_name = remote_agent_name
        self._remote_handle = remote_dlist_handle

        self._task_counter = 0
        # task_id -> (xfer_handle, remote_addresses)
        self._tasks: dict[int, tuple] = {}
        self._lock = threading.Lock()

    def submit_read(
        self,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> int:
        """Submit a read transfer from the remote addresses to the local addresses.

        Args:
            local_addresses: The local addresses to read into.
            remote_addresses: The remote addresses to read from.

        Returns:
            A unique task ID for this transfer.
        """
        if len(local_addresses) != len(remote_addresses):
            raise ValueError(
                "local_addresses and remote_addresses must have equal length "
                f"({len(local_addresses)} != {len(remote_addresses)})"
            )

        local_idx = self._ctx.addresses_to_indices(local_addresses)
        remote_idx = self._ctx.addresses_to_indices(remote_addresses)

        agent = self._ctx.agent
        handle = agent.make_prepped_xfer(
            "READ",
            self._ctx.local_handle,
            local_idx,
            self._remote_handle,
            remote_idx,
        )
        agent.transfer(handle)

        with self._lock:
            task_id = self._task_counter
            self._task_counter += 1
            self._tasks[task_id] = (handle, list(remote_addresses))
        return task_id

    def query_read_status(self, task_id: int) -> TransferChannelReadResult:
        """
        Query the status of a previously submitted read transfer.

        Args:
            task_id: The ID of the transfer task to query.

        Returns:
            A TransferChannelReadResult indicating whether the transfer is finished
            and which objects succeeded (if finished).
        """
        with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"Unknown read task id: {task_id}")
            handle, remote_addresses = self._tasks[task_id]

        status = self._ctx.agent.check_xfer_state(handle)
        if status == "PROC":
            return TransferChannelReadResult(finished=False, succeeded_mask=[])

        # Terminal state (DONE or ERR): release the handle and report.
        with self._lock:
            self._tasks.pop(task_id, None)
        self._ctx.agent.release_xfer_handle(handle)

        if status == "DONE":
            return TransferChannelReadResult(
                finished=True, succeeded_mask=[True] * len(remote_addresses)
            )

        # status == "ERR" (or any unexpected state): finished but nothing succeeded.
        return TransferChannelReadResult(
            finished=True, succeeded_mask=[False] * len(remote_addresses)
        )

    def close(self) -> None:
        """
        Close the transfer channel, releasing any pending tasks and the remote handle.

        Note:
            This function does not send any notifications to the server side. The
            server-side teardown needs to be done by other ways (if needed).
        """
        with self._lock:
            tasks = list(self._tasks.values())
            self._tasks.clear()
        for handle, _ in tasks:
            try:
                self._ctx.agent.release_xfer_handle(handle)
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        if self._remote_handle is not None:
            try:
                self._ctx.agent.release_dlist_handle(self._remote_handle)
            except Exception:  # noqa: BLE001
                pass
            self._remote_handle = None


############################################################
# Server
############################################################
class NixlTransferChannelServer(TransferChannelServer):
    """
    Note:
        The server uses a ZMQ REP socket for the metadata handshake.
        It's not using LMCache MQ because we think that's an overkill
        and we don't want to register the transfer-channel specific
        functions into the global LMCache MQ.
    """

    def __init__(
        self,
        listen_url: str,
        advertise_url: str,
        l1_memory_desc: L1MemoryDesc,
        context: "NixlTransferChannelContext",
    ) -> None:
        self._ctx = context
        self._listen_url = listen_url
        self._advertise_url = advertise_url
        self._l1_memory_desc = l1_memory_desc

        self._running = True
        self._socket = self._ctx.zmq_context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        host, port = _parse_url(listen_url)
        self._socket.bind(f"tcp://{host}:{port}")

        self._thread = threading.Thread(
            target=self._serve_loop, name="tc-nixl-server", daemon=True
        )
        self._thread.start()

    def _serve_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while self._running:
            try:
                events = dict(poller.poll(timeout=1000))  # ms
                if self._socket not in events:
                    continue
                req_bytes = self._socket.recv()
                req = msgspec.msgpack.decode(req_bytes, type=HandshakeMsg)
                resp = self._handle_msg(req)
                self._socket.send(msgspec.msgpack.encode(resp))
            except Exception:  # noqa: BLE001
                if self._running:
                    logger.exception("Error in transfer channel server loop")

    def _handle_msg(self, req: HandshakeMsg) -> HandshakeMsg:
        agent = self._ctx.agent
        if isinstance(req, InitReq):
            # Learn the connecting peer's agent (idempotent on repeat).
            agent.add_remote_agent(req.agent_meta)
            return InitResp(
                agent_name=self._ctx.agent_name,
                agent_meta=agent.get_agent_metadata(),
            )
        elif isinstance(req, MemRegReq):
            remote_xfer_dlist = agent.deserialize_descs(req.xfer_descs)
            remote_handle = agent.prep_xfer_dlist(
                req.sender_agent_name, remote_xfer_dlist
            )

            self._ctx.register_client(
                key=req.sender_advertise_url,
                client=NixlTransferChannelClient(
                    context=self._ctx,
                    remote_agent_name=req.sender_agent_name,
                    remote_dlist_handle=remote_handle,
                ),
            )
            return MemRegResp(xfer_descs=self._ctx.serialized_xfer_descs)
        else:
            raise ValueError(f"Unexpected handshake message: {type(req)}")

    def close(self) -> None:
        """
        Close the transfer channel server, stopping the serve loop and
        closing the socket.
        """
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        self._socket.close(linger=0)


############################################################
# Context
############################################################
class NixlTransferChannelContext(TransferChannelContext):
    """Owns the single nixl agent, the registered L1 buffer, and support
    the address translation.
    """

    def __init__(
        self,
        l1_memory_desc: L1MemoryDesc,
        listen_url: str,
        advertise_url: str,
        backends: Optional[list[str]] = None,
    ) -> None:
        """
        Creates the transfer channel context using nixl.

        Args:
            l1_memory_desc: The description of the local L1 memory buffer to register.
            listen_url: The URL to listen on for incoming connections.
            advertise_url: The URL to advertise to peers for them to connect to us.
            backends: Optional list of nixl backends to use (e.g., ["UCX"])
        """
        nixl_agent, nixl_agent_config = _load_nixl()

        self._l1_memory_desc = l1_memory_desc
        self._align = l1_memory_desc.align_bytes
        self.listen_url = listen_url
        self.advertise_url = advertise_url
        backends = backends if backends else ["UCX"]

        self.agent_name = str(uuid.uuid4())
        self.agent = nixl_agent(self.agent_name, nixl_agent_config(backends=backends))

        # Register the whole L1 buffer once (CPU/DRAM, fixed nixl dev_id=0).
        ptr, size = l1_memory_desc.ptr, l1_memory_desc.size
        self._reg_descs = self.agent.get_reg_descs([(ptr, size, 0, "")], "cpu")
        self.agent.register_memory(self._reg_descs)

        # Build + prep a page-granular local xfer dlist over the whole buffer.
        xfer_list = [
            (addr, self._align, 0) for addr in range(ptr, ptr + size, self._align)
        ]
        self._xfer_descs = self.agent.get_xfer_descs(xfer_list, "cpu")
        self.local_handle = self.agent.prep_xfer_dlist("", self._xfer_descs, "cpu")
        self.serialized_xfer_descs = self.agent.get_serialized_descs(self._xfer_descs)

        self.zmq_context = zmq.Context.instance()

        # Clients keyed by the peer's advertised url. Populated either actively
        # (we dialed the peer) or reactively (the peer connected to our server).
        self._clients: dict[str, NixlTransferChannelClient] = {}
        self._lock = threading.Lock()

        # Exactly one server per context, bound eagerly at construction.
        self._server = NixlTransferChannelServer(
            listen_url=listen_url,
            advertise_url=advertise_url,
            l1_memory_desc=l1_memory_desc,
            context=self,
        )

    ############################################################
    # Address translation
    ############################################################
    def get_transfer_channel_address(
        self,
        lmcache_addresses: list[tuple[int, int]],
    ) -> list[TransferChannelAddress]:
        """
        Validate the given LMCache addresses (offset, size) against the
        registered L1 memory region and convert it to TransferChannelAddress

        Args:
            lmcache_addresses: List of (offset, size) tuples representing the LMCache
                addresses.

        Returns:
            A list of TransferChannelAddress corresponding to the given LMCache
            addresses.
        """
        size = self._l1_memory_desc.size
        out = []
        for offset, obj_size in lmcache_addresses:
            if offset < 0 or offset + obj_size > size:
                raise ValueError(
                    f"Object [{offset:#x}, {offset + obj_size:#x}) is outside the "
                    f"registered L1 region [0x0, {size:#x})"
                )
            out.append(TransferChannelAddress(offset=offset, size=obj_size))
        return out

    def addresses_to_indices(
        self, addresses: list[TransferChannelAddress]
    ) -> list[int]:
        """Translate (offset, size) addresses into page indices in the prepped dlist."""
        indices: list[int] = []
        for a in addresses:
            if a.offset % self._align != 0:
                raise ValueError(
                    f"Address offset {a.offset} is not aligned to {self._align}"
                )
            start = a.offset // self._align
            n_pages = max(1, math.ceil(a.size / self._align))
            indices.extend(range(start, start + n_pages))
        return indices

    ############################################################
    # Server / client management
    ############################################################
    def get_transfer_channel_server(self) -> NixlTransferChannelServer:
        return self._server

    def get_transfer_channel_client(
        self,
        peer_advertise_url: str,
    ) -> NixlTransferChannelClient:
        with self._lock:
            client = self._clients.get(peer_advertise_url)
            if client is not None:
                return client

        # Not yet known: actively connect to the peer and perform the handshake.
        client = self._connect(peer_advertise_url)
        return self.register_client(peer_advertise_url, client)

    def get_num_connected_clients(self) -> int:
        with self._lock:
            return len(self._clients)

    def register_client(
        self, key: str, client: NixlTransferChannelClient
    ) -> NixlTransferChannelClient:
        """
        Register a client for the given key (peer advertise url).

        A client already registered for ``key`` is kept; ``client`` is then a
        redundant duplicate (the active connect and the peer's inbound
        connection can race) and is closed.

        Args:
            key: The peer advertise url to register the client under.
            client: A freshly created NixlTransferChannelClient for the peer.

        Returns:
            The canonical client for ``key``: the previously registered one if
            present, otherwise ``client``.
        """
        with self._lock:
            existing = self._clients.get(key)
            if existing is None:
                self._clients[key] = client
                return client
            if existing is client:
                return client

        logger.debug("Reusing existing transfer channel client for %s", key)
        try:
            client.close()
        except Exception:  # noqa: BLE001
            logger.exception(
                "Error closing duplicate transfer channel client for %s", key
            )
        return existing

    def remove_transfer_channel_client(self, peer_advertise_url: str) -> None:
        """Discard the client for ``peer_advertise_url`` and free its resources.

        Call this when reads from the peer are no longer needed (e.g. the
        owning L2 adapter is being removed). Any task ids previously returned by
        that client become invalid. A later ``get_transfer_channel_client`` for
        the same peer returns a fresh client. The peer is not notified.

        Calling this for a peer with no current client does nothing.

        Args:
            peer_advertise_url: The ``host:port`` of the peer, as passed to
                ``get_transfer_channel_client``.
        """
        with self._lock:
            client = self._clients.pop(peer_advertise_url, None)
        if client is None:
            return
        try:
            client.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            logger.exception(
                "Error closing transfer channel client for %s", peer_advertise_url
            )

    ############################################################
    # Cleanup
    ############################################################
    def close(self) -> None:
        with self._lock:
            server = self._server
            clients = list(self._clients.values())
            self._clients.clear()

        if server is not None:
            server.close()
        for client in clients:
            client.close()

        try:
            self.agent.release_dlist_handle(self.local_handle)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.agent.deregister_memory(self._reg_descs)
        except Exception:  # noqa: BLE001
            pass

    ############################################################
    # Helper functions
    ############################################################
    def _recv_handshake(self, socket: "zmq.Socket", server_url: str) -> bytes:
        """Receive one handshake reply, mapping a timeout to a clear error.

        Args:
            socket: The REQ socket awaiting the server's reply.
            server_url: The peer url being dialed (for the error message).

        Returns:
            The raw reply bytes.

        Raises:
            TimeoutError: If no reply arrives within the handshake timeout
                (e.g. the url is wrong/unreachable or the port is blocked).
        """
        try:
            return socket.recv()
        except zmq.Again as err:
            raise LMCacheTimeoutError(
                f"Timed out after {_HANDSHAKE_TIMEOUT_MS / 1000:.0f}s waiting for "
                f"a transfer-channel handshake reply from {server_url!r}. Check "
                f"that the peer is running and that the url/port is correct and "
                f"reachable."
            ) from err

    def _connect(self, server_url: str) -> NixlTransferChannelClient:
        socket = self.zmq_context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, _HANDSHAKE_TIMEOUT_MS)
        host, port = _parse_url(server_url)
        socket.connect(f"tcp://{host}:{port}")
        try:
            # Stage 1: exchange agent metadata.
            socket.send(
                msgspec.msgpack.encode(
                    InitReq(
                        agent_name=self.agent_name,
                        agent_meta=self.agent.get_agent_metadata(),
                    )
                )
            )
            init_resp = msgspec.msgpack.decode(
                self._recv_handshake(socket, server_url), type=HandshakeMsg
            )
            assert isinstance(init_resp, InitResp)
            server_agent_name = self.agent.add_remote_agent(init_resp.agent_meta)

            # Stage 2: exchange transfer-descriptor lists.
            socket.send(
                msgspec.msgpack.encode(
                    MemRegReq(
                        sender_agent_name=self.agent_name,
                        sender_advertise_url=self.advertise_url,
                        xfer_descs=self.serialized_xfer_descs,
                    )
                )
            )
            memreg_resp = msgspec.msgpack.decode(
                self._recv_handshake(socket, server_url), type=HandshakeMsg
            )
            assert isinstance(memreg_resp, MemRegResp)
            remote_xfer_dlist = self.agent.deserialize_descs(memreg_resp.xfer_descs)
            remote_handle = self.agent.prep_xfer_dlist(
                server_agent_name, remote_xfer_dlist
            )
        finally:
            socket.close(linger=0)

        return NixlTransferChannelClient(
            context=self,
            remote_agent_name=server_agent_name,
            remote_dlist_handle=remote_handle,
        )


############################################################
# Factory registration
############################################################
def create_nixl_transfer_channel_context(
    l1_memory_desc: L1MemoryDesc,
    listen_url: str,
    advertise_url: str,
    **kwargs,
) -> NixlTransferChannelContext:
    """Create a ``NixlTransferChannelContext``.

    Args:
        l1_memory_desc: Describes the L1 memory region to register.
        listen_url: ``host:port`` this peer's server binds to.
        advertise_url: ``host:port`` this peer advertises as its identity.
        **kwargs: Accepts ``backends`` (an optional list of nixl backends,
            e.g. ``["UCX"]``).

    Returns:
        A new ``NixlTransferChannelContext`` instance.
    """
    return NixlTransferChannelContext(
        l1_memory_desc=l1_memory_desc,
        listen_url=listen_url,
        advertise_url=advertise_url,
        backends=kwargs.get("backends"),
    )


register_transfer_channel_factory("nixl", create_nixl_transfer_channel_context)
