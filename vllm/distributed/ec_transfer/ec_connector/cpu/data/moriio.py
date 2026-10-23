# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoRI-IO-backed data-plane transport for ECCPUConnector.

The consumer READs encoder outputs out of the producer's CPU region over the
MoRI-IO RDMA backend. MoRI-IO has no equivalent of a NIXL notification that
rides on the transfer, so completion is signalled to the producer over a ZMQ
PUSH/PULL pair owned by this transport; the producer drains it through
get_new_notifs() exactly as it drains NIXL notifications.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any

import msgspec
import torch
import zmq

from vllm.distributed.ec_transfer.ec_connector.cpu.data.base import (
    DataTransport,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import (
    build_block_descs,
    deserialize_mem_descriptor,
    serialize_mem_descriptor,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket

logger = init_logger(__name__)

try:
    from mori.io import (
        BackendType,
        EngineDesc,
        IOEngine,
        IOEngineConfig,
        MemoryDesc,
        PollCqMode,
        RdmaBackendConfig,
    )
except ImportError:
    IOEngine = None  # type: ignore[assignment,misc]


class _PeerMetadata(msgspec.Struct):
    engine_desc: bytes
    mem_desc: bytes
    notify_path: str


@dataclass
class _Peer:
    engine_desc: Any
    session: Any
    notify: zmq.Socket


@dataclass
class _MoriXfer:
    status: Any
    notify: zmq.Socket
    notif_msg: bytes
    # Keeps the session alive while the transfer can still touch it.
    session: Any
    notified: bool = field(default=False)


def _merge_segments(
    local_offsets: list[int], remote_offsets: list[int], block_size: int
) -> tuple[list[int], list[int], list[int]]:
    """Coalesce block pairs that are contiguous on both sides."""
    out_local: list[int] = []
    out_remote: list[int] = []
    sizes: list[int] = []
    for local, remote in zip(local_offsets, remote_offsets):
        if (
            sizes
            and out_local[-1] + sizes[-1] == local
            and out_remote[-1] + sizes[-1] == remote
        ):
            sizes[-1] += block_size
        else:
            out_local.append(local)
            out_remote.append(remote)
            sizes.append(block_size)
    return out_local, out_remote, sizes


class MoRIIODataTransport(DataTransport):
    """Adapts a MoRI-IO IOEngine to the DataTransport interface.

    Not thread-safe; like NixlDataTransport, every call runs on the scheduler
    thread.

    MoRI-IO writes transfer results into TransferStatus objects it holds by
    raw pointer, so a status (and its session) must outlive the transfer even
    after the caller releases the handle; such handles are parked until they
    leave the in-progress state.
    """

    def __init__(
        self,
        agent_name: str,
        region: torch.Tensor,
        block_size_bytes: int,
        host: str,
        port: int,
        notify_port: int,
        qp_per_transfer: int = 1,
        num_workers: int = 1,
    ) -> None:
        if IOEngine is None:
            raise RuntimeError(
                "ec_transport=moriio requires MoRI-IO; install `amd-mori` or "
                "set a different ec_transport."
            )
        self._block_size = block_size_bytes
        self._peers: dict[str, _Peer] = {}
        self._parked: list[_MoriXfer] = []
        self._retired_sessions: list[Any] = []
        self._local_mem: Any = None
        self._ctx = zmq.Context()
        self._pull: zmq.Socket | None = None
        self._push_by_path: dict[str, zmq.Socket] = {}

        try:
            self._engine = IOEngine(
                f"ec:{agent_name}:{uuid.uuid4().hex[:8]}", IOEngineConfig(host, port)
            )
            self._engine.create_backend(
                BackendType.RDMA,
                RdmaBackendConfig(
                    qp_per_transfer,
                    -1,
                    num_workers,
                    PollCqMode.POLLING,
                    # Completion is signalled over ZMQ; MoRI's own target-side
                    # notification is not consumed, as in MoRIIOConnector.
                    enable_notification=False,
                ),
            )
            notify_path = make_zmq_path("tcp", host, notify_port)
            self._pull = make_zmq_socket(
                ctx=self._ctx, path=notify_path, socket_type=zmq.PULL, bind=True
            )
            self._local_mem = self._engine.register_torch_tensor(region)
        except Exception:
            self.deregister()
            raise
        self._base = region.data_ptr()
        self._agent_metadata = msgspec.msgpack.encode(
            _PeerMetadata(
                engine_desc=self._engine.get_engine_desc().pack(),
                mem_desc=self._local_mem.pack(),
                notify_path=notify_path,
            )
        )
        self._mem_descriptor_bytes = serialize_mem_descriptor(
            build_block_descs(self._base, region.shape[0], block_size_bytes)
        )

    def get_agent_metadata(self) -> bytes:
        return self._agent_metadata

    def get_mem_descriptor(self) -> bytes:
        return self._mem_descriptor_bytes

    def add_remote_peer(self, metadata: bytes, mem_descriptor: bytes) -> str:
        meta = msgspec.msgpack.decode(metadata, type=_PeerMetadata)
        engine_desc = EngineDesc.unpack(meta.engine_desc)
        remote_mem = MemoryDesc.unpack(meta.mem_desc)
        agent_name = engine_desc.key
        if agent_name in self._peers:
            self.remove_remote_peer(agent_name)
        remote_blocks = deserialize_mem_descriptor(mem_descriptor)
        # post_read addresses remote block i at i * block_size from the start
        # of the producer's registered region.
        if remote_blocks and remote_blocks[0][0] != remote_mem.data:
            raise ValueError(
                f"EC: producer {agent_name} block descriptors do not start at "
                "its registered region"
            )
        self._engine.register_remote_engine(engine_desc)
        try:
            session = self._engine.create_session(self._local_mem, remote_mem)
            if session is None:
                raise RuntimeError(f"MoRI-IO could not open a session to {agent_name}")
        except Exception:
            self._deregister_remote_engine(engine_desc)
            raise
        self._peers[agent_name] = _Peer(
            engine_desc=engine_desc,
            session=session,
            notify=self._push_socket(meta.notify_path),
        )
        return agent_name

    def _deregister_remote_engine(self, engine_desc: Any) -> None:
        try:
            self._engine.deregister_remote_engine(engine_desc)
        except Exception:
            logger.warning(
                "EC: deregister_remote_engine failed for %s",
                engine_desc.key,
                exc_info=True,
            )

    def _push_socket(self, path: str) -> zmq.Socket:
        sock = self._push_by_path.get(path)
        if sock is None:
            sock = make_zmq_socket(
                ctx=self._ctx, path=path, socket_type=zmq.PUSH, bind=False, linger=0
            )
            self._push_by_path[path] = sock
        return sock

    def remove_remote_peer(self, agent_name: str) -> None:
        peer = self._peers.pop(agent_name, None)
        if peer is None:
            return
        self._retired_sessions.append(peer.session)
        self._deregister_remote_engine(peer.engine_desc)

    def post_read(
        self,
        local_indices: list[int],
        agent_name: str,
        remote_indices: list[int],
        notif_msg: bytes,
    ) -> Any:
        if len(local_indices) != len(remote_indices):
            raise ValueError(
                f"EC: local/remote block count mismatch "
                f"({len(local_indices)} vs {len(remote_indices)})"
            )
        peer = self._peers[agent_name]
        bs = self._block_size
        local_offsets, remote_offsets, sizes = _merge_segments(
            [i * bs for i in local_indices],
            [i * bs for i in remote_indices],
            bs,
        )
        status = peer.session.batch_read(
            local_offsets,
            remote_offsets,
            sizes,
            self._engine.allocate_transfer_uid(),
        )
        return _MoriXfer(
            status=status,
            notify=peer.notify,
            notif_msg=notif_msg,
            session=peer.session,
        )

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        self._reap_parked()
        msgs: list[bytes] = []
        while self._pull is not None:
            try:
                msgs.append(self._pull.recv(flags=zmq.NOBLOCK))
            except zmq.Again:
                break
        return {"moriio": msgs} if msgs else {}

    def check_xfer_state(self, handle: Any) -> str:
        status = handle.status
        if status.Succeeded():
            if not handle.notified:
                handle.notified = True
                try:
                    handle.notify.send(handle.notif_msg, flags=zmq.NOBLOCK)
                except zmq.ZMQError:
                    # The producer's pin lease releases the blocks instead.
                    logger.warning("EC: completion notification not sent")
            return "DONE"
        if status.Failed():
            return f"ERR: {status.Message()} (code={status.Code()})"
        return "PROC"

    def release_xfer_handle(self, handle: Any) -> None:
        if handle.status.InProgress():
            self._parked.append(handle)
        self._reap_parked()

    def _reap_parked(self) -> None:
        if self._parked:
            self._parked = [h for h in self._parked if h.status.InProgress()]

    def deregister(self) -> None:
        if self._local_mem is not None:
            try:
                self._engine.deregister_memory(self._local_mem)
            except Exception:
                logger.warning("EC: deregister failed", exc_info=True)
            self._local_mem = None
        if self._pull is not None:
            self._pull.close(linger=0)
            self._pull = None
        for sock in self._push_by_path.values():
            sock.close(linger=0)
        self._push_by_path.clear()
        if not self._ctx.closed:
            self._ctx.term()
