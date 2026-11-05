# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IOP2PConnector — ZMQ-backed P2P implementation of IOConnectorBase.

Moves arbitrary payloads (MultiModalKwargsItem or torch.Tensor) between
disaggregated IO roles (preprocessor / worker / postprocessor) over ZMQ
DEALER/ROUTER sockets without unnecessary serialization copies.

Design notes
------------
  * ``put_data()`` (sender side) is non-blocking: it encodes the payload
    using MsgpackEncoder (which yields a header buffer + zero or more
    tensor backing-memory views) and stores all buffers in the engine's
    local send buffer, then immediately returns a ``DataRef`` encoding the
    sender's own address.

  * ``get_data()`` (receiver side) uses the sender address from the
    ``DataRef`` to issue a GET request to the sender's engine, which
    serves the buffered buffers on demand.  MsgpackDecoder reconstructs
    the original object from the received buffers using the type tag stored
    in the wire envelope.

  * ``cleanup_data()`` discards the buffered payload from the sender's
    engine if the receiver never calls get_data().

Zero-copy path
--------------
MsgpackEncoder encodes the payload into a list of buffers:
  bufs[0]   — msgpack header (small, inlined metadata + tensor shape/dtype)
  bufs[1:]  — direct memoryview pointers into each tensor's backing storage

These buffers are handed to the engine, which sends each one as a separate
ZMQ frame with ``zmq.Frame(buf, copy=False)``.  ZMQ holds a reference to
the backing memory and puts it on the wire without an additional memcpy.

On the receive side the frame bytes are fed directly to MsgpackDecoder
with ``share_mem=True`` (the default), so tensors are reconstructed as
views over the received bytes rather than copying them again.

Engine surface used (see p2p_engine.py):
    put_object(object_id, frames)
    fetch_object(object_id, remote_address) -> list[bytes] | None   [sync]
    fetch_object_future(object_id, remote_address) -> Future         [async-bridge]
    delete_object(object_id)
    shutdown()
"""

import asyncio
import threading
from concurrent.futures import Executor
from typing import TYPE_CHECKING, Any

import msgspec
import torch

from vllm.entrypoints.pooling.pooling.protocol import (
    DataLocation,
    DataObject,
    DataRef,
    MetadataRef,
)
from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.utils.network_utils import get_ip
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from .base import IOConnectorBase, IOConnectorRole
from .p2p_engine import P2pEngine

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Allowed payload types and the tag written into the wire envelope.
# The tag is used on the receive side to pick the right MsgpackDecoder type.
_TYPE_TAG: dict[type, str] = {
    MultiModalKwargsItem: "MultiModalKwargsItem",
    torch.Tensor: "Tensor",
}
_TAG_TO_TYPE: dict[str, type] = {v: k for k, v in _TYPE_TAG.items()}

# Thread-local storage for encoder and decoder instances.
# MsgpackEncoder / MsgpackDecoder are not thread-safe (they use mutable
# instance state — aux_buffers — during encode/decode), but they are safe to
# reuse within a single thread sequentially.  Constructing a new Encoder or
# Decoder on every call is expensive because msgspec compiles the type schema
# at construction time.  thread-local instances pay that cost once per thread
# and then reuse the compiled schema on every subsequent call.
_local = threading.local()


def _get_encoder() -> MsgpackEncoder:
    enc = getattr(_local, "encoder", None)
    if enc is None:
        enc = MsgpackEncoder()
        _local.encoder = enc
    return enc


def _get_decoder(tag: str) -> MsgpackDecoder:
    decoders: dict[str, MsgpackDecoder] = getattr(_local, "decoders", None) or {}
    dec = decoders.get(tag)
    if dec is None:
        t = _TAG_TO_TYPE.get(tag)
        if t is None:
            raise ValueError(f"IOP2PConnector: unknown type tag {tag!r}")
        dec = MsgpackDecoder(t=t, share_mem=False)
        decoders[tag] = dec
        _local.decoders = decoders
    return dec


class IOP2PConnector(IOConnectorBase):
    """Peer-to-peer transport for arbitrary vLLM payloads.

    Both SENDER and RECEIVER roles (and BOTH) share the same engine
    instance.

    Serialization uses MsgpackEncoder / MsgpackDecoder so that large
    tensors are transferred as raw memory views rather than base64 blobs.
    The encoder emits one buffer per tensor (plus a small header buffer);
    each buffer becomes its own ZMQ frame sent with ``copy=False``.
    """

    def __init__(self, vllm_config: "VllmConfig", role: IOConnectorRole) -> None:
        super().__init__(vllm_config, role)

        cfg = vllm_config.io_transfer_config
        extra_cfg = getattr(cfg, "io_connector_extra_config", {})
        self._host = extra_cfg.get("host") or get_ip()
        self._bind_host = extra_cfg.get("bind_host") or "0.0.0.0"
        self._port = int(extra_cfg.get("port", 5000))
        self._bind_port = int(extra_cfg.get("bind_port", self._port))

        # Engine handles both directions; construct once regardless of role.
        self._engine = P2pEngine(self._bind_host, self._bind_port)

    def _pack(self, data: Any) -> list:
        """Encode *data* into a list of ZMQ-ready buffers.

        Wire layout:
          buf 0  — msgpack envelope: ``{"type": <tag>}``
          buf 1  — msgpack header produced by MsgpackEncoder (tensor shapes/dtypes
                   and indices into the out-of-band buffers)
          buf 2+ — raw tensor backing-memory views (one per large tensor)
        """
        tag = _TYPE_TAG.get(type(data))
        if tag is None:
            raise TypeError(
                f"IOP2PConnector: unsupported payload type {type(data).__name__}. "
                f"Supported: {list(_TYPE_TAG)}"
            )
        data_bufs = _get_encoder().encode(data)
        envelope = msgspec.msgpack.encode({"type": tag})
        return [envelope, *data_bufs]

    def _unpack(self, frames: list) -> Any:
        """Inverse of _pack.  Returns the decoded data object."""
        envelope = msgspec.msgpack.decode(frames[0].buffer)
        if not isinstance(envelope, dict):
            raise ValueError(
                f"IOP2PConnector: corrupt payload: expected a dict envelope, "
                f"got {type(envelope).__name__!r}"
            )
        tag: str = envelope["type"]
        t = _TAG_TO_TYPE.get(tag)
        if t is None:
            raise ValueError(f"IOP2PConnector: unknown type tag {tag!r}")
        # frames[1]  — msgpack header (decoded by msgspec, no frombuffer)
        # frames[2+] — raw tensor backing buffers; copy into writable bytearray
        #              so that torch.frombuffer sees a writable buffer and does
        #              not emit a UserWarning about non-writable memory.
        decodable = [frames[1].buffer, *[memoryview(f.buffer) for f in frames[2:]]]
        return _get_decoder(tag).decode(decodable)

    @staticmethod
    def _address(host: str, port: int) -> str:
        return f"{host}:{port}"

    @staticmethod
    def _parse_locator(locator: str) -> str:
        """Strip the ``tcp://`` scheme from *locator* and return ``host:port``."""
        return locator.replace("tcp://", "")

    def put_data(
        self,
        object_id: str,
        data: Any,
    ) -> DataRef:
        """Encode *data* into raw buffers and buffer them locally.

        Returns a DataRef immediately; the receiver calls get_data(ref) later
        to pull the buffers from this instance over ZMQ.
        """
        if not self.is_io_sender:
            raise RuntimeError(
                "This IOP2PConnector is not configured as a sender "
                "(is_io_sender=False). Check your io_transfer_config."
            )

        frames = self._pack(data)
        self._engine.put_object(object_id, frames)

        logger.debug(
            "IOP2PConnector: buffered object_id=%s (%d frames)",
            object_id,
            len(frames),
        )

        own_address = self._address(self._host, self._port)
        return DataRef(
            object_id=object_id,
            location=DataLocation(locator=f"tcp://{own_address}"),
        )

    def get_data(
        self,
        ref: DataRef,
    ) -> DataObject:
        """Pull the payload for *ref* from the sender and return it."""
        if not self.is_io_receiver:
            raise RuntimeError(
                "This IOP2PConnector is not configured as a receiver "
                "(is_io_receiver=False). Check your io_transfer_config."
            )

        remote_address = self._parse_locator(ref.location.locator)
        frames = self._engine.fetch_object(ref.object_id, remote_address)

        if frames is None:
            raise KeyError(
                f"IOP2PConnector: no data found for object_id="
                f"{ref.object_id!r} at {remote_address}"
            )

        data = self._unpack(frames)
        logger.debug(
            "IOP2PConnector: fetched object_id=%s from %s",
            ref.object_id,
            remote_address,
        )
        return DataObject(data=data)

    async def get_data_async(
        self,
        ref: DataRef,
        executor: Executor | None = None,
    ) -> DataObject:
        """Zero-thread async fetch: awaits the ZMQ reply without occupying an OS thread.

        Wraps the ``concurrent.futures.Future`` returned by
        ``engine.fetch_object_future`` into an awaitable on the calling event
        loop using ``asyncio.wrap_future``.  The calling coroutine yields while
        the ZMQ reply is in flight; no executor thread is held during the RTT.
        """
        if not self.is_io_receiver:
            raise RuntimeError(
                "This IOP2PConnector is not configured as a receiver "
                "(is_io_receiver=False). Check your io_transfer_config."
            )

        remote_address = self._parse_locator(ref.location.locator)
        frames = await asyncio.wrap_future(
            self._engine.fetch_object_future(ref.object_id, remote_address)
        )

        if frames is None:
            raise KeyError(
                f"IOP2PConnector: no data found for object_id="
                f"{ref.object_id!r} at {remote_address}"
            )

        # _unpack is CPU-bound (MsgpackDecoder + bytearray copy); run in the
        # executor so it does not block the event loop.
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(executor, self._unpack, frames)
        logger.debug(
            "IOP2PConnector: fetched object_id=%s from %s",
            ref.object_id,
            remote_address,
        )
        return DataObject(data=data)

    async def put_data_async(
        self,
        object_id: str,
        data: Any,
        executor: Executor | None = None,
    ) -> DataRef:
        """Async put: encoding is CPU-bound, offloaded to *executor*.

        ``put_object`` (in-memory store) is called after encoding completes,
        still inside the executor, so the event loop is never blocked.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.put_data, object_id, data)

    def cleanup_data(self, ref: DataRef) -> None:
        """Discard a buffered payload that was never fetched.

        A no-op if get_data() already consumed the payload (the engine's
        listener thread removes it from the buffer at serve time).
        """
        if not self.is_io_sender:
            raise RuntimeError(
                "This IOP2PConnector is not configured as a sender "
                "(is_io_sender=False). Check your io_transfer_config."
            )

        self._engine.delete_object(ref.object_id)

    def put_metadata(
        self,
        request_id: str,
        metadata: dict,
    ) -> MetadataRef:
        """Buffer per-request *metadata* under *request_id* in the P2P engine."""
        if not self.is_io_sender:
            raise RuntimeError(
                "This IOP2PConnector is not configured as a sender "
                "(is_io_sender=False). Check your io_transfer_config."
            )
        packed = msgspec.msgpack.encode(metadata)
        frames: list[bytes | memoryview] = [packed]
        self._engine.put_object(f"meta:{request_id}", frames)
        logger.debug("IOP2PConnector: buffered metadata for request_id=%s", request_id)

        own_address = self._address(self._host, self._port)
        return MetadataRef(
            request_id=request_id,
            location=DataLocation(locator=f"tcp://{own_address}"),
        )

    def get_metadata(
        self,
        ref: MetadataRef,
    ) -> dict:
        """Fetch per-request metadata from the sender's P2P engine."""
        if not self.is_io_receiver:
            raise RuntimeError(
                "This IOP2PConnector is not configured as a receiver "
                "(is_io_receiver=False). Check your io_transfer_config."
            )
        remote_address = self._parse_locator(ref.location.locator)
        frames = self._engine.fetch_object(f"meta:{ref.request_id}", remote_address)
        if frames is None:
            return {}
        result = msgspec.msgpack.decode(frames[0].buffer)
        if not isinstance(result, dict):
            raise ValueError(
                f"IOP2PConnector: corrupt metadata for request_id={ref.request_id!r}: "
                f"expected a dict, got {type(result).__name__!r}"
            )
        return result

    def cleanup_metadata(self, ref: "MetadataRef") -> None:
        """Discard buffered metadata that was never fetched."""
        self._engine.delete_object(f"meta:{ref.request_id}")

    def shutdown(self) -> None:
        self._engine.shutdown()
        super().shutdown()
