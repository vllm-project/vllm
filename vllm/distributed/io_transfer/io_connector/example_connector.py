# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example IO connector that stores/loads base64-encoded files on a shared filesystem.

This is intentionally simple — suitable for debugging and single-node
setups — and mirrors the structure of ECExampleConnector.
"""

import os
import struct
import threading
from typing import TYPE_CHECKING, Any

import msgspec
import torch

from vllm.distributed.io_transfer.io_connector.base import (
    IOConnectorBase,
    IOConnectorRole,
)
from vllm.entrypoints.pooling.pooling.protocol import (
    DataLocation,
    DataObject,
    DataRef,
    MetadataRef,
)
from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

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

# ------------------------------------------------------------------
# Wire framing helpers
#
# Format written to disk:
#   [4B LE] number of frames  N
#   for each frame:
#     [4B LE] frame length L
#     [L bytes] frame data
#
# Frame 0 is always the UTF-8 type tag.
# Frames 1..N-1 are the raw buffers returned by MsgpackEncoder.encode().
#
# This avoids re-packing all tensor side-buffers into a single merged
# allocation; writelines() issues a writev()-style write for each chunk.
# ------------------------------------------------------------------


def _pack_frames(tag: str, bufs: list) -> list:
    """Return iovec-style chunks for writelines() — no buffer merging."""
    tag_bytes = tag.encode()
    n_frames = 1 + len(bufs)
    header = struct.pack("<I", n_frames)
    tag_frame = struct.pack("<I", len(tag_bytes)) + tag_bytes
    chunks: list = [header, tag_frame]
    for b in bufs:
        b = bytes(b)  # materialise memoryview without a second copy
        chunks.append(struct.pack("<I", len(b)))
        chunks.append(b)
    return chunks


def _unpack_frames(raw: bytes) -> tuple[str, list[bytes]]:
    """Inverse of _pack_frames — O(n_frames) with no extra allocation."""
    pos = 0
    (n_frames,) = struct.unpack_from("<I", raw, pos)
    pos += 4
    frames: list[bytes] = []
    for _ in range(n_frames):
        (length,) = struct.unpack_from("<I", raw, pos)
        pos += 4
        frames.append(raw[pos : pos + length])
        pos += length
    if not frames:
        raise ValueError("IOExampleConnector: empty frame list in payload")
    tag = frames[0].decode()
    return tag, frames[1:]


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
            raise ValueError(f"IOExampleConnector: unknown type tag {tag!r}")
        dec = MsgpackDecoder(t=t, share_mem=False)
        decoders[tag] = dec
        _local.decoders = decoders
    return dec


class IOExampleConnector(IOConnectorBase):
    """Filesystem-backed IO connector.

    Layout on disk::

        <shared_storage_path>/
          <object_id>.msg
          <request_id>.meta

    The payload is encoded using MsgpackEncoder (which efficiently handles
    tensors as zero-copy memory views) and written as a single file.  Reading
    it back constructs a :class:`~vllm.entrypoints.pooling.pooling.protocol.DataObject`.
    """

    def __init__(self, vllm_config: "VllmConfig", role: IOConnectorRole) -> None:
        super().__init__(vllm_config=vllm_config, role=role)
        cfg = vllm_config.io_transfer_config
        assert cfg is not None
        self._storage_path: str = cfg.get_from_extra_config(
            "shared_storage_path", "/tmp/io_transfer"
        )
        logger.debug("IOExampleConnector storage_path=%s", self._storage_path)

        self.created_data_refs: list[DataRef] = []

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def put_data(
        self,
        object_id: str,
        data: Any,
    ) -> DataRef:
        """Encode *data* and write it to ``<storage_path>/<object_id>.msg``.

        Returns:
            A ``DataRef`` whose ``location.locator`` points to the written file.

        Raises:
            RuntimeError: If this connector is not configured as a sender.

        """
        if not self.is_io_sender:
            raise RuntimeError(
                "This IOExampleConnector is not configured as a sender "
                "(is_io_sender=False). Check your io_transfer_config."
            )

        filepath = os.path.join(self._storage_path, f"{object_id}.msg")

        # Attempt to create the file exclusively before encoding.
        # O_CREAT | O_EXCL is atomic: no separate stat(), no TOCTOU race.
        # If the file already exists we skip encoding entirely (fix #2).
        try:
            fd = os.open(filepath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return DataRef(object_id=object_id, location=DataLocation(locator=filepath))

        tag = _TYPE_TAG.get(type(data))
        if tag is None:
            os.close(fd)
            raise TypeError(
                f"IOExampleConnector: unsupported payload type {type(data).__name__}. "
                f"Supported: {list(_TYPE_TAG)}"
            )
        encoder = _get_encoder()
        data_bufs = list(encoder.encode(data))
        chunks = _pack_frames(tag, data_bufs)

        with os.fdopen(fd, "wb") as f:
            f.writelines(chunks)

        logger.debug("Saved IO data with object_id=%s", object_id)

        data_ref = DataRef(object_id=object_id, location=DataLocation(locator=filepath))
        self.created_data_refs.append(data_ref)

        return data_ref

    def get_data(
        self,
        ref: DataRef,
    ) -> DataObject:
        """Read and deserialize the file referenced by *ref* using MsgpackDecoder.

        Returns:
            The ``DataObject`` stored by the corresponding ``put_data`` call.

        Raises:
            RuntimeError: If this connector is not configured as a receiver.
            KeyError: If no file exists at the path recorded in *ref*.

        """
        if not self.is_io_receiver:
            raise RuntimeError(
                "This IOExampleConnector is not configured as a receiver "
                "(is_io_receiver=False). Check your io_transfer_config."
            )

        filepath = os.path.join(self._storage_path, f"{ref.object_id}.msg")

        # Single open() — no preceding exists() stat(), no TOCTOU race.
        try:
            with open(filepath, "rb") as f:
                raw_bytes = f.read()
        except FileNotFoundError:
            raise KeyError(
                f"No IO transfer payload found for object_id={ref.object_id!r} "
                f"(expected at {filepath})"
            ) from None

        tag, frames = _unpack_frames(raw_bytes)
        decoder = _get_decoder(tag)
        data = decoder.decode(frames)

        return DataObject(data=data)

    def cleanup_data(self, ref: DataRef) -> None:
        """Delete the file written by ``put_data`` for *ref*.

        A no-op when the file has already been removed.

        Raises:
            RuntimeError: If this connector is not configured as a sender.

        """
        filepath = os.path.join(self._storage_path, f"{ref.object_id}.msg")
        try:
            os.remove(filepath)
            logger.debug("Removed IO data file %s", filepath)
        except FileNotFoundError:
            pass

    def put_metadata(
        self,
        request_id: str,
        metadata: dict,
    ) -> MetadataRef:
        """Write per-request *metadata* to ``<storage_path>/<request_id>.meta``."""
        if not self.is_io_sender:
            raise RuntimeError(
                "This IOExampleConnector is not configured as a sender "
                "(is_io_sender=False). Check your io_transfer_config."
            )
        filepath = os.path.join(self._storage_path, f"{request_id}.meta")
        with open(filepath, "wb") as f:
            packed_metadata = msgspec.msgpack.encode(metadata)
            f.write(packed_metadata)
        logger.debug("Saved metadata for request_id=%s", request_id)

        return MetadataRef(
            request_id=request_id,
            location=DataLocation(locator=filepath),
        )

    def get_metadata(
        self,
        ref: MetadataRef,
    ) -> dict:
        """Read per-request metadata from ``<storage_path>/<request_id>.meta``."""
        if not self.is_io_receiver:
            raise RuntimeError(
                "This IOExampleConnector is not configured as a receiver "
                "(is_io_receiver=False). Check your io_transfer_config."
            )
        filepath = ref.location.locator
        if not os.path.exists(filepath):
            return {}
        with open(filepath, "rb") as f:
            result = msgspec.msgpack.decode(f.read())
        if not isinstance(result, dict):
            raise ValueError(
                "IOExampleConnector: corrupt metadata for "
                f"request_id={ref.request_id!r}: expected a dict, "
                f"got {type(result).__name__!r}"
            )
        return result

    def cleanup_metadata(self, ref: "MetadataRef") -> None:
        """Delete the metadata file for *ref.request_id*."""
        filepath = os.path.join(self._storage_path, f"{ref.request_id}.meta")
        try:
            os.remove(filepath)
            logger.debug("Removed metadata file %s", filepath)
        except FileNotFoundError:
            pass

    def shutdown(self) -> None:
        for data_ref in self.created_data_refs:
            self.cleanup_data(data_ref)
        self.created_data_refs = []
