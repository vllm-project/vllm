# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Serialization utilities for paged shared memory cache.

This module provides functions to encode multi-modal cache items into chunks
that can be stored in shared memory blocks, and to decode them back.

The serialization format uses a separate metadata chunk prepended to the data chunks.
The overall layout in shared memory is:

    Chunks:   [ Meta ]   [   Data 0   ]        [ Data 1  ...]
    Blocks:   +----------+----------+----------+----------+ ...
              Block 0    Block 1    Block 2    Block 3

    - Each chunk starts at a new block boundary (block-aligned).
    - A chunk may occupy multiple consecutive blocks.
    - Padding (unused space) may exist at the end of the last block of each chunk.

The Metadata Chunk consists of:
    1. A 10‑byte header:
       - 2 bytes: magic number "M0"
           (identifies vLLM paged shared memory format version 0)
       - 4 bytes: size of the entire metadata chunk (including this header)
       - 4 bytes: total number of chunks (metadata chunk + all data chunks)
    2. For each data chunk, a metadata entry:
       - 4 bytes: original length of the data chunk (unsigned int, little‑endian)
       - 1 byte : type flag
           0 -> bytes‑like data (returned as memoryview)
           1 -> torch.Tensor (returned as CPU tensor on requested device)

The encoding function returns None when only one data chunk exists (i.e., no large
tensor), indicating that shared memory transfer is unnecessary.
"""

import struct
from collections.abc import Sequence
from enum import IntEnum
from typing import Any

import numpy as np
import torch
from msgspec import msgpack

from vllm.multimodal.cache import (
    MultiModalProcessorCacheInItem,
    MultiModalProcessorCacheOutItem,
)
from vllm.multimodal.paged_shm.storage import PagedShmStorage
from vllm.multimodal.paged_shm.types import PagedShmCacheOutItem
from vllm.utils.torch_utils import DeviceLikeType
from vllm.v1.serial_utils import (
    CUSTOM_TYPE_RAW_VIEW,
    MsgpackDecoder,
    MsgpackEncoder,
)
from vllm.v1.utils import tensor_data

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

MAGIC = b"M0"
MAGIC_LEN = len(MAGIC)
HEADER_SIZE = 10  # 2 (magic) + 4 (meta_chunk_size) + 4 (num_total_chunks)
ENTRY_STRUCT = struct.Struct("<IB")  # 4-byte length + 1-byte type flag
ENTRY_SIZE = ENTRY_STRUCT.size


# Type flags for data chunks
class ChunkType(IntEnum):
    """Type flag for a data chunk stored in shared memory."""

    BYTES = 0  # bytes-like data (returned as memoryview)
    TENSOR = 1  # torch.Tensor


# -----------------------------------------------------------------------------
# Encoder / Decoder for MessagePack (with out-of-band tensor handling)
# -----------------------------------------------------------------------------


class PagedShmEncoder(MsgpackEncoder):
    """MessagePack encoder that optionally places large tensors out-of-band."""

    def _encode_tensor(
        self, obj: torch.Tensor
    ) -> tuple[str, tuple[int, ...], int | memoryview]:
        # Small CPU tensors are encoded inline
        if obj.nbytes < self.size_threshold and obj.is_cpu:
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, tensor_data(obj))
        else:
            assert self.aux_buffers is not None
            data = len(self.aux_buffers)
            # hack: We directly put the tensor into aux_buffers.
            self.aux_buffers.append(obj)
        dtype = str(obj.dtype).removeprefix("torch.")
        return dtype, obj.shape, data


class PagedShmDecoder(MsgpackDecoder):
    """MessagePack decoder that retrieves out-of-band tensors from aux buffers."""

    def _decode_tensor(self, arr: Any) -> torch.Tensor | None:
        dtype, shape, data = arr
        is_aux = isinstance(data, int)
        buffer = self.aux_buffers[data] if is_aux else data
        if buffer is None:
            return None
        if not isinstance(buffer, torch.Tensor):
            return super()._decode_tensor(arr)
        torch_dtype = getattr(torch, dtype)
        assert isinstance(torch_dtype, torch.dtype)
        return buffer.view(torch_dtype).view(shape)


# -----------------------------------------------------------------------------
# Helpers for metadata chunk construction / parsing
# -----------------------------------------------------------------------------


def _build_metadata_chunk(chunk_lengths: list[int], chunk_types: list[int]) -> bytes:
    """
    Build the metadata chunk bytes.

    Args:
        chunk_lengths: Original lengths (in bytes) of each data chunk.
        chunk_types: Type flag (from ChunkType) for each data chunk.

    Returns:
        Bytes of the complete metadata chunk (header + entries).
    """
    num_data = len(chunk_lengths)
    if num_data != len(chunk_types):
        raise ValueError("chunk_lengths and chunk_types must have the same length")

    # Build body: pairs of (length, type)
    body = b"".join(
        ENTRY_STRUCT.pack(length, typ)
        for length, typ in zip(chunk_lengths, chunk_types)
    )
    meta_chunk_size = HEADER_SIZE + len(body)
    num_total_chunks = 1 + num_data
    header = MAGIC + struct.pack("<II", meta_chunk_size, num_total_chunks)
    return header + body


def _parse_metadata_chunk(meta_bytes: bytes) -> tuple[list[int], list[int]]:
    """
    Parse the metadata chunk bytes.

    Args:
        meta_bytes: Raw bytes of the metadata chunk (including header).

    Returns:
        A tuple (lengths, types) where each is a list of length num_data_chunks.

    Raises:
        ValueError: If magic number is invalid or data is malformed.
    """
    if len(meta_bytes) < HEADER_SIZE:
        raise ValueError("Metadata chunk too short to contain header")

    magic = meta_bytes[:MAGIC_LEN]
    if magic != MAGIC:
        raise ValueError(f"Invalid magic number: expected {MAGIC!r}, got {magic!r}")

    meta_chunk_size, num_total_chunks = struct.unpack(
        "<II", meta_bytes[MAGIC_LEN:HEADER_SIZE]
    )
    if len(meta_bytes) != meta_chunk_size:
        raise ValueError(
            f"Metadata chunk size mismatch: expected "
            f"{meta_chunk_size}, got {len(meta_bytes)}"
        )

    num_data_chunks = num_total_chunks - 1
    lengths: list[int] = []
    types: list[int] = []
    pos = HEADER_SIZE
    for _ in range(num_data_chunks):
        if pos + ENTRY_SIZE > len(meta_bytes):
            raise ValueError("Metadata chunk truncated before all entries parsed")
        length, typ = ENTRY_STRUCT.unpack_from(meta_bytes, pos)
        lengths.append(length)
        types.append(typ)
        pos += ENTRY_SIZE
    return lengths, types


# -----------------------------------------------------------------------------
# Public encoding / writing functions
# -----------------------------------------------------------------------------


def encode_item(
    mm_item: MultiModalProcessorCacheInItem,
    encoder: PagedShmEncoder,
) -> tuple[list[bytes | np.ndarray | torch.Tensor], list[int]] | None:
    """
    Encode a multi-modal item into chunks suitable for shared memory storage.

    When the encoded data consists of a single chunk (no large tensor), returns
    None to indicate that shared memory transfer is unnecessary.

    Args:
        mm_item: Input multi-modal item (kwargs_item, prompt_updates).
        encoder: Encoder instance for serializing the item.

    Returns:
        Either (chunks, lengths) where:
            - chunks: List of bytes, numpy arrays, or torch tensors.
            - lengths: List of byte sizes for each chunk (for easy block allocation).
        Or None if only one data chunk exists.
    """
    if mm_item is None:
        return None

    # Encode the item into a list of raw chunks (bytes, np.ndarray, or torch.Tensor)
    encoded = encoder.encode(
        PagedShmCacheOutItem(kwargs_item=mm_item[0], prompt_updates=mm_item[1])
    )

    if len(encoded) == 1:
        # Only one data chunk → no need for shared memory transfer
        return None

    # Normalize chunk types: first chunk → bytes, others → np.ndarray (view)
    converted: list[bytes | np.ndarray | torch.Tensor] = []
    for idx, ch in enumerate(encoded):
        if isinstance(ch, torch.Tensor):
            converted.append(ch)
        elif isinstance(ch, np.ndarray):
            converted.append(ch)  # keep as is
        else:
            # bytes, bytearray, memoryview, etc.
            if idx == 0:
                converted.append(bytes(ch))
            else:
                converted.append(np.frombuffer(ch, dtype=np.uint8))

    # Compute lengths and types for each data chunk
    chunk_lengths: list[int] = []
    chunk_types: list[int] = []
    for ch in converted:
        if isinstance(ch, torch.Tensor):
            chunk_lengths.append(ch.nbytes)
            chunk_types.append(ChunkType.TENSOR)
        else:  # bytes or np.ndarray
            length = ch.nbytes if isinstance(ch, np.ndarray) else len(ch)
            chunk_lengths.append(length)
            chunk_types.append(ChunkType.BYTES)

    # Build the metadata chunk
    meta_chunk = _build_metadata_chunk(chunk_lengths, chunk_types)

    # Assemble all chunks
    chunks: list[bytes | np.ndarray | torch.Tensor] = [meta_chunk] + converted
    # Lengths for all chunks (including metadata)
    all_lengths = [len(meta_chunk)] + chunk_lengths
    return chunks, all_lengths


def write_encoded_to_blocks(
    storage: PagedShmStorage,
    chunks: Sequence[bytes | np.ndarray | torch.Tensor],
    blocks: Sequence[int],
) -> None:
    """
    Write encoded chunks into shared memory blocks.

    Args:
        storage: The shared memory storage instance.
        chunks: List of chunks (bytes, ndarray, tensor) to write.
        blocks: List of block indices, must have enough blocks for all chunks.

    Raises:
        ValueError: If blocks list is empty or insufficient for any chunk.
    """
    if not blocks:
        raise ValueError("Blocks list cannot be empty")

    block_size = storage.block_size
    block_idx = 0

    for chunk_idx, ch in enumerate(chunks):
        # Compute size in bytes
        size = ch.nbytes if isinstance(ch, (torch.Tensor, np.ndarray)) else len(ch)
        num_blocks = (size + block_size - 1) // block_size
        sub_blocks = blocks[block_idx : block_idx + num_blocks]
        if len(sub_blocks) < num_blocks:
            raise ValueError(
                f"Not enough blocks for chunk {chunk_idx}: need {num_blocks}, "
                f"got {len(sub_blocks)} remaining (total blocks: {len(blocks)})"
            )
        storage.write(ch, list(sub_blocks))
        block_idx += num_blocks


def read_decoded_from_blocks(
    storage: PagedShmStorage,
    blocks: Sequence[int],
    block_size: int,
    decoder: PagedShmDecoder,
    skip_tensor_payload: bool = False,
    device: DeviceLikeType = "cpu",
) -> MultiModalProcessorCacheOutItem:
    """
    Read and decode data from shared memory blocks.

    The layout must match the format produced by `encode_item()`.

    Args:
        storage: PagedShmStorage instance.
        blocks: List of block indices containing the data.
        block_size: Size of each block in bytes.
        decoder: MessagePack decoder instance for deserialization.
        skip_tensor_payload: If True, do not read tensor data; return None
                             for tensor chunks (useful for metadata-only access).
        device: Device to load tensor chunks onto (default: "cpu").

    Returns:
        A tuple (kwargs_item, prompt_updates) as decoded.

    Raises:
        ValueError: If blocks are insufficient, magic number invalid, or data malformed.
    """
    if not blocks:
        raise ValueError("Blocks list cannot be empty")

    # --------------------------------------------------------------------
    # 1. Read and validate the header (first 10 bytes)
    # --------------------------------------------------------------------
    header_bytes = storage.read_to_numpy(HEADER_SIZE, [blocks[0]]).tobytes()
    if len(header_bytes) < HEADER_SIZE:
        raise ValueError("Not enough data to read header")

    magic = header_bytes[:MAGIC_LEN]
    if magic != MAGIC:
        raise ValueError(f"Invalid magic number: expected {MAGIC!r}, got {magic!r}")

    meta_chunk_size, num_total_chunks = struct.unpack(
        "<II", header_bytes[MAGIC_LEN:HEADER_SIZE]
    )
    num_data_chunks = num_total_chunks - 1

    # --------------------------------------------------------------------
    # 2. Read the entire metadata chunk
    # --------------------------------------------------------------------
    blocks_needed_meta = (meta_chunk_size + block_size - 1) // block_size
    if len(blocks) < blocks_needed_meta:
        raise ValueError(
            f"Insufficient blocks for metadata chunk: need {blocks_needed_meta}, "
            f"got {len(blocks)}"
        )

    meta_bytes = storage.read_to_numpy(
        meta_chunk_size, list(blocks[:blocks_needed_meta])
    ).tobytes()
    if len(meta_bytes) != meta_chunk_size:
        raise ValueError(
            f"Metadata chunk size mismatch: expected {meta_chunk_size}, "
            f"got {len(meta_bytes)}"
        )

    # Parse the metadata to get lengths and types
    chunk_lengths, chunk_types = _parse_metadata_chunk(meta_bytes)

    # --------------------------------------------------------------------
    # 3. Read each data chunk
    # --------------------------------------------------------------------
    raw_chunks: list[memoryview | torch.Tensor | None] = []
    block_offset = blocks_needed_meta  # start after metadata

    # First data chunk: MessagePack body (always bytes)
    if num_data_chunks < 1:
        raise ValueError("No data chunks found after metadata")
    first_len = chunk_lengths[0]
    if chunk_types[0] != ChunkType.BYTES:
        raise ValueError("First data chunk must be of type BYTES")
    blocks_needed = (first_len + block_size - 1) // block_size
    if block_offset + blocks_needed > len(blocks):
        raise ValueError("Insufficient blocks for MessagePack body")
    msgpack_body = storage.read_to_numpy(
        first_len,
        list(blocks[block_offset : block_offset + blocks_needed]),
    )
    raw_chunks.append(memoryview(msgpack_body))
    block_offset += blocks_needed

    # Remaining data chunks
    for i in range(1, num_data_chunks):
        if skip_tensor_payload:
            # Skip reading tensor data, but advance block offset
            length = chunk_lengths[i]
            blocks_needed = (length + block_size - 1) // block_size
            block_offset += blocks_needed
            raw_chunks.append(None)
            continue

        length = chunk_lengths[i]
        blocks_needed = (length + block_size - 1) // block_size
        if block_offset + blocks_needed > len(blocks):
            raise ValueError(
                f"Insufficient blocks for data chunk {i}: need {blocks_needed}, "
                f"available from offset {block_offset}: "
                f"{len(blocks) - block_offset}"
            )
        chunk_device = device if chunk_types[i] == ChunkType.TENSOR else "cpu"
        chunk_data = storage.read_to_tensor(
            length,
            list(blocks[block_offset : block_offset + blocks_needed]),
            device=chunk_device,
        )
        if chunk_types[i] == ChunkType.TENSOR:
            raw_chunks.append(chunk_data)
        else:  # BYTES
            raw_chunks.append(memoryview(chunk_data.numpy()))
        block_offset += blocks_needed

    # --------------------------------------------------------------------
    # 4. Decode and return
    # --------------------------------------------------------------------
    decoded = decoder.decode(raw_chunks)
    return decoded.kwargs_item, decoded.prompt_updates
