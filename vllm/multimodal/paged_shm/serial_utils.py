# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Shared-memory serialization utilities for paged SHM cache.

This module provides low-level functions to encode/decode ShmItem objects
to/from a block-based layout in shared memory. It depends only on
PagedShmStorage for actual I/O, not on PagedShmClient.

Layout:
  - Block 0: meta block (fixed size = block_size)
    Contains: num_chunks (4 bytes) + chunk_lengths (4 bytes each)
  - Blocks 1..N: data blocks, storing encoded chunks sequentially.
    Each chunk occupies a whole number of blocks (except maybe the last),
    and chunks are placed one after another without gaps at the block level.
    The unused tail of a chunk's last block is not used by subsequent chunks.
"""

import struct
from collections.abc import Sequence

from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.multimodal.paged_shm.storage import PagedShmStorage
from vllm.multimodal.paged_shm.types import ShmItem
from vllm.multimodal.processing.processor import ResolvedPromptUpdate
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


def encode_item(
    item: ShmItem,
    block_size: int,
    encoder: MsgpackEncoder,
) -> tuple[bytes, Sequence[bytes], Sequence[int]]:
    """
    Encode a ShmItem into a meta block and a tuple of data chunks.

    Returns:
        - meta_block_data: raw bytes for the meta block (padded to block_size)
        - chunks: tuple of bytes chunks as produced by MsgpackEncoder.encode
        - chunk_lengths: list of original chunk lengths
    """
    chunks = encoder.encode(item)  # tuple of bytes
    num_chunks = len(chunks)
    chunk_lengths = [len(chunk) for chunk in chunks]

    # Build meta data: num_chunks (4 bytes) + lengths (4 bytes each)
    meta_data = struct.pack("<I", num_chunks)
    for length in chunk_lengths:
        meta_data += struct.pack("<I", length)

    # Pad meta data to block_size
    meta_block_data = bytearray(block_size)
    meta_block_data[: len(meta_data)] = meta_data

    return bytes(meta_block_data), chunks, chunk_lengths


def write_encoded_to_blocks(
    storage: PagedShmStorage,
    meta_block_data: bytes,
    chunks: Sequence[bytes],
    blocks: list[int],
) -> None:
    """
    Write an encoded item to shared memory using the provided block list.

    The first block in `blocks` is used for meta, the rest for data.
    Each chunk is written to consecutive blocks (possibly spanning multiple blocks),
    and chunks are placed sequentially.
    """
    if not blocks:
        raise ValueError("blocks list cannot be empty")

    # Write meta block
    storage.write(meta_block_data, [blocks[0]])

    block_index = 1
    block_size = storage.block_size

    for chunk in chunks:
        if not isinstance(chunk, bytes):
            chunk = bytes(chunk)
        chunk_len = len(chunk)
        if chunk_len == 0:
            continue
        num_blocks_needed = (chunk_len + block_size - 1) // block_size
        if block_index + num_blocks_needed > len(blocks):
            raise ValueError(
                f"Insufficient blocks: need {block_index + num_blocks_needed}, "
                f"have {len(blocks)}"
            )
        chunk_blocks = blocks[block_index : block_index + num_blocks_needed]
        storage.write(chunk, chunk_blocks)
        block_index += num_blocks_needed


def read_decoded_from_blocks(
    storage: PagedShmStorage,
    blocks: list[int],
    block_size: int,
    decoder: MsgpackDecoder,
) -> tuple[MultiModalKwargsItem, Sequence[ResolvedPromptUpdate]]:
    """
    Read and decode an item from shared memory using the provided block list.

    Assumes the same layout: first block = meta, rest = data.
    Each chunk is read independently from its allocated block group.
    """
    if not blocks:
        raise ValueError("blocks list cannot be empty")

    # Read meta block
    meta_np = storage.read_to_numpy(block_size, [blocks[0]])

    # Parse meta
    offset = 0
    num_chunks = struct.unpack("<I", meta_np[offset : offset + 4].tobytes())[0]
    offset += 4
    chunk_lengths = []
    for _ in range(num_chunks):
        length = struct.unpack("<I", meta_np[offset : offset + 4].tobytes())[0]
        offset += 4
        chunk_lengths.append(length)

    if num_chunks == 0:
        return MultiModalKwargsItem({}), []

    # Read each chunk independently
    data_blocks = blocks[1:]
    if not data_blocks:
        raise ValueError("Data blocks missing for non-zero chunk count")

    chunks = []
    block_index = 0
    for length in chunk_lengths:
        if length == 0:
            chunks.append(b"")
            continue
        num_blocks_needed = (length + block_size - 1) // block_size
        if block_index + num_blocks_needed > len(data_blocks):
            raise ValueError(
                f"Insufficient data blocks: need {block_index + num_blocks_needed}, "
                f"have {len(data_blocks)}"
            )
        chunk_blocks = data_blocks[block_index : block_index + num_blocks_needed]
        chunk_data = storage.read_to_numpy(length, chunk_blocks)
        chunks.append(chunk_data.tobytes())
        block_index += num_blocks_needed

    # Decode the full chunk tuple
    item = decoder.decode(tuple(chunks))
    return item.kwargs_item, item.prompt_updates
