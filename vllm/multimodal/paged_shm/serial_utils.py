# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Serialization utilities for paged shared memory cache.

This module provides functions to encode multi-modal cache items into chunks
that can be stored in shared memory blocks, and to decode them back.

The serialization format uses a separate metadata chunk prepended to the data chunks.
The overall layout in shared memory is:

    Chunks:   [ Meta ]   [   Data 0   ]         [ Data 1  ...]
    Blocks:   +----------+----------+----------+----------+ ...
              Block 0    Block 1    Block 2    Block 3

    - Each chunk starts at a new block boundary (block-aligned).
    - A chunk may occupy multiple consecutive blocks.
    - Padding (unused space) may exist at the end of the last block of each chunk.

The Metadata Chunk consists of:
    1. An 8‑byte header:
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

import numpy as np
import torch

from vllm.multimodal.cache import MultiModalProcessorCacheInItem
from vllm.multimodal.paged_shm.storage import PagedShmStorage
from vllm.multimodal.paged_shm.types import PagedShmCacheOutItem
from vllm.utils.torch_utils import DeviceLikeType
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


def encode_item(
    mm_item: MultiModalProcessorCacheInItem,
    encoder: MsgpackEncoder,
) -> tuple[Sequence[bytes | np.ndarray | torch.Tensor], Sequence[int]] | None:
    """Encode a multi-modal item into chunks for shared memory storage."""
    raw_chunks = encoder.encode(
        PagedShmCacheOutItem(kwargs_item=mm_item[0], prompt_updates=mm_item[1])
    )

    if len(raw_chunks) == 1:
        # Single data chunk: no need for shared memory transfer.
        return None

    # Convert non-tensor chunks: first chunk -> bytes, others -> np.ndarray (view)
    converted = []
    for idx, ch in enumerate(raw_chunks):
        if isinstance(ch, torch.Tensor):
            converted.append(ch)
        elif isinstance(ch, np.ndarray):
            # Keep as is (no copy)
            converted.append(ch)
        else:
            # bytes, bytearray, memoryview, etc.
            if idx == 0:
                converted.append(bytes(ch))
            else:
                converted.append(np.frombuffer(ch, dtype=np.uint8))

    # Build metadata body: (length, type) for each data chunk.
    meta_body = b""
    for ch in converted:
        if isinstance(ch, torch.Tensor):
            length = ch.nbytes
            flag = 1
        elif isinstance(ch, np.ndarray):
            length = ch.nbytes
            flag = 0
        else:
            length = len(ch)
            flag = 0
        meta_body += struct.pack("<I", length) + struct.pack("<B", flag)

    # Metadata chunk: 8‑byte header + body.
    meta_chunk_size = 8 + len(meta_body)
    num_total_chunks = 1 + len(converted)
    header = struct.pack("<I", meta_chunk_size) + struct.pack("<I", num_total_chunks)
    meta_chunk = header + meta_body

    chunks = [meta_chunk] + converted
    lengths = [len(meta_chunk)] + [
        ch.nbytes if isinstance(ch, (torch.Tensor, np.ndarray)) else len(ch)
        for ch in converted
    ]
    return chunks, lengths


def write_encoded_to_blocks(
    storage: PagedShmStorage,
    chunks: Sequence[bytes | np.ndarray | torch.Tensor],
    blocks: Sequence[int],
) -> None:
    """Write encoded chunks into shared memory blocks."""
    if not blocks:
        raise ValueError("Blocks list cannot be empty")

    block_size = storage.block_size
    block_idx = 0
    for chunk_idx, ch in enumerate(chunks):
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
    decoder: MsgpackDecoder,
    device: DeviceLikeType = "cpu",
) -> MultiModalProcessorCacheInItem:
    """
    Read and decode data from shared memory blocks using the metadata format.

    The layout is expected to match the encoding produced by encode_item():
        - First, the metadata chunk (with 8‑byte header and per‑chunk entries).
        - Then, the data chunks.

    The read process:
        1. Read the first 8 bytes to get metadata chunk size and total chunk count.
        2. Read the entire metadata chunk.
        3. Parse per‑chunk lengths and type flags.
        4. Read each data chunk, placing torch.Tensor data on the specified device
           and bytes‑like data as memoryview on CPU.

    Type flag 0: chunk is bytes-like -> returned as memoryview.
    Type flag 1: chunk is torch.Tensor -> returned as a tensor on the specified device.

    Args:
        storage: PagedShmStorage instance providing `read_to_tensor`.
        blocks: List of block indices containing the data.
        block_size: Size of each block in bytes.
        decoder: MessagePack decoder instance.
        device: Device to load tensor chunks onto (default: "cpu").

    Returns:
        A tuple (kwargs_item, prompt_updates) as decoded.

    Raises:
        ValueError: If blocks list is empty or insufficient for any chunk.
    """
    if not blocks:
        raise ValueError("Blocks list cannot be empty")

    # Read the first 8 bytes to get metadata chunk size and total chunk count.
    first8 = storage.read_to_tensor(8, [blocks[0]], device="cpu").numpy().tobytes()
    meta_chunk_size, num_total_chunks = struct.unpack("<II", first8)
    num_data_chunks = num_total_chunks - 1

    # Read the entire metadata chunk.
    blocks_needed_meta = (meta_chunk_size + block_size - 1) // block_size
    if len(blocks) < blocks_needed_meta:
        raise ValueError(
            f"Insufficient blocks for metadata chunk: need {blocks_needed_meta}, "
            f"got {len(blocks)}"
        )
    meta_tensor = storage.read_to_tensor(
        meta_chunk_size, list(blocks[:blocks_needed_meta]), device="cpu"
    )
    meta_bytes = meta_tensor.numpy().tobytes()

    # Parse per‑data‑chunk lengths and types.
    pos = 8
    original_lengths = []
    type_flags = []
    for _ in range(num_data_chunks):
        length = struct.unpack("<I", meta_bytes[pos : pos + 4])[0]
        pos += 4
        flag = struct.unpack("<B", meta_bytes[pos : pos + 1])[0]
        pos += 1
        original_lengths.append(length)
        type_flags.append(flag)

    # Read each data chunk.
    block_offset = blocks_needed_meta
    raw_chunks = []
    for i in range(num_data_chunks):
        length = original_lengths[i]
        blocks_needed = (length + block_size - 1) // block_size
        if block_offset + blocks_needed > len(blocks):
            raise ValueError(
                f"Insufficient blocks for data chunk {i}: need {blocks_needed}, "
                f"available from offset {block_offset} is {len(blocks) - block_offset}"
            )

        # Read chunk data. For bytes‑like data, we read to CPU; for tensors,
        # we read directly to the requested device.
        chunk_device = device if type_flags[i] == 1 else "cpu"
        chunk_data = storage.read_to_tensor(
            length,
            list(blocks[block_offset : block_offset + blocks_needed]),
            device=chunk_device,
        )

        if type_flags[i] == 1:
            raw_chunks.append(chunk_data)
        else:
            raw_chunks.append(memoryview(chunk_data.numpy()))

        block_offset += blocks_needed

    # Decode and return.
    decoded = decoder.decode(raw_chunks)
    return decoded.kwargs_item, decoded.prompt_updates
