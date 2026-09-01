# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Serialization utilities for paged shared memory cache.

This module provides functions to encode multi-modal cache items into chunks
that can be stored in shared memory blocks, and to decode them back.

The serialization format uses a metadata header prepended to the first chunk.
The header contains:
- Number of chunks (4 bytes, unsigned int, little-endian)
- For each chunk: original length (4 bytes, unsigned int) and type flag (1 byte)

Type flag meanings:
- 0: bytes-like data (returned as bytes or memoryview)
- 1: torch.Tensor (returned as CPU tensor)

The encoding function returns None when only one chunk exists (i.e., no large
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
    """
    Encode a multi-modal item into chunks and prepare metadata for shared memory.

    The encoder serializes the item into a sequence of chunks. If the encoded data
    consists of only one chunk (no large tensor), returns None to indicate that
    shared memory transfer is unnecessary. Otherwise, prepends a metadata header
    to the first chunk and returns the modified chunks and their lengths.

    Metadata header format (little-endian):
        - num_chunks: unsigned int (4 bytes)
        - For each chunk:
            - original_length: unsigned int (4 bytes)
            - type_flag: unsigned char (1 byte):
                0 -> data is bytes-like
                     (returned as bytes for first, np.ndarray for others)
                1 -> data is torch.Tensor

    The first returned chunk is a bytes object containing the header followed by
    the original first chunk data. Other chunks remain as either np.ndarray or
    torch.Tensor (if they were tensors originally).

    Args:
        mm_item: Input item, expected to be a tuple (kwargs_item, prompt_updates).
        encoder: MessagePack encoder instance.

    Returns:
        A tuple (chunks, lengths) if multiple chunks exist, otherwise None.
        - chunks: List of chunk data (mixed types).
        - lengths: List of byte lengths for each chunk.

    Raises:
        TypeError: If chunk type is unsupported during conversion.
    """
    raw_chunks = encoder.encode(
        PagedShmCacheOutItem(kwargs_item=mm_item[0], prompt_updates=mm_item[1])
    )

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

    num_chunks = len(converted)

    if num_chunks == 1:
        # No tensor larger than block_size found, not using shm transfer.
        return None

    first = converted[0]
    if not isinstance(first, bytes):
        first = bytes(first)

    # Build metadata: num_chunks (4 bytes) + (length (4) + type (1)) for each chunk
    original_lengths = []
    type_flags = []
    for ch in converted:
        if isinstance(ch, torch.Tensor):
            length = ch.nbytes
            type_flags.append(1)
        elif isinstance(ch, np.ndarray):
            length = ch.nbytes
            type_flags.append(0)
        else:
            length = len(ch)
            type_flags.append(0)
        original_lengths.append(length)

    meta_data = struct.pack("<I", num_chunks)
    for length, flag in zip(original_lengths, type_flags):
        meta_data += struct.pack("<I", length)
        meta_data += struct.pack("<B", flag)

    merged_first = meta_data + first
    new_chunks = [merged_first] + converted[1:]
    new_lengths = [len(merged_first)] + original_lengths[1:]
    return new_chunks, new_lengths


def write_encoded_to_blocks(
    storage: PagedShmStorage,
    chunks: Sequence[bytes | np.ndarray | torch.Tensor],
    blocks: Sequence[int],
) -> None:
    """
    Write encoded chunks into shared memory blocks.

    Each chunk starts at a new block boundary; blocks are consumed sequentially
    from the provided list. The number of blocks allocated must be sufficient to
    hold all chunks. Any unused blocks are ignored.

    Args:
        storage: PagedShmStorage instance providing `write(data, block_indices)`.
        chunks: List of chunk data (first chunk includes metadata header).
        blocks: List of block indices, ordered and contiguous.

    Raises:
        ValueError: If blocks list is empty or insufficient for any chunk.
        TypeError: If a chunk type is not accepted by `storage.write`.
    """
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
    Read and decode data from shared memory blocks using metadata header.

    The first chunk contains a header that specifies the number of chunks and
    the original length and type of each chunk. This function reads chunk by
    chunk, strips the header, and reconstructs the original data types.

    - Type flag 0: chunk is bytes-like -> returned as memoryview.
    - Type flag 1: chunk is torch.Tensor -> returned as a CPU tensor on the
      specified device (the header chunk is always read on CPU, but subsequent
      tensor chunks are placed on the requested device via `device` parameter).

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

    # Read first block to get header
    header_data = storage.read_to_tensor(block_size, list(blocks[:1]), device="cpu")
    header_np = header_data.numpy()  # view of the CPU tensor

    # Parse header (small data, copy via tobytes is fine)
    num_chunks = struct.unpack("<I", header_np[:4].tobytes())[0]
    pos = 4
    original_lengths = []
    type_flags = []
    for _ in range(num_chunks):
        length = struct.unpack("<I", header_np[pos : pos + 4].tobytes())[0]
        pos += 4
        flag = struct.unpack("<B", header_np[pos : pos + 1].tobytes())[0]
        pos += 1
        original_lengths.append(length)
        type_flags.append(flag)
    header_size = pos  # 4 + num_chunks * 5

    # Read first chunk (header + original data)
    first_chunk_total = header_size + original_lengths[0]
    blocks_needed_first = (first_chunk_total + block_size - 1) // block_size

    if first_chunk_total <= block_size:
        # header_np already contains the entire first chunk
        first_original = memoryview(
            header_np[header_size : header_size + original_lengths[0]]
        )
    else:
        if len(blocks) < blocks_needed_first:
            raise ValueError(
                f"Insufficient blocks for first chunk: need {blocks_needed_first}, "
                f"got {len(blocks)}"
            )
        first_chunk_data = storage.read_to_tensor(
            first_chunk_total,
            list(blocks[:blocks_needed_first]),
            device="cpu",
        )
        first_chunk_np = first_chunk_data.numpy()
        first_original = memoryview(
            first_chunk_np[header_size : header_size + original_lengths[0]]
        )

    raw_chunks: list[memoryview | torch.Tensor] = [first_original]

    # Read remaining chunks
    block_offset = blocks_needed_first
    for i in range(1, num_chunks):
        length = original_lengths[i]
        blocks_needed = (length + block_size - 1) // block_size
        if block_offset + blocks_needed > len(blocks):
            raise ValueError(
                f"Insufficient blocks for chunk {i}: need {blocks_needed}, "
                f"available from offset {block_offset} is {len(blocks) - block_offset}"
            )

        chunk_data = storage.read_to_tensor(
            length,
            list(blocks[block_offset : block_offset + blocks_needed]),
            device=device,
        )

        if type_flags[i] == 1:
            raw_chunks.append(chunk_data)
        else:
            raw_chunks.append(memoryview(chunk_data.numpy()))

        block_offset += blocks_needed

    # Decode using mixed types (memoryview, tensor)
    decoded = decoder.decode(raw_chunks)
    return decoded.kwargs_item, decoded.prompt_updates
