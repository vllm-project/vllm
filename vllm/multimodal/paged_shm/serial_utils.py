# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Serialization utilities for paged shared memory cache.
"""

import struct
from collections.abc import Sequence

import numpy as np
import torch

from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.multimodal.paged_shm.storage import PagedShmStorage
from vllm.multimodal.paged_shm.types import ShmItem
from vllm.multimodal.processing.processor import ResolvedPromptUpdate
from vllm.utils.torch_utils import DeviceLikeType
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


def encode_item(
    kwargs_item: MultiModalKwargsItem,
    prompt_updates: Sequence[ResolvedPromptUpdate],
    encoder: MsgpackEncoder,
) -> tuple[Sequence[bytes | np.ndarray | torch.Tensor], Sequence[int]] | None:
    """
    Encode a multi-modal item into chunks and prepare metadata.

    If the encoded data consists of only one chunk (no large tensor),
    returns None to indicate no shared memory transfer is needed.
    Otherwise, prepends a metadata header to the first chunk and returns
    the modified chunks and their lengths.

    The metadata header contains:
    - num_chunks (4 bytes, little-endian unsigned int)
    - For each chunk:
        - original length (4 bytes, little-endian)
        - type flag (1 byte): 0 for bytes, 1 for torch.Tensor

    The first returned chunk is a bytes object that includes the header
    followed by the original first chunk data. Other chunks remain as
    either np.ndarray or torch.Tensor (if they were tensors originally).

    Returns:
        (chunks, lengths) or None.
    """
    item = ShmItem(kwargs_item=kwargs_item, prompt_updates=prompt_updates)
    raw_chunks = encoder.encode(item)

    # Convert non-tensor chunks: first chunk -> bytes, others -> np.ndarray (view)
    converted = []
    for idx, ch in enumerate(raw_chunks):
        if isinstance(ch, torch.Tensor):
            converted.append(ch)
        else:
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
    Write chunks into shared memory blocks, each chunk starting at a new
    block boundary. The blocks for each chunk are contiguous and taken
    sequentially from the provided block list.

    Args:
        storage: PagedShmStorage instance.
        chunks: List of chunk data (first chunk includes metadata header).
        blocks: List of block indices allocated for the entire data.
    """
    if not blocks:
        raise ValueError("Blocks list cannot be empty")

    block_size = storage.block_size
    block_idx = 0
    for ch in chunks:
        size = ch.nbytes if isinstance(ch, (torch.Tensor, np.ndarray)) else len(ch)

        num_blocks = (size + block_size - 1) // block_size
        sub_blocks = blocks[block_idx : block_idx + num_blocks]
        if len(sub_blocks) < num_blocks:
            raise ValueError(
                f"Not enough blocks allocated for chunk: need {num_blocks}, "
                f"got {len(sub_blocks)} remaining"
            )
        storage.write(ch, list(sub_blocks))
        block_idx += num_blocks


def read_decoded_from_blocks(
    storage: PagedShmStorage,
    blocks: Sequence[int],
    block_size: int,
    decoder: MsgpackDecoder,
    device: DeviceLikeType = "cpu",
):
    """
    Read data from shared memory blocks, parse metadata, and decode into
    kwargs_item and prompt_updates.

    This function reads data chunk by chunk, using precise lengths (not
    padded block sizes). The first chunk contains a metadata header which
    is stripped before decoding.

    The type flags are used to restore each chunk to its original type:
    - flag=0: the chunk is bytes (returned as memoryview)
    - flag=1: the chunk is a torch.Tensor (returned as CPU tensor)

    Returns:
        (kwargs_item, prompt_updates)
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
    item = decoder.decode(raw_chunks)
    return item.kwargs_item, item.prompt_updates
