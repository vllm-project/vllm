# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Protocol

# Ordered spans carry capacity when advertised and valid bytes after compression.
BufferSpan = tuple[int, int]


class CompressedChunk(Protocol):
    buffers: tuple[BufferSpan, ...]
    descriptor: bytes
    event: Any


class ChunkIO(Protocol):
    """Runtime operations on slots reserved by the connector memory plan."""

    @property
    def wire_profile(self) -> str: ...

    def slot_buffers(self, slot_id: int) -> tuple[BufferSpan, ...]: ...

    def plan(
        self,
        slot_id: int,
        blocks: list[list[int]],
        kv_part: int,
        valid_tokens: dict[int, int],
    ) -> Any: ...

    def compress(self, plan: Any) -> CompressedChunk:
        """Return readable spans owned by the slot until transport completes."""
        ...

    def decompress(
        self, plan: Any, lengths: tuple[int, ...], descriptor: bytes
    ) -> Any: ...
