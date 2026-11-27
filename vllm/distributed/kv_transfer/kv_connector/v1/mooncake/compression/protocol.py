# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal

import msgspec

from vllm.distributed.kv_transfer.kv_connector.v1.compression import BufferSpan

if TYPE_CHECKING:
    from zmq.asyncio import Socket

KVPart = Literal[0, 1]
ReqId = str
TransferId = str


class MooncakeXferMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    req_blocks: dict[ReqId, tuple[TransferId, list[list[int]]]]
    kv_caches_base_addr: list[int]
    block_lens: list[int]
    kv_block_lens: list[int]
    registered_layer_names: list[str] = msgspec.field(default_factory=list)
    registered_layer_indices: list[int] = msgspec.field(default_factory=list)
    registered_group_indices: list[int] = msgspec.field(default_factory=list)
    # Parallel to registered regions. Each entry is the groups that share
    # that region. Empty means the peer did not send the field.
    registered_shared_group_ids: list[list[int]] = msgspec.field(default_factory=list)
    # Byte offset of each registered view from its block row. -1 means the
    # peer did not send the field, so the run must not be promoted.
    registered_row_offsets: list[int] = msgspec.field(default_factory=list)
    remote_pp_size: int = 1
    # Zero keeps the full-layout protocol used by compression sessions.
    layout_id: int = 0
    requires_draft_context_kv: bool = False


class CompressionSessionOpen(  # type: ignore[call-arg]
    msgspec.Struct, tag=True, frozen=True
):
    metadata: MooncakeXferMetadata
    nonce: str = ""
    profile: str = ""


class CompressionSessionOpened(  # type: ignore[call-arg]
    msgspec.Struct, tag=True, frozen=True
):
    logical_block_count: int
    total_chunks: int
    error: str | None = None
    profile: str = ""


class CompressionChunkPull(  # type: ignore[call-arg]
    msgspec.Struct, tag=True, frozen=True
):
    transfer_id: TransferId
    d_req_id: ReqId
    chunk_id: int
    buffers: tuple[BufferSpan, ...]
    nonce: str = ""


class CompressionChunkReady(  # type: ignore[call-arg]
    msgspec.Struct, tag=True, frozen=True
):
    chunk_id: int
    lengths: tuple[int, ...] = ()
    descriptor: bytes = b""
    error: str | None = None


class CompressionAbort(  # type: ignore[call-arg]
    msgspec.Struct, tag=True, frozen=True
):
    transfer_id: TransferId
    d_req_id: ReqId
    reason: str
    nonce: str = ""


class CompressionAbortAck(  # type: ignore[call-arg]
    msgspec.Struct, tag=True, frozen=True
):
    transfer_id: TransferId
    d_req_id: ReqId
    nonce: str
    error: str | None = None


CompressionRequest = CompressionSessionOpen | CompressionChunkPull | CompressionAbort
CompressionResponse = (
    CompressionSessionOpened | CompressionChunkReady | CompressionAbortAck
)


async def wait_for_abort_ack(
    socket: Socket,
    decoder: msgspec.msgpack.Decoder,
    abort: CompressionAbort,
    timeout: float,
) -> None:
    """Wait for the producer to confirm that it has drained the session."""
    async with asyncio.timeout(timeout):
        while True:
            reply = decoder.decode(await socket.recv())
            # Replies to operations issued before Abort can already be queued.
            if isinstance(reply, (CompressionSessionOpened, CompressionChunkReady)):
                continue
            if (
                isinstance(reply, CompressionAbortAck)
                and reply.transfer_id == abort.transfer_id
                and reply.d_req_id == abort.d_req_id
                and reply.nonce == abort.nonce
                and not reply.error
            ):
                return
            raise RuntimeError("Producer did not acknowledge safe session cancellation")
