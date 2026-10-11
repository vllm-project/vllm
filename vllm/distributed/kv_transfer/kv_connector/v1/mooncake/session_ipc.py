# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IPC messages for the out-of-process compression session helper.

The compression session machinery (ZMQ transport, protocol state machine,
slot scheduling, timeout handling and completion accounting) runs in a
helper process spawned per TP rank.  The engine process keeps the KV cache,
the compression arena and the C++/CUDA encode-decode path; it only executes
short jobs and publishes per-step metadata deltas.

Messages are msgspec structs so both sides can share the definitions.  Keep
this module free of engine-only imports (torch, vllm internals): the helper
may be started before the connector wires everything up.
"""

from __future__ import annotations

import os

import msgspec

SESSION_PROCESS_ENV = "KVSERVE_SESSION_PROCESS"


def validate_session_process_mode() -> None:
    """Validate that the retired in-process compression backend is not selected."""
    if os.environ.get(SESSION_PROCESS_ENV) == "0":
        raise ValueError(
            "KVSERVE_SESSION_PROCESS=0 is no longer supported; unset it to use "
            "the compression session helper. Raw Mooncake transport is unaffected."
        )


def handshake_target_ranks(
    tp_rank: int, tp_size: int, remote_tp_size: int
) -> list[int]:
    """Remote TP ranks this local rank pairs with (DCP size 1)."""
    if tp_size >= remote_tp_size:
        return [tp_rank // (tp_size // remote_tp_size)]
    ratio = remote_tp_size // tp_size
    return [tp_rank * ratio + index for index in range(ratio)]


class SlotAddresses(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    payload: int
    aux: int
    tail: int


class InitMessage(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    """One-time engine to helper configuration."""

    needs_sender: bool
    needs_receiver: bool
    register_with_bootstrap: bool
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    engine_id: str
    dp_rank: int
    hostname: str
    rpc_port: int
    bootstrap_host: str
    bootstrap_port: int
    slot_addresses: list[SlotAddresses]
    kv_caches_base_addr: list[int]
    block_lens: list[int]
    kv_block_lens: list[int]
    registered_layer_names: list[str]
    registered_layer_indices: list[int]
    registered_group_indices: list[int]
    registered_shared_group_ids: list[list[int]] = msgspec.field(default_factory=list)
    registered_row_offsets: list[int] = msgspec.field(default_factory=list)
    abort_timeout_s: float = 480.0
    compression_profile: str = ""
    logical_blocks_per_chunk: int = 1
    fa_group_block_sizes: dict[int, int] = msgspec.field(default_factory=dict)


class HelperReady(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    side_channel_port: int = 0


class HelperHeartbeat(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    pass


class EngineStop(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    pass


class HelperFailed(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    reason: str


class Stopped(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    pass


class ProducerAnnouncement(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    p_req_id: str
    transfer_id: str
    local_block_ids: list[list[int]]
    num_tokens: int | None = None


class ProducerPublish(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    p_req_id: str
    transfer_id: str


class ProducerMetadata(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    announced: list[ProducerAnnouncement] = msgspec.field(default_factory=list)
    published: list[ProducerPublish] = msgspec.field(default_factory=list)
    dropped: list[str] = msgspec.field(default_factory=list)


class ConsumerRequest(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    d_req_id: str
    transfer_id: str
    local_block_ids: list[list[int]]
    num_tokens: int | None
    remote_engine_id: str
    remote_bootstrap_addr: str


class ConsumerMetadata(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    requests: list[ConsumerRequest] = msgspec.field(default_factory=list)


class FinalEvent(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    """A request finished sending/receiving; kinds: sending, sending_expired."""

    kind: str
    req_id: str


class RecvDone(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    """One consumer request finished on every remote worker it pulled from."""

    d_req_id: str
    failed_blocks: list[int] = msgspec.field(default_factory=list)


class OpenProducerJob(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    job_id: int
    transfer_id: str
    d_req_id: str
    nonce: str
    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    remote_pp_size: int
    local_block_ids: list[list[int]]
    remote_block_ids: list[list[int]]
    num_tokens: int | None
    kv_caches_base_addr: list[int]
    block_lens: list[int]
    kv_block_lens: list[int]
    registered_layer_names: list[str]
    registered_layer_indices: list[int]
    registered_group_indices: list[int]
    registered_shared_group_ids: list[list[int]] = msgspec.field(default_factory=list)
    registered_row_offsets: list[int] = msgspec.field(default_factory=list)


class EncodeChunkJob(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    job_id: int
    transfer_id: str
    d_req_id: str
    nonce: str
    chunk_id: int
    slot_id: int
    local_block_ids: list[list[int]]
    valid_tokens_by_group: dict[int, int]
    remote_session: str
    payload_addr: int
    aux_addr: int
    tail_addr: int


class DecodeChunkJob(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    job_id: int
    transfer_id: str
    d_req_id: str
    nonce: str
    chunk_id: int
    slot_id: int
    local_block_ids: list[list[int]]
    valid_tokens_by_group: dict[int, int]
    codec_bytes: int
    u8_bytes: int


class DrainSlotsJob(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    job_id: int
    slot_ids: list[int]


class JobDone(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    job_id: int
    error: str | None = None
    logical_block_count: int = 0
    total_chunks: int = 0
    local_block_ids: list[list[int]] = msgspec.field(default_factory=list)
    valid_tokens_by_group: dict[int, int] = msgspec.field(default_factory=dict)
    u8_bytes: int = 0
    codec_bytes: int = 0
    aux_bytes: int = 0


HelperJob = OpenProducerJob | EncodeChunkJob | DecodeChunkJob | DrainSlotsJob
EngineToHelper = ProducerMetadata | ConsumerMetadata | JobDone | EngineStop
HelperToEngine = (
    HelperJob
    | HelperReady
    | HelperHeartbeat
    | HelperFailed
    | Stopped
    | FinalEvent
    | RecvDone
)
