# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire format of the Mooncake pull protocol.

The decoder asks for block ids, reads the bytes itself, and reports back; the
prefiller answers with its address book and the block ids it handed out. These
are the messages both sides encode, kept apart from the worker that decides
what to do with them.
"""

from enum import IntEnum

import msgspec

from vllm.distributed.kv_transfer.kv_connector.utils import EngineId

# Wire marker of the pull protocol. Push and pull messages are not decodable by
# the other mode, and this marker turns a mixed deployment into an explicit
# error instead of a decode failure deep inside a handler.
PULL_PROTOCOL = "mooncake_pull_v1"

# The pull protocol shares these two identities with the push connector but not
# its implementation, so it declares them here rather than importing the
# connector module.
ReqId = str  # Internal scheduler request ID
TransferId = str  # KV transfer coordination ID (shared by P/D)

# Identifies one reader worker: (engine_id, tp_rank, pp_rank).
ReaderKey = tuple[EngineId, int, int]


class PullStatus(IntEnum):
    """Status of one pull response."""

    FINISH = 0
    CONTINUE = 1
    ERROR = 2


class MooncakePullLayout(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    """A prefiller worker's KV cache address book.

    Sent once per peer and cached by the reader. Every response carries the
    `layout_id`, so a reader notices a peer that came back with another layout.
    """

    engine_id: EngineId
    hostname: str
    rpc_port: int
    tp_rank: int
    pp_rank: int
    tp_size: int
    pp_size: int
    block_size: int
    physical_blocks_per_logical: int
    use_mla: bool
    has_mamba: bool
    producer_cache_replicated: bool
    kv_caches_base_addr: list[int]
    block_lens: list[int]
    kv_block_lens: list[int]
    layer_names: list[str]
    layer_indices: list[int]
    group_indices: list[int]
    shared_group_ids: list[list[int]] = msgspec.field(default_factory=list)
    row_offsets: list[int] = msgspec.field(default_factory=list)
    layout_id: str = ""


class MooncakePullRequest(
    msgspec.Struct,
    tag="read",
    tag_field="kind",
    omit_defaults=True,  # type: ignore[call-arg]
):
    """Decoder to prefiller: these requests are about to be read."""

    protocol: str
    compat_hash: str
    engine_id: EngineId
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    # d_req_id -> (transfer_id, whether this reader needs any bytes)
    reqs: dict[ReqId, tuple[TransferId, bool]]
    want_layout: bool = False


class MooncakePullDone(
    msgspec.Struct,
    tag="done",
    tag_field="kind",
    omit_defaults=True,  # type: ignore[call-arg]
):
    """Decoder to prefiller: these transfers are finished on the reader."""

    protocol: str
    engine_id: EngineId
    tp_rank: int
    pp_rank: int
    transfer_ids: list[TransferId]
    failed_transfers: list[TransferId] | None = None


class MooncakePullReading(
    msgspec.Struct,
    tag="reading",
    tag_field="kind",
    omit_defaults=True,  # type: ignore[call-arg]
):
    """Decoder to prefiller: the read of these transfers is still running."""

    protocol: str
    engine_id: EngineId
    tp_rank: int
    pp_rank: int
    transfer_ids: list[TransferId]


class MooncakePullResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    status: PullStatus
    layout_id: str | None = None
    compat_hash: str | None = None
    layout: MooncakePullLayout | None = None
    # d_req_id -> (transfer_id, producer block ids)
    resolved: dict[ReqId, tuple[TransferId, list[list[int]]]] | None = None
    err_reqs: list[ReqId] | None = None
    err_msg: str | None = None


class MooncakePullAck(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    ok: bool = True


PullMessage = MooncakePullRequest | MooncakePullDone | MooncakePullReading
