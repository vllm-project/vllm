# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import IntEnum

import msgspec

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.metadata import (
    ReqId,
    TransferId,
)


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
    # Random per register_kv_caches, so ids never repeat across restarts.
    # 0 means the peer sends the layout on every pull.
    layout_id: int = 0


class MooncakeXferResponseStatus(IntEnum):
    # Transfer finished
    FINISH = 0
    # Continue to receive
    CONTINUE = 1
    # Something wrong, see err_msg
    ERROR = 2
    # P does not hold the layout for this layout_id; resend it.
    # Only sent for pulls with a nonzero layout_id and no layout.
    LAYOUT_MISS = 3


class MooncakeXferResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    status: MooncakeXferResponseStatus
    ok_reqs: list[ReqId] | None = None
    err_reqs: list[ReqId] | None = None
    err_msg: str | None = None
    # The pull's layout_id once P holds that layout. D omits the layout only
    # after seeing it, so a P that predates this field always gets it.
    layout_id: int = 0
