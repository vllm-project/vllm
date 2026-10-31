# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Messages a decoder sends to the prefiller it reads cache blocks from.

The prefiller answers with the push connector's `MooncakeXferResponse`, whose
`metadata` carries its registered KV cache regions and the blocks it resolved
for this wave. Sharing that envelope keeps one response type for both
directions and lets a mixed deployment fail with the peer's message.
"""

import msgspec

from vllm.distributed.kv_transfer.kv_connector.utils import EngineId

# The pull protocol uses the same identities as the push connector but declares
# them here, so this module does not import the connector.
ReqId = str  # Internal scheduler request ID
TransferId = str  # KV transfer coordination ID (shared by P/D)

# Identifies one reader worker: (engine_id, tp_rank, pp_rank).
ReaderKey = tuple[EngineId, int, int]


class MooncakePullRequest(
    msgspec.Struct,
    tag="read",
    tag_field="kind",
    omit_defaults=True,  # type: ignore[call-arg]
):
    """Decoder to prefiller: these requests are about to be read."""

    engine_id: EngineId
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    # d_req_id -> (transfer_id, whether this reader needs any bytes)
    reqs: dict[ReqId, tuple[TransferId, bool]]
    # The reader asks for the registered regions on its first pull from this
    # peer and reuses them afterwards.
    want_regions: bool = False


class MooncakePullDone(
    msgspec.Struct,
    tag="done",
    tag_field="kind",
    omit_defaults=True,  # type: ignore[call-arg]
):
    """Decoder to prefiller: these transfers are finished on the reader."""

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

    engine_id: EngineId
    tp_rank: int
    pp_rank: int
    transfer_ids: list[TransferId]


PullMessage = MooncakePullRequest | MooncakePullDone | MooncakePullReading
