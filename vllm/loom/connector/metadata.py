# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.v1.kv_offload.worker.worker import TransferSpec
from vllm.v1.core.kv_cache_utils import BlockHash

ReqId = str


class RequestPhase(str, Enum):
    PREFILL = "PREFILL"
    DECODE = "DECODE"


@dataclass
class RegenSpec:
    block_hashes: list[BlockHash]
    dst_block_ids: list[int]
    extra: dict[str, Any]


@dataclass
class LoomConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: dict[ReqId, TransferSpec]
    reqs_to_store: dict[ReqId, TransferSpec]
    reqs_to_regen: dict[ReqId, RegenSpec]


@dataclass(frozen=True)
class LoomSharedPrefixHandshake(KVConnectorHandshakeMetadata):
    extents: list[dict[str, int]]
