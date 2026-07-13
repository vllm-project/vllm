# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in diagnostics for Mamba/MTP block lifetime issues."""

import json
import os
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

DEBUG_ENV = "VLLM_ASCEND_DEBUG_MAMBA_MTP_BLOCKS"
SAMPLE_ENV = "VLLM_ASCEND_DEBUG_MAMBA_MTP_BLOCKS_SAMPLE"
LIMIT_ENV = "VLLM_ASCEND_DEBUG_MAMBA_MTP_BLOCKS_LIMIT"
LOG_PREFIX = "[MAMBA_MTP_BLOCK_DEBUG]"


@lru_cache(maxsize=1)
def debug_enabled() -> bool:
    return os.getenv(DEBUG_ENV, "").lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def sample_enabled() -> bool:
    return os.getenv(SAMPLE_ENV, "").lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def debug_limit() -> int:
    raw = os.getenv(LIMIT_ENV, "8")
    try:
        return max(1, int(raw))
    except ValueError:
        return 8


def _trim(values: Sequence[Any] | list[Any], limit: int | None = None) -> list[Any]:
    if limit is None:
        limit = debug_limit()
    if len(values) <= limit:
        return list(values)
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    return list(values[:head]) + ["..."] + list(values[-tail:])


def block_id(block: Any) -> int | None:
    if block is None:
        return None
    if isinstance(block, int):
        return block
    return getattr(block, "block_id", None)


def block_ids_tail(blocks: Any, limit: int | None = None) -> Any:
    if not debug_enabled() or blocks is None:
        return None
    if isinstance(blocks, tuple):
        return [block_ids_tail(group, limit) for group in blocks]
    if isinstance(blocks, list):
        return _trim([block_id(block) for block in blocks], limit)
    if isinstance(blocks, Iterable) and not isinstance(blocks, (str, bytes, dict)):
        return block_ids_tail(list(blocks), limit)
    return block_id(blocks)


def block_meta(block: Any) -> dict[str, Any]:
    if block is None:
        return {"id": None}
    block_hash = getattr(block, "block_hash", None)
    return {
        "id": block_id(block),
        "ref": getattr(block, "ref_cnt", None),
        "hash": str(block_hash) if block_hash is not None else None,
        "null": bool(getattr(block, "is_null", False)),
    }


def blocks_meta_tail(blocks: Any, limit: int | None = None) -> Any:
    if not debug_enabled() or blocks is None:
        return None
    if isinstance(blocks, tuple):
        return [blocks_meta_tail(group, limit) for group in blocks]
    if isinstance(blocks, list):
        return _trim([block_meta(block) for block in blocks], limit)
    if isinstance(blocks, Iterable) and not isinstance(blocks, (str, bytes, dict)):
        return blocks_meta_tail(list(blocks), limit)
    return block_meta(blocks)


def debug_log(event: str, **payload: Any) -> None:
    if not debug_enabled():
        return
    payload = {"event": event, **payload}
    logger.warning("%s %s", LOG_PREFIX, json.dumps(payload, sort_keys=True, default=str))
