# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in runtime trace for vocab-state materialization and compact paths."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import torch

import vllm.envs as envs
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

_TRACE_LOCK = threading.Lock()


def tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def tp_size_for_trace() -> int:
    try:
        return int(get_tensor_model_parallel_world_size())
    except Exception:
        return 1


def tp_rank_for_trace() -> int:
    try:
        return int(get_tensor_model_parallel_rank())
    except Exception:
        return 0


def emit_consumer_state_trace(record: dict[str, Any]) -> None:
    path = envs.VLLM_CONSUMER_STATE_TRACE_JSONL
    if not path:
        return

    event = {
        "schema_version": 1,
        "timestamp_unix_s": time.time(),
        "framework": "vllm",
        **record,
    }
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = json.dumps(event, sort_keys=True, separators=(",", ":"))
    with _TRACE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def emit_full_vocab_trace(
    *,
    component: str,
    full_logits: torch.Tensor | None,
    vocab_size: int,
    consumer_contract: str,
    fallback_reason: str | None = None,
    tp_size: int | None = None,
    rank: int | None = None,
) -> None:
    tp_size = tp_size_for_trace() if tp_size is None else int(tp_size)
    rank = tp_rank_for_trace() if rank is None else int(rank)
    rows = int(full_logits.shape[0]) if full_logits is not None else 0
    emit_consumer_state_trace(
        {
            "component": component,
            "path": "full_vocab_materialized",
            "consumer_contract": consumer_contract,
            "rank": rank,
            "tp_size": tp_size,
            "rows": rows,
            "vocab_size": int(vocab_size),
            "local_vocab_size": int(full_logits.shape[-1])
            if full_logits is not None and full_logits.ndim > 0
            else 0,
            "full_vocab_materialized_bytes": tensor_nbytes(full_logits),
            "full_vocab_reread_bytes": 0,
            "compact_state_bytes": 0,
            "tp_gather_bytes": tensor_nbytes(full_logits) if tp_size > 1 else 0,
            "fallback_reason": fallback_reason,
            "exact_replay_status": "runtime_metadata_only",
        }
    )


def emit_compact_vocab_state_trace(
    *,
    component: str,
    rows: int,
    vocab_size: int,
    local_vocab_size: int,
    local_vocab_materialized_bytes: int,
    compact_state_bytes: int,
    consumer_contract: str,
    tp_gather_bytes: int = 0,
    fallback_reason: str | None = None,
    dtype_bytes: int = 2,
    tp_size: int | None = None,
    rank: int | None = None,
) -> None:
    tp_size = tp_size_for_trace() if tp_size is None else int(tp_size)
    rank = tp_rank_for_trace() if rank is None else int(rank)
    rows = int(rows)
    emit_consumer_state_trace(
        {
            "component": component,
            "path": "consumer_sufficient_compact",
            "consumer_contract": consumer_contract,
            "rank": rank,
            "tp_size": tp_size,
            "rows": rows,
            "vocab_size": int(vocab_size),
            "local_vocab_size": int(local_vocab_size),
            "local_vocab_materialized_bytes": int(local_vocab_materialized_bytes),
            "full_vocab_materialized_bytes": 0,
            "full_vocab_reread_bytes": 0,
            "avoidable_full_vocab_materialized_bytes": rows
            * int(vocab_size)
            * int(dtype_bytes),
            "compact_state_bytes": int(compact_state_bytes),
            "tp_gather_bytes": int(tp_gather_bytes),
            "fallback_reason": fallback_reason,
            "exact_replay_status": "runtime_metadata_only",
        }
    )
