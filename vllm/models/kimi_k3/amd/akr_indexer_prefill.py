# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""amd_kernels routes of the FP4 prefill indexer (`ops/fp4_indexer_prefill.py`).

Entries `indexer/mxfp4_indexer_common.mxfp4_prefill_score` / `mxfp4_prefill_topk`; a
tile their `*_unsupported_reason` rejects takes the aiter op
(`ops/fp4_indexer_prefill.prefill_indexer`).

Sub-switches (env, default on; the `indexer` route of `akr_config` must be on for the
first two):
  VLLM_KIMI_AKR_INDEXER_SCORE=0/1
      amd_kernels `mxfp4_prefill_score` instead of aiter's FP4 prefill scorer
  VLLM_KIMI_AKR_INDEXER_TOPK=0/1
      amd_kernels `mxfp4_prefill_topk` instead of aiter `top_k_per_row_prefill` +
      epilogue
  VLLM_KIMI_AKR_PREFILL_TABLE_BUCKET=0/1
      power-of-two block-table width for aiter's scorer (no per-step JIT); independent
      of the route (host-only fix of the stock path)
"""

import functools
import os

import torch

from . import akr_config

# Below this mean window (keys per row) aiter's single-launch top-k wins (kernel report:
# 80 us floor of the four-launch select vs 68 us at a mean window of 42K).
TOPK_MIN_MEAN_WINDOW = 60_000
TOPK = 2048  # the Kimi-K3.1 indexer's top-k (vLLM's output width)


def _env(name: str, default: bool = True) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v == "":
        return default
    return v not in ("0", "false", "off", "no")


@functools.cache
def _ok() -> bool:
    try:
        from amd_kernels.indexer.mxfp4_indexer_common import available

        return bool(available())
    except Exception:  # noqa: BLE001 - any import problem means the stock path
        return False


@functools.cache
def _topk_covered(out_width: int) -> bool:
    from amd_kernels.indexer.mxfp4_indexer_common import mxfp4_prefill_topk_shapes

    return (out_width,) in mxfp4_prefill_topk_shapes()


def table_bucket_on() -> bool:
    return _env("VLLM_KIMI_AKR_PREFILL_TABLE_BUCKET")


def score_on() -> bool:
    return akr_config.on("indexer") and _env("VLLM_KIMI_AKR_INDEXER_SCORE") and _ok()


def topk_on() -> bool:
    return (
        akr_config.on("indexer")
        and _env("VLLM_KIMI_AKR_INDEXER_TOPK")
        and _ok()
        and _topk_covered(TOPK)
    )


def mean_window(chunk) -> float:
    """Host estimate of the mean key window of a row of the chunk (causal rows of
    requests of `total_seq_lens / num_reqs` keys with `queries / num_reqs` new tokens
    each)."""
    reqs = max(chunk.num_reqs, 1)
    queries = chunk.token_end - chunk.token_start
    return chunk.total_seq_lens / reqs - queries / (2 * reqs)


def topk_for_chunk(chunk, out_width: int) -> bool:
    return (
        topk_on()
        and _topk_covered(out_width)
        and mean_window(chunk) >= TOPK_MIN_MEAN_WINDOW
    )


@functools.cache
def identity_page_table(device, width: int) -> torch.Tensor:
    return torch.arange(width, device=device, dtype=torch.int32).unsqueeze(0)


def page_table_width(pages: int) -> int:
    # aiter compiles one scorer per block-table width and reads at most pages + 4
    # entries: bucket to powers of two >= 4096 (262144-token contexts; 8192 -> 524288,
    # 16384 -> 1M).
    return max(4096, 1 << (pages + 4 - 1).bit_length())


def topk_workspace_bytes(max_keys: int, row_chunk_bytes: int = 1024**3) -> int:
    """Scratch of the top-k select for the largest logits tile (<= 1 GiB, <= 8192 rows)
    of a chunk holding at most `max_keys` keys."""
    from amd_kernels.indexer.mxfp4_indexer_common import prefill_topk_workspace_bytes

    width = -(-max_keys // 64) * 64
    rows = max(1, min(8192, (row_chunk_bytes - 1) // (4 * width)))
    if rows >= 128:
        rows = rows // 128 * 128
    return prefill_topk_workspace_bytes(rows, width, topk=TOPK)
