# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-decode-step Top-K recall probe for ZoomKV retrieval quality.

Enable with ``VLLM_ZOOMKV_RECALL_LOG=/path/to/dir``.  Every worker process
appends one JSONL record per (decode step, layer, request) to
``<dir>/recall.<pid>.jsonl``.

Ground truth is the exact attention distribution computed from the full Key
cache: for each KV head, softmax(q.K/sqrt(d)) is evaluated per query head over
the whole sequence, summed inside each GQA group, and the Top-K tokens of the
retrieval zone under that importance are compared with the token set ZoomKV
retrieved.  This requires the full Key to be GPU-resident, so the probe only
supports GPU-only mode (not ``zoomkv_enable_offload``).

The probe synchronizes the GPU on every record and must never be enabled in
production runs.
"""

from __future__ import annotations

import json
import os
from typing import TextIO

import torch

_LOG_DIR = os.environ.get("VLLM_ZOOMKV_RECALL_LOG", "")


def enabled() -> bool:
    return bool(_LOG_DIR)


@torch.no_grad()
def compute_recall_record(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    block_size: int,
    seq_len: int,
    start_block: int,
    end_block: int,
    topk_logical: torch.Tensor,
    scale: float,
    retrieval_query: torch.Tensor | None = None,
) -> dict | None:
    """Compare retrieved Top-K tokens with the exact-attention Top-K.

    Args:
        query: [1, num_q_heads, head_dim] decode query (all query heads).
        key_cache: [num_blocks, block_size, num_kv_heads, head_dim].
        block_table_row: [max_blocks] physical block ids of this request.
        seq_len: current sequence length (includes the decoding token).
        start_block / end_block: retrieval zone in logical block coords,
            matching ``ZoomKVRetriever.retrieval_block_range``.
        topk_logical: [num_kv_heads, final_topk] absolute token indices
            retrieved by ZoomKV, -1 for padding.
        scale: softmax scale of the attention layer.
        retrieval_query: optional [1, num_kv_heads, head_dim] aggregated query
            actually used by the retriever (``raw_q``).  When given, the
            record also contains ``recall_vs_rq``: recall against the exact
            q.K Top-K under this query, i.e. pipeline-only loss (Quest
            pruning + KIVI quantization) with the query-aggregation mismatch
            factored out.

    Returns:
        Per-head recall / attention-mass statistics, or None when the
        retrieval zone is empty.
    """
    ret_start = start_block * block_size
    ret_end = min(end_block * block_size, seq_len)
    zone_len = ret_end - ret_start
    if zone_len <= 0:
        return None

    num_q_heads = query.shape[1]
    num_kv_heads = key_cache.shape[2]
    group = num_q_heads // num_kv_heads

    # Full-sequence Key in logical order: [num_kv_heads, seq_len, head_dim].
    n_blocks = (seq_len + block_size - 1) // block_size
    phys = block_table_row[:n_blocks].to(torch.int64)
    keys = (
        key_cache[phys]
        .reshape(n_blocks * block_size, num_kv_heads, -1)[:seq_len]
        .permute(1, 0, 2)
        .to(torch.float32)
    )

    # Exact per-query-head softmax over the full sequence, then aggregate
    # token importance inside each GQA group.  Softmax over the full sequence
    # (not just the zone) keeps per-head normalization faithful to the real
    # attention output.
    q = query[0].to(torch.float32).view(num_kv_heads, group, -1)
    scores = torch.einsum("kgd,ktd->kgt", q, keys) * scale
    probs = scores.softmax(dim=-1)
    importance = probs.sum(dim=1)  # [num_kv_heads, seq_len]

    zone = importance[:, ret_start:ret_end]
    zone_mass = zone.sum(dim=-1)
    total_mass = importance.sum(dim=-1)

    k_eff = min(int(topk_logical.shape[-1]), zone_len)
    true_val, true_pos = zone.topk(k_eff, dim=-1)
    true_abs = true_pos + ret_start  # [num_kv_heads, k_eff]

    # Pipeline-aligned ground truth: exact q.K Top-K under the retriever's own
    # aggregated query (softmax is monotone per head, so raw scores suffice).
    rq_abs = None
    if retrieval_query is not None:
        rq = retrieval_query.reshape(num_kv_heads, -1).to(torch.float32)
        rq_scores = torch.einsum("kd,ktd->kt", rq, keys)
        rq_zone = rq_scores[:, ret_start:ret_end]
        rq_abs = rq_zone.topk(k_eff, dim=-1).indices + ret_start

    recall: list[float] = []
    recall_vs_rq: list[float] = []
    mass_cov: list[float] = []
    oracle_mass_cov: list[float] = []
    zone_mass_frac: list[float] = []
    for h in range(num_kv_heads):
        retrieved = topk_logical[h]
        retrieved = retrieved[retrieved >= 0]
        hits = int(torch.isin(true_abs[h], retrieved).sum().item())
        recall.append(hits / k_eff)
        if rq_abs is not None:
            rq_hits = int(torch.isin(rq_abs[h], retrieved).sum().item())
            recall_vs_rq.append(rq_hits / k_eff)
        zm = float(zone_mass[h].item())
        if retrieved.numel() and zm > 0:
            mass_cov.append(float(importance[h, retrieved].sum().item()) / zm)
        else:
            mass_cov.append(0.0)
        oracle_mass_cov.append(float(true_val[h].sum().item()) / zm if zm > 0 else 0.0)
        tm = float(total_mass[h].item())
        zone_mass_frac.append(zm / tm if tm > 0 else 0.0)

    rec = {
        "seq_len": int(seq_len),
        "zone_tokens": int(zone_len),
        "k": int(k_eff),
        "recall": recall,
        "recall_mean": sum(recall) / len(recall),
        # Fraction of the zone's exact attention mass covered by the
        # retrieved set, and by the oracle Top-K (upper bound).
        "mass_coverage": mass_cov,
        "oracle_mass_coverage": oracle_mass_cov,
        # Fraction of total attention mass that lives in the retrieval zone
        # (the rest goes to sink/local, which are always attended).
        "zone_mass_frac": zone_mass_frac,
    }
    if recall_vs_rq:
        rec["recall_vs_rq"] = recall_vs_rq
        rec["recall_vs_rq_mean"] = sum(recall_vs_rq) / len(recall_vs_rq)
    return rec


class _RecallProbe:
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self._fh: TextIO | None = None

    def _file(self):
        if self._fh is None:
            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(self.log_dir, f"recall.{os.getpid()}.jsonl")
            self._fh = open(path, "a", encoding="utf-8")  # noqa: SIM115
        return self._fh

    def record(
        self,
        *,
        layer_name: str,
        req_idx: int,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        block_table_row: torch.Tensor,
        block_size: int,
        seq_len: int,
        start_block: int,
        end_block: int,
        topk_logical: torch.Tensor,
        scale: float,
        retrieval_query: torch.Tensor | None = None,
    ) -> None:
        rec = compute_recall_record(
            query,
            key_cache,
            block_table_row,
            block_size,
            seq_len,
            start_block,
            end_block,
            topk_logical,
            scale,
            retrieval_query=retrieval_query,
        )
        if rec is None:
            return
        rec["layer"] = layer_name
        rec["req"] = int(req_idx)
        fh = self._file()
        fh.write(json.dumps(rec) + "\n")
        fh.flush()


_probe: _RecallProbe | None = None


def get_probe() -> _RecallProbe | None:
    global _probe
    if not _LOG_DIR:
        return None
    if _probe is None:
        _probe = _RecallProbe(_LOG_DIR)
    return _probe
