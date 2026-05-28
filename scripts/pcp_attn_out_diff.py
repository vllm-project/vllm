#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diff prefill attention output between PCP=1 baseline and PCP=N.

Reads dumps written by ``forward_mha`` in mla_attention.py (gated by
``VLLM_PCP_ATTN_DUMP_DIR``). For each layer, maps each PCP=N rank's
rank-local attention output rows to their global Q positions via the
DualChunkSwap pattern, then compares against PCP=1's output at the
same global position.

Usage::

    VLLM_PCP_ATTN_DUMP_DIR=/tmp/attn_p1 ./run_pcp1.py
    VLLM_PCP_ATTN_DUMP_DIR=/tmp/attn_p4 VLLM_PCP_QKV_SELECT_BACKEND=torch ./run_pcp4.py
    python scripts/pcp_attn_out_diff.py /tmp/attn_p1 /tmp/attn_p4
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import torch


def _per_rank_index_to_global_pos(
    local_idx: int,
    pcp_rank: int,
    pcp_world_size: int,
    chunk_per_req: list[int],
    req_starts_padded: list[int],
) -> int:
    """Given a row index in a rank's attention-output tensor (which is in
    rank-local Q order: [head_req0, tail_req0, head_req1, tail_req1, ...]),
    return the global position in the padded layout it corresponds to.

    Layout per request:
        local_idx = 2*chunk_prev + j  where j in [0, 2*chunk_r):
          j in [0, chunk_r)            -> head: global = r_start + pcp_rank*chunk_r + j
          j in [chunk_r, 2*chunk_r)    -> tail: global = r_start + (2*W - pcp_rank - 1)*chunk_r + (j - chunk_r)
    """
    cum = 0
    for r, chunk_r in enumerate(chunk_per_req):
        if local_idx < cum + 2 * chunk_r:
            j = local_idx - cum
            r_start = req_starts_padded[r]
            if j < chunk_r:
                return r_start + pcp_rank * chunk_r + j
            else:
                return (
                    r_start
                    + (2 * pcp_world_size - pcp_rank - 1) * chunk_r
                    + (j - chunk_r)
                )
        cum += 2 * chunk_r
    raise IndexError(f"local_idx {local_idx} out of range")


def _load_layer(dump_dir: str, rank: int, layer: int) -> dict | None:
    path = os.path.join(dump_dir, f"rank{rank}_layer{layer:02d}.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir", help="VLLM_PCP_ATTN_DUMP_DIR for PCP=1")
    parser.add_argument("other_dir", help="VLLM_PCP_ATTN_DUMP_DIR for PCP=N")
    parser.add_argument("--max-layers", type=int, default=27)
    parser.add_argument("--show-positions", type=int, default=8)
    args = parser.parse_args()

    p1_layer0 = _load_layer(args.baseline_dir, 0, 0)
    if p1_layer0 is None:
        print(f"ERROR: no baseline dump at {args.baseline_dir}/rank0_layer00.pt")
        return 1
    p1_seq_len = p1_layer0["attn_output"].shape[0] - p1_layer0["num_mqa_tokens"]
    print(f"PCP=1 seq_len (prefill new tokens): {p1_seq_len}")

    # Probe pcp_world_size from other_dir's rank0_layer00.
    p4_layer0 = _load_layer(args.other_dir, 0, 0)
    if p4_layer0 is None:
        print(f"ERROR: no other dump at {args.other_dir}/rank0_layer00.pt")
        return 1
    pcp_ws = p4_layer0["pcp_world_size"]
    print(f"PCP={pcp_ws}")

    # Compute chunk_per_req and req_starts_padded from cu_seqlens.
    # For PCP=N, query_start_loc is LOCAL: each entry is sum of pcp_tokens[r].
    # We need padded per-req lengths to derive req_starts_padded. Since
    # padded = pcp_tokens * pcp_world_size, we can compute it from the
    # local cu directly.
    local_cu = p4_layer0["query_start_loc"].tolist()
    pcp_tokens_per_req = [local_cu[i + 1] - local_cu[i] for i in range(len(local_cu) - 1)]
    # In _run_prefill_new_tokens_pcp, the kernel uses cu_seqlens_q = local_cu // 2
    # so chunk_r = pcp_tokens_per_req[r] // 2.
    chunk_per_req = [n // 2 for n in pcp_tokens_per_req]
    padded_per_req = [n * pcp_ws for n in pcp_tokens_per_req]
    cum = 0
    req_starts_padded = []
    for p in padded_per_req:
        req_starts_padded.append(cum)
        cum += p
    print(f"chunk_per_req: {chunk_per_req}, padded_per_req: {padded_per_req}")

    # For each layer that exists in both dumps, compare.
    summary = []
    for layer in range(args.max_layers):
        p1 = _load_layer(args.baseline_dir, 0, layer)
        if p1 is None:
            continue
        p1_out = p1["attn_output"][p1["num_mqa_tokens"] :].flatten(1)  # (seq_len, H*D)

        # Collect each PCP=N rank's output at each global position.
        global_to_pcpN_out: dict[int, torch.Tensor] = {}
        for rank in range(pcp_ws):
            p4 = _load_layer(args.other_dir, rank, layer)
            if p4 is None:
                continue
            out = p4["attn_output"][p4["num_mqa_tokens"] :].flatten(1)  # (local_total, H*D)
            for i in range(out.shape[0]):
                g = _per_rank_index_to_global_pos(
                    i, rank, pcp_ws, chunk_per_req, req_starts_padded
                )
                if g < p1_seq_len:
                    global_to_pcpN_out[g] = out[i]

        # Compute per-position diff
        diffs: list[tuple[int, float]] = []
        for g in sorted(global_to_pcpN_out):
            d = (p1_out[g] - global_to_pcpN_out[g]).abs().max().item()
            diffs.append((g, d))
        if not diffs:
            continue
        max_diff = max(d for _, d in diffs)
        bad = [(g, d) for g, d in diffs if d > 1e-2]
        summary.append((layer, max_diff, len(bad), len(diffs)))
        if layer < 3:  # show details for first few layers
            print(f"layer {layer:2d}: max diff = {max_diff:.4g}, "
                  f"{len(bad)}/{len(diffs)} positions diverge")
            for g, d in diffs[: args.show_positions]:
                marker = "  DIFF" if d > 1e-2 else ""
                print(f"  pos {g:3d}: diff = {d:.4g}{marker}")
            if len(bad) > 0:
                print("  diverging positions (first 16):")
                for g, d in bad[:16]:
                    print(f"    pos {g:3d}: diff = {d:.4g}")
        else:
            mark = "  <<< DIVERGES" if max_diff > 1e-2 else ""
            print(f"layer {layer:2d}: max diff = {max_diff:.4g}, "
                  f"{len(bad)}/{len(diffs)} positions diverge{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
