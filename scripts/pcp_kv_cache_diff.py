#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic: dump K/V cache contents from PCP=1 vs PCP=4 prefill and diff.

Hooks ``MLACommonImpl.do_kv_cache_update`` and records, per layer, the
(k_c_normed, k_pe, slot_mapping) tuples each rank's prefill writes into
its cache. Also dumps the resulting cache contents at the prompt's slots.

Run once with --pcp 1 to produce baseline.pt, then with --pcp 4 to
produce pcp4_rank{r}.pt. Then `python -m vllm... compare` (or just run
this script with --compare baseline.pt pcp4_rank0.pt) to diff the
recorded writes per layer.

Usage::

    .venv/bin/python scripts/pcp_kv_cache_diff.py --pcp 1 --out /tmp/pcp1.pt
    VLLM_PCP_QKV_SELECT_BACKEND=torch \\
      .venv/bin/python scripts/pcp_kv_cache_diff.py --pcp 4 --out /tmp/pcp4.pt
    .venv/bin/python scripts/pcp_kv_cache_diff.py --compare /tmp/pcp1.pt /tmp/pcp4.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch


def _add_repo_to_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    if repo not in sys.path:
        sys.path.insert(0, repo)


# The prompt we run for both configurations. Short enough for fast iteration
# but long enough that PCP=4 has to actually split across ranks.
PROMPT = (
    "Question: Sarah had 15 apples. She gave 3 to her brother and 4 to her sister. "
    "How many apples does Sarah have left?\nAnswer:"
)


def install_kv_dump_hook(records: dict) -> None:
    """Monkeypatch ``MLACommonImpl.do_kv_cache_update`` to record its inputs.

    Records {layer_name: list of (k_c_normed.cpu(), k_pe.cpu(),
    slot_mapping.cpu())}. We only record the FIRST call per layer (i.e.,
    the prefill iteration) — subsequent decode writes are skipped to keep
    the dump small.
    """
    from vllm.model_executor.layers.attention import mla_attention as ma

    _orig = ma.MLACommonImpl.do_kv_cache_update
    recorded_layers: set[str] = set()

    def hooked(
        self,
        kv_c_normed,
        k_pe,
        kv_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
    ):
        # Identify the layer via the calling layer name. The layer name
        # isn't on the impl directly, but the kv_cache buffer's id is
        # unique per layer — use that as a stable key.
        layer_key = id(kv_cache)
        if layer_key not in recorded_layers:
            recorded_layers.add(layer_key)
            records.setdefault(layer_key, []).append(
                {
                    "k_c_normed": kv_c_normed.detach().cpu().to(torch.float32),
                    "k_pe": k_pe.detach().cpu().to(torch.float32),
                    "slot_mapping": slot_mapping.detach().cpu().to(torch.int64),
                    "kv_cache_shape": tuple(kv_cache.shape),
                    "kv_cache_dtype": kv_cache_dtype,
                }
            )
        return _orig(
            self, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale
        )

    ma.MLACommonImpl.do_kv_cache_update = hooked
    print(f"[hook] installed kv dump hook at rank={os.environ.get('RANK','?')}")


def run_dump(args: argparse.Namespace) -> int:
    _add_repo_to_path()
    from vllm import LLM, SamplingParams

    records: dict = {}
    install_kv_dump_hook(records)

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=1,
        prefill_context_parallel_size=args.pcp,
        decode_context_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=4,
        dtype="bfloat16",
        enforce_eager=True,
        distributed_executor_backend="mp",
        trust_remote_code=True,
        seed=0,
    )
    llm = LLM(**llm_kwargs)
    # max_tokens=1 — just prefill + 1 decode so we capture the prefill writes.
    out = llm.generate(
        [PROMPT], SamplingParams(max_tokens=1, temperature=0)
    )
    print(f"[run] generated: {out[0].outputs[0].text!r}")

    # Serialize. Records are keyed by id(kv_cache) which differs between
    # processes; sort layers by first-write order so they line up
    # between runs (assumes layer ordering is deterministic, which it
    # is since model.forward calls layers sequentially).
    ordered = [records[k] for k in records]
    torch.save({"layers": ordered, "pcp": args.pcp, "prompt": PROMPT}, args.out)
    print(f"[run] wrote {len(ordered)} layers to {args.out}")
    return 0


def run_compare(args: argparse.Namespace) -> int:
    base = torch.load(args.compare[0], map_location="cpu", weights_only=False)
    other = torch.load(args.compare[1], map_location="cpu", weights_only=False)
    print(f"baseline: pcp={base['pcp']} layers={len(base['layers'])}")
    print(f"other:    pcp={other['pcp']} layers={len(other['layers'])}")

    n_layers = min(len(base["layers"]), len(other["layers"]))
    for li in range(n_layers):
        b = base["layers"][li][0]
        o = other["layers"][li][0]
        # Each record is the prefill call. Compare slot_mapping coverage
        # and the K/V values at matching slots.
        b_slot = b["slot_mapping"]
        o_slot = o["slot_mapping"]

        # Map slot -> K/V row in each record.
        # baseline (PCP=1) has slot_mapping length = N (real tokens).
        # PCP=4 has slot_mapping length = padded_total (4*local_total)
        # with -1 at padding positions.
        b_real = b_slot >= 0
        o_real = o_slot >= 0

        b_slots = set(b_slot[b_real].tolist())
        o_slots = set(o_slot[o_real].tolist())
        missing_in_o = b_slots - o_slots
        extra_in_o = o_slots - b_slots
        common = b_slots & o_slots

        # Build slot -> k_c value
        def to_dict(rec):
            slot = rec["slot_mapping"]
            kc = rec["k_c_normed"]
            kp = rec["k_pe"]
            d_kc = {}
            d_kp = {}
            for i in range(slot.shape[0]):
                s = int(slot[i].item())
                if s < 0:
                    continue
                d_kc[s] = kc[i]
                d_kp[s] = kp[i]
            return d_kc, d_kp

        b_kc, b_kp = to_dict(b)
        o_kc, o_kp = to_dict(o)

        max_diff_kc = 0.0
        max_diff_kp = 0.0
        diff_slots = []
        for s in sorted(common):
            d_kc = (b_kc[s] - o_kc[s]).abs().max().item()
            d_kp = (b_kp[s] - o_kp[s]).abs().max().item()
            if d_kc > 1e-2 or d_kp > 1e-2:
                diff_slots.append((s, d_kc, d_kp))
            max_diff_kc = max(max_diff_kc, d_kc)
            max_diff_kp = max(max_diff_kp, d_kp)

        marker = ""
        if max_diff_kc > 0.5 or max_diff_kp > 0.5:
            marker = "  <<< DIVERGES"
        print(
            f"layer {li:2d}: common_slots={len(common)} "
            f"missing_in_other={len(missing_in_o)} extra_in_other={len(extra_in_o)} "
            f"max_diff_kc={max_diff_kc:.4g} max_diff_kp={max_diff_kp:.4g}{marker}"
        )
        if diff_slots and li < 3:
            print(f"          first 5 diverging slots: {diff_slots[:5]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    parser.add_argument("--pcp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--out", default=None, help="Output .pt path for --pcp run")
    parser.add_argument(
        "--compare",
        nargs=2,
        default=None,
        metavar=("BASELINE", "OTHER"),
        help="Two .pt files to diff",
    )
    args = parser.parse_args()
    if args.compare:
        return run_compare(args)
    if not args.out:
        parser.error("--out is required when not using --compare")
    return run_dump(args)


if __name__ == "__main__":
    sys.exit(main())
