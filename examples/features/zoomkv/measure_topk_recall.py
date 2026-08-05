#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure per-decode-step Top-K retrieval recall of the ZoomKV backend.

Enables the ``VLLM_ZOOMKV_RECALL_LOG`` probe, runs one long prompt, then
aggregates the JSONL records the workers wrote:

- recall@k: fraction of the exact Q.K Top-K under the group-mean retrieval
  query (per KV head, restricted to the retrieval zone) that ZoomKV retrieved,
- mass coverage: fraction of the zone's exact attention mass covered by the
  retrieved tokens (oracle = best achievable with k tokens),
- zone mass fraction: how much attention mass lives in the retrieval zone at
  all (the rest goes to the always-attended sink/local windows).

GPU-only mode only (the probe needs GPU-resident Keys).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from collections import defaultdict

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--threshold", type=int, default=2000)
    parser.add_argument("--final-topk", type=int, default=100)
    parser.add_argument(
        "--prompt-sentences",
        type=int,
        default=600,
        help="Number of distinct filler sentences in the long prompt.",
    )
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument(
        "--dataset-jsonl",
        default=None,
        help="Optional LongBench-style JSONL with context/input fields.",
    )
    parser.add_argument(
        "--target-prompt-tokens",
        type=int,
        default=None,
        help="Build a deterministic multi-document prompt of this token length.",
    )
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--output-json", default=None)
    # Retrieval knobs (defaults mirror ZoomKVRuntimeConfig).
    parser.add_argument("--quest-large-ratio", type=float, default=0.5)
    parser.add_argument("--quest-small-ratio", type=float, default=0.3)
    parser.add_argument("--dense-ratio", type=float, default=0.4)
    parser.add_argument("--dense-topk", type=int, default=8)
    parser.add_argument("--sparse-topk", type=int, default=4)
    return parser.parse_args()


def build_prompt(n_sentences: int) -> str:
    """Varied filler + a needle so retrieval is non-degenerate."""
    rng = random.Random(1234)
    subjects = [
        "the northern harbor",
        "an old observatory",
        "the mountain railway",
        "a coastal village",
        "the glass factory",
        "an ancient library",
        "the river delta",
        "a desert outpost",
        "the botanical garden",
        "an island lighthouse",
        "the city archive",
        "a forest cabin",
    ]
    verbs = [
        "recorded",
        "stored",
        "shipped",
        "produced",
        "measured",
        "collected",
        "restored",
        "monitored",
        "exported",
        "catalogued",
    ]
    objects = [
        "rare manuscripts",
        "copper instruments",
        "weather logs",
        "ceramic tiles",
        "salted fish",
        "silk banners",
        "iron tools",
        "glass lenses",
        "grain samples",
        "maps of the coastline",
    ]
    lines = []
    needle_at = n_sentences // 2
    for i in range(n_sentences):
        if i == needle_at:
            lines.append(
                "Important: the secret access code for the vault is 7-4-1-9-2."
            )
            continue
        s = rng.choice(subjects)
        v = rng.choice(verbs)
        o = rng.choice(objects)
        lines.append(f"Entry {i}: {s} {v} {o} in year {1800 + rng.randint(0, 200)}.")
    lines.append(
        "Question: what is the secret access code for the vault? "
        "Answer with the digits only."
    )
    return " ".join(lines)


def build_dataset_prompt(path: str, target_tokens: int, tokenizer) -> str:
    """Build a fixed-length prompt from real LongBench document contexts."""
    contexts: list[str] = []
    question = ""
    total_chars = 0
    # Collect enough real text before tokenizing once. Multiple records form a
    # realistic multi-document context when one sample is shorter than 64K.
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            context = str(rec.get("context", ""))
            if not context:
                continue
            contexts.append(context)
            total_chars += len(context)
            if not question:
                question = str(rec.get("input", ""))
            if total_chars >= target_tokens * 6:
                break
    if not contexts:
        raise RuntimeError(f"No context records found in {path}")

    suffix = f"\n\nQuestion: {question}\nAnswer:"
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    body_ids = tokenizer.encode(
        "\n\n--- Next document ---\n\n".join(contexts),
        add_special_tokens=False,
    )
    body_budget = target_tokens - len(suffix_ids)
    if body_budget <= 0 or len(body_ids) < body_budget:
        raise RuntimeError(
            f"Dataset supplied {len(body_ids)} body tokens, need {body_budget}"
        )
    return tokenizer.decode(body_ids[:body_budget]) + suffix


def aggregate(log_dir: str, prompt_tokens: int) -> dict:
    records = []
    for path in glob.glob(os.path.join(log_dir, "recall.*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    if not records:
        raise RuntimeError(
            f"No probe records in {log_dir}. Did the run enter sparse decode?"
        )

    by_step: dict[int, list[dict]] = defaultdict(list)
    by_layer: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        step = rec["seq_len"] - prompt_tokens
        by_step[step].append(rec)
        by_layer[rec["layer"]].extend(rec["recall"])

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    steps = []
    for step in sorted(by_step):
        recs = by_step[step]
        recalls = [r for rec in recs for r in rec["recall"]]
        attention = [
            r for rec in recs for r in rec.get("attention_recall", [])
        ]
        cov = [c for rec in recs for c in rec["mass_coverage"]]
        oracle = [c for rec in recs for c in rec["oracle_mass_coverage"]]
        zone = [z for rec in recs for z in rec["zone_mass_frac"]]
        layer_means = [_mean(rec["recall"]) for rec in recs]
        steps.append(
            {
                "step": step,
                "seq_len": recs[0]["seq_len"],
                "records": len(recs),
                "recall_mean": _mean(recalls),
                "attention_recall_mean": _mean(attention),
                "recall_min_layer": min(layer_means),
                "recall_max_layer": max(layer_means),
                "mass_coverage_mean": _mean(cov),
                "oracle_mass_coverage_mean": _mean(oracle),
                "zone_mass_frac_mean": _mean(zone),
            }
        )

    layers = {
        layer: _mean(vals)
        for layer, vals in sorted(by_layer.items(), key=lambda kv: _mean(kv[1]))
    }
    all_recalls = [r for rec in records for r in rec["recall"]]
    all_attention = [
        r for rec in records for r in rec.get("attention_recall", [])
    ]
    return {
        "num_records": len(records),
        "overall_recall_mean": _mean(all_recalls),
        "overall_attention_recall_mean": _mean(all_attention),
        "per_step": steps,
        "per_layer_recall_mean": layers,
    }


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir or f"/tmp/zoomkv_recall_{int(time.time())}"
    os.makedirs(log_dir, exist_ok=True)
    for old in glob.glob(os.path.join(log_dir, "recall.*.jsonl")):
        os.remove(old)
    # Must be set before engine workers are spawned.
    os.environ["VLLM_ZOOMKV_RECALL_LOG"] = log_dir

    from vllm import LLM, SamplingParams
    from vllm.config.attention import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        block_size=16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
        attention_config=AttentionConfig(
            backend=AttentionBackendEnum.ZOOMKV,
            zoomkv_sink_size=64,
            zoomkv_local_size=256,
            zoomkv_final_topk=args.final_topk,
            zoomkv_quest_chunk=16,
            zoomkv_quest_large_chunk=256,
            zoomkv_quest_large_ratio=args.quest_large_ratio,
            zoomkv_quest_small_ratio=args.quest_small_ratio,
            zoomkv_dense_ratio=args.dense_ratio,
            zoomkv_dense_topk=args.dense_topk,
            zoomkv_sparse_topk=args.sparse_topk,
            zoomkv_full_attention_threshold=args.threshold,
        ),
    )
    sampling = SamplingParams(
        max_tokens=args.output_tokens,
        temperature=0.0,
        ignore_eos=True,
    )
    if args.dataset_jsonl:
        if args.target_prompt_tokens is None:
            raise ValueError("--dataset-jsonl requires --target-prompt-tokens")
        prompt = build_dataset_prompt(
            args.dataset_jsonl,
            args.target_prompt_tokens,
            llm.get_tokenizer(),
        )
    else:
        prompt = build_prompt(args.prompt_sentences)
    output = llm.generate([prompt], sampling)[0]
    prompt_tokens = len(output.prompt_token_ids)
    if prompt_tokens < args.threshold:
        raise RuntimeError(
            f"Prompt has {prompt_tokens} tokens < threshold {args.threshold}; "
            "increase --prompt-sentences"
        )

    summary = aggregate(log_dir, prompt_tokens)
    summary["model"] = args.model
    summary["prompt_tokens"] = prompt_tokens
    summary["output_tokens"] = args.output_tokens
    summary["final_topk"] = args.final_topk
    summary["retrieval_config"] = {
        "quest_large_ratio": args.quest_large_ratio,
        "quest_small_ratio": args.quest_small_ratio,
        "dense_ratio": args.dense_ratio,
        "dense_topk": args.dense_topk,
        "sparse_topk": args.sparse_topk,
    }
    summary["dataset_jsonl"] = args.dataset_jsonl
    summary["generated_text"] = output.outputs[0].text
    summary["log_dir"] = log_dir

    print(f"\nprompt_tokens={prompt_tokens}  output_tokens={args.output_tokens}")
    print(f"generated: {output.outputs[0].text!r}\n")
    header = (
        f"{'step':>4} {'seq_len':>8} {'recs':>5} {'recall':>7} {'attn-rec':>8} "
        f"{'min-layer':>9} {'max-layer':>9} {'coverage':>8} "
        f"{'oracle':>7} {'zone-mass':>9}"
    )
    print(header)
    for s in summary["per_step"]:
        print(
            f"{s['step']:>4} {s['seq_len']:>8} {s['records']:>5} "
            f"{s['recall_mean']:>7.3f} {s['attention_recall_mean']:>8.3f} "
            f"{s['recall_min_layer']:>9.3f} "
            f"{s['recall_max_layer']:>9.3f} {s['mass_coverage_mean']:>8.3f} "
            f"{s['oracle_mass_coverage_mean']:>7.3f} "
            f"{s['zone_mass_frac_mean']:>9.3f}"
        )
    print(f"\noverall recall@{args.final_topk}: {summary['overall_recall_mean']:.3f}")
    print(
        f"auxiliary all-query-head attention recall: "
        f"{summary['overall_attention_recall_mean']:.3f}"
    )
    print("weakest layers by recall:")
    for layer, r in list(summary["per_layer_recall_mean"].items())[:5]:
        print(f"  {layer}: {r:.3f}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print(f"\nsummary written to {args.output_json}")


if __name__ == "__main__":
    main()
