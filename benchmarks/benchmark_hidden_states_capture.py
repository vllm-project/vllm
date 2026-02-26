#!/usr/bin/env python3
"""Benchmark hidden states capture: baseline vs online connector.

Runs each config sequentially in a subprocess for clean GPU isolation.
  1. Baseline  — no capture, no drafter, pure serving throughput
  2. Online    — OnlineHiddenStatesConnector (async GPU→CPU + disk I/O)
  3. Sync      — ExampleHiddenStatesConnector (blocking, opt-in)

Usage (from /tmp on H200):
    python /home/ubuntu/vllm/benchmarks/bench_connectors.py \
        --model Qwen/Qwen3-8B --num-prompts 100 --decode-tokens 256

    # Include sync for 3-way comparison:
    python /home/ubuntu/vllm/benchmarks/bench_connectors.py \
        --include-sync --decode-tokens 256

    # Run only one config:
    python /home/ubuntu/vllm/benchmarks/bench_connectors.py \
        --only online --decode-tokens 256

    # Use TP (disables parallel GPU execution):
    python /home/ubuntu/vllm/benchmarks/bench_connectors.py \
        --tensor-parallel-size 8 --model Qwen/Qwen3-32B
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import torch


# ── prompt loading ──────────────────────────────────────────────

def load_prompts(data_files: list[str], num_prompts: int,
                 max_chars: int = 0) -> list[str]:
    prompts: list[str] = []
    for path in data_files:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                if len(prompts) >= num_prompts:
                    break
                data = json.loads(line.strip())
                text = data.get("prompt") or data.get("text", "")
                if not text:
                    for turn in data.get("conversations", []):
                        role = turn.get("from") or turn.get("role", "")
                        if role in ("human", "user"):
                            text = (turn.get("value")
                                    or turn.get("content", ""))
                            break
                if text:
                    if max_chars > 0 and len(text) > max_chars:
                        continue
                    prompts.append(text)
        if len(prompts) >= num_prompts:
            break
    if not prompts:
        prompts = [f"Explain topic {i} in detail." for i in range(num_prompts)]
    return prompts[:num_prompts]


# ── helpers ─────────────────────────────────────────────────────

def count_output(output_dir: str) -> dict:
    files = glob.glob(os.path.join(output_dir, "**", "*.safetensors*"),
                       recursive=True)
    total_bytes = sum(os.path.getsize(f) for f in files)
    return {"num_files": len(files), "total_mb": total_bytes / 1024 / 1024}


def cleanup_gpu():
    torch.cuda.empty_cache()
    import gc; gc.collect()
    time.sleep(2)


# ── single-config runner (called as subprocess or directly) ─────

def run_single(cfg: str, args, prompts: list[str],
               warmup_prompts: list[str]) -> dict:
    """Run one benchmark config and return results dict."""
    from vllm import LLM, SamplingParams

    mt = args.decode_tokens
    layer_ids = [int(x) for x in args.layer_ids.split(",")]
    tag = cfg
    output_dir = f"/tmp/bench_{cfg}"

    # Build LLM kwargs
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    if cfg in ("online", "sync"):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        connector = ("OnlineHiddenStatesConnector" if cfg == "online"
                     else "ExampleHiddenStatesConnector")
        extra = {"shared_storage_path": output_dir}

        llm_kwargs["speculative_config"] = {
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": layer_ids,
                }
            },
        }
        llm_kwargs["kv_transfer_config"] = {
            "kv_connector": connector,
            "kv_role": "kv_producer",
            "kv_connector_extra_config": extra,
        }

    print(f"  [{tag}] Loading model...", flush=True)
    t_load = time.perf_counter()
    llm = LLM(**llm_kwargs)
    print(f"  [{tag}] Model loaded in {time.perf_counter()-t_load:.1f}s",
          flush=True)

    sp = SamplingParams(max_tokens=mt)

    if warmup_prompts:
        print(f"  [{tag}] Warmup ({len(warmup_prompts)} prompts)...",
              flush=True)
        _ = llm.generate(warmup_prompts, sp)
        if cfg in ("online", "sync"):
            time.sleep(1)
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)

    print(f"  [{tag}] Generating {len(prompts)} prompts, "
          f"max_tokens={mt}...", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_gen = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  [{tag}] Done: {elapsed:.2f}s, {total_gen} tokens, "
          f"{total_gen/elapsed:.1f} tok/s", flush=True)

    file_stats = {"num_files": 0, "total_mb": 0.0}
    if cfg in ("online", "sync"):
        time.sleep(3)
        file_stats = count_output(output_dir)
        print(f"  [{tag}] Output: {file_stats['num_files']} files, "
              f"{file_stats['total_mb']:.1f} MB", flush=True)

    print(f"  [{tag}] Shutting down...", flush=True)
    llm.llm_engine.engine_core.shutdown()
    del llm
    cleanup_gpu()

    return {
        "num_prompts": len(prompts),
        "max_tokens": mt,
        "elapsed_s": elapsed,
        "prompts_per_s": len(prompts) / elapsed,
        "avg_latency_ms": elapsed / len(prompts) * 1000,
        "total_gen_tokens": total_gen,
        "gen_tok_per_s": total_gen / elapsed,
        **file_stats,
    }


# ── subprocess entry point ──────────────────────────────────────

def _subprocess_main():
    """Entry point when this script is invoked as a subprocess for one config."""
    import pickle
    cfg = os.environ["BENCH_CONFIG"]
    args_path = os.environ["BENCH_ARGS"]
    prompts_path = os.environ["BENCH_PROMPTS"]
    result_path = os.environ["BENCH_RESULT"]

    with open(args_path, "rb") as f:
        args = pickle.load(f)
    with open(prompts_path, "rb") as f:
        prompts_data = pickle.load(f)

    result = run_single(cfg, args, prompts_data["bench"], prompts_data["warmup"])

    with open(result_path, "w") as f:
        json.dump(result, f)


# ── sequential launcher with subprocess isolation ───────────────

def launch_configs(configs: list[str], args, bench_prompts, warmup_prompts):
    """Run each config sequentially in its own subprocess for clean GPU isolation."""
    import pickle

    tmpdir = tempfile.mkdtemp(prefix="bench_")
    args_path = os.path.join(tmpdir, "args.pkl")
    prompts_path = os.path.join(tmpdir, "prompts.pkl")
    with open(args_path, "wb") as f:
        pickle.dump(args, f)
    with open(prompts_path, "wb") as f:
        pickle.dump({"bench": bench_prompts, "warmup": warmup_prompts}, f)

    results = {}
    for cfg in configs:
        result_path = os.path.join(tmpdir, f"result_{cfg}.json")
        log_path = os.path.join(tmpdir, f"log_{cfg}.txt")

        env = os.environ.copy()
        env["BENCH_CONFIG"] = cfg
        env["BENCH_ARGS"] = args_path
        env["BENCH_PROMPTS"] = prompts_path
        env["BENCH_RESULT"] = result_path

        print("\n" + "=" * 60)
        print(f"  Running [{cfg}]...")
        print("=" * 60, flush=True)

        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                [sys.executable, __file__, "--_subprocess"],
                env=env, stdout=log_f, stderr=subprocess.STDOUT,
            )
            proc.wait()

        with open(log_path) as f:
            print(f.read(), flush=True)

        if proc.returncode != 0:
            print(f"  [{cfg}] FAILED (exit code {proc.returncode})",
                  flush=True)
            continue

        try:
            with open(result_path) as f:
                results[cfg] = json.load(f)
        except Exception as e:
            print(f"  [{cfg}] Failed to read results: {e}", flush=True)

    shutil.rmtree(tmpdir, ignore_errors=True)
    return results


# ── output ──────────────────────────────────────────────────────

def print_table(results: dict[str, dict]):
    cols = list(results.keys())
    rows = [
        ("max_tokens",       "max_tokens",       "{:>10}"),
        ("elapsed (s)",      "elapsed_s",         "{:>10.2f}"),
        ("prompts/s",        "prompts_per_s",     "{:>10.1f}"),
        ("avg latency (ms)", "avg_latency_ms",    "{:>10.1f}"),
        ("gen tokens",       "total_gen_tokens",  "{:>10d}"),
        ("gen tok/s",        "gen_tok_per_s",     "{:>10.1f}"),
        ("output files",     "num_files",         "{:>10d}"),
        ("output (MB)",      "total_mb",          "{:>10.1f}"),
    ]

    label_w = max(len(r[0]) for r in rows) + 2
    col_w = max(max(len(c) for c in cols), 10) + 2
    header = " " * label_w + "".join(c.rjust(col_w) for c in cols)
    sep = "-" * len(header)

    print("\n" + sep)
    print("BENCHMARK RESULTS".center(len(header)))
    print(sep)
    print(header)
    print(sep)

    for label, key, fmt in rows:
        line = label.ljust(label_w)
        for col in cols:
            val = results[col].get(key, 0)
            line += fmt.format(val).rjust(col_w)
        print(line)

    print(sep)

    if "baseline" in results and results["baseline"]["elapsed_s"] > 0:
        base_t = results["baseline"]["elapsed_s"]
        print()
        for col in cols:
            if col == "baseline":
                continue
            overhead = results[col]["elapsed_s"] - base_t
            pct = overhead / base_t * 100
            print(f"  {col} overhead vs baseline: "
                  f"{overhead:+.2f}s ({pct:+.1f}%)")
    print()


# ── main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark hidden states capture: baseline vs online")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--num-warmup", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--layer-ids", default="1,2,3,4")
    parser.add_argument("--data-files", nargs="+", default=[
        "/home/ubuntu/downloads/test_mix_dataset_5k.jsonl",
        "/home/ubuntu/downloads/test_ultrachat_5k.jsonl",
    ])
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-online", action="store_true")
    parser.add_argument("--include-sync", action="store_true",
                        help="Include sync ExampleHiddenStatesConnector")
    parser.add_argument("--only", choices=["baseline", "online", "sync"])
    parser.add_argument("--_subprocess", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Subprocess mode: run single config and exit
    if args._subprocess:
        _subprocess_main()
        return

    total_needed = args.num_prompts + args.num_warmup
    all_prompts = load_prompts(args.data_files, total_needed, args.max_chars)
    if len(all_prompts) < total_needed:
        print(f"WARNING: only {len(all_prompts)} prompts, need {total_needed}")
    bench_prompts = all_prompts[:args.num_prompts]
    warmup_prompts = all_prompts[args.num_prompts:
                                 args.num_prompts + args.num_warmup]

    print(f"Prompts: {len(bench_prompts)} bench + {len(warmup_prompts)} warmup")
    print(f"Model: {args.model}  TP: {args.tensor_parallel_size}  "
          f"max_model_len: {args.max_model_len}")
    print(f"Decode tokens: {args.decode_tokens}")
    print()

    if args.only:
        configs = [args.only]
    else:
        configs = []
        if not args.skip_baseline:
            configs.append("baseline")
        if not args.skip_online:
            configs.append("online")
        if args.include_sync:
            configs.append("sync")

    bench_t0 = time.perf_counter()
    results = launch_configs(configs, args, bench_prompts, warmup_prompts)
    total_elapsed = time.perf_counter() - bench_t0

    if results:
        print_table(results)
        print(f"Total benchmark time: {total_elapsed:.0f}s")

    raw_path = "/tmp/bench_connectors_results.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Raw results saved to {raw_path}")


if __name__ == "__main__":
    main()
