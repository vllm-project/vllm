"""Compare a fixed prefill chunk with a step-time budget on one GPU.

The model has to fit in one 16GB card. Qwen2.5-3B in bf16 is about 6GB of
weights, which leaves room for an 8k KV cache. A 7B bf16 checkpoint does not.

From the repo root, on the A100 node:

    python -m prefill_budget.bench --model Qwen/Qwen2.5-3B-Instruct

One GPU is the right test. The scheduler runs once per engine; tensor
parallel does not change the chunk decision. Point CUDA_VISIBLE_DEVICES at
whichever card you want.

What you should see after the warmup chunks: with a fixed 1024-token chunk,
step time climbs as the prefix grows. With the budget (default 30ms), the
chunk shrinks and the step time stays nearer 30ms.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _prompt_ids(tokenizer, n_tokens: int) -> list[int]:
    piece = tokenizer.encode("The quick brown fox jumps over the lazy dog. ")
    if not piece:
        raise RuntimeError("tokenizer returned an empty piece")
    ids: list[int] = []
    while len(ids) < n_tokens:
        ids.extend(piece)
    return ids[:n_tokens]


def _run_engine(args: argparse.Namespace, mode: str, log_path: Path) -> None:
    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["VLLM_PREFILL_MODE"] = mode
    os.environ["VLLM_PREFILL_TIME_BUDGET_MS"] = str(args.budget_ms)
    os.environ["VLLM_PREFILL_FIXED_CHUNK"] = str(args.fixed_chunk)
    os.environ["VLLM_PREFILL_MIN_CHUNK"] = str(args.min_chunk)
    os.environ["VLLM_PREFILL_MAX_CHUNK"] = str(args.max_chunk)
    os.environ["VLLM_PREFILL_WARMUP_STEPS"] = str(args.warmup_steps if mode == "budget" else 0)
    os.environ["VLLM_PREFILL_BUDGET_LOG"] = str(log_path)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_chunk * 2,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=True,
        compilation_config=0,
        gpu_memory_utilization=args.gpu_memory_utilization,
        scheduler_cls="prefill_budget.scheduler.TimeBudgetScheduler",
    )
    prompt_ids = _prompt_ids(llm.get_tokenizer(), args.prompt_tokens)
    llm.generate(
        [{"prompt_token_ids": prompt_ids}],
        SamplingParams(temperature=0.0, max_tokens=1),
    )
    del llm


def _load(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _summary(name: str, rows: list[dict], c_min: int) -> str:
    chosen = [r for r in rows if r["usable"] and r["c"] >= c_min and r["phase"] != "warmup"]
    if not chosen:
        return f"{name}: no usable steps with c>={c_min}"
    elapsed = [r["elapsed_ms"] for r in chosen]
    qs = [r["q"] for r in chosen]
    cs = [r["c"] for r in chosen]
    elapsed_sorted = sorted(elapsed)
    p90 = elapsed_sorted[min(len(elapsed_sorted) - 1, int(0.9 * (len(elapsed_sorted) - 1)))]
    return (
        f"{name}: steps={len(chosen)} c={min(cs)}..{max(cs)} "
        f"q median={statistics.median(qs):.0f} "
        f"step_ms median={statistics.median(elapsed):.1f} p90={p90:.1f}"
    )


def _compare(fixed_rows: list[dict], budget_rows: list[dict], budget_ms: float) -> None:
    # Compare the prefix region the budget policy actually controlled.
    budget_body = [r for r in budget_rows if r["usable"] and r["phase"] == "budget"]
    if not budget_body:
        print("Budget mode never left warmup. Raise --prompt-tokens or lower --warmup-steps.")
        return
    c_min = min(r["c"] for r in budget_body)
    print(_summary("fixed ", fixed_rows, c_min))
    print(_summary("budget", budget_rows, c_min))
    body_ms = [r["elapsed_ms"] for r in budget_body]
    med = statistics.median(body_ms)
    print(f"budget median step {med:.1f} ms, target {budget_ms:.1f} ms")
    # Longer prefixes should get shorter chunks once the q*c term matters.
    if len(budget_body) >= 4:
        mid = len(budget_body) // 2
        early_q = statistics.median(r["q"] for r in budget_body[:mid])
        late_q = statistics.median(r["q"] for r in budget_body[mid:])
        print(f"budget chunk median early={early_q:.0f} late={late_q:.0f}")
        if late_q < early_q:
            print("chunk shrank as the prefix grew")
        else:
            print(
                "chunk did not shrink. The target is at or above the cost of "
                "--max-chunk on this GPU. Lower --budget-ms and run again."
            )


def _parent(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    fixed_log = log_dir / "fixed.jsonl"
    budget_log = log_dir / "budget.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "prefill_budget.bench",
        "--run",
        "fixed",
        "--model",
        args.model,
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--budget-ms",
        str(args.budget_ms),
        "--fixed-chunk",
        str(args.fixed_chunk),
        "--min-chunk",
        str(args.min_chunk),
        "--max-chunk",
        str(args.max_chunk),
        "--warmup-steps",
        str(args.warmup_steps),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--log-dir",
        str(log_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    print("running fixed chunk ...", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    cmd[cmd.index("fixed")] = "budget"
    print("running time budget ...", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    _compare(_load(fixed_log), _load(budget_log), args.budget_ms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=("both", "fixed", "budget"), default="both")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--prompt-tokens", type=int, default=6144)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--budget-ms", type=float, default=30.0)
    parser.add_argument("--fixed-chunk", type=int, default=1024)
    parser.add_argument("--min-chunk", type=int, default=128)
    parser.add_argument("--max-chunk", type=int, default=2048)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--log-dir", default="prefill_budget_logs")
    args = parser.parse_args()
    if args.prompt_tokens >= args.max_model_len:
        parser.error("--prompt-tokens must be smaller than --max-model-len")
    if args.run == "both":
        _parent(args)
        return
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    log_path = log_dir / f"{args.run}.jsonl"
    _run_engine(args, args.run, log_path.resolve())


if __name__ == "__main__":
    main()
