"""
Profile c_draft = draft_forward_time / target_forward_time.
Runs target-only and draft-only in separate subprocesses.
"""
import json
import os
import sys
import subprocess
import time

# ── hardware constants ────────────────────────────────────────────
TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"

N_WARMUP = 3
N_MEASURE = 30  # single-token generations per run
MAX_TOKENS = 1


def make_runner_code(model: str, label: str) -> str:
    return f"""
import os, json, time, gc, torch
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from vllm.sampling_params import SamplingParams
from vllm.entrypoints.llm import LLM

sp = SamplingParams(temperature=0.0, max_tokens=1)
llm = LLM(
    model={model!r},
    tensor_parallel_size=1,
    gpu_memory_utilization=0.80,
    max_model_len=512,
    enforce_eager=True,
    cpu_offload_gb=3,
)

# Warmup
_ = llm.generate(["warmup"] * 2, sp)

# Timed single-token generations
times = []
for i in range({N_MEASURE}):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = llm.generate(["profile a b c d e f g h i j"], sp)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    times.append(elapsed)

avg_ms = (sum(times) / len(times)) * 1000
result = {{
    "label": {label!r},
    "avg_s": sum(times) / len(times),
    "avg_ms": round(avg_ms, 3),
    "raw_seconds": [round(t, 6) for t in times],
}}
print(json.dumps(result))
del llm; gc.collect(); torch.cuda.empty_cache()
"""


def run_single(model: str, label: str) -> dict:
    code = make_runner_code(model, label)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=600,
    )
    # Find last JSON line in output
    for line in reversed(proc.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    print(f"--- stderr ({label}) ---", file=sys.stderr)
    print(proc.stderr, file=sys.stderr)
    raise RuntimeError(f"No JSON result for {label}")


def main():
    print("Profiling target model (7B AWQ)...", file=sys.stderr)
    target = run_single(TARGET_MODEL, "target")
    print(f"  Target avg: {target['avg_ms']:.2f} ms", file=sys.stderr)

    print("Profiling draft model (0.5B AWQ)...", file=sys.stderr)
    draft = run_single(DRAFT_MODEL, "draft")
    print(f"  Draft avg:  {draft['avg_ms']:.2f} ms", file=sys.stderr)

    c_draft = draft["avg_s"] / target["avg_s"]
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"c_draft = {c_draft:.4f}  (draft={draft['avg_ms']:.1f}ms, "
          f"target={target['avg_ms']:.1f}ms)", file=sys.stderr)
    print(f"c_draft (ms) = {draft['avg_ms']:.2f} / {target['avg_ms']:.2f} = {c_draft:.4f}",
          file=sys.stderr)

    # Output JSON for machine consumption
    print(json.dumps({
        "c_draft": round(c_draft, 4),
        "draft_ms": draft["avg_ms"],
        "target_ms": target["avg_ms"],
        "draft_raw": draft["raw_seconds"],
        "target_raw": target["raw_seconds"],
    }))


if __name__ == "__main__":
    main()
