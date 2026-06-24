# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline diagnostic probe for cumem sleep/wake allocator residual.

Useful for verifying fixes for issue #36651 and similar cumem /
caching-allocator leaks across wake/sleep cycles. Spins up a small vLLM
``LLM`` with ``enable_sleep_mode=True``, runs N wake/sleep cycles, and
after each cycle captures both:

  * ``torch.cuda.memory_stats()`` from inside the worker process via
    ``collective_rpc`` -- PyTorch's per-process view of the caching
    allocator.
  * ``nvidia-smi`` per-PID resident memory for the same worker -- the
    driver-level physical footprint, which includes pages held by the
    cumem allocator outside of PyTorch's bookkeeping.

Interpretation:

  * ``Δ reserved_bytes.all.current`` across slept snapshots ~=
    caching-allocator growth.
  * ``Δ allocated_bytes.all.current`` across slept snapshots ~= live
    tensors leaked across sleep.
  * ``Δ nvidia-smi`` across slept snapshots ~= total physical footprint
    growth (cumem + caching allocator + driver/NCCL state).

If ``reserved_bytes`` grows in lockstep with ``nvidia-smi``, the leak is
in the caching allocator. If ``reserved_bytes`` is flat but
``nvidia-smi`` grows, the leak is outside the caching allocator (cumem
or other CUDA subsystem state).

Per-cycle records are written incrementally to JSONL so a crash mid-run
does not lose data.

Requirements:
  * ``vllm`` installed (offline ``LLM`` API).
  * ``nvidia-smi`` on PATH (optional; if absent, those fields are -1).
  * ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` must be set in the
    environment so ``collective_rpc`` can ship a Python callable to the
    worker.

Example:
  $ VLLM_ALLOW_INSECURE_SERIALIZATION=1 python \\
        examples/features/pause_resume/cumem_probe_offline.py \\
        --model Qwen/Qwen3-Embedding-0.6B \\
        --cycles 20 \\
        --gpu-mem-util 0.4 \\
        --kv-cache-bytes 1073741824 \\
        --runner pooling \\
        --inference-between-cycles
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Probe vLLM sleep/wake allocator residual across cycles."
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="HF model id. Small models keep the probe runtime short.",
    )
    p.add_argument(
        "--cycles",
        type=int,
        default=20,
        help="Number of wake/sleep cycles to run.",
    )
    p.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.10,
        help=(
            "vLLM gpu_memory_utilization. Keep small if sharing the GPU "
            "with other workloads."
        ),
    )
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-num-seqs", type=int, default=2)
    p.add_argument(
        "--kv-cache-bytes",
        type=int,
        default=0,
        help=(
            "If >0, pass kv_cache_memory_bytes (bypasses "
            "gpu_memory_utilization KV-cache sizing)."
        ),
    )
    p.add_argument(
        "--runner",
        default="pooling",
        choices=["pooling", "generate"],
        help="vLLM runner type: 'pooling' for embed/rerank, 'generate' for LLM.",
    )
    p.add_argument(
        "--inference-between-cycles",
        action="store_true",
        help="Run a tiny inference between each wake and the next sleep.",
    )
    p.add_argument(
        "--out",
        default="/tmp/cumem-probe.jsonl",
        help="Path to JSONL output file.",
    )
    p.add_argument(
        "--sleep-secs",
        type=float,
        default=2.0,
        help="Pause between operations so the allocator can settle.",
    )
    return p.parse_args()


def _worker_memory_stats(self) -> dict:
    """Collect per-worker memory stats. Runs INSIDE each vLLM worker.

    ``self`` is the ``Worker`` instance bound when ``collective_rpc``
    ships the callable into the worker process.

    Returns:
        A dict of selected ``torch.cuda.memory_stats()`` keys plus
        ``_pid`` so the caller can correlate with ``nvidia-smi``.
    """
    import torch

    stats = torch.cuda.memory_stats()
    # Full dict is ~200 keys; pull just what is relevant for sleep/wake
    # residual analysis.
    keys_of_interest = [
        "reserved_bytes.all.current",
        "reserved_bytes.all.peak",
        "reserved_bytes.all.allocated",
        "reserved_bytes.all.freed",
        "allocated_bytes.all.current",
        "allocated_bytes.all.peak",
        "allocated_bytes.all.allocated",
        "allocated_bytes.all.freed",
        "active_bytes.all.current",
        "inactive_split_bytes.all.current",
        "num_alloc_retries",
        "num_ooms",
        "num_device_alloc",
        "num_device_free",
    ]
    out = {k: stats.get(k, 0) for k in keys_of_interest}
    out["_pid"] = os.getpid()
    return out


def nvidia_smi_for_pid(pid: int) -> int:
    """Return MiB used by ``pid`` across all GPUs, per ``nvidia-smi``.

    Args:
        pid: PID to query.

    Returns:
        Resident GPU memory in MiB, or -1 if ``nvidia-smi`` is missing
        or fails.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return -1
    total = 0
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(pid):
            try:
                total += int(parts[1])
            except ValueError:
                continue
    return total


def snapshot(llm, label: str, cycle: int) -> dict:
    """Capture per-worker memory_stats + nvidia-smi residual.

    Args:
        llm: The offline ``LLM`` instance.
        label: One of ``"fresh-awake"``, ``"slept"``, ``"awake"``.
        cycle: 0-based cycle index (0 = baseline, 1..N = wake/sleep cycles).

    Returns:
        A JSON-serializable dict ready to write to the JSONL output.
    """
    worker_stats = llm.collective_rpc(_worker_memory_stats)
    smi_per_worker = []
    for ws in worker_stats:
        pid = ws.get("_pid", -1)
        smi_per_worker.append({"pid": pid, "smi_mib": nvidia_smi_for_pid(pid)})
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "cycle": cycle,
        "label": label,
        "worker_stats": worker_stats,
        "nvidia_smi": smi_per_worker,
        # Driver-side PID (this process) for cross-correlation.
        "self_pid": os.getpid(),
        "self_smi_mib": nvidia_smi_for_pid(os.getpid()),
    }


def _print_cycle_line(label: str, cycle: int, rec: dict, call_label: str,
                      call_dur: float) -> None:
    ws = rec["worker_stats"][0]
    rmib = ws["reserved_bytes.all.current"] / (1024**2)
    amib = ws["allocated_bytes.all.current"] / (1024**2)
    smib = rec["nvidia_smi"][0]["smi_mib"]
    print(
        f"[probe] cycle={cycle} {label} "
        f"reserved={rmib:.0f}MiB allocated={amib:.0f}MiB smi={smib}MiB "
        f"({call_label}={call_dur:.2f}s)",
        flush=True,
    )


def _print_summary(records: list) -> None:
    print("\n[probe] === SUMMARY ===", flush=True)
    slept_recs = [r for r in records if r["label"] == "slept"]
    if len(slept_recs) < 2:
        return
    first = slept_recs[0]["worker_stats"][0]
    last = slept_recs[-1]["worker_stats"][0]
    first_smi = slept_recs[0]["nvidia_smi"][0]["smi_mib"]
    last_smi = slept_recs[-1]["nvidia_smi"][0]["smi_mib"]
    d_res = (
        last["reserved_bytes.all.current"] - first["reserved_bytes.all.current"]
    ) / (1024**2)
    d_all = (
        last["allocated_bytes.all.current"] - first["allocated_bytes.all.current"]
    ) / (1024**2)
    d_smi = last_smi - first_smi
    print(
        f"[probe] Across slept cycles {slept_recs[0]['cycle']} -> "
        f"{slept_recs[-1]['cycle']}:",
        flush=True,
    )
    print(f"[probe]   delta reserved_bytes : {d_res:+.1f} MiB", flush=True)
    print(f"[probe]   delta allocated_bytes: {d_all:+.1f} MiB", flush=True)
    print(f"[probe]   delta nvidia-smi     : {d_smi:+d} MiB", flush=True)
    print(
        f"[probe]   num_alloc_retries: {last['num_alloc_retries']} "
        f"(was {first['num_alloc_retries']})",
        flush=True,
    )
    print(
        f"[probe]   num_device_alloc : {last['num_device_alloc']} "
        f"(was {first['num_device_alloc']})",
        flush=True,
    )
    print(
        f"[probe]   num_device_free  : {last['num_device_free']} "
        f"(was {first['num_device_free']})",
        flush=True,
    )


def main() -> int:
    args = parse_args()

    # Import vLLM lazily so ``--help`` works without vllm installed.
    from vllm import LLM
    from vllm.sampling_params import SamplingParams

    print(
        f"[probe] model={args.model} cycles={args.cycles} "
        f"util={args.gpu_mem_util} runner={args.runner}",
        flush=True,
    )
    print(f"[probe] output -> {args.out}", flush=True)

    llm_kwargs = dict(
        model=args.model,
        enable_sleep_mode=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
    )
    if args.runner != "generate":
        llm_kwargs["runner"] = args.runner
    if args.kv_cache_bytes > 0:
        llm_kwargs["kv_cache_memory_bytes"] = args.kv_cache_bytes

    t0 = time.time()
    llm = LLM(**llm_kwargs)
    init_secs = time.time() - t0
    print(f"[probe] LLM initialized in {init_secs:.1f}s", flush=True)

    records = []

    # Baseline (cycle 0, fresh awake).
    rec = snapshot(llm, "fresh-awake", 0)
    records.append(rec)
    ws = rec["worker_stats"][0]
    print(
        f"[probe] cycle=0 fresh-awake "
        f"reserved={ws['reserved_bytes.all.current'] / (1024**2):.0f}MiB "
        f"allocated={ws['allocated_bytes.all.current'] / (1024**2):.0f}MiB "
        f"smi={rec['nvidia_smi'][0]['smi_mib']}MiB",
        flush=True,
    )

    for c in range(1, args.cycles + 1):
        # Sleep.
        time.sleep(args.sleep_secs)
        t_s = time.time()
        llm.sleep(level=1)
        sleep_dur = time.time() - t_s
        time.sleep(args.sleep_secs)
        rec = snapshot(llm, "slept", c)
        records.append(rec)
        _print_cycle_line("slept", c, rec, "sleep_call", sleep_dur)

        # Wake.
        time.sleep(args.sleep_secs)
        t_w = time.time()
        llm.wake_up()
        wake_dur = time.time() - t_w
        time.sleep(args.sleep_secs)
        rec = snapshot(llm, "awake", c)
        records.append(rec)
        _print_cycle_line("awake", c, rec, "wake_call", wake_dur)

        # Optional: exercise the model briefly to provoke transient
        # tensor allocation between wake and the next sleep.
        if args.inference_between_cycles:
            try:
                if args.runner == "generate":
                    sp = SamplingParams(max_tokens=8, temperature=0.0)
                    llm.generate(
                        ["hello world"], sampling_params=sp, use_tqdm=False
                    )
                else:  # pooling
                    llm.embed(
                        ["hello world", "the quick brown fox"], use_tqdm=False
                    )
            except Exception as e:
                import traceback

                tb = traceback.format_exc()
                print(
                    f"[probe] cycle={c} inference failed: "
                    f"{type(e).__name__}: {e!r}\n{tb}",
                    flush=True,
                )

        # Flush JSONL incrementally so a crash does not lose data.
        with open(args.out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    _print_summary(records)
    print(f"[probe] wrote {len(records)} records to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
