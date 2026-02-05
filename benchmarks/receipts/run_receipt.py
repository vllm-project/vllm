# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run an arbitrary benchmark command and write a receipt JSON.

This is intentionally generic: it wraps any command (including `vllm bench ...`)
and records:
- command + args
- wall time
- stdout/stderr logs saved alongside the receipt
- optional GPU telemetry (NVML or nvidia-smi)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .energy_sampling import EnergySampler, integrate_energy_j

MAX_EMBED_JSON_SIZE_BYTES = 2_000_000


def _best_effort_git_head(repo_root: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            return None
        s = (r.stdout or "").strip()
        return s if s else None
    except Exception:
        return None


def _repo_root_from_here() -> Path:
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        # Prefer .git when available (developer checkout).
        if (parent / ".git").is_dir():
            return parent
        # Fallback for environments where .git isn't present (e.g., sdist).
        if (parent / "pyproject.toml").is_file() and (parent / "vllm").is_dir():
            return parent
    raise RuntimeError("Could not find repository root from benchmarks/receipts")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _json_dump(path: Path, obj: dict[str, Any]) -> None:
    _write_text(path, json.dumps(obj, indent=2) + "\n")


def _summarize_sampler(samples: list[Any]) -> dict[str, Any]:
    util_vals = [
        s.gpu_util_pct for s in samples if getattr(s, "gpu_util_pct", None) is not None
    ]
    power_vals = [s.power_w for s in samples if getattr(s, "power_w", None) is not None]
    temp_vals = [s.temp_c for s in samples if getattr(s, "temp_c", None) is not None]

    def _avg(xs: list[float]) -> float | None:
        return (float(sum(xs)) / float(len(xs))) if xs else None

    return {
        "num_samples": int(len(samples)),
        "avg_gpu_util_pct": _avg([float(x) for x in util_vals]),
        "avg_power_w": _avg([float(x) for x in power_vals]),
        "avg_temp_c": _avg([float(x) for x in temp_vals]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", type=Path, required=True, help="Receipt JSON path to write"
    )
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU id for telemetry sampling (default: 0)",
    )
    ap.add_argument(
        "--sample-interval-s",
        type=float,
        default=0.1,
        help="Telemetry sampling interval (default: 0.1s)",
    )
    ap.add_argument(
        "--attach-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file produced by the benchmark to embed in the receipt "
            "(path recorded + sha256)."
        ),
    )
    ap.add_argument(
        "--no-telemetry", action="store_true", help="Disable GPU telemetry sampling"
    )

    ap.add_argument(
        "cmd", nargs=argparse.REMAINDER, help="Command to run (prefix with --)"
    )
    args = ap.parse_args()

    if not args.cmd or args.cmd[0] != "--" or len(args.cmd) < 2:
        print("error: command must be provided after --", file=sys.stderr)
        return 2

    cmd = args.cmd[1:]
    if not cmd or cmd[0] == "--":
        print("error: command must be provided after --", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    stdout_log = out.with_suffix(out.suffix + ".stdout.log")
    stderr_log = out.with_suffix(out.suffix + ".stderr.log")

    repo_root = _repo_root_from_here()
    git_head = _best_effort_git_head(repo_root)

    env = os.environ.copy()
    t0_wall = time.time()
    t0 = time.perf_counter()

    sampler: EnergySampler | None = None
    if not args.no_telemetry:
        sampler = EnergySampler(
            gpu_id=int(args.gpu_id), sample_interval_s=float(args.sample_interval_s)
        )

    with (
        stdout_log.open("w", encoding="utf-8") as out_f,
        stderr_log.open("w", encoding="utf-8") as err_f,
    ):
        if sampler is not None:
            sampler.start()
        try:
            try:
                p = subprocess.run(
                    cmd,
                    cwd=str(repo_root),
                    env=env,
                    stdout=out_f,
                    stderr=err_f,
                    text=True,
                    check=False,
                )
                rc = int(p.returncode)
            except FileNotFoundError as e:
                print(f"error: command not found: {cmd[0]} ({e})", file=sys.stderr)
                rc = 127
        finally:
            if sampler is not None:
                sampler.stop()

    dt_s = time.perf_counter() - t0
    t1_wall = time.time()

    receipt: dict[str, Any] = {
        "schema": "vllm.receipt.v1",
        "timestamp_start_unix_s": float(t0_wall),
        "timestamp_end_unix_s": float(t1_wall),
        "duration_s": float(dt_s),
        "command": {
            "argv": cmd,
            "argv_shell_escaped": " ".join(shlex.quote(c) for c in cmd),
        },
        "result": {
            "returncode": rc,
            "stdout_log": str(stdout_log.name),
            "stderr_log": str(stderr_log.name),
        },
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_head": git_head,
        },
    }

    if sampler is not None:
        energy_j = integrate_energy_j(sampler.samples, duration_s=float(dt_s))
        receipt["telemetry"] = {
            "backend": sampler.backend,
            "interval_s": float(sampler.sample_interval_s),
            "metadata": dict(sampler.metadata),
            "summary": _summarize_sampler(sampler.samples),
            "energy_joules": float(energy_j) if energy_j is not None else None,
        }
    else:
        receipt["telemetry"] = None

    if args.attach_json is not None:
        attach = Path(args.attach_json)
        attach_info: dict[str, Any] = {"path": str(attach)}
        try:
            raw = attach.read_bytes()
            import hashlib

            attach_info["sha256"] = hashlib.sha256(raw).hexdigest()
            # Embed only if reasonably small.
            if len(raw) <= MAX_EMBED_JSON_SIZE_BYTES:
                attach_info["json"] = json.loads(raw.decode("utf-8"))
            else:
                attach_info["json"] = None
                attach_info["note"] = "json not embedded (too large); see sha256 + path"
        except Exception as e:
            attach_info["error"] = str(e)
        receipt["attached_json"] = attach_info

    _json_dump(out, receipt)
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    raise SystemExit(main())
