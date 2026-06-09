# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep tuning configs for the ViT TRITON_ATTN kernel on gfx1151 for the
Gemma3-4B SigLIP shape (B=1, S=4096, H=16, D=72, bf16, non-causal).

Calls bench.py for each config. The bench uses triton.testing.do_bench,
which clears the L2 cache before every measurement iteration -- so results
approximate the kernel's behavior inside a real transformer block (where
surrounding qkv-proj / MLP / norm work evicts q/k/v between layer calls).

Phases:
  axis    - vary one parameter at a time around baseline
  refine  - cross-product around top-K axis-best (small)
  custom  - read configs from JSON file
"""

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BENCH = SCRIPT_DIR / "bench.py"
TMP = Path(
    os.environ.get("VLLM_VIT_SWEEP_OUT", Path(tempfile.gettempdir()) / "vit_attn_sweep")
)
TMP.mkdir(parents=True, exist_ok=True)

# Baseline from triton_prefill_attention.py:get_block_size/get_num_warps on RDNA bf16:
# BLOCK_M=BLOCK_N=32, num_warps=8, num_stages=1, no waves_per_eu override.
BASELINE = dict(bm=32, bn=32, nw=8, ns=1, we=None)


def _gpu_lock_cmd(cmd: list[str]) -> list[str]:
    gpu_lock = shutil.which("gpu-lock")
    if gpu_lock:
        return [gpu_lock] + cmd
    return cmd


def _bench_cmd(cfg: dict) -> list[str]:
    cmd = [
        sys.executable,
        str(BENCH),
        "--bm",
        str(cfg["bm"]),
        "--bn",
        str(cfg["bn"]),
        "--nw",
        str(cfg["nw"]),
        "--ns",
        str(cfg["ns"]),
    ]
    if cfg.get("we") is not None:
        cmd += ["--we", str(cfg["we"])]
    return cmd


def run_one(cfg: dict) -> dict:
    tag = f"bm{cfg['bm']}_bn{cfg['bn']}_nw{cfg['nw']}_ns{cfg['ns']}_we{cfg.get('we')}"
    log_out = TMP / f"{tag}.log"
    json_out = TMP / f"{tag}.json"
    cmd = _gpu_lock_cmd(_bench_cmd(cfg))
    t0 = time.time()
    with open(log_out, "w") as f:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=f, timeout=600)
    elapsed = time.time() - t0
    if p.returncode != 0:
        return {
            "tag": tag,
            "error": f"rc={p.returncode}, log={log_out}",
            "elapsed": elapsed,
        }
    text = p.stdout.decode().strip().splitlines()
    if not text:
        return {"tag": tag, "error": f"no stdout, log={log_out}", "elapsed": elapsed}
    try:
        data = json.loads(text[-1])
    except json.JSONDecodeError as e:
        return {
            "tag": tag,
            "error": f"bad json: {e}, log={log_out}",
            "elapsed": elapsed,
        }
    json_out.write_text(json.dumps(data, indent=2))
    return {"tag": tag, **data, "elapsed": elapsed}


def axis_sweep() -> list[dict]:
    base = BASELINE
    grid: list[dict] = []
    for bm in (16, 32, 64, 128):
        c = dict(base)
        c["bm"] = bm
        c["bn"] = bm
        grid.append(c)
    for nw in (2, 4, 8, 16):
        c = dict(base)
        c["nw"] = nw
        grid.append(c)
    for ns in (1, 2, 3):
        c = dict(base)
        c["ns"] = ns
        grid.append(c)
    for we in (1, 2, 4, 6, 8):
        c = dict(base)
        c["we"] = we
        grid.append(c)
    seen = set()
    uniq = []
    for c in grid:
        key = tuple(sorted(c.items(), key=lambda kv: kv[0]))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def refine_grid(seed_configs: list[dict]) -> list[dict]:
    bms = sorted({c["bm"] for c in seed_configs} | {16, 32, 64})
    nws = sorted({c["nw"] for c in seed_configs} | {4, 8})
    nss = sorted({c["ns"] for c in seed_configs} | {1})
    wes = sorted(
        {c.get("we") for c in seed_configs if c.get("we") is not None} | {2, 4}
    )
    return [
        dict(bm=bm, bn=bm, nw=nw, ns=ns, we=we)
        for bm, nw, ns, we in itertools.product(bms, nws, nss, wes)
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("axis", "refine", "custom"), default="axis")
    p.add_argument("--configs", default=None)
    p.add_argument(
        "--seed-configs",
        default=None,
        help="for refine: JSON list of axis-best configs",
    )
    p.add_argument("--out", default=str(TMP / "sweep_results.json"))
    args = p.parse_args()

    if args.phase == "axis":
        grid = axis_sweep()
    elif args.phase == "refine":
        with open(args.seed_configs) as f:
            seeds = json.load(f)
        grid = refine_grid(seeds)
    else:
        with open(args.configs) as f:
            grid = json.load(f)

    print(f"Configs to sweep: {len(grid)}", file=sys.stderr)

    results: list[dict] = []
    for i, c in enumerate(grid):
        print(
            f"[{i + 1}/{len(grid)}] bm={c['bm']} bn={c['bn']} nw={c['nw']} "
            f"ns={c['ns']} we={c.get('we')}",
            file=sys.stderr,
            flush=True,
        )
        r = run_one(c)
        results.append({"input_cfg": c, **r})
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        if "error" in r:
            print(f"  ERROR: {r['error']}", file=sys.stderr)
        else:
            print(
                f"  per_call median = {r['per_call_ms_median']:.3f} ms  "
                f"min = {r['per_call_ms_min']:.3f} ms  "
                f"total/image = {r['total_per_image_ms_median']:.1f} ms",
                file=sys.stderr,
            )

    ok = [r for r in results if "error" not in r]
    ok.sort(key=lambda r: r["per_call_ms_median"])
    print("\nTop 10:", file=sys.stderr)
    for r in ok[:10]:
        c = r["input_cfg"]
        print(
            f"  bm={c['bm']:>3} bn={c['bn']:>3} nw={c['nw']:>2} ns={c['ns']} "
            f"we={c.get('we')}  "
            f"median={r['per_call_ms_median']:.3f} ms  "
            f"min={r['per_call_ms_min']:.3f} ms",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
