#!/usr/bin/env python3
"""
SP+AsyncTP 2×2 Matrix Benchmark: SP on/off × CUDA Graph on/off

Key design decisions for fair comparison:
  - NUM_PROMPTS=256 so decode BS=256 naturally (matches compile_sizes=[256])
  - compile_sizes=[256] for ALL configs (ensures compiled code is always used)
  - random-output-len=128 (fixed, so BS stays at 256 throughout decode)
  - CG configs: cudagraph_capture_sizes=[256] (pads BS to 256)
  - No-CG configs: cudagraph_mode=0 (BS=256 naturally from 256 prompts)
  - All SP configs: enable_sp=True, fuse_gemm_comms=True (+rms_norm auto-added)
  - All baseline configs: enable_sp=False, fuse_gemm_comms=False, custom_ops=["+rms_norm"]

Usage:
  cd /data/users/tianren/vllm && \
    CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 \
    /home/tianren/.conda/envs/vllm/bin/python benchmarks/run_sp_cg_2x2.py
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

MODEL = "meta-llama/Llama-2-7b-hf"
TP_SIZE = 2
MAX_MODEL_LEN = 4096
PORT = 8000
PYTHON = sys.executable

NUM_PROMPTS = 256
INPUT_LEN = 256
OUTPUT_LEN = 128

CONFIGS = {
    # ── 2×2 matrix: SP on/off × CG on/off ──
    "baseline_cg": {
        "compile_sizes": [256],
        "cudagraph_capture_sizes": [256],
        "custom_ops": ["+rms_norm"],
        "pass_config": {"enable_sp": False, "fuse_gemm_comms": False},
    },
    "baseline_no_cg": {
        "compile_sizes": [256],
        "cudagraph_mode": 0,
        "custom_ops": ["+rms_norm"],
        "pass_config": {"enable_sp": False, "fuse_gemm_comms": False},
    },
    "sp_asynctp_cg": {
        "compile_sizes": [256],
        "cudagraph_capture_sizes": [256],
        "pass_config": {"enable_sp": True, "fuse_gemm_comms": True},
    },
    "sp_asynctp_no_cg": {
        "compile_sizes": [256],
        "cudagraph_mode": 0,
        "pass_config": {"enable_sp": True, "fuse_gemm_comms": True},
    },
}

CONFIG_ORDER = ["baseline_cg", "baseline_no_cg", "sp_asynctp_cg", "sp_asynctp_no_cg"]


def kill_server(server):
    """Kill server and all child worker processes."""
    try:
        pgid = os.getpgid(server.pid)
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(2)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
    try:
        server.kill()
    except Exception:
        pass
    try:
        server.wait(timeout=10)
    except Exception:
        pass
    subprocess.run(
        ["pkill", "-9", "-f", "VLLM::Worker"],
        capture_output=True, timeout=5,
    )
    time.sleep(5)


def run_one(config_name, config):
    config_json = json.dumps(config)
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0,1"

    cmd = [
        PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--tensor-parallel-size", str(TP_SIZE),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.85",
        "--enforce-eager", "false",
        "--compilation-config", config_json,
        "--port", str(PORT),
    ]

    print(f"\n{'='*70}", flush=True)
    print(f"CONFIG: {config_name}", flush=True)
    print(f"  compilation-config: {config_json}", flush=True)
    print(f"{'='*70}", flush=True)

    server = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        start_new_session=True,
    )

    for i in range(600):
        time.sleep(1)
        if i % 30 == 0:
            print(f"  Waiting for server... {i}s", flush=True)
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=1)
            print(f"  Server ready after {i}s", flush=True)
            break
        except Exception:
            if server.poll() is not None:
                stderr = server.stderr.read().decode(errors="replace")[-3000:]
                print(f"  SERVER DIED!\n{stderr}", flush=True)
                kill_server(server)
                return None
    else:
        print("  Timeout waiting for server (600s)!", flush=True)
        kill_server(server)
        return None

    bench_cmd = [
        PYTHON, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "vllm", "--model", MODEL,
        "--dataset-name", "random",
        "--random-input-len", str(INPUT_LEN),
        "--random-output-len", str(OUTPUT_LEN),
        "--num-prompts", str(NUM_PROMPTS),
        "--request-rate", "inf",
        "--port", str(PORT),
    ]

    print(f"  Running benchmark ({NUM_PROMPTS} prompts, "
          f"in={INPUT_LEN}, out={OUTPUT_LEN})...", flush=True)
    try:
        bench = subprocess.run(
            bench_cmd, env=env, capture_output=True, text=True, timeout=600,
        )
        output = bench.stdout + "\n" + bench.stderr
    except subprocess.TimeoutExpired:
        output = ""
        print("  Benchmark timed out!", flush=True)

    result = {"config": config_name}
    for line in output.split("\n"):
        ll = line.lower().strip()
        if "output token throughput" in ll and "peak" not in ll:
            try:
                result["tok_tput"] = float(line.split(":")[-1].strip().split()[0])
            except Exception:
                pass
        elif "request throughput" in ll:
            try:
                result["req_tput"] = float(line.split(":")[-1].strip().split()[0])
            except Exception:
                pass
        elif "mean ttft" in ll:
            try:
                result["mean_ttft_ms"] = float(line.split(":")[-1].strip().split()[0])
            except Exception:
                pass
        elif "mean itl" in ll:
            try:
                result["mean_itl_ms"] = float(line.split(":")[-1].strip().split()[0])
            except Exception:
                pass
        elif "mean tpot" in ll:
            try:
                result["mean_tpot_ms"] = float(line.split(":")[-1].strip().split()[0])
            except Exception:
                pass

    kill_server(server)

    if "tok_tput" in result:
        print(f"  ✓ Throughput: {result['tok_tput']:.1f} tok/s | "
              f"ITL: {result.get('mean_itl_ms', 'N/A')} ms | "
              f"TTFT: {result.get('mean_ttft_ms', 'N/A')} ms", flush=True)
    else:
        print(f"  ✗ No throughput parsed. Raw output (last 2000):", flush=True)
        print(output[-2000:], flush=True)

    return result


def main():
    results = []
    ts = int(time.time())
    start = time.time()

    print(f"SP+AsyncTP 2×2 Matrix Benchmark", flush=True)
    print(f"Model: {MODEL} | TP={TP_SIZE} | prompts={NUM_PROMPTS} "
          f"| in={INPUT_LEN} out={OUTPUT_LEN}", flush=True)
    print(f"Configs: {CONFIG_ORDER}", flush=True)

    for config_name in CONFIG_ORDER:
        r = run_one(config_name, CONFIGS[config_name])
        if r and "tok_tput" in r:
            results.append(r)

    elapsed = time.time() - start
    out = f"/tmp/sp_2x2_results_{ts}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out} (total: {elapsed:.0f}s)", flush=True)

    # ── Summary table ──
    print(f"\n{'='*80}", flush=True)
    print("SUMMARY: SP+AsyncTP 2×2 Matrix (Llama-2-7b-hf TP=2)", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Config':<25} {'Tok/s':>10} {'ITL(ms)':>10} "
          f"{'TTFT(ms)':>10} {'TPOT(ms)':>10}", flush=True)
    print("-" * 80, flush=True)
    for r in results:
        tok = f"{r['tok_tput']:.1f}" if "tok_tput" in r else "N/A"
        itl = f"{r['mean_itl_ms']:.2f}" if "mean_itl_ms" in r else "N/A"
        ttft = f"{r['mean_ttft_ms']:.2f}" if "mean_ttft_ms" in r else "N/A"
        tpot = f"{r['mean_tpot_ms']:.2f}" if "mean_tpot_ms" in r else "N/A"
        print(f"{r['config']:<25} {tok:>10} {itl:>10} "
              f"{ttft:>10} {tpot:>10}", flush=True)

    # ── Speedup analysis (2×2 decomposition) ──
    by_name = {r["config"]: r for r in results}
    print(f"\n{'='*80}", flush=True)
    print("SPEEDUP ANALYSIS (2×2 decomposition)", flush=True)
    print(f"{'='*80}", flush=True)

    pairs = [
        ("SP effect (CG on)", "baseline_cg", "sp_asynctp_cg"),
        ("SP effect (CG off)", "baseline_no_cg", "sp_asynctp_no_cg"),
        ("CG effect (no SP)", "baseline_no_cg", "baseline_cg"),
        ("CG effect (with SP)", "sp_asynctp_no_cg", "sp_asynctp_cg"),
    ]
    for label, base, target in pairs:
        if base in by_name and target in by_name:
            base_t = by_name[base]["tok_tput"]
            target_t = by_name[target]["tok_tput"]
            ratio = target_t / base_t
            print(f"  {label:<25}: {target} / {base} = {ratio:.2f}x "
                  f"({base_t:.0f} → {target_t:.0f} tok/s)", flush=True)


if __name__ == "__main__":
    main()
