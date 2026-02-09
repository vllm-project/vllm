#!/usr/bin/env python3
"""
Reproduction script for CPU memory leak with multimodal models (issue #28726).

Uses the EXACT reproduction from the issue report:
  - vllm serve Qwen2.5-VL-3B-Instruct --max-model-len 25000
  - vllm bench serve with lmarena-ai/VisionArena-Chat dataset (1000 prompts)

Measures the ENGINE CORE subprocess RSS specifically (the EngineCore runs in
a child process in V1 architecture, and that's where Request objects live).

The leak is caused by a reference cycle in Request objects:
    Request -> partial(block_hasher, self) -> Request

Usage:
    python test_mm_memory_leak.py          # run on current branch
    python test_mm_memory_leak.py --rounds 3  # fewer rounds

Requirements:
    - GPU with enough VRAM for Qwen2.5-VL-3B-Instruct (~8GB)
    - Model cached at ~/.cache/huggingface/hub/
    - Internet access (first run downloads VisionArena-Chat dataset)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
PORT = 29345
HOST = f"http://localhost:{PORT}"


def get_process_tree_rss(root_pid: int) -> dict[int, float]:
    """Get RSS for every process in the tree. Returns {pid: rss_gb}."""
    result = {}
    pids = _get_all_descendant_pids(root_pid)
    pids.add(root_pid)
    for pid in pids:
        try:
            with open(f"/proc/{pid}/status") as f:
                rss_kb = 0
                cmdline = ""
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
                        break
            # Also get command name
            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_text().replace("\0", " ")[:100]
            except Exception:
                cmdline = "?"
            result[pid] = (rss_kb / (1024 * 1024), cmdline)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return result


def get_tree_total_rss_gb(root_pid: int) -> float:
    tree = get_process_tree_rss(root_pid)
    return sum(rss for rss, _ in tree.values())


def get_engine_core_rss_gb(root_pid: int) -> float:
    """Find and return RSS of the EngineCore subprocess."""
    tree = get_process_tree_rss(root_pid)
    for pid, (rss_gb, cmdline) in tree.items():
        if pid != root_pid and "python" in cmdline.lower() and rss_gb > 0.1:
            # The engine core subprocess is the largest Python child process
            return rss_gb
    # Fallback: return total tree RSS
    return sum(rss for rss, _ in tree.values())


def _get_all_descendant_pids(pid: int) -> set[int]:
    children = set()
    try:
        result = subprocess.run(
            ["pgrep", "-P", str(pid)], capture_output=True, text=True,
        )
        for line in result.stdout.strip().splitlines():
            if line.strip():
                child = int(line.strip())
                children.add(child)
                children.update(_get_all_descendant_pids(child))
    except Exception:
        pass
    return children


def wait_for_server(timeout: int = 300) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{HOST}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(3)
    return False


def run_bench(num_prompts: int) -> tuple[float, bool]:
    """Run vllm bench serve with VisionArena-Chat. Returns (elapsed, success)."""
    cmd = [
        sys.executable, "-c",
        "import sys; "
        f"sys.argv = ['vllm', 'bench', 'serve', "
        f"'--backend', 'openai-chat', "
        f"'--model', '{MODEL}', "
        f"'--endpoint', '/v1/chat/completions', "
        f"'--dataset-name', 'hf', "
        f"'--dataset-path', 'lmarena-ai/VisionArena-Chat', "
        f"'--hf-split', 'train', "
        f"'--num-prompts', '{num_prompts}', "
        f"'--port', '{PORT}']; "
        "from vllm.entrypoints.cli.main import main; main()",
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    elapsed = time.time() - t0
    success = result.returncode == 0
    if not success:
        err = (result.stderr or "")[-500:] + (result.stdout or "")[-500:]
        print(f"  Bench output: {err}")
    return elapsed, success


def print_process_tree(root_pid: int, label: str):
    """Print RSS breakdown of all processes in the tree."""
    tree = get_process_tree_rss(root_pid)
    print(f"\n  Process tree ({label}):")
    for pid, (rss_gb, cmdline) in sorted(tree.items()):
        short_cmd = cmdline[:60]
        print(f"    PID {pid:>7}: {rss_gb:6.2f} GB  {short_cmd}")
    total = sum(rss for rss, _ in tree.values())
    print(f"    {'TOTAL':>11}: {total:6.2f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--prompts", type=int, default=1000)
    args = parser.parse_args()

    num_rounds = args.rounds
    prompts_per_round = args.prompts

    print("=" * 70)
    print(" CPU Memory Leak Reproduction - Multimodal + Prefix Caching")
    print(" (using lmarena-ai/VisionArena-Chat dataset)")
    print("=" * 70)

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, cwd=repo_dir,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=repo_dir,
    ).stdout.strip()
    print(f"Branch:      {branch} ({commit})")
    print(f"Model:       {MODEL}")
    print(f"Config:      {num_rounds} rounds x {prompts_per_round} prompts")
    print(f"             prefix caching ON, max-model-len 25000")
    print()

    # --- Start vLLM server ---
    log_file = tempfile.NamedTemporaryFile(
        mode="w", prefix="vllm_server_", suffix=".log", delete=False,
    )
    log_path = log_file.name
    print(f"Starting vLLM server (log: {log_path})...")

    server_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(PORT),
        "--max-model-len", "25000",
        "--gpu-memory-utilization", "0.95",
        "--limit-mm-per-prompt", json.dumps({"image": 1, "video": 0}),
        "--mm-processor-kwargs", json.dumps({
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        }),
    ]

    server_proc = subprocess.Popen(
        server_cmd, stdout=log_file, stderr=subprocess.STDOUT,
    )

    try:
        if not wait_for_server(timeout=300):
            print("ERROR: Server failed to start within timeout.")
            server_proc.kill()
            server_proc.wait(timeout=10)
            log_file.close()
            print("Server log (last 3000 chars):")
            print(Path(log_path).read_text()[-3000:])
            return 1

        server_pid = server_proc.pid
        rss_idle = get_tree_total_rss_gb(server_pid)
        ec_idle = get_engine_core_rss_gb(server_pid)
        print(f"Server ready (PID {server_pid}).")
        print_process_tree(server_pid, "idle")

        # --- Run benchmark rounds ---
        print(f"\n{'Round':<7} {'Reqs':<8} {'Total(GB)':<11} {'EC(GB)':<9} "
              f"{'EC delta':<10} {'EC round':<10} {'Time'}")
        print("-" * 70)
        print(f"{'idle':<7} {0:<8} {rss_idle:<11.2f} {ec_idle:<9.2f} "
              f"{'---':<10} {'---':<10}")

        ec_history = [ec_idle]
        rss_history = [rss_idle]

        for round_num in range(1, num_rounds + 1):
            elapsed, success = run_bench(prompts_per_round)

            if not success:
                print(f"  Round {round_num} bench had errors, continuing...")

            time.sleep(3)

            rss_now = get_tree_total_rss_gb(server_pid)
            ec_now = get_engine_core_rss_gb(server_pid)
            ec_delta = ec_now - ec_idle
            ec_round = ec_now - ec_history[-1]
            ec_history.append(ec_now)
            rss_history.append(rss_now)

            total_reqs = round_num * prompts_per_round
            print(f"{round_num:<7} {total_reqs:<8} {rss_now:<11.2f} "
                  f"{ec_now:<9.2f} "
                  f"{'+' if ec_delta >= 0 else ''}{ec_delta:<9.2f} "
                  f"{'+' if ec_round >= 0 else ''}{ec_round:<9.2f} "
                  f"{elapsed:.0f}s"
                  f"{' (FAIL)' if not success else ''}")

        # Print final tree.
        print_process_tree(server_pid, "final")

        # --- Summary ---
        print()
        print("=" * 70)
        ec_growth = ec_history[-1] - ec_history[1]
        ec_avg = ec_growth / max(num_rounds - 1, 1)
        total_growth = rss_history[-1] - rss_history[1]
        total_avg = total_growth / max(num_rounds - 1, 1)

        if ec_avg > 0.3 or total_avg > 0.5:
            verdict = "\033[91mLEAK DETECTED\033[0m"
        else:
            verdict = "\033[92mMEMORY STABLE\033[0m"

        print(f"Result: {verdict}")
        print(f"  Branch:               {branch} ({commit})")
        print(f"  Total RSS idle:       {rss_idle:.2f} GB")
        print(f"  Total RSS after R1:   {rss_history[1]:.2f} GB  (warmup)")
        print(f"  Total RSS after R{num_rounds}:   {rss_history[-1]:.2f} GB")
        print(f"  Total growth R2-{num_rounds}:    {total_growth:+.2f} GB "
              f"(avg {total_avg:+.2f} GB/round)")
        print()
        print(f"  EngineCore RSS idle:  {ec_idle:.2f} GB")
        print(f"  EngineCore after R1:  {ec_history[1]:.2f} GB  (warmup)")
        print(f"  EngineCore after R{num_rounds}:  {ec_history[-1]:.2f} GB")
        print(f"  EngineCore growth:    {ec_growth:+.2f} GB "
              f"(avg {ec_avg:+.2f} GB/round)")

        if ec_avg > 0.3 or total_avg > 0.5:
            print()
            print("  The EngineCore subprocess RSS keeps growing.")
            print("  This confirms the CPU memory leak from the Request")
            print("  reference cycle (partial(block_hasher, self)).")
        else:
            print()
            print("  Memory is stable after warmup - no leak.")

        print("=" * 70)
        return 0

    finally:
        print("\nShutting down server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait(timeout=5)
        log_file.close()
        print("Done.")


if __name__ == "__main__":
    sys.exit(main())
