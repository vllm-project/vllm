#!/usr/bin/env python3
"""Live OOM resilience demo: vanilla crashes, flash-maxsim survives.

Run:
  # Terminal 1 — start vanilla server:
  VLLM_FORCE_VANILLA_MAXSIM=1 VLLM_DISABLE_ZEROCOPY=1 \
  HF_HOME=/dccstor/mm-rag/pony/hf_home python -m vllm.entrypoints.openai.api_server \
    --model vidore/colpali-v1.3-hf --runner pooling --port 9256 \
    --enforce-eager --max-model-len 4096 --max-num-batched-tokens 131072 \
    --dtype half --trust-remote-code --gpu-memory-utilization 0.7

  # Terminal 2 — run this script:
  python tests/v1/worker/demo_oom_resilience.py --port 9256

  # Then kill server, restart WITHOUT env vars (flash mode), run again.

Or use --auto to run both automatically:
  python tests/v1/worker/demo_oom_resilience.py --auto
"""
import argparse
import concurrent.futures
import os
import signal
import subprocess
import sys
import time

import requests

MODEL = "vidore/colpali-v1.3-hf"
DOC_TEXT = (
    "Machine learning is a subfield of artificial intelligence that gives "
    "computers the ability to learn without being explicitly programmed. "
    "It focuses on developing algorithms that can access data and use it "
    "to learn for themselves."
)


def gpu_mem():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        return int(out.split()[0])
    except Exception:
        return 0


def send_request(url, model, n_docs, timeout=120):
    payload = {
        "model": model,
        "text_1": "What is machine learning and how does it work?",
        "text_2": [DOC_TEXT] * n_docs,
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def check_server(port):
    try:
        r = requests.get(f"http://localhost:{port}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def wait_for_server(port, timeout=300):
    for i in range(timeout):
        if check_server(port):
            return True
        time.sleep(1)
    return False


def run_escalating_load(port, model, label):
    url = f"http://localhost:{port}/v1/score"
    conc = 64

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print("  Escalating docs/request until OOM or completion")
    print(f"{'=' * 60}")
    print(f"  Baseline GPU mem: {gpu_mem()} MiB\n")

    # Warmup
    print("  Warming up...", end=" ", flush=True)
    for warmup_docs in [5, 20]:
        def _warmup(_i, _d=warmup_docs):
            return send_request(url, model, _d, timeout=60)
        with concurrent.futures.ThreadPoolExecutor(16) as ex:
            list(ex.map(_warmup, range(30)))
    print("done.")
    print(f"  GPU mem after warmup: {gpu_mem()} MiB\n")

    print(f"  {'Docs':>6} {'Reqs':>6} {'Conc':>6} {'OK':>6} "
          f"{'Fail':>6} {'GPU MiB':>10} {'Status':>12}")
    print(f"  {'-' * 58}")

    for n_docs in [50, 100, 200, 500]:
        n_reqs = 50

        def _send(_i, _d=n_docs):
            return send_request(url, model, _d)

        with concurrent.futures.ThreadPoolExecutor(conc) as ex:
            results = list(ex.map(_send, range(n_reqs)))

        ok = sum(1 for r in results if r)
        fail = n_reqs - ok
        mem_after = gpu_mem()

        alive = check_server(port)
        if ok == n_reqs:
            status = "OK"
        elif alive:
            status = f"ERRORS({fail})"
        else:
            status = "*** CRASHED ***"

        print(f"  {n_docs:>6} {n_reqs:>6} {conc:>6} {ok:>6} "
              f"{fail:>6} {mem_after:>10} {status:>12}")

        if not alive:
            print(f"\n  Server crashed at {n_docs} docs/request!")
            print(f"  GPU mem at crash: {mem_after} MiB")
            return False

    print(f"\n  Survived all levels! Final GPU mem: {gpu_mem()} MiB")
    return True


def start_server(port, flash=True, gpu_util=0.7):
    env = os.environ.copy()
    env["HF_HOME"] = "/dccstor/mm-rag/pony/hf_home"
    if not flash:
        env["VLLM_FORCE_VANILLA_MAXSIM"] = "1"
        env["VLLM_DISABLE_ZEROCOPY"] = "1"

    python = os.path.join(os.path.dirname(sys.executable), "python")
    cmd = [
        python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--runner", "pooling",
        "--port", str(port),
        "--enforce-eager",
        "--max-model-len", "4096",
        "--max-num-batched-tokens", "131072",
        "--dtype", "half",
        "--trust-remote-code",
        "--gpu-memory-utilization", str(gpu_util),
        "--disable-log-stats",
    ]

    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc


def kill_server(proc):
    """Kill the spawned server process and its children.

    Intentionally scoped to ``proc`` and its subprocesses — on shared
    GPU nodes, killing every pid from ``nvidia-smi`` would also kill
    unrelated users' jobs.
    """
    if not proc:
        return
    try:
        # Kill the process group (server + any child workers).
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception:
        proc.kill()
    proc.wait()
    time.sleep(3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9256)
    parser.add_argument("--auto", action="store_true",
                        help="Automatically start/stop servers")
    parser.add_argument("--gpu-util", type=float, default=0.7,
                        help="GPU memory utilization (lower = more constrained)")
    args = parser.parse_args()

    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total",
         "--format=csv,noheader"], text=True,
    ).strip()
    print(f"GPU: {gpu_info}")
    print(f"Model: {MODEL}")

    if args.auto:
        # --- Vanilla (should crash) ---
        print("\nStarting VANILLA server...")
        proc = start_server(args.port, flash=False, gpu_util=args.gpu_util)
        if not wait_for_server(args.port):
            print("ERROR: Vanilla server failed to start")
            kill_server(proc)
            return
        print(f"Server ready. GPU mem: {gpu_mem()} MiB")

        vanilla_survived = run_escalating_load(
            args.port, MODEL, "VANILLA MaxSim (padded bmm)"
        )
        kill_server(proc)

        # --- Flash (should survive) ---
        print("\n\nStarting FLASH-MAXSIM server...")
        proc = start_server(args.port, flash=True, gpu_util=args.gpu_util)
        if not wait_for_server(args.port):
            print("ERROR: Flash server failed to start")
            kill_server(proc)
            return
        print(f"Server ready. GPU mem: {gpu_mem()} MiB")

        flash_survived = run_escalating_load(
            args.port, MODEL, "FLASH-MAXSIM (zero-copy, 0 extra HBM)"
        )
        kill_server(proc)

        # --- Summary ---
        print(f"\n{'=' * 60}")
        print("  RESULT")
        print(f"{'=' * 60}")
        print(f"  Vanilla:     {'SURVIVED' if vanilla_survived else 'CRASHED'}")
        print(f"  Flash-MaxSim: {'SURVIVED' if flash_survived else 'CRASHED'}")
    else:
        # Manual mode — server already running
        if not check_server(args.port):
            print(f"No server on port {args.port}. "
                  f"Use --auto or start manually.")
            return

        run_escalating_load(args.port, MODEL, "Server on port " + str(args.port))


if __name__ == "__main__":
    main()
