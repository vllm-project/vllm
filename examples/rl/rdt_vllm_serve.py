# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Driver-side helpers to run the inference fleet as a standard ``vllm serve``
process for the sharded-RDT examples.

The trainer reaches the server's weight-sync control plane over the RLHF HTTP
routes (``HTTPVLLMWeightSyncClient``); generation uses ``/v1/completions``. RDT
still needs the workers to be RayExecutorV2 tensor-transport actors sharing the
trainer's Ray cluster, so the server is launched with the Ray v2 executor and
inherits this process's env (``address=auto``). These helpers run only on the
driver — they are not shipped to or imported by the Ray actors.
"""

import json
import os
import subprocess
import sys
import time

import requests


def launch_vllm_serve(
    model: str,
    *,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
    enable_expert_parallel: bool = False,
    port: int = 8000,
    gpu_memory_utilization: float = 0.7,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """Start ``vllm serve`` for the RDT inference fleet.

    ``VLLM_SERVER_DEV_MODE`` exposes the RLHF weight-sync routes; the Ray v2
    executor makes the workers tensor-transport actors (RDT's data plane). The
    child inherits our env, so it joins the same Ray cluster and picks up the
    editable vLLM install via the venv interpreter.
    """
    vllm_bin = os.path.join(os.path.dirname(sys.executable), "vllm")
    cmd = [
        vllm_bin,
        "serve",
        model,
        "--port",
        str(port),
        "--enforce-eager",
        "--load-format",
        "dummy",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--distributed-executor-backend",
        "ray",
        "--weight-transfer-config",
        json.dumps({"backend": "sharded_rdt"}),
    ]
    if tensor_parallel_size > 1:
        cmd += ["--tensor-parallel-size", str(tensor_parallel_size)]
    # Only engage the DP-ray backend when actually data-parallel: DP=1 with
    # data_parallel_backend=ray forces vLLM's DP-placement path, which fails.
    if data_parallel_size > 1:
        cmd += [
            "--data-parallel-size",
            str(data_parallel_size),
            "--data-parallel-backend",
            "ray",
        ]
    if enable_expert_parallel:
        cmd.append("--enable-expert-parallel")
    if extra_args:
        cmd += extra_args
    env = dict(
        os.environ,
        VLLM_SERVER_DEV_MODE="1",
        VLLM_USE_RAY_V2_EXECUTOR_BACKEND="1",
    )
    # Let the server's Ray workers use the ambient interpreter/install rather
    # than a workspace snapshot working_dir. On an Anyscale dev workspace the
    # RAY_RUNTIME_ENV_HOOK injects a git snapshot as the workers' working_dir,
    # which shadows an editable vLLM install (its compiled .so is gitignored, so
    # the snapshot lacks it). No-op off Anyscale (the var is unset there).
    env.pop("RAY_RUNTIME_ENV_HOOK", None)
    print(f"[serve] launching: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, env=env)


def wait_for_server(
    endpoint: str, proc: subprocess.Popen, timeout: float = 1800
) -> None:
    """Block until ``/health`` returns 200, failing fast if the server exits."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm serve exited early (code {proc.returncode})")
        try:
            if requests.get(f"{endpoint}/health", timeout=5).status_code == 200:
                print("[serve] server is healthy", flush=True)
                return
        except requests.RequestException:
            pass
        time.sleep(3)
    raise RuntimeError("vllm serve did not become healthy in time")


def http_generate(
    endpoint: str, model: str, prompts: list[str], max_tokens: int = 16
) -> list[str]:
    """Greedy ``/v1/completions`` for each prompt; returns the generated texts."""
    outs = []
    for p in prompts:
        r = requests.post(
            f"{endpoint}/v1/completions",
            json={
                "model": model,
                "prompt": p,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=120,
        )
        r.raise_for_status()
        outs.append(r.json()["choices"][0]["text"])
    return outs


def pause_generation(endpoint: str, mode: str = "abort") -> None:
    requests.post(f"{endpoint}/pause", params={"mode": mode}, timeout=60)


def resume_generation(endpoint: str) -> None:
    requests.post(f"{endpoint}/resume", timeout=60)


def shutdown_server(proc: subprocess.Popen) -> None:
    print("[serve] shutting down vllm serve", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
