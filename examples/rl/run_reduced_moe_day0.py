# SPDX-License-Identifier: Apache-2.0
"""Run reduced-model cold/warm/cold validation using the actual day0 publisher."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=28491)
    args = parser.parse_args()
    print(sys.executable, sys.prefix, flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    checkout = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=f"{Path(__file__).parent}:{checkout}",
               VLLM_SERVER_DEV_MODE="1", VLLM_PLUGINS="", HF_HUB_OFFLINE="1",
               VLLM_USE_DEEP_GEMM="0", NCCL_DEBUG="INFO", NCCL_SOCKET_IFNAME="lo",
               VLLM_HOST_IP="127.0.0.1",
               FLASHINFER_WORKSPACE_BASE=str(checkout / "jit-cuda130"))
    base = f"http://127.0.0.1:{args.port}"
    evidence = {}

    def inspect(arm=False):
        response = requests.post(base + "/collective_rpc", json={
            "method": "inspect_moe_reload", "args": [arm]}, timeout=120)
        response.raise_for_status()
        return response.json()["results"][0]

    def generate():
        results = []
        for prompt in ("The capital of France is", "1 + 1 =", "Write a short greeting:"):
            response = requests.post(base + "/v1/completions", json={
                "model": "experiment", "prompt": prompt, "temperature": 0,
                "max_tokens": 16, "logprobs": 5, "seed": 0}, timeout=120)
            response.raise_for_status()
            choice = response.json()["choices"][0]
            results.append({"text": choice["text"], "logprobs": choice["logprobs"]})
        return results

    for variant in ("a", "b"):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        with (args.output / f"server-{variant}.log").open("w") as log:
            server = subprocess.Popen([
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(args.checkpoints / variant), "--served-model-name", "experiment",
                "--port", str(args.port), "--max-model-len", "128", "--enforce-eager",
                "--gpu-memory-utilization", "0.2", "--kv-cache-memory-bytes", "268435456",
                "--moe-backend", "flashinfer_cutlass", "--no-enable-flashinfer-autotune",
                "--worker-extension-cls", "moe_reload_evidence.MoEReloadEvidence",
                "--weight-transfer-config", '{"backend":"nccl"}',
            ], env=dict(env, CUDA_VISIBLE_DEVICES="1"), stdout=log, stderr=log,
                start_new_session=True)
            try:
                for _ in range(240):
                    if server.poll() is not None:
                        raise RuntimeError(f"Server exited: server-{variant}.log")
                    try:
                        if requests.get(base + "/health", timeout=2).ok:
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(1)
                else:
                    raise TimeoutError("Server did not become healthy")
                evidence[f"cold_{variant}"] = inspect(variant == "a")
                evidence[f"cold_{variant}_output"] = generate()
                if variant == "a":
                    with (args.output / "client.log").open("w") as client_log:
                        subprocess.run([
                            sys.executable, str(args.kit / "scripts/vllm_weight_update_client/run_vllm_weight_update.py"),
                            "--base-url", base, "--model", str(args.checkpoints / "b"),
                            "--revision", "main", "--checkpoint-path", str(args.checkpoints / "b"),
                            "--device", "cuda:0", "--output", str(args.output / "update.json"),
                        ], env=dict(env, CUDA_VISIBLE_DEVICES="2"), stdout=client_log,
                            stderr=client_log, timeout=600, check=True)
                    requests.get(base + "/health", timeout=30).raise_for_status()
                    evidence["warm_b"] = inspect()
                    evidence["warm_b_output"] = generate()
            finally:
                (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2))
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait()
    assert evidence["cold_a"].keys() == evidence["warm_b"].keys() == evidence["cold_b"].keys()
    changed = False
    for name, warm in evidence["warm_b"].items():
        old, cold = evidence["cold_a"][name], evidence["cold_b"][name]
        for key in ("method", "kernel", "config"):
            assert warm[key] == old[key]
        assert warm["tensors"].keys() == cold["tensors"].keys() == old["tensors"].keys()
        for key, tensor in warm["tensors"].items():
            assert tensor["id"] == old["tensors"][key]["id"]
            assert tensor["ptr"] == old["tensors"][key]["ptr"]
            assert tensor["hash"] == cold["tensors"][key]["hash"]
            changed |= tensor["hash"] != old["tensors"][key]["hash"]
    assert changed
    assert evidence["warm_b_output"] == evidence["cold_b_output"]
    result = {"status": "PASS", "layers": len(evidence["warm_b"]),
              "runtime_changed": changed, "warm_matches_cold": True}
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
