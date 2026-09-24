# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold-A / warm-B / cold-B validation through production NCCL or IPC APIs.

Use a local day0-kit checkout at 152c2c0 with its publisher's legacy NCCL imports
redirected to day0_nccl_compat. IPC uses the same checkpoint reader and the
current native IPC trainer. The service is local, single-rank, without EPLB.
"""

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


def publish_ipc(args):
    import torch

    from vllm.distributed.weight_transfer import HTTPVLLMWeightSyncClient
    from vllm.distributed.weight_transfer.ipc_engine import (
        IPCTrainerInitInfo,
        IPCTrainerWeightTransferEngine,
    )

    sys.path.insert(0, str(args.kit / "scripts/vllm_weight_update_client"))
    from hf_checkpoint_nccl_publisher import (
        CheckpointManifest,
        SafetensorsCheckpointSource,
    )

    manifest = CheckpointManifest.load(
        model=str(args.checkpoints / "b"),
        revision="main",
        checkpoint_path=str(args.checkpoints / "b"),
    )
    engine = IPCTrainerWeightTransferEngine.trainer_init(
        IPCTrainerInitInfo(
            rank=0,
            packed=True,
            packed_buffer_size_bytes=max(
                1 << 30, max(tensor.nbytes for tensor in manifest.tensors)
            ),
        ),
        client=HTTPVLLMWeightSyncClient(f"http://127.0.0.1:{args.port}", timeout=600),
        source=SafetensorsCheckpointSource(manifest, torch.device("cuda:0")),
    )
    engine.send_weights()
    (args.output / "update.json").write_text(
        json.dumps({"backend": "ipc", "tensor_count": manifest.tensor_count})
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=28491)
    parser.add_argument("--backend", choices=["nccl", "ipc"], default="nccl")
    parser.add_argument("--moe-backend", default="flashinfer_cutlass")
    parser.add_argument("--server-gpu", default="0")
    parser.add_argument("--publisher-gpu", default="2")
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--publish-ipc", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--batch-probe",
        action="store_true",
        help="Use one fixed prompt batch for cold-A and warm-B comparison.",
    )
    args = parser.parse_args()
    print(sys.executable, sys.prefix, flush=True)
    if args.publish_ipc:
        publish_ipc(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    checkout = Path(__file__).resolve().parents[2]
    env = dict(
        os.environ,
        PYTHONPATH=f"{Path(__file__).parent}:{checkout}",
        VLLM_SERVER_DEV_MODE="1",
        VLLM_PLUGINS="",
        HF_HUB_OFFLINE="1",
        VLLM_USE_DEEP_GEMM="1",
        NCCL_SOCKET_IFNAME="lo",
        VLLM_HOST_IP="127.0.0.1",
    )
    if args.backend == "ipc":
        env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    base = f"http://127.0.0.1:{args.port}"
    evidence = {}

    def inspect(arm=False):
        response = requests.post(
            base + "/collective_rpc",
            json={"method": "inspect_reload_trace", "args": [arm]},
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["results"][0]

    def inspect_parameters():
        response = requests.post(
            base + "/collective_rpc",
            json={"method": "inspect_model_parameters", "args": []},
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["results"][0]

    def generate():
        if args.batch_probe:
            prompts = [
                "The capital of France is",
                "1 + 1 =",
                "Write a short greeting:",
                "The largest planet is",
                "Water freezes at",
                "Complete: reload testing is",
                "Name one primary color:",
                "A triangle has",
            ]
            choices = []
            for start in (0, 4):
                response = requests.post(
                    base + "/v1/completions",
                    json={
                        "model": "experiment",
                        "prompt": prompts[start : start + 4],
                        "temperature": 0,
                        "top_k": 1,
                        "max_tokens": 1,
                        "seed": 0,
                    },
                    timeout=180,
                )
                response.raise_for_status()
                choices.extend(response.json()["choices"])
            return choices
        results = []
        for prompt in (
            "The capital of France is",
            "1 + 1 =",
            "Write a short greeting:",
        ):
            response = requests.post(
                base + "/v1/completions",
                json={
                    "model": "experiment",
                    "prompt": prompt,
                    "temperature": 0,
                    "max_tokens": 16,
                    "logprobs": 5,
                    "seed": 0,
                },
                timeout=180,
            )
            response.raise_for_status()
            choice = response.json()["choices"][0]
            results.append({"text": choice["text"], "logprobs": choice["logprobs"]})
        return results

    for variant in ("a", "b"):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        with (args.output / f"server-{variant}.log").open("w") as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--model",
                    str(args.checkpoints / variant),
                    "--served-model-name",
                    "experiment",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.port),
                    "--max-model-len",
                    "4096",
                    "--enforce-eager",
                    "--gpu-memory-utilization",
                    "0.8",
                    "--kv-cache-memory-bytes",
                    "268435456",
                    "--moe-backend",
                    args.moe_backend,
                    "--no-enable-flashinfer-autotune",
                    "--kernel-config",
                    (
                        '{"enable_cutedsl_warmup":false,'
                        '"linear_backend_per_quant":{"mxfp8":"emulation"}}'
                    ),
                    "--worker-extension-cls",
                    "reload_trace_evidence.ReloadTraceEvidence",
                    "--weight-transfer-config",
                    json.dumps({"backend": args.backend, "reload_mode": "trace"}),
                ],
                env=dict(env, CUDA_VISIBLE_DEVICES=args.server_gpu),
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + args.startup_timeout
                while time.monotonic() < deadline:
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
                evidence[f"cold_{variant}_parameters"] = inspect_parameters()
                evidence[f"cold_{variant}_output"] = generate()
                if variant == "a":
                    requests.post(
                        base + "/pause?mode=wait&clear_cache=true", timeout=180
                    ).raise_for_status()
                    if args.backend == "nccl":
                        command = [
                            sys.executable,
                            str(Path(__file__).with_name("run_day0_nccl_publisher.py")),
                            str(
                                args.kit / "scripts/vllm_weight_update_client/"
                                "run_vllm_weight_update.py"
                            ),
                            "--base-url",
                            base,
                            "--model",
                            str(args.checkpoints / "b"),
                            "--revision",
                            "main",
                            "--checkpoint-path",
                            str(args.checkpoints / "b"),
                            "--device",
                            "cuda:0",
                            "--output",
                            str(args.output / "update.json"),
                        ]
                        if args.batch_probe:
                            command.append("--freeze-engram-lookup")
                        publisher_gpu = args.publisher_gpu
                    else:
                        command = [
                            sys.executable,
                            __file__,
                            *sys.argv[1:],
                            "--publish-ipc",
                        ]
                        publisher_gpu = args.server_gpu
                    with (args.output / "client.log").open("w") as client_log:
                        subprocess.run(
                            command,
                            env=dict(env, CUDA_VISIBLE_DEVICES=publisher_gpu),
                            stdout=client_log,
                            stderr=client_log,
                            timeout=900,
                            check=True,
                        )
                    evidence["warm_b"] = inspect()
                    evidence["warm_b_parameters"] = inspect_parameters()
                    if args.batch_probe:
                        requests.post(
                            base + "/reset_prefix_cache", timeout=180
                        ).raise_for_status()
                        requests.post(
                            base + "/reset_encoder_cache", timeout=180
                        ).raise_for_status()
                    requests.post(base + "/resume", timeout=180).raise_for_status()
                    evidence["warm_b_output"] = generate()
            finally:
                (args.output / "evidence.json").write_text(
                    json.dumps(evidence, indent=2)
                )
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait()

    assert evidence["cold_a"].keys() == evidence["warm_b"].keys()
    assert evidence["warm_b"].keys() == evidence["cold_b"].keys()
    assert (
        evidence["cold_a_parameters"].keys()
        == evidence["warm_b_parameters"].keys()
        == evidence["cold_b_parameters"].keys()
    )
    for name, warm in evidence["warm_b_parameters"].items():
        cold_a = evidence["cold_a_parameters"][name]
        cold_b = evidence["cold_b_parameters"][name]
        for key in ("shape", "dtype", "numel"):
            assert warm[key] == cold_b[key] == cold_a[key], (name, key)
        assert warm["hash"] == cold_b["hash"], name
        assert warm["ptr"] == cold_a["ptr"], name
    changed = False
    for name, warm in evidence["warm_b"].items():
        old, cold = evidence["cold_a"][name], evidence["cold_b"][name]
        assert warm["complete"] and not warm["staging"], name
        for key in ("method", "kernel", "config"):
            assert warm[key] == old[key], (name, key)
        assert warm["tensors"].keys() == cold["tensors"].keys() == old["tensors"].keys()
        for key, tensor in warm["tensors"].items():
            assert tensor["id"] == old["tensors"][key]["id"], (name, key)
            assert tensor["ptr"] == old["tensors"][key]["ptr"], (name, key)
            assert tensor["hash"] == cold["tensors"][key]["hash"], (name, key)
            changed |= tensor["hash"] != old["tensors"][key]["hash"]
    assert changed
    assert evidence["warm_b_output"] == evidence["cold_b_output"]
    result = {
        "status": "PASS",
        "backend": args.backend,
        "reload_mode": "trace",
        "layers": len(evidence["warm_b"]),
        "runtime_changed": changed,
        "warm_matches_cold": True,
    }
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
