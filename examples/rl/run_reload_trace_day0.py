# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold-A / warm-B / cold-B validation through production NCCL or IPC APIs.

Use a local day0-kit checkout at 152c2c0 with its publisher's legacy NCCL imports
redirected to day0_nccl_compat. IPC uses the same checkpoint reader and the
current native IPC trainer. Optional probes cover EPLB and fixed-order batches.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from contextlib import suppress
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
        freeze_engram_lookup=True,
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
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument(
        "--enable-eplb",
        action="store_true",
        help="Enable EPLB and collect expert placement evidence.",
    )
    parser.add_argument(
        "--swap-eplb-first-two",
        action="store_true",
        help="Swap the first two physical EPLB slots before warm-B reload.",
    )
    parser.add_argument(
        "--native-rearrange-cold-b",
        action="store_true",
        help="Move cold-B expert tensors to the swapped mapping with native EPLB.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--eplb-swap-slots", type=int, nargs=2, default=(0, 1))
    parser.add_argument(
        "--eplb-swap-pairs",
        type=int,
        nargs="+",
        default=None,
        help="Even list of physical slots to swap in one mapping update.",
    )
    parser.add_argument(
        "--expect-mapping-rejection",
        action="store_true",
        help="Treat an unsupported cross-rank mapping rejection as success.",
    )
    parser.add_argument(
        "--probe-eplb-reload-gate",
        action="store_true",
        help="Probe EPLB/reload serialization before the real reload.",
    )
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--publish-ipc", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--freeze-engram-lookup",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--batch-probe",
        action="store_true",
        help="Use one fixed prompt batch for cold-A and warm-B comparison.",
    )
    parser.add_argument(
        "--batch-max-tokens",
        type=int,
        default=16,
        help="Maximum generated tokens for the fixed batch probe.",
    )
    parser.add_argument(
        "--capture-forward",
        action="store_true",
        help="Capture hashes of language-model layer outputs for each batch.",
    )
    parser.add_argument(
        "--repeat-inference-control",
        action="store_true",
        help="Repeat warm-B and cold-B inference after cache reset, without reload.",
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
    forward_captures = {}

    def wait_for_port_release(timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    probe.bind(("127.0.0.1", args.port))
                except OSError:
                    time.sleep(1)
                    continue
                return
        raise TimeoutError(f"Port {args.port} was not released")

    def rank_results(response):
        results = response.json()["results"]
        gathered = results[0]
        expected = args.tensor_parallel_size * args.data_parallel_size
        assert set(gathered) == {str(rank) for rank in range(expected)}
        flattened = {}
        for rank, result in gathered.items():
            if isinstance(result, dict):
                flattened.update(
                    {f"rank{rank}:{name}": value for name, value in result.items()}
                )
            else:
                flattened[f"rank{rank}"] = result
        return flattened

    def inspect(arm=False):
        response = requests.post(
            base + "/collective_rpc",
            json={"method": "inspect_reload_trace", "args": [arm]},
            timeout=600,
        )
        response.raise_for_status()
        return rank_results(response)

    def inspect_parameters():
        response = requests.post(
            base + "/collective_rpc",
            json={"method": "inspect_model_parameters", "args": []},
            timeout=600,
        )
        response.raise_for_status()
        return rank_results(response)

    def inspect_workspace():
        response = requests.post(
            base + "/collective_rpc",
            json={"method": "inspect_workspace", "args": []},
            timeout=600,
        )
        response.raise_for_status()
        return rank_results(response)

    def inspect_eplb():
        response = requests.post(
            base + "/collective_rpc",
            json={"method": "inspect_eplb", "args": []},
            timeout=600,
        )
        response.raise_for_status()
        return rank_results(response)

    def inspect_expert_slots():
        response = requests.post(
            base + "/collective_rpc",
            json={"method": "inspect_expert_slots", "args": []},
            timeout=600,
        )
        response.raise_for_status()
        return rank_results(response)

    def set_eplb_mapping(mapping):
        response = requests.post(
            base + "/collective_rpc",
            json={"method": "set_eplb_mapping", "args": [mapping]},
            timeout=600,
        )
        response.raise_for_status()
        return rank_results(response)

    def swap_eplb_mapping():
        if args.eplb_swap_pairs is None:
            method = "swap_eplb_first_two"
            rpc_args = list(args.eplb_swap_slots)
        else:
            if not args.eplb_swap_pairs or len(args.eplb_swap_pairs) % 2:
                raise ValueError("--eplb-swap-pairs must contain an even number")
            method = "swap_eplb_pairs"
            rpc_args = [args.eplb_swap_pairs]
        response = requests.post(
            base + "/collective_rpc",
            json={"method": method, "args": rpc_args},
            timeout=600,
        )
        response.raise_for_status()
        return rank_results(response)

    def probe_eplb_reload_gate():
        response = requests.post(
            base + "/collective_rpc",
            json={"method": "probe_eplb_reload_gate", "args": []},
            timeout=600,
        )
        response.raise_for_status()
        return rank_results(response)

    def rearrange_eplb_mapping(mapping):
        response = requests.post(
            base + "/collective_rpc",
            json={"method": "rearrange_eplb_mapping", "args": [mapping]},
            timeout=1800,
        )
        response.raise_for_status()
        return rank_results(response)

    def inspect_memory():
        response = requests.post(
            base + "/collective_rpc",
            json={"method": "inspect_memory", "args": []},
            timeout=600,
        )
        response.raise_for_status()
        return rank_results(response)

    def capture_forward(label):
        if not args.capture_forward:
            return
        response = requests.post(
            base + "/collective_rpc",
            json={"method": "arm_forward_capture", "args": []},
            timeout=600,
        )
        response.raise_for_status()
        forward_captures[label] = None

    def finish_forward_capture(label):
        if not args.capture_forward:
            return
        response = requests.post(
            base + "/collective_rpc",
            json={"method": "read_forward_capture", "args": []},
            timeout=600,
        )
        response.raise_for_status()
        forward_captures[label] = rank_results(response)

    def reset_caches():
        for endpoint in ("reset_prefix_cache", "reset_encoder_cache"):
            requests.post(base + "/" + endpoint, timeout=180).raise_for_status()

    def repeat_inference(label):
        if not args.repeat_inference_control:
            return
        reset_caches()
        evidence[f"{label}_repeat_output"] = generate(f"{label}_repeat")
        if args.capture_forward:
            evidence[f"{label}_repeat_forward"] = {
                key: value
                for key, value in forward_captures.items()
                if key.startswith(f"{label}_repeat.")
            }

    def generate(label="output"):
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
                capture_forward(f"{label}.batch{start // 4}")
                response = requests.post(
                    base + "/v1/completions",
                    json={
                        "model": "experiment",
                        "prompt": prompts[start : start + 4],
                        "temperature": 0,
                        "top_k": 1,
                        "max_tokens": args.batch_max_tokens,
                        "logprobs": 5,
                        "seed": 0,
                    },
                    timeout=180,
                )
                response.raise_for_status()
                batch = sorted(response.json()["choices"], key=lambda x: x["index"])
                assert [item["index"] for item in batch] == list(range(4))
                choices.extend(
                    {
                        "prompt": prompt,
                        "text": item["text"],
                        "logprobs": item["logprobs"],
                    }
                    for prompt, item in zip(prompts[start : start + 4], batch)
                )
                finish_forward_capture(f"{label}.batch{start // 4}")
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
        wait_for_port_release()
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
                    "--tensor-parallel-size",
                    str(args.tensor_parallel_size),
                    "--data-parallel-size",
                    str(args.data_parallel_size),
                    *(
                        ["--enable-expert-parallel"]
                        if args.enable_expert_parallel
                        else []
                    ),
                    *(["--enable-eplb"] if args.enable_eplb else []),
                    "--gpu-memory-utilization",
                    str(args.gpu_memory_utilization),
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
                evidence[f"cold_{variant}_workspace"] = inspect_workspace()
                evidence[f"cold_{variant}_memory"] = inspect_memory()
                if args.enable_eplb:
                    evidence[f"cold_{variant}_eplb"] = inspect_eplb()
                    evidence[f"cold_{variant}_expert_slots"] = inspect_expert_slots()
                    if variant == "a" and args.probe_eplb_reload_gate:
                        evidence["eplb_reload_gate_probe"] = probe_eplb_reload_gate()
                    if (
                        variant == "b"
                        and args.native_rearrange_cold_b
                        and args.swap_eplb_first_two
                    ):
                        swapped_entries = evidence["eplb_after_swap"]["rank0"]
                        mapping = [
                            entry["physical_to_logical"] for entry in swapped_entries
                        ]
                        evidence["cold_b_eplb_before_native_rearrange"] = evidence[
                            "cold_b_eplb"
                        ]
                        evidence["cold_b_eplb"] = rearrange_eplb_mapping(mapping)
                        evidence["cold_b_expert_slots"] = inspect_expert_slots()
                evidence[f"cold_{variant}_output"] = generate(f"cold_{variant}")
                if args.capture_forward:
                    evidence[f"cold_{variant}_forward"] = {
                        key: forward_captures[key]
                        for key in forward_captures
                        if key.startswith(f"cold_{variant}.")
                    }
                if variant == "b":
                    repeat_inference("cold_b")
                if variant == "a":
                    # Clear request-side caches before pausing and receiving
                    # the replacement checkpoint. This is explicit here so
                    # reload tests do not rely only on pause's clear_cache
                    # implementation.
                    requests.post(
                        base + "/reset_prefix_cache", timeout=180
                    ).raise_for_status()
                    requests.post(
                        base + "/reset_encoder_cache", timeout=180
                    ).raise_for_status()
                    if args.swap_eplb_first_two:
                        evidence["eplb_before_swap"] = inspect_eplb()
                        try:
                            swapped = swap_eplb_mapping()
                        except requests.HTTPError as error:
                            if not args.expect_mapping_rejection:
                                raise
                            response = error.response
                            assert response is not None
                            evidence["mapping_rejection"] = {
                                "status": response.status_code,
                                "detail": response.text,
                            }
                            (args.output / "comparison.json").write_text(
                                json.dumps(
                                    {
                                        "status": "PASS",
                                        "mapping_change": "rejected",
                                        "reason": response.text,
                                    },
                                    indent=2,
                                )
                            )
                            return
                        evidence["eplb_after_swap"] = swapped
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
                            # Frozen lookup tables are excluded regardless of
                            # how inference prompts are batched.
                            "--freeze-engram-lookup",
                        ]
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
                    evidence["warm_b_workspace"] = inspect_workspace()
                    evidence["warm_b_memory"] = inspect_memory()
                    if args.enable_eplb:
                        evidence["warm_b_eplb"] = inspect_eplb()
                        evidence["warm_b_expert_slots"] = inspect_expert_slots()
                    if args.batch_probe:
                        requests.post(
                            base + "/reset_prefix_cache", timeout=180
                        ).raise_for_status()
                        requests.post(
                            base + "/reset_encoder_cache", timeout=180
                        ).raise_for_status()
                    requests.post(base + "/resume", timeout=180).raise_for_status()
                    evidence["warm_b_output"] = generate("warm_b")
                    if args.capture_forward:
                        evidence["warm_b_forward"] = {
                            key: forward_captures[key]
                            for key in forward_captures
                            if key.startswith("warm_b.")
                        }
                    repeat_inference("warm_b")
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
                if variant == "a":
                    wait_for_port_release()

    (args.output / "runner_complete.json").write_text(
        json.dumps({"keys": sorted(evidence)}, indent=2)
    )
    if args.expect_mapping_rejection:
        raise AssertionError("Expected the mapping change to be rejected")
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
        if "routed_experts." not in name:
            assert warm["hash"] == cold_b["hash"], name
        assert warm["ptr"] == cold_a["ptr"], name
    reference_slots = {}
    for rank_key, entries in evidence.get("cold_b_expert_slots", {}).items():
        rank = int(rank_key.removeprefix("rank").split(":", 1)[0])
        for entry in entries:
            count = entry["local_num_experts"]
            placement = entry["physical_to_logical"][rank * count : (rank + 1) * count]
            for role, data in entry["slot_hashes"].items():
                for index, logical in enumerate(placement):
                    slot_key = (entry["module"], role, logical)
                    digest = data["slots"][str(index)]
                    assert reference_slots.setdefault(slot_key, digest) == digest
    for key, warm_entries in evidence.get("warm_b_expert_slots", {}).items():
        for warm_entry in warm_entries:
            rank = int(key.removeprefix("rank").split(":", 1)[0])
            num_local_slots = warm_entry["local_num_experts"]
            start = rank * num_local_slots
            warm_map = dict(
                enumerate(
                    warm_entry["physical_to_logical"][start : start + num_local_slots]
                )
            )
            for role, warm_data in warm_entry["slot_hashes"].items():
                for index, logical in warm_map.items():
                    assert (
                        warm_data["slots"][str(index)]
                        == reference_slots[(warm_entry["module"], role, logical)]
                    ), (key, warm_entry["module"], role, index)
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
            # Routed expert storage is physically reordered by EPLB. Its
            # logical-slot hashes are checked above; comparing raw layouts
            # here would reject a correct mapping change.
            if "routed_experts" not in name:
                assert tensor["hash"] == cold["tensors"][key]["hash"], (
                    name,
                    key,
                )
            changed |= tensor["hash"] != old["tensors"][key]["hash"]
    assert changed
    output_comparison = {
        "warm_matches_cold": evidence["warm_b_output"] == evidence["cold_b_output"],
        "prompts": [
            {
                "index": index,
                "text_equal": warm["text"] == cold["text"],
                "tokens_equal": warm["logprobs"]["tokens"]
                == cold["logprobs"]["tokens"],
                "logprobs_equal": warm["logprobs"] == cold["logprobs"],
            }
            for index, (warm, cold) in enumerate(
                zip(evidence["warm_b_output"], evidence["cold_b_output"], strict=True)
            )
        ],
    }
    for label in ("warm_b", "cold_b"):
        if f"{label}_repeat_output" in evidence:
            output_comparison[f"{label}_repeat_equal"] = (
                evidence[f"{label}_output"] == evidence[f"{label}_repeat_output"]
            )
    (args.output / "output_comparison.json").write_text(
        json.dumps(output_comparison, indent=2)
    )
    assert evidence["warm_b_output"] == evidence["cold_b_output"]
    result = {
        "status": "PASS",
        "backend": args.backend,
        "reload_mode": "trace",
        "tensor_parallel_size": args.tensor_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "expert_parallel": args.enable_expert_parallel,
        "layers": len(evidence["warm_b"]),
        "runtime_changed": changed,
        "warm_matches_cold": True,
    }
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        output = None
        if len(sys.argv) > 1:
            with suppress(ValueError, IndexError):
                output = Path(sys.argv[sys.argv.index("--output") + 1])
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            (output / "runner_error.log").write_text(error)
        raise
