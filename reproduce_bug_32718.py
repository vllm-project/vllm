#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce vLLM issue #32718: reload_weights returns cached weights."""

import argparse
import os
import shutil
import subprocess
import sys
import time

import httpx
import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()

    # Get model cache
    from huggingface_hub import snapshot_download

    hf_folder = snapshot_download(args.model)
    weight_files = [
        os.path.join(hf_folder, f)
        for f in os.listdir(hf_folder)
        if f.endswith((".bin", ".safetensors"))
    ]
    is_safetensors = weight_files[0].endswith(".safetensors")

    # Read original weights from disk
    state = (
        load_file(weight_files[0])
        if is_safetensors
        else torch.load(weight_files[0], map_location="cpu", weights_only=True)
    )
    param = list(state.keys())[0]
    disk_original_mean = state[param].mean().item()

    # Start server
    dp_size = min(torch.cuda.device_count(), 4) if torch.cuda.device_count() >= 2 else 1
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--port",
        "8000",
        "--max-model-len",
        "512",
        "--disable-log-requests",
        "--enable-sleep-mode",
        "--worker-extension-cls",
        "test_worker_extension.WeightInspectorExtension",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if dp_size > 1:
        cmd.extend(["--data-parallel-size", str(dp_size)])

    env = os.environ.copy()
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
    )
    client = httpx.Client(timeout=httpx.Timeout(300.0))

    for i in range(120):
        try:
            if client.get("http://localhost:8000/health").status_code == 200:
                break
        except Exception:
            pass
        if process.poll():
            sys.exit("Server died")
        time.sleep(1)

    def rpc(method, params=None):
        resp = client.post(
            "http://localhost:8000/collective_rpc",
            json={"method": method, "params": params or {}},
        )
        if resp.status_code != 200:
            return None
        if method == "reload_weights":
            return True
        data = resp.json()
        return (
            data["results"][0]
            if "results" in data
            else (data[0] if isinstance(data, list) else data)
        )

    try:
        # Get loaded weights before modification
        loaded_original = rpc("get_weight_stats", {"param_name": param})
        loaded_original_mean = loaded_original["mean"]
        loaded_original_first5 = loaded_original["first_5"]

        # Modify weights on disk
        backups = []
        for wf in weight_files:
            backup = wf + ".backup"
            shutil.copy2(wf, backup)
            backups.append(backup)
            state = (
                load_file(wf)
                if is_safetensors
                else torch.load(wf, map_location="cpu", weights_only=True)
            )
            modified = {k: torch.randn_like(v) * 0.1 + 5.0 for k, v in state.items()}
            save_file(modified, wf) if is_safetensors else torch.save(modified, wf)

        # Read modified weights from disk
        state = (
            load_file(weight_files[0])
            if is_safetensors
            else torch.load(weight_files[0], map_location="cpu", weights_only=True)
        )
        disk_modified_mean = state[param].mean().item()

        try:
            # Reload and check loaded weights
            rpc("reload_weights")
            loaded_reloaded = rpc("get_weight_stats", {"param_name": param})
            loaded_reloaded_mean = loaded_reloaded["mean"]
            loaded_reloaded_first5 = loaded_reloaded["first_5"]

            # Results
            diff_disk = abs(loaded_reloaded_mean - disk_modified_mean)
            diff_orig = abs(loaded_reloaded_mean - disk_original_mean)

            print("\n=== RESULTS ===")
            print(f"Disk original:         {disk_original_mean:.6f}")
            print(f"Disk modified:         {disk_modified_mean:.6f}")
            print(f"Loaded original (RPC): {loaded_original_mean:.6f}")
            print(f"Loaded reloaded (RPC): {loaded_reloaded_mean:.6f}\n")
            first5_before = [f"{v:.3f}" for v in loaded_original_first5]
            first5_after = [f"{v:.3f}" for v in loaded_reloaded_first5]
            print(f"First 5 before reload: {first5_before}")
            print(f"First 5 after reload:  {first5_after}\n")

            if diff_orig < 0.1 and diff_disk > 1.0:
                print("❌ BUG #32718: Loaded weights match ORIGINAL!\n")
                return 1
            elif diff_disk < 0.1:
                print("✓ NO BUG: Loaded weights match modified disk\n")
                return 0
            else:
                print("⚠ UNEXPECTED\n")
                return 1
        finally:
            for wf, backup in zip(weight_files, backups):
                shutil.copy2(backup, wf)
                os.remove(backup)
    finally:
        process.terminate()
        process.wait(timeout=10) if process.poll() is None else None
        client.close()


if __name__ == "__main__":
    sys.exit(main())
