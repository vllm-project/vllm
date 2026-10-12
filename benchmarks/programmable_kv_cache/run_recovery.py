# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Own real servers to validate cross-engine reuse, Store outage and restart."""

import argparse
import asyncio
import gzip
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

from benchmarks.programmable_kv_cache.client import envelope
from benchmarks.programmable_kv_cache.rollout import RolloutClient
from benchmarks.programmable_kv_cache.run_rollout_matrix import (
    require_free_port,
    source_hashes,
    stop,
)


async def recover(args, report, restart, stop_store):
    base = f"http://127.0.0.1:{args.port}"
    client = RolloutClient(base, "priority")
    replacement = None
    report["requests"] = client.rows
    report["controls"] = client.controls
    try:
        await client.initialize()
        report["before"] = client.invocation
        with gzip.open(args.trace, "rt") as file:
            prompt = json.load(file)["requests"][0]["prompt_ids"]
        prompt = prompt[: client.boundary + 128]
        original = await client.generate(
            "before-restart", prompt, {"temperature": 0, "max_tokens": 16}
        )
        logical = client.targets["before-restart"]
        resident = await client.call("residency", {"target": logical})
        target = {
            "kind": "hashes",
            "namespace": client.invocation["namespace"],
            "blocks": [
                {
                    k: row[k]
                    for k in ("block_hash", "group_id", "start_token", "end_token")
                }
                for row in resident["items"]
                if row["resident_copies"]
            ],
        }
        assert target["blocks"]
        stored = await client.action(
            "vllm.copy_to",
            {"target": target, "src": {"tier": "gpu"}, "dst": {"tier": "mooncake"}},
        )
        assert stored["status"] == "completed", stored
        evicted = await client.action("evict", {"target": target}, version="1")
        assert evicted["status"] == "completed", evicted
        before = await client.call("capacity", {})
        stop_store()
        message = uuid.uuid4().hex
        await client.call(
            "submit",
            envelope(
                message,
                "vllm.copy_to",
                {"target": target, "src": {"tier": "mooncake"}, "dst": {"tier": "gpu"}},
            ),
        )
        failed = await client.wait(message, timeout=120)
        report["outage_result"] = failed
        assert failed["status"] == "failed", failed
        after = await client.call("capacity", {})
        report["outage_capacity"] = after
        assert (
            after["transfers"]["failed_transfers"]
            > before["transfers"]["failed_transfers"]
        ), after
        resident = await client.call("residency", {"target": target})
        assert all(
            not r["resident_copies"] and not r["loading"] for r in resident["items"]
        ), resident
        fallback = await client.generate(
            "outage-fallback", prompt, {"temperature": 0, "max_tokens": 16}
        )
        assert fallback["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        assert fallback["token_ids"] == original["token_ids"]
        report["saved_associations"] = await client.call("associations", {})
        assert report["saved_associations"]["items"]
        await client.http.aclose()
        restart()
        replacement = RolloutClient(base, "priority")
        replacement.run_id = client.run_id
        await replacement.initialize()
        report["after"] = replacement.invocation
        assert replacement.invocation["engine_id"] == client.invocation["engine_id"]
        assert (
            replacement.invocation["engine_epoch"] != client.invocation["engine_epoch"]
        )
        stale = await replacement.http.post(
            "/v1/kv/control",
            json={
                "method": "associations",
                "invocation": client.invocation,
                "payload": {},
            },
        )
        report["stale_epoch"] = stale.json()
        assert stale.status_code == 409
        assert stale.json()["error"]["code"] == "stale_epoch"
        fresh = await replacement.call("associations", {})
        report["fresh_snapshot"] = fresh
        assert not fresh["items"]
        missing = await replacement.action(
            "vllm.copy_to",
            {"target": target, "src": {"tier": "mooncake"}, "dst": {"tier": "gpu"}},
        )
        assert missing["status"] == "missing", missing
        rebuilt = await replacement.generate(
            "before-restart", prompt, {"temperature": 0, "max_tokens": 16}
        )
        assert rebuilt["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        assert rebuilt["token_ids"] == original["token_ids"]
        report["reconstructed"] = await replacement.call("associations", {})
        assert any(
            row["session_id"] == logical["session_id"]
            for row in report["reconstructed"]["items"]
        )
        report["restart_requests"] = replacement.rows
        report["restart_controls"] = replacement.controls
    finally:
        await client.http.aclose()
        if replacement is not None:
            await replacement.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,3")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--store-port", type=int, default=19151)
    parser.add_argument("--evaluation-samples", type=int, default=0)
    args = parser.parse_args()
    for port in (args.port, args.port + 1, args.store_port, args.store_port + 2):
        require_free_port(port)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = args.output_dir.resolve() / "mooncake-config.json"
    config.write_text(
        json.dumps(
            {
                "metadata_server": "P2PHANDSHAKE",
                "master_server_address": f"127.0.0.1:{args.store_port}",
                "protocol": "tcp",
                "device_name": "",
                "global_segment_size": "4GB",
                "local_buffer_size": "256MB",
            }
        )
    )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=args.gpus,
        OMP_NUM_THREADS="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_SERVER_DEV_MODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        MOONCAKE_CONFIG_PATH=str(config),
    )
    report = {"passed": False, "commands": [], "source_hashes": source_hashes()}
    owned = []

    def launch(command, label):
        report["commands"].append(command)
        with (args.output_dir / f"{label}.log").open("w") as log:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        owned.append(process)
        return process

    def store(label):
        return launch(
            [
                str(Path(sys.executable).parent / "mooncake_master"),
                "--rpc_address=127.0.0.1",
                f"--rpc_port={args.store_port}",
                "--metrics_host=127.0.0.1",
                f"--metrics_port={args.store_port + 2}",
                "--enable_metric_reporting=false",
            ],
            label,
        )

    def server(port, engine, label):
        settings = {
            "engine_id": engine,
            "policy_decode_lookahead_tokens": 512,
            "max_policy_wait_seconds": 30,
            "max_inflight_load_blocks": 32,
            "policies": [
                {
                    "name": "quota",
                    "implementation": "quota",
                    "options": {
                        "reserve_running_growth": True,
                        "borrow_idle_capacity": True,
                        "reuse_score_weight": 0.5,
                        "max_wait_seconds": 30,
                    },
                }
            ],
        }
        connector = {
            "kv_connector": "MooncakeStoreConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "enable_lookup": False,
                "enable_save": False,
                "save_decode_cache": False,
            },
        }
        process = launch(
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                str(args.model),
                "--served-model-name",
                "qwen-rollout",
                "--tensor-parallel-size",
                "2",
                "--max-model-len",
                "4096",
                "--max-num-seqs",
                "8",
                "--max-num-batched-tokens",
                "2048",
                "--kv-cache-memory-bytes",
                str(2 << 30),
                "--gpu-memory-utilization",
                "0.45",
                "--mamba-cache-mode",
                "align",
                "--dtype",
                "bfloat16",
                "--enable-prefix-caching",
                "--enable-prompt-tokens-details",
                "--enforce-eager",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--disable-uvicorn-access-log",
                "--kv-control-config",
                json.dumps(settings),
                "--kv-transfer-config",
                json.dumps(connector),
            ],
            label,
        )
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"{label} exited: {process.returncode}")
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=3).ok:
                    return process
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError(f"{label} startup")

    def http_check(label, extra):
        command = [
            sys.executable,
            "-m",
            "benchmarks.programmable_kv_cache.http_control",
            "--base-url",
            f"http://127.0.0.1:{args.port}",
            "--dataset",
            str(args.dataset),
            "--primitive-actions",
            "--validate-movement",
            "mooncake",
            "--output",
            str(args.output_dir / f"{label}.json"),
            *extra,
        ]
        process = launch(command, label)
        if process.wait() != 0:
            raise RuntimeError(f"{label} failed; inspect its log")

    try:
        master = store("store")
        source = server(args.port, "recovery-source", "source")
        destination = server(args.port + 1, "recovery-destination", "destination")
        http_check(
            "cross-engine", ["--destination-url", f"http://127.0.0.1:{args.port + 1}"]
        )
        stop(destination)
        if args.evaluation_samples:
            http_check(
                "model-quality", ["--evaluation-samples", str(args.evaluation_samples)]
            )

        def restart():
            stop(source)
            store("restarted-store")
            server(args.port, "recovery-source", "restarted-source")

        asyncio.run(recover(args, report, restart, lambda: stop(master)))
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for process in reversed(owned):
            if process.poll() is None:
                stop(process)
        with gzip.open(args.output_dir / "recovery.json.gz", "wt") as file:
            json.dump(report, file)


if __name__ == "__main__":
    main()
