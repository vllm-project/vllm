# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real four-GPU 2E2PD SSD serving test; requires a dedicated external Store.

MODEL=/local/model MOONCAKE_CONFIG_PATH=/path/store.json \
    .venv/bin/python -m pytest -v -s \
    tests/v1/ec_connector/integration/test_cross_encoder_store_ssd.py

Start an empty standalone Store with eager SSD offload, promotion disabled,
and a small owner RAM pool (e.g. 256 MiB). For low-volume correctness tests,
set MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT=1 on the owner. Enable the master admin
HTTP endpoint and set SSD_E2E_MASTER_ADMIN_URL if it is not localhost:9003.
No native Store calls are mocked.
"""

import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import pybase64 as base64
import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm.utils.network_utils import get_open_port


def _rpc(server, method):
    response = requests.post(
        server.url_for("collective_rpc"),
        json={"method": method},
        timeout=10 if method == "ssd_test_state" else 180,
    )
    assert response.ok, (
        f"collective_rpc {method}: HTTP {response.status_code}: {response.text}"
    )
    results = response.json()["results"]
    assert len(results) == 1, "This test requires TP1"
    return results[0]


def _wait(fetch, predicate, timeout=120, *, phase="service readiness", evidence=None):
    start = time.monotonic()
    deadline = start + timeout
    report_at = start
    print(f"[SSD E2E] {phase}: begin; timeout={timeout}s", flush=True)
    while True:
        result = fetch()
        if evidence is not None:
            evidence.write_text(json.dumps({"phase": phase, "state": result}, indent=2))
        now = time.monotonic()
        if now >= report_at:
            summary = result
            if isinstance(result, dict):
                summary = {
                    "encoded_items": result.get("encoded_items"),
                    "pending_puts": result.get("pending_puts"),
                    "published": len(result.get("published", {})),
                    "replicas": list(result.get("replicas", {}).values()),
                    "put_errors": result.get("put_errors", {}),
                }
            print(f"[SSD E2E] {phase}: {now - start:.1f}s {summary}", flush=True)
            report_at = now + 5
        if isinstance(result, dict) and result.get("put_errors"):
            pytest.fail(f"{phase}: Store PUT failed: {result['put_errors']}")
        if predicate(result):
            print(f"[SSD E2E] {phase}: complete in {now - start:.1f}s", flush=True)
            return result
        if now >= deadline:
            pytest.fail(
                f"{phase}: condition not reached; last observed state: {result}"
            )
        time.sleep(min(1, deadline - now))


@contextmanager
def _proxy(encoder, pd, control, tmp_path, index):
    source = Path(__file__).resolve().parents[4]
    port = get_open_port()
    script = source / "examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py"
    command = [
        sys.executable,
        str(script),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--encode-servers-urls",
        encoder.url_root,
        "--prefill-servers-urls",
        "disable",
        "--decode-servers-urls",
        pd.url_root,
        "--ec-consumer-zmq-addrs",
        control,
    ]
    url = f"http://127.0.0.1:{port}"
    with (tmp_path / f"proxy{index}.log").open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:

        def healthy():
            assert process.poll() is None, f"Proxy exited; inspect {tmp_path}"
            try:
                return requests.get(url + "/health", timeout=2).status_code == 200
            except requests.ConnectionError:
                return False

        _wait(healthy, bool)
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def test_2e2pd_disk_only_hit_skips_encoder_and_preserves_output(tmp_path):
    """E0 computes; after real eviction E1 restores the exact bytes without encoding."""
    model = os.environ.get("MODEL")
    config_path = os.environ.get("MOONCAKE_CONFIG_PATH")
    assert model and Path(model).is_dir(), "Set MODEL to local immutable VL weights"
    assert config_path, "Start a dedicated SSD Store and set MOONCAKE_CONFIG_PATH"
    config = json.loads(Path(config_path).read_text())
    assert (
        config.get("mode") == "standalone-store"
        and config.get("enable_offload") is True
    )
    gpus = os.environ.get("SSD_E2E_GPUS", "0,1,2,3").split(",")
    assert len(gpus) == len(set(gpus)) == 4
    namespace = "ssd-e2e-" + uuid.uuid4().hex
    common = [
        "--disable-uvicorn-access-log",
        "--tensor-parallel-size",
        "1",
        "--enforce-eager",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "8192",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb",
        "1",
        "--mm-tensor-ipc",
        "torch_shm",
        "--limit-mm-per-prompt",
        '{"image":1,"video":0}',
    ]
    controls, pds, encoders = [], [], []
    with ExitStack() as stack:
        for index in range(4):
            producer = index >= 2
            port = get_open_port()
            extra: dict[str, Any] = {"mooncake_protocol": "tcp"}
            if producer:
                extra.update(
                    {
                        "cross_encoder_cache": True,
                        "embedding_cache_prefix": namespace,
                        "store_read_buffer_bytes": 268435456,
                        "store_disk_staging_budget_bytes": 268435456,
                    }
                )
            ec = {
                "ec_connector": "ECMooncakeConnector",
                "ec_role": "ec_producer" if producer else "ec_consumer",
                "ec_ip": "127.0.0.1",
                "ec_port": port,
                "ec_connector_extra_config": extra,
            }
            args = common + [
                "--ec-transfer-config",
                json.dumps(ec),
                "--gpu-memory-utilization",
                "0.45" if producer else "0.8",
            ]
            if producer:
                args += [
                    "--compilation-config",
                    '{"cudagraph_mm_encoder":false}',
                    "--worker-extension-cls",
                    "tests.v1.ec_connector.integration.ssd_test_worker.SSDTestWorkerExtension",
                ]
            else:
                args += ["--enable-mm-embeds"]
            server = stack.enter_context(
                RemoteOpenAIServer(
                    model,
                    args,
                    env_dict={
                        "CUDA_VISIBLE_DEVICES": gpus[
                            index - 2 if producer else index + 2
                        ],
                        "VLLM_USE_V2_MODEL_RUNNER": "1",
                        "VLLM_SERVER_DEV_MODE": "1",
                        "VLLM_BATCH_INVARIANT": "1",
                        "MC_FORCE_TCP": "1",
                        "MOONCAKE_CONFIG_PATH": str(Path(config_path).resolve()),
                    },
                    max_wait_seconds=1200,
                )
            )
            if producer:
                encoders.append(server)
                assert _rpc(server, "ssd_test_arm")
            else:
                pds.append(server)
                controls.append(f"tcp://127.0.0.1:{port}")
        routes = [
            stack.enter_context(_proxy(encoders[i], pds[i], controls[i], tmp_path, i))
            for i in range(2)
        ]
        image = base64.b64encode(
            (Path(__file__).parent / "hato.jpg").read_bytes()
        ).decode()
        payload = {
            "model": model,
            "temperature": 0,
            "max_tokens": 32,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + image},
                        },
                        {"type": "text", "text": "Describe the image briefly."},
                    ],
                }
            ],
        }

        def request(url):
            response = requests.post(
                url + "/v1/chat/completions", json=payload, timeout=600
            )
            assert response.ok, (
                f"HTTP {response.status_code} from {url}: {response.text}"
            )
            return response.json()

        print("[SSD E2E] E0 cold request: begin", flush=True)
        cold = request(routes[0])
        seeded = _wait(
            lambda: _rpc(encoders[0], "ssd_test_state"),
            lambda state: state["pending_puts"] == 0,
            phase="E0 asynchronous Store publication",
            evidence=tmp_path / "publication-state.json",
        )
        assert seeded["encoded_items"] > 0 and not seeded["loaded"]
        assert seeded["published"], f"E0 did not publish any Store object: {seeded}"
        receiver = _rpc(encoders[1], "ssd_test_state")
        for key, sizes in seeded["published_sizes"].items():
            assert sizes["object_bytes"] <= receiver["read_buffer_bytes"], (
                f"Object exceeds E1 read buffer: {key}: {sizes}; limits={receiver}"
            )
            assert (
                sizes["disk_staging_bytes"] <= receiver["disk_staging_budget_bytes"]
            ), f"Object exceeds E1 disk allowance: {key}: {sizes}; limits={receiver}"
            print(f"[SSD E2E] object sizes: {sizes}", flush=True)
        keys = set(seeded["published"])
        _wait(
            lambda: _rpc(encoders[0], "ssd_test_state"),
            lambda state: all("local_disk" in state["replicas"][key] for key in keys),
            phase="Store LOCAL_DISK persistence (no read-lease refresh)",
            evidence=tmp_path / "persistence-state.json",
        )
        pressure_limit = int(os.environ.get("SSD_E2E_PRESSURE_MAX_BYTES", str(1 << 30)))
        written = 0
        while True:
            state = _rpc(encoders[0], "ssd_test_state")
            if all(
                state["replicas"][key] and set(state["replicas"][key]) == {"local_disk"}
                for key in keys
            ):
                break
            assert written + (16 << 20) <= pressure_limit, (
                f"No disk-only eviction within pressure limit: {state}"
            )
            outcome = _rpc(encoders[0], "ssd_test_pressure")
            written += outcome["written_bytes"]
            print(f"[SSD E2E] eviction pressure: {written >> 20} MiB", flush=True)
            if outcome.get("rejection"):
                print(
                    f"[SSD E2E] pressure rejected: {outcome['rejection']}", flush=True
                )
                state = _wait(
                    lambda: _rpc(encoders[0], "ssd_test_state"),
                    lambda current: all(
                        current["replicas"][key]
                        and set(current["replicas"][key]) == {"local_disk"}
                        for key in keys
                    ),
                    phase="Store eviction after pressure rejection",
                    evidence=tmp_path / "eviction-state.json",
                )
                break
            (tmp_path / "eviction-state.json").write_text(
                json.dumps({"pressure_bytes": written, "state": state}, indent=2)
            )
        before = _rpc(encoders[1], "ssd_test_state")
        assert before["encoded_items"] == 0 and not before["loaded"]
        print("[SSD E2E] E1 disk-only restore request: begin", flush=True)
        warm = request(routes[1])
        restored = _rpc(encoders[1], "ssd_test_state")
        (tmp_path / "evidence.json").write_text(
            json.dumps(
                {
                    "seeded": seeded,
                    "disk_only": state,
                    "restored": restored,
                    "pressure_bytes": written,
                    "cold": cold,
                    "warm": warm,
                },
                indent=2,
            )
        )
        assert restored["encoded_items"] == 0, "E1 recomputed instead of using SSD"
        assert restored["loaded"] == seeded["published"], (
            "Restored embeddings differ byte-for-byte"
        )
        assert all(
            restored["get_replicas"][key]
            and set(restored["get_replicas"][key]) == {"local_disk"}
            for key in keys
        )
        assert warm["choices"][0]["message"] == cold["choices"][0]["message"]
        assert warm["usage"]["completion_tokens"] == cold["usage"]["completion_tokens"]
