# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared ROCm P/D accuracy harness for Mooncake and NIXL integration tests.

Keep the models, TP pairs, full GSM8K workload, and acceptance checks of the
Mooncake and FlashInfer/NIXL sweeps. Mooncake uses same-host HIP IPC; NIXL uses
AITER unified attention. These cases do not exercise RDMA or FlashInfer.
"""

import contextlib
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
from prometheus_client.parser import text_string_to_metric_families

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

_ROOT = Path(__file__).resolve().parents[3]
QWEN_MODEL = "Qwen/Qwen3-0.6B"
DEEPSEEK_MODEL = "deepseek-ai/deepseek-vl2-tiny"
_REVISIONS = {
    QWEN_MODEL: "c1899de289a04d12100db370d81485cdf75e47ca",
    DEEPSEEK_MODEL: "66c54660eae7e90c9ba259bfdf92d07d6e3ce8aa",
}


def _metric_total(server, name: str, **labels: str) -> float:
    response = requests.get(server.url_for("metrics"), timeout=10)
    response.raise_for_status()
    return sum(
        sample.value
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
        if sample.name == name
        and all(sample.labels.get(key) == value for key, value in labels.items())
    )


@contextlib.contextmanager
def _proxy(connector: str, prefill, decode, bootstrap_port: int):
    port = get_open_port()
    if connector == "mooncake":
        script = (
            _ROOT / "examples/disaggregated/mooncake_connector/"
            "mooncake_connector_proxy.py"
        )
        args = [
            "--prefill",
            prefill.url_root,
            str(bootstrap_port),
            "--decode",
            decode.url_root,
        ]
    else:
        script = Path(__file__).parent / "nixl_integration/toy_proxy_server.py"
        args = [
            "--prefiller-hosts",
            "127.0.0.1",
            "--prefiller-ports",
            str(prefill.port),
            "--decoder-hosts",
            "127.0.0.1",
            "--decoder-ports",
            str(decode.port),
        ]
    process = subprocess.Popen(
        [sys.executable, str(script), "--port", str(port), *args],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            assert process.poll() is None, "P/D proxy exited during startup"
            try:
                requests.get(f"http://127.0.0.1:{port}/", timeout=1)
                break
            except requests.ConnectionError:
                time.sleep(0.2)
        else:
            raise TimeoutError("P/D proxy did not start")
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


def run_rocm_pd_accuracy(connector, model, prefill_tp, decode_tp, monkeypatch):
    """Transferred KV must feed decoding while full upstream accuracy holds."""
    # Preserve the caller's allocation instead of querying host-wide SMI indices.
    selectors = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    selector = next((key for key in selectors if key in os.environ), selectors[0])
    visible = os.getenv(selector)
    devices = (
        visible.split(",")
        if visible is not None
        else [str(i) for i in range(current_platform.device_count())]
    )
    assert len(devices) >= prefill_tp + decode_tp, "Insufficient allocated GPUs"
    for key in selectors:
        if key != selector:
            monkeypatch.delenv(key, raising=False)

    if connector == "mooncake":
        import mooncake.engine

        assert mooncake.engine.SUPPORT_HIP, "Mooncake requires its ROCm/HIP build"
        connector_name = "MooncakeConnector"
        extra = {"mooncake_protocol": "hip", "device_name": ""}
    else:
        from vllm.distributed.nixl_utils import NixlWrapper

        assert NixlWrapper is not None, "NIXL ROCm bindings must import"
        connector_name = "NixlConnector"
        extra = {}

    common_args = [
        "--revision",
        _REVISIONS[model],
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--distributed-executor-backend",
        "mp",
        "--block-size",
        "128",
        "--max-num-seqs",
        "128",
        "--max-num-batched-tokens",
        "4096",
        "--gpu-memory-utilization",
        "0.2",
        "--kv-cache-memory-bytes",
        str(8 * 1024**3),
        "--moe-backend",
        "triton",
        "--attention-backend",
        "ROCM_AITER_UNIFIED_ATTN",
    ]
    common_env = {
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "VLLM_HOST_IP": "127.0.0.1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_ROCM_USE_AITER": "1",
        "VLLM_KV_CACHE_LAYOUT": "HND",
    }
    bootstrap_port = get_open_port()
    with contextlib.ExitStack() as stack:
        servers: list[RemoteOpenAIServer] = []
        stack.callback(RemoteOpenAIServer.shutdown_many, servers)
        for role, tp, assigned in (
            ("kv_producer", prefill_tp, devices[:prefill_tp]),
            ("kv_consumer", decode_tp, devices[prefill_tp : prefill_tp + decode_tp]),
        ):
            config = {
                "kv_connector": connector_name,
                "kv_role": role,
                "kv_connector_extra_config": extra,
            }
            env = {
                **common_env,
                selector: ",".join(assigned),
                "VLLM_PORT": str(get_open_port()),
                "VLLM_NIXL_SIDE_CHANNEL_PORT": str(get_open_port()),
                "VLLM_MOONCAKE_BOOTSTRAP_PORT": str(bootstrap_port),
            }
            servers.append(
                RemoteOpenAIServer(
                    model,
                    [
                        *common_args,
                        "--tensor-parallel-size",
                        str(tp),
                        "--kv-transfer-config",
                        json.dumps(config),
                    ],
                    env_dict=env,
                    max_wait_seconds=1200,
                )
            )
        prefill, decode = servers
        proxy_url = stack.enter_context(
            _proxy(connector, prefill, decode, bootstrap_port)
        )

        # Reuse the full upstream evaluation and its model-specific assertions.
        accuracy = importlib.import_module(
            f"tests.v1.kv_connector.{connector}_integration.test_accuracy"
        )
        monkeypatch.setattr(accuracy, "MODEL_NAME", model)
        monkeypatch.setattr(accuracy, "BASE_URL", proxy_url)
        assert model in accuracy.EXPECTED_VALUES

        def transfer_totals():
            totals = {
                "external_tokens": _metric_total(
                    decode,
                    "vllm:prompt_tokens_by_source_total",
                    source="external_kv_transfer",
                )
            }
            if connector == "nixl":
                totals["nixl_bytes"] = _metric_total(
                    decode, "vllm:nixl_bytes_transferred_sum"
                )
            return totals

        simple_evaluate = accuracy.lm_eval.simple_evaluate

        def evaluate_with_transfer_checks(*args, **kwargs):
            # The upstream helper runs a smoke prompt before GSM8K. Wait until
            # its counters appear, then require new transfers during evaluation.
            deadline = time.monotonic() + 15
            before = transfer_totals()
            while not all(value > 0 for value in before.values()):
                assert time.monotonic() < deadline, (
                    f"Smoke prompt did not transfer KV: {before}"
                )
                time.sleep(1)
                before = transfer_totals()

            # The API adapter loads a local tokenizer because these proxies do
            # not expose /tokenize. Match it to the server's pinned checkpoint.
            kwargs["model_args"] += f",revision={_REVISIONS[model]}"
            results = simple_evaluate(*args, **kwargs)
            deadline = time.monotonic() + 15
            after = transfer_totals()
            while not all(after[key] > value for key, value in before.items()):
                assert time.monotonic() < deadline, (
                    f"GSM8K did not transfer KV: before={before}, after={after}"
                )
                time.sleep(1)
                after = transfer_totals()
            print(f"GSM8K transfer counters: before={before}, after={after}")
            return results

        monkeypatch.setattr(
            accuracy.lm_eval, "simple_evaluate", evaluate_with_transfer_checks
        )
        accuracy.test_accuracy()
