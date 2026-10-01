# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-transfer metrics scraped from a real two-API-server deployment."""

from pathlib import Path

import requests
import torch
from prometheus_client.multiprocess import MultiProcessCollector
from prometheus_client.parser import text_string_to_metric_families
from transformers import AutoModelForCausalLM

from tests.entrypoints.serve.dev.rlhf.conftest import MODEL_NAME
from tests.utils import RemoteOpenAIServer
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    ModuleSource,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

DURATION = "vllm:rl_weight_update_operation_duration_seconds"
IN_FLIGHT = "vllm:rl_weight_update_operations_in_flight"
# The API servers accept from one shared listening socket, so the kernel may
# hand one process most connections. Sync until both have recorded, within a cap.
MIN_SYNCS = 8
MAX_SYNCS = 64


def scrape(url: str) -> dict[tuple[str, str], float]:
    text = requests.get(f"{url}/metrics", timeout=30).text
    return {
        (sample.name, sample.labels["operation"]): sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name in (f"{DURATION}_count", IN_FLIGHT)
    }


def pids_that_recorded(multiproc_dir: Path) -> set[str]:
    return {
        path.stem.rsplit("_", 1)[1]
        for path in multiproc_dir.glob("histogram_*.db")
        if any(
            metric.name == DURATION
            for metric in MultiProcessCollector.merge([str(path)])
        )
    }


def test_weight_sync_metrics_aggregate_across_api_servers(tmp_path):
    args = [
        "--load-format",
        "dummy",
        "--enforce-eager",
        "--max-model-len",
        "512",
        "--gpu-memory-utilization",
        "0.5",
        "--api-server-count",
        "2",
        "--weight-transfer-config",
        '{"backend": "ipc"}',
    ]
    env = {
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "PROMETHEUS_MULTIPROC_DIR": str(tmp_path),
    }
    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=env) as server:
        url = server.url_root
        trainer = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16
        ).cuda()
        engine = WeightTransferTrainerFactory.trainer_init(
            IPCTrainerInitInfo(rank=0),
            client=HTTPVLLMWeightSyncClient(url),
            source=ModuleSource(trainer),
        )
        syncs = 0
        while syncs < MIN_SYNCS or (
            len(pids_that_recorded(tmp_path)) < 2 and syncs < MAX_SYNCS
        ):
            engine.send_weights()
            syncs += 1
        metrics = scrape(url)

    calls = {"init": 1, "start": syncs, "update": syncs, "finish": syncs}
    for operation, count in calls.items():
        assert metrics[(f"{DURATION}_count", operation)] == count, operation
        assert metrics[(IN_FLIGHT, operation)] == 0, operation
    # The counts above are sums over processes only if both servers recorded.
    assert len(pids_that_recorded(tmp_path)) == 2
