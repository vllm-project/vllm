# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest
import torch

from tests.v1.kv_connector.unit.utils import create_vllm_config
from vllm import SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
)
from vllm.distributed.kv_transfer.kv_transfer_state import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.worker_base import WorkerBase


class CacheReadyConnector(ExampleConnector):
    """Defer registration until startup preparation has finished."""

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self.kv_caches = {}
        self.ready_count = 0

    def register_kv_caches(self, kv_caches):
        self.kv_caches = kv_caches

    def on_kv_cache_ready(self):
        assert self.role == KVConnectorRole.WORKER
        assert self.kv_caches
        if torch.cuda.is_available():
            assert not torch.cuda.is_current_stream_capturing()
        extra = self._kv_transfer_config.kv_connector_extra_config
        if extra.get("fail_ready", False):
            raise RuntimeError("deferred registration failed")
        self.ready_count += 1
        rank = self._vllm_config.parallel_config.rank
        path = Path(extra["shared_storage_path"]) / f"{extra['name']}-{rank}.json"
        path.write_text(
            json.dumps(
                {
                    "notifications": self.ready_count,
                    "num_blocks": self._kv_cache_config.num_blocks,
                }
            )
        )

    def start_load_kv(self, forward_context, **kwargs):
        assert self.ready_count == 1
        super().start_load_kv(forward_context, **kwargs)


def _child_config(path: Path, name: str, *, fail_ready: bool = False):
    return {
        "kv_connector": "CacheReadyConnector",
        "kv_connector_module_path": __name__,
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "shared_storage_path": str(path),
            "name": name,
            "fail_ready": fail_ready,
        },
    }


@pytest.mark.cpu_test
@pytest.mark.parametrize("fail_ready", [False, True])
def test_connector_deferred_registration(tmp_path, fail_ready, dist_init):
    """The worker notifies all children and propagates registration errors."""
    config = create_vllm_config(
        kv_connector="MultiConnector",
        kv_connector_extra_config={
            "connectors": [
                {"kv_connector": "ExampleConnector", "kv_role": "kv_both"},
                _child_config(tmp_path, "first"),
                _child_config(tmp_path, "second", fail_ready=fail_ready),
            ]
        },
    )
    worker = WorkerBase(config, 0, 0, "env://")
    ensure_kv_transfer_initialized(config, KVCacheConfig(2, [], []))
    try:
        get_kv_transfer_group().register_kv_caches({"layer": torch.zeros(2, 1, 16, 4)})
        if fail_ready:
            with pytest.raises(RuntimeError, match="deferred registration failed"):
                worker.notify_kv_cache_ready()
            assert not (tmp_path / "second-0.json").exists()
        else:
            worker.notify_kv_cache_ready()
            assert json.loads((tmp_path / "second-0.json").read_text()) == {
                "notifications": 1,
                "num_blocks": 2,
            }
        assert json.loads((tmp_path / "first-0.json").read_text()) == {
            "notifications": 1,
            "num_blocks": 2,
        }
    finally:
        ensure_kv_transfer_shutdown()


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_cache_ready_before_first_request(vllm_runner, tmp_path, enforce_eager):
    """Deferred registration finishes before LLM returns, with or without graphs."""
    transfer_config = KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                _child_config(tmp_path, "first"),
                _child_config(tmp_path, "second"),
            ]
        },
    )
    with vllm_runner(
        "facebook/opt-125m",
        enforce_eager=enforce_eager,
        kv_transfer_config=transfer_config,
        gpu_memory_utilization=0.5,
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        block_size=32,
    ) as runner:
        num_blocks = runner.llm.llm_engine.vllm_config.cache_config.num_gpu_blocks
        for name in ("first", "second"):
            assert json.loads((tmp_path / f"{name}-0.json").read_text()) == {
                "notifications": 1,
                "num_blocks": num_blocks,
            }
        outputs = runner.llm.generate(
            ["Hello"], SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        )
        assert len(outputs[0].outputs[0].token_ids) == 1
