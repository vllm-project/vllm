# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec
import pytest

from vllm.v1.engine import EngineCoreReadyResponse
from vllm.v1.engine.core_client import MPClient
from vllm.v1.metrics.loggers import PrometheusStatLogger

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _ready_response(dp_rank: int, capacity_bytes: int) -> EngineCoreReadyResponse:
    return EngineCoreReadyResponse(
        max_model_len=8192,
        num_gpu_blocks=100,
        block_size=16,
        dp_stats_address=None,
        dtype="bfloat16",
        vllm_version="test",
        world_size=1,
        data_parallel_size=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_rank=dp_rank,
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        instance_id="test-instance",
        supports_lora=False,
        max_loras=0,
        kv_cache_capacity_bytes=capacity_bytes,
    )


def test_kv_cache_capacity_is_preserved_per_dp_engine(monkeypatch):
    cache_config = SimpleNamespace(
        block_size=16,
        num_gpu_blocks=0,
        kv_cache_capacity_bytes={},
    )
    vllm_config = SimpleNamespace(
        cache_config=cache_config,
        model_config=SimpleNamespace(
            is_diffusion=False,
            max_model_len=8192,
            served_model_name="test-model",
        ),
        observability_config=SimpleNamespace(
            kv_cache_metrics=False,
            show_hidden_metrics=False,
        ),
        speculative_config=None,
        lora_config=None,
    )
    client = object.__new__(MPClient)
    client.vllm_config = vllm_config
    client.stats_update_address = None

    # Apply rank 1 first to ensure ready-message arrival order is irrelevant.
    for response in (_ready_response(1, 222), _ready_response(0, 111)):
        client._apply_ready_response(msgspec.msgpack.encode(response))

    assert cache_config.kv_cache_capacity_bytes == {0: 111, 1: 222}

    monkeypatch.setattr(PrometheusStatLogger, "_spec_decoding_cls", MagicMock())
    monkeypatch.setattr(PrometheusStatLogger, "_kv_connector_cls", MagicMock())
    monkeypatch.setattr(PrometheusStatLogger, "_perf_metrics_cls", MagicMock())
    stat_logger = PrometheusStatLogger(vllm_config, engine_indexes=[0, 1])

    assert stat_logger.gauge_kv_cache_capacity[0]._value.get() == 111
    assert stat_logger.gauge_kv_cache_capacity[1]._value.get() == 222
