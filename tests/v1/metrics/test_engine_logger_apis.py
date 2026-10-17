# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import logging

import pytest

from tests.plugins.vllm_add_dummy_stat_logger.dummy_stat_logger.dummy_stat_logger import (  # noqa E501
    DummyStatLogger,
)
from tests.utils import wait_for_memory_to_settle
from vllm import SamplingParams
from vllm.platforms import current_platform
from vllm.v1.engine.async_llm import AsyncEngineArgs, AsyncLLM
from vllm.v1.metrics.ray_wrappers import RayPrometheusStatLogger


@pytest.fixture
def log_stats_enabled_engine_args():
    """Shared fixture providing common AsyncEngineArgs configuration
    used across multiple tests.
    """
    return AsyncEngineArgs(
        model="distilbert/distilgpt2",
        dtype="half",
        disable_log_stats=False,
        enforce_eager=True,
    )


@pytest.mark.asyncio
async def test_async_llm_replace_default_loggers(log_stats_enabled_engine_args):
    """RayPrometheusStatLogger should replace the default PrometheusStatLogger."""
    engine = AsyncLLM.from_engine_args(
        log_stats_enabled_engine_args, stat_loggers=[RayPrometheusStatLogger]
    )
    try:
        assert isinstance(
            engine.logger_manager.stat_loggers[0], RayPrometheusStatLogger
        )
    finally:
        engine.shutdown()
        wait_for_memory_to_settle(
            threshold_ratio=1.0 - log_stats_enabled_engine_args.gpu_memory_utilization
        )


@pytest.mark.asyncio
async def test_async_llm_add_to_default_loggers(log_stats_enabled_engine_args):
    """It's still possible to use custom stat loggers exclusively by passing
    disable_log_stats=True in addition to a list of custom stat loggers.
    """
    # Create engine_args with disable_log_stats=True for this test
    disabled_log_engine_args = copy.deepcopy(log_stats_enabled_engine_args)
    disabled_log_engine_args.disable_log_stats = True

    # Disable default loggers; pass custom stat logger to the constructor
    engine = AsyncLLM.from_engine_args(
        disabled_log_engine_args, stat_loggers=[DummyStatLogger]
    )

    try:
        assert len(engine.logger_manager.stat_loggers) == 2
        assert len(engine.logger_manager.stat_loggers[0].per_engine_stat_loggers) == 1
        assert isinstance(
            engine.logger_manager.stat_loggers[0].per_engine_stat_loggers[0],
            DummyStatLogger,
        )

        # log_stats is still True, since custom stat loggers are used
        assert engine.log_stats
    finally:
        engine.shutdown()
        wait_for_memory_to_settle(
            threshold_ratio=1.0 - disabled_log_engine_args.gpu_memory_utilization
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA graphs")
@pytest.mark.parametrize("cudagraph_metrics", [False, True])
@pytest.mark.asyncio
async def test_async_llm_cudagraph_runtime_metrics(
    monkeypatch, caplog_vllm, cudagraph_metrics
):
    """MRV2 runtime graph statistics must reach the frontend logging table."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen3-0.6B",
        dtype="half",
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=1,
        gpu_memory_utilization=0.3,
        disable_log_stats=False,
        cudagraph_metrics=cudagraph_metrics,
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1],
        },
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    try:
        # Engine initialization can reconfigure the vLLM logger.
        monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
        caplog_vllm.clear()
        outputs = []
        async for output in engine.generate(
            "Hello, my name is",
            SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
            request_id="cudagraph-metrics",
        ):
            outputs.append(output)
        assert outputs[-1].finished
        assert len(outputs[-1].outputs[0].token_ids) == 8

        await engine.do_log_stats()
        tables = [
            message
            for message in caplog_vllm.messages
            if "**CUDAGraph Stats:**" in message
        ]
        if not cudagraph_metrics:
            assert not tables
            return

        assert tables, "No runtime CUDA graph statistics reached the frontend"
        rows = [
            [cell.strip() for cell in line.strip("|").split("|")]
            for table in tables
            for line in table.splitlines()
            if line.startswith("|")
        ]
        assert any(
            row[:4] == ["1", "1", "0", "FULL"] and int(row[4]) > 0 for row in rows
        ), "Expected a nonzero count of single-token FULL decode graph executions"
    finally:
        engine.shutdown()
        wait_for_memory_to_settle(
            threshold_ratio=1.0 - engine_args.gpu_memory_utilization
        )
