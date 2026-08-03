# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import os

import pytest

from tests.plugins.vllm_add_dummy_stat_logger.dummy_stat_logger.dummy_stat_logger import (  # noqa E501
    DummyStatLogger,
)
from tests.utils import wait_for_memory_to_settle
from vllm import SamplingParams
from vllm.v1.engine.async_llm import AsyncEngineArgs, AsyncLLM
from vllm.v1.engine.core_client import SyncMPClient
from vllm.v1.metrics.external import (
    register_external_metrics_provider,
    unregister_external_metrics_provider,
)
from vllm.v1.metrics.loggers import PerEngineStatLoggerAdapter
from vllm.v1.metrics.ray_wrappers import RayPrometheusStatLogger


def register_test_external_metrics_provider():
    calls = 0

    def collect():
        nonlocal calls
        calls += 1
        return {"calls": calls, "pid": os.getpid()}

    register_external_metrics_provider("lifecycle", collect, collection_interval_s=3600)


@pytest.fixture
def external_metrics_plugin(monkeypatch, tmp_path):
    metadata = tmp_path / "external_metrics_test-0.1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: external_metrics_test\nVersion: 0.1\n"
    )
    (metadata / "entry_points.txt").write_text(
        "[vllm.general_plugins]\nexternal_metrics_test = "
        "tests.v1.metrics.test_engine_logger_apis:"
        "register_test_external_metrics_provider\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path), prepend=os.pathsep)
    monkeypatch.setenv("VLLM_PLUGINS", "external_metrics_test")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    yield
    unregister_external_metrics_provider("lifecycle")


def test_llm_final_external_metrics_on_completion_and_abort(
    vllm_runner, external_metrics_plugin
):
    """A spawned engine must deliver its final provider payload before returning."""
    with vllm_runner(
        "hmellor/tiny-random-LlamaForCausalLM",
        dtype="half",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.2,
        enable_prefix_caching=False,
        disable_log_stats=False,
    ) as runner:
        engine = runner.llm.llm_engine
        assert isinstance(engine.engine_core, SyncMPClient)
        assert engine.logger_manager is not None
        adapter = PerEngineStatLoggerAdapter(engine.vllm_config, [0], DummyStatLogger)
        engine.logger_manager.stat_loggers.append(adapter)
        logger = adapter.per_engine_stat_loggers[0]
        assert isinstance(logger, DummyStatLogger)
        (completed,) = runner.llm.generate(
            "Hello",
            SamplingParams(
                temperature=0,
                max_tokens=4,
                ignore_eos=True,
                skip_special_tokens=False,
            ),
            use_tqdm=False,
        )
        assert completed.finished
        assert completed.outputs[0].finish_reason == "length"
        snapshots = [stats for stats, *_ in logger.recorded if stats.external_metrics]
        first, final = snapshots
        payload = final.external_metrics["lifecycle"]
        assert first.external_metrics == {
            "lifecycle": {"calls": 1, "pid": payload["pid"]}
        }
        assert payload == {"calls": 2, "pid": payload["pid"]}
        assert payload["pid"] != os.getpid()
        assert final.num_running_reqs == final.num_waiting_reqs == 0
        assert final.kv_cache_usage == 0

        # Stop strings are handled by the frontend, which sends an abort to the core.
        stop = completed.outputs[0].text
        assert stop
        logger.recorded.clear()
        (stopped,) = runner.llm.generate(
            "Hello",
            SamplingParams(
                temperature=0,
                max_tokens=32,
                ignore_eos=True,
                skip_special_tokens=False,
                stop=stop,
            ),
            use_tqdm=False,
        )
        assert stopped.finished
        assert stopped.outputs[0].stop_reason == stop
        assert not runner.llm.llm_engine.has_unfinished_requests()
        (final,) = [stats for stats, *_ in logger.recorded if stats.external_metrics]
        assert final.external_metrics == {
            "lifecycle": {"calls": 3, "pid": payload["pid"]}
        }
        assert final.num_running_reqs == final.num_waiting_reqs == 0
        assert final.kv_cache_usage == 0


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
        assert engine.logger_manager is not None
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
        assert engine.logger_manager is not None
        assert len(engine.logger_manager.stat_loggers) == 2
        # DummyStatLogger is a per-engine logger, so the manager wraps it.
        per_engine_adapter = engine.logger_manager.stat_loggers[0]
        assert isinstance(per_engine_adapter, PerEngineStatLoggerAdapter)
        assert len(per_engine_adapter.per_engine_stat_loggers) == 1
        assert isinstance(
            per_engine_adapter.per_engine_stat_loggers[0],
            DummyStatLogger,
        )

        # log_stats is still True, since custom stat loggers are used
        assert engine.log_stats
    finally:
        engine.shutdown()
        wait_for_memory_to_settle(
            threshold_ratio=1.0 - disabled_log_engine_args.gpu_memory_utilization
        )
