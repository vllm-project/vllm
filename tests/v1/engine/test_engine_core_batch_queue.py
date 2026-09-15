# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import patch

import pytest

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput

from ...utils import create_new_process_for_each_test

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.", allow_module_level=True)

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


class ThreadedUniProcExecutor(UniProcExecutor):
    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        super().initialize_from_config(kv_cache_configs)
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.sample_futures: list[Future[ModelRunnerOutput | None]] = []

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        non_block: bool = False,
    ) -> Future[ModelRunnerOutput | None]:
        assert non_block

        def execute() -> ModelRunnerOutput | None:
            output = self.collective_rpc("execute_model", args=(scheduler_output,))
            return output[0]

        return self.thread_pool.submit(execute)

    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None,
        non_block: bool = False,
    ) -> Future[ModelRunnerOutput | None]:
        assert non_block

        def sample() -> ModelRunnerOutput | None:
            output = self.collective_rpc("sample_tokens", args=(grammar_output,))
            return output[0]

        future = self.thread_pool.submit(sample)
        self.sample_futures.append(future)
        return future

    def shutdown(self) -> None:
        self.thread_pool.shutdown(wait=True)
        super().shutdown()


def make_request() -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="request",
        external_req_id="request",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


@create_new_process_for_each_test()
def test_batch_queue_propagates_model_runner_execution_error() -> None:
    vllm_config = EngineArgs(
        model=MODEL_NAME,
        enforce_eager=True,
        async_scheduling=True,
    ).create_engine_config()
    engine_core = EngineCore(
        vllm_config=vllm_config,
        executor_class=ThreadedUniProcExecutor,
        log_stats=False,
    )

    execution_error = RuntimeError("model execution failed")
    model_runner = engine_core.model_executor.driver_worker.model_runner
    with patch.object(model_runner.model, "forward", side_effect=execution_error):
        try:
            engine_core.add_request(*engine_core.preprocess_add_request(make_request()))
            assert engine_core.step_with_batch_queue()[0] is None
            sample_future = engine_core.model_executor.sample_futures[0]

            with pytest.raises(
                RuntimeError, match="model execution failed"
            ) as exc_info:
                engine_core.step_with_batch_queue()

            assert exc_info.value is execution_error
            sample_output = sample_future.result()
        finally:
            engine_core.shutdown()

    assert sample_output is None
