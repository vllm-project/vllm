# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import tempfile

import pytest
import torch

from tests.v1.worker.test_gpu_model_runner import _schedule_new_request
from vllm.config import VllmConfig
from vllm.distributed import (cleanup_dist_env_and_memory,
                              init_distributed_environment,
                              initialize_model_parallel)
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.engine.core import get_kv_cache_config
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

model_dir = "meta-llama/Llama-3.1-8B-Instruct"
eagle_dir = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"


@pytest.fixture()
def should_do_global_cleanup_after_test(request) -> bool:
    # So we can share the DraftModelProposer between tests
    return False


@pytest.fixture(scope="class")
def monkeyclass():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(scope="class")
def spec_decode_vllm_config_and_env_setup(monkeyclass: pytest.MonkeyPatch):
    with monkeyclass.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        vllm_config = EngineArgs(model=model_dir,
                                 max_model_len=256,
                                 cuda_graph_sizes=[1, 2, 4],
                                 gpu_memory_utilization=0.8,
                                 speculative_config={
                                     "model": eagle_dir,
                                     "method": "eagle",
                                     "num_speculative_tokens": 2,
                                 }).create_engine_config()
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temp_file}",
            local_rank=0,
            backend="nccl",
        )
        initialize_model_parallel(1, 1)
        yield vllm_config
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="class")
def mock_spec_decode_model_runner(
        spec_decode_vllm_config_and_env_setup: VllmConfig):
    model_runner = GPUModelRunner(spec_decode_vllm_config_and_env_setup,
                                  torch.device("cuda"))
    model_runner.load_model()
    kv_cache_spec = model_runner.get_kv_cache_spec()

    kv_cache_config = get_kv_cache_config(
        spec_decode_vllm_config_and_env_setup, kv_cache_spec, 1024**3)  # 1GB
    model_runner.initialize_kv_cache(kv_cache_config)
    yield model_runner


class TestSpecDecodeScheduling:

    def test_spec_decode_partial_scheduling(
            self, mock_spec_decode_model_runner: GPUModelRunner):
        """Make sure we don't crash when the scheduler schedules only a subset
        of the requests.

        Four iterations:
        1. Schedule both req1 (w/ 0 draft) and req2 (w/ 0 draft)
        2. Schedule only req1 (w/ 1 draft)
        3. Schedule both req1 (w/ 1 draft) and req2 (w/ 2 draft)
        4. Terminate req1 and req2
        """
        # Schedule both req1 and req2 on the first iteration
        scheduler_output = _schedule_new_request("req1", "req2")
        mock_spec_decode_model_runner.execute_model(scheduler_output)

        # Only schedule req1 on the second iteration
        cached_req_data = CachedRequestData(
            req_ids=["req1"],
            resumed_from_preemption=[False],
            new_token_ids=[[3]],
            new_block_ids=[([], )],
            num_computed_tokens=[3],
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached_req_data,
            num_scheduled_tokens={"req1": 2},
            total_num_scheduled_tokens=2,
            scheduled_spec_decode_tokens={"req1": [1001]},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )
        mock_spec_decode_model_runner.execute_model(scheduler_output)

        # Schedule both req1 and req2 on the third iteration
        cached_req_data = CachedRequestData(
            req_ids=["req1", "req2"],
            resumed_from_preemption=[False, False],
            new_token_ids=[[10], [11]],
            new_block_ids=[([], ), ([], )],
            num_computed_tokens=[4, 3],
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached_req_data,
            num_scheduled_tokens={
                "req1": 2,
                "req2": 3
            },
            total_num_scheduled_tokens=5,
            scheduled_spec_decode_tokens={
                "req1": [1001],
                "req2": [2001, 2002]
            },
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )
        mock_spec_decode_model_runner.execute_model(scheduler_output)

        # Terminate both req1 and req2
        cached_req_data = CachedRequestData(
            req_ids=[],
            resumed_from_preemption=[],
            new_token_ids=[],
            new_block_ids=[],
            num_computed_tokens=[],
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached_req_data,
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids={"req1", "req2"},
            free_encoder_mm_hashes=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )
        mock_spec_decode_model_runner.execute_model(scheduler_output)

    def test_spec_decode_preemption_scheduling(
            self, mock_spec_decode_model_runner: GPUModelRunner):
        """Make sure we don't crash when the scheduler preempts a request.

        Four iterations:
        1. Schedule req1 (w/ 0 draft) and req2 (w/ 0 draft)
        2. Schedule req1 (w/ 1 draft) and preempt req2
        3. Schedule req1 (w/ 1 draft) and resume req2 (w/ 2 draft)
        4. Terminate req1 and req2
        """
        # Schedule both req1 and req2 on the first iteration
        scheduler_output = _schedule_new_request("req1", "req2")
        mock_spec_decode_model_runner.execute_model(scheduler_output)

        # Only schedule req1 on the second iteration
        cached_req_data = CachedRequestData(
            req_ids=["req1"],
            resumed_from_preemption=[False],
            new_token_ids=[[3]],
            new_block_ids=[([], )],
            num_computed_tokens=[3],
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached_req_data,
            num_scheduled_tokens={"req1": 2},
            total_num_scheduled_tokens=2,
            scheduled_spec_decode_tokens={"req1": [1001]},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )
        mock_spec_decode_model_runner.execute_model(scheduler_output)

        # Schedule both req1 and req2 on the third iteration
        cached_req_data = CachedRequestData(
            req_ids=["req1", "req2"],
            resumed_from_preemption=[False, True],
            new_token_ids=[[10], [11]],
            new_block_ids=[([], ), ([0], )],
            num_computed_tokens=[4, 0],
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached_req_data,
            num_scheduled_tokens={
                "req1": 2,
                "req2": 6
            },
            total_num_scheduled_tokens=8,
            scheduled_spec_decode_tokens={
                "req1": [1001],
                "req2": [2001, 2002]
            },
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )
        mock_spec_decode_model_runner.execute_model(scheduler_output)

        # Terminate both req1 and req2
        cached_req_data = CachedRequestData(
            req_ids=[],
            resumed_from_preemption=[],
            new_token_ids=[],
            new_block_ids=[],
            num_computed_tokens=[],
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached_req_data,
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids={"req1", "req2"},
            free_encoder_mm_hashes=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )
        mock_spec_decode_model_runner.execute_model(scheduler_output)
