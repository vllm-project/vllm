# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor

import pytest
from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    ECTransferConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput

from ...utils import create_new_process_for_each_test, multi_gpu_test

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.", allow_module_level=True)

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
# test_engine_core_concurrent_batches assumes exactly 12 tokens per prompt.
# Adjust prompt if changing model to maintain 12-token length.
PROMPT = "I am Gyoubu Masataka Oniwa"
PROMPT_TOKENS = TOKENIZER(PROMPT).input_ids

_REQUEST_COUNTER = 0


def make_request() -> EngineCoreRequest:
    global _REQUEST_COUNTER
    _REQUEST_COUNTER += 1
    request_id = f"request-{_REQUEST_COUNTER}"
    return EngineCoreRequest(
        request_id=request_id,
        external_req_id=f"{request_id}-{uuid.uuid4()}",
        prompt_token_ids=PROMPT_TOKENS,
        mm_features=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


@create_new_process_for_each_test()
def test_engine_core():
    """Setup the EngineCore."""
    engine_args = EngineArgs(model=MODEL_NAME)
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)

    with set_default_torch_num_threads(1):
        engine_core = EngineCore(
            vllm_config=vllm_config, executor_class=executor_class, log_stats=True
        )
    """Test basic request lifecycle."""

    # First request.
    engine_core.add_request(*engine_core.preprocess_add_request(make_request()))
    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 0

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 1

    # Second request.
    engine_core.add_request(*engine_core.preprocess_add_request(make_request()))
    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 1

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2

    # Add two requests in a row.
    engine_core.add_request(*engine_core.preprocess_add_request(make_request()))
    engine_core.add_request(*engine_core.preprocess_add_request(make_request()))
    assert len(engine_core.scheduler.waiting) == 2
    assert len(engine_core.scheduler.running) == 2

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 4

    # Loop through until they are all done.
    while (outs := engine_core.step_fn()[0].get(0)) and outs.outputs:
        pass

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 0
    """Test abort cycle."""

    # Basic abort.
    req = make_request()
    request_id = req.request_id

    engine_core.add_request(*engine_core.preprocess_add_request(req))
    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 0
    assert engine_core.scheduler.has_unfinished_requests()
    assert not engine_core.scheduler.has_finished_requests()

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 1
    assert engine_core.scheduler.has_unfinished_requests()
    assert not engine_core.scheduler.has_finished_requests()

    engine_core.abort_requests([request_id])
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 0
    assert not engine_core.scheduler.has_unfinished_requests()
    assert engine_core.scheduler.has_finished_requests()

    _ = engine_core.step_fn()
    assert not engine_core.scheduler.has_unfinished_requests()
    assert not engine_core.scheduler.has_finished_requests()

    # Add, step, abort 1 of the 3.
    req0 = make_request()
    req1 = make_request()
    req2 = make_request()

    engine_core.add_request(*engine_core.preprocess_add_request(req0))
    engine_core.add_request(*engine_core.preprocess_add_request(req1))
    assert len(engine_core.scheduler.waiting) == 2
    assert len(engine_core.scheduler.running) == 0

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2

    engine_core.add_request(*engine_core.preprocess_add_request(req2))
    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 2

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 3

    # Abort just one.
    engine_core.abort_requests([req1.request_id])
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2

    # Abort the other requests at the same time.
    engine_core.abort_requests([req2.request_id, req0.request_id])
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 0

    # Sending duplicate requests with same request_id
    req0 = make_request()
    req1 = make_request()
    req0.request_id = req1.request_id = "test"
    engine_core.add_request(*engine_core.preprocess_add_request(req0))

    while engine_core.scheduler.has_work():
        engine_core.step_fn()

    engine_core.add_request(*engine_core.preprocess_add_request(req1))
    while engine_core.scheduler.has_work():
        engine_core.step_fn()

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 0


@create_new_process_for_each_test()
def test_engine_core_advanced_sampling():
    """
    A basic end-to-end test to verify that the engine functions correctly
    when additional sampling parameters, such as top_p, min_tokens, and
    presence_penalty, are set.
    """
    """Setup the EngineCore."""
    engine_args = EngineArgs(model=MODEL_NAME)
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)

    with set_default_torch_num_threads(1):
        engine_core = EngineCore(
            vllm_config=vllm_config, executor_class=executor_class, log_stats=True
        )
    """Test basic request lifecycle."""
    # First request.
    request: EngineCoreRequest = make_request()
    request.sampling_params = SamplingParams(
        min_tokens=4,
        presence_penalty=1.0,
        frequency_penalty=1.0,
        repetition_penalty=0.1,
        stop_token_ids=[1001, 1002],
    )
    engine_core.add_request(*engine_core.preprocess_add_request(request))

    def _check_engine_state():
        assert len(engine_core.scheduler.waiting) == 1
        assert len(engine_core.scheduler.running) == 0
        # Loop through until they are all done.
        while engine_core.scheduler.has_work():
            engine_core.step_fn()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 0

    _check_engine_state()

    # Second request.
    request2 = make_request()
    request2.sampling_params = SamplingParams(
        top_p=0.99,
        top_k=50,
    )
    engine_core.add_request(*engine_core.preprocess_add_request(request2))
    _check_engine_state()


@create_new_process_for_each_test()
def test_engine_core_concurrent_batches():
    """
    Test that the engine can handle multiple concurrent batches.
    """

    def make_request_with_max_tokens(req_id: str, max_tokens: int) -> EngineCoreRequest:
        request = make_request()
        request.request_id = req_id
        request.sampling_params.max_tokens = max_tokens
        return request

    class DummyExecutor(UniProcExecutor):
        def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
            super().initialize_from_config(kv_cache_configs)

            # Create a thread pool with a single worker
            self.thread_pool = ThreadPoolExecutor(max_workers=1)

        def execute_model(
            self,
            scheduler_output,
            non_block=False,
        ) -> Future[ModelRunnerOutput | None]:
            """Make execute_model non-blocking."""

            # DummyExecutor used only for testing async case.
            assert non_block

            def _execute():
                output = self.collective_rpc("execute_model", args=(scheduler_output,))
                # Make a copy because output[0] may be reused
                # by the next batch.
                return copy.deepcopy(output[0])

            # Use the thread pool instead of creating a new thread
            return self.thread_pool.submit(_execute)

        def sample_tokens(
            self, grammar_output, non_block=False
        ) -> Future[ModelRunnerOutput]:
            """Make sample_tokens non-blocking."""

            # DummyExecutor used only for testing async case.
            assert non_block

            def _execute():
                output = self.collective_rpc("sample_tokens", args=(grammar_output,))
                # Make a copy because output[0] may be reused
                # by the next batch.
                return copy.deepcopy(output[0])

            # Use the thread pool instead of creating a new thread
            return self.thread_pool.submit(_execute)

        @property
        def max_concurrent_batches(self) -> int:
            return 2

        def shutdown(self):
            if hasattr(self, "thread_pool"):
                self.thread_pool.shutdown(wait=False)

    engine_args = EngineArgs(
        model=MODEL_NAME,
        # To test concurrent batches.
        max_num_seqs=2,
        # Avoid all requests being scheduled once.
        enable_prefix_caching=False,
        max_num_batched_tokens=10,
        # Reduce startup time.
        enforce_eager=True,
        # Test concurrent batch behaviour independently of async scheduling.
        async_scheduling=False,
    )
    vllm_config = engine_args.create_engine_config()
    with set_default_torch_num_threads(1):
        engine_core = EngineCore(
            vllm_config=vllm_config, log_stats=False, executor_class=DummyExecutor
        )
    assert engine_core.batch_queue is not None

    # Add two requests in a row. Each request have 12 prompt tokens.
    req0 = make_request_with_max_tokens("0", 5)
    engine_core.add_request(*engine_core.preprocess_add_request(req0))
    req1 = make_request_with_max_tokens("1", 5)
    engine_core.add_request(*engine_core.preprocess_add_request(req1))

    # Schedule Batch 1: (10, req0)
    assert engine_core.step_with_batch_queue()[0] is None
    assert len(engine_core.batch_queue) == 1
    scheduler_output = engine_core.batch_queue[-1][1]
    assert scheduler_output.num_scheduled_tokens["0"] == 10
    # num_computed_tokens should have been updated immediately.
    assert engine_core.scheduler.requests[req0.request_id].num_computed_tokens == 10

    # Schedule Batch 2: (2, req0), (8, req1)
    assert engine_core.step_with_batch_queue()[0] == {}
    assert len(engine_core.batch_queue) == 1
    scheduler_output = engine_core.batch_queue[-1][1]
    assert scheduler_output.num_scheduled_tokens["0"] == 2
    assert scheduler_output.num_scheduled_tokens["1"] == 8
    # num_computed_tokens should have been updated immediately.
    assert engine_core.scheduler.requests["0"].num_computed_tokens == 12
    assert engine_core.scheduler.requests["1"].num_computed_tokens == 8

    assert engine_core.scheduler.get_num_unfinished_requests() == 2

    # Finish Batch 1 and schedule Batch 3: (4, req1).
    # Note that req0 cannot be scheduled
    # because it is in the decoding stage now.
    engine_core.step_with_batch_queue()
    assert len(engine_core.batch_queue) == 1
    scheduler_output = engine_core.batch_queue[-1][1]
    assert scheduler_output.num_scheduled_tokens["1"] == 4

    # Finish Batch 2. Get first token of req0.
    # Schedule Batch 4: (1, req0).
    output = engine_core.step_with_batch_queue()[0].get(0)
    assert output is not None
    assert len(output.outputs) == 1
    assert engine_core.scheduler.requests[req0.request_id].num_tokens == 13
    scheduler_output = engine_core.batch_queue[-1][1]
    assert scheduler_output.num_scheduled_tokens["0"] == 1

    # Finish Batch 3. Get first token of req1. Schedule Batch 5: (1, req1).
    output = engine_core.step_with_batch_queue()[0].get(0)
    assert output is not None
    assert len(output.outputs) == 1
    assert engine_core.scheduler.requests[req1.request_id].num_tokens == 13
    scheduler_output = engine_core.batch_queue[-1][1]
    assert scheduler_output.num_scheduled_tokens["1"] == 1

    # Loop until req0 is finished.
    req_id = 0
    expected_num_tokens = [
        engine_core.scheduler.requests["0"].num_tokens + 1,
        engine_core.scheduler.requests["1"].num_tokens + 1,
    ]
    while engine_core.scheduler.get_num_unfinished_requests() == 2:
        output = engine_core.step_with_batch_queue()[0]
        # Every step consumes an output.
        assert output is not None
        assert len(output[0].outputs) == 1
        if req_id in engine_core.scheduler.requests:
            assert (
                engine_core.scheduler.requests[req_id].num_tokens
                == expected_num_tokens[req_id]
            )
        expected_num_tokens[req_id] += 1
        req_id = (req_id + 1) % 2


@multi_gpu_test(num_gpus=2)
def test_engine_core_tp():
    """
    Test engine can initialize worker in tp properly
    """

    """Setup the EngineCore."""
    engine_args = EngineArgs(
        model=MODEL_NAME,
        tensor_parallel_size=2,
        # Reduce startup time.
        enforce_eager=True,
    )
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)

    with set_default_torch_num_threads(1):
        engine_core = EngineCore(
            vllm_config=vllm_config, executor_class=executor_class, log_stats=True
        )

    def get_worker_cache_config_field(worker, key: str):
        return getattr(worker.cache_config, key)

    num_gpu_blocks = engine_core.collective_rpc(
        get_worker_cache_config_field, args=("num_gpu_blocks",)
    )
    num_cpu_blocks = engine_core.collective_rpc(
        get_worker_cache_config_field, args=("num_cpu_blocks",)
    )
    assert all(x is not None for x in num_gpu_blocks)
    assert all(x is not None for x in num_cpu_blocks)


@create_new_process_for_each_test()
def test_engine_core_invalid_request_id_type():
    """Test that engine raises TypeError for non-string request_id."""
    engine_args = EngineArgs(model=MODEL_NAME)
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)

    with set_default_torch_num_threads(1):
        engine_core = EngineCore(
            vllm_config=vllm_config, executor_class=executor_class, log_stats=True
        )

    # Test with UUID object (common mistake)
    uuid_request = make_request()
    uuid_request.request_id = uuid.uuid4()  # UUID object instead of string

    with pytest.raises(TypeError, match="request_id must be a string, got.*UUID"):
        engine_core.add_request(*engine_core.preprocess_add_request(uuid_request))

    # Test with integer
    int_request = make_request()
    int_request.request_id = 12345

    with pytest.raises(TypeError, match="request_id must be a string, got.*int"):
        engine_core.add_request(*engine_core.preprocess_add_request(int_request))

    # Test with None
    none_request = make_request()
    none_request.request_id = None

    with pytest.raises(TypeError, match="request_id must be a string, got.*NoneType"):
        engine_core.add_request(*engine_core.preprocess_add_request(none_request))

    # Verify engine is still functional after errors
    valid_request = make_request()
    engine_core.add_request(*engine_core.preprocess_add_request(valid_request))
    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 0


@create_new_process_for_each_test()
@pytest.mark.parametrize(
    ("ec_role", "gpu_memory_utilization", "enable_prefix_caching"),
    [
        ("ec_producer", 0.01, False),
        # NOTE: ec_producer never allows prefix caching
        ("ec_consumer", 0.7, True),
        ("ec_consumer", 0.7, False),
    ],
)
@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_encoder_instance_zero_kv_cache(
    ec_role: str,
    gpu_memory_utilization: float,
    enable_prefix_caching: bool,
    use_kv_connector: bool,
):
    """EPD (Encoder-Prefill-Decode) Encoder-cache-specific tests

    This test verifies encoder-only instance initializes with 0 KV cache blocks.
    Under EPD disagg mode, Encoder instances (EC producer role) only execute
    vision encoder, so they don't need KV cache for text generation.
    """
    # Form vllm config
    model_config = ModelConfig(
        model="llava-hf/llava-1.5-7b-hf",  # Multimodal model
        enforce_eager=True,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
        disable_hybrid_kv_cache_manager=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=gpu_memory_utilization,
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    kv_transfer_config = (
        KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": "local_storage"},
        )
        if use_kv_connector
        else None
    )
    ec_transfer_config = ECTransferConfig(
        ec_connector="ECExampleConnector",
        ec_role=ec_role,
        ec_connector_extra_config={"shared_storage_path": "/tmp/ec_test_encoder"},
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        kv_transfer_config=kv_transfer_config,
        ec_transfer_config=ec_transfer_config,
    )

    executor_class = Executor.get_class(vllm_config)
    print(f"executor_class: {executor_class}")

    with set_default_torch_num_threads(1):
        engine_core = EngineCore(
            vllm_config=vllm_config, executor_class=executor_class, log_stats=True
        )

    # Check encoder cache manager exists
    assert engine_core.scheduler.encoder_cache_manager is not None, (
        "encoder_cache_manager should exist"
    )

    if ec_role == "ec_producer":
        # Check 1: num_blocks should be 0
        # NOTE: num_blocks=1 as BlockPool always needs a null_block.
        kv_cache_config = engine_core.scheduler.kv_cache_manager.kv_cache_config
        print(f"kv_cache_config: {kv_cache_config}")
        assert kv_cache_config.num_blocks == 1, (
            f"ec_producer should only have 1 KV blocks, "
            f"got {kv_cache_config.num_blocks}"
        )

        # Check 2: kv_cache_groups should be empty
        assert len(kv_cache_config.kv_cache_groups) == 0, (
            f"ec_producer should have 0 KV cache groups, "
            f"got {len(kv_cache_config.kv_cache_groups)}"
        )

        # Check 3: kv_cache_tensors should be empty
        assert len(kv_cache_config.kv_cache_tensors) == 0, (
            f"Encoder instance should have 0 KV cache tensors, "
            f"got {len(kv_cache_config.kv_cache_tensors)}"
        )

        # Check 4: Verify EC connector is initialized and is producer
        assert engine_core.scheduler.ec_connector is not None, (
            "Encoder instance should have EC connector"
        )
        assert engine_core.scheduler.ec_connector.is_producer, (
            "Encoder instance EC connector should be producer"
        )

        # Check 5: Verify chunked prefill is disabled
        assert not vllm_config.scheduler_config.enable_chunked_prefill, (
            "Encoder instance should disable chunked prefill (no KV cache)"
        )

    elif ec_role == "ec_consumer":
        # Check 1: num_blocks should be > 1
        kv_cache_config = engine_core.scheduler.kv_cache_manager.kv_cache_config
        print(f"kv_cache_config: {kv_cache_config}")
        assert kv_cache_config.num_blocks > 1, (
            f"ec_consumer should have >1 KV blocks, got {kv_cache_config.num_blocks}"
        )

        # Check 2: kv_cache_groups should NOT be empty
        assert len(kv_cache_config.kv_cache_groups) > 0, (
            f"ec_consumer should have KV cache groups, "
            f"got {len(kv_cache_config.kv_cache_groups)}"
        )

        # Check 3: Verify EC connector is consumer
        assert engine_core.scheduler.ec_connector is not None, (
            "Consumer instance should have EC connector"
        )
        assert not engine_core.scheduler.ec_connector.is_producer, (
            "Consumer instance EC connector should be consumer"
        )
