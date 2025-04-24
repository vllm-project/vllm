# SPDX-License-Identifier: Apache-2.0

import copy
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor

import pytest
from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor, UniProcExecutor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput

from ...utils import create_new_process_for_each_test

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.",
                allow_module_level=True)

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
PROMPT = "Hello my name is Robert and I love quantization kernels"
PROMPT_TOKENS = TOKENIZER(PROMPT).input_ids


def make_request() -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=uuid.uuid4(),
        prompt=PROMPT,
        prompt_token_ids=PROMPT_TOKENS,
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=SamplingParams(),
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
    )


@create_new_process_for_each_test()
def test_engine_core(monkeypatch: pytest.MonkeyPatch):

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        """Setup the EngineCore."""
        engine_args = EngineArgs(model=MODEL_NAME)
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)

        engine_core = EngineCore(vllm_config=vllm_config,
                                 executor_class=executor_class,
                                 log_stats=True)
        """Test basic request lifecycle."""

        # First request.
        engine_core.add_request(make_request())
        assert len(engine_core.scheduler.waiting) == 1
        assert len(engine_core.scheduler.running) == 0

        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 1

        # Second request.
        engine_core.add_request(make_request())
        assert len(engine_core.scheduler.waiting) == 1
        assert len(engine_core.scheduler.running) == 1

        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 2

        # Add two requests in a row.
        engine_core.add_request(make_request())
        engine_core.add_request(make_request())
        assert len(engine_core.scheduler.waiting) == 2
        assert len(engine_core.scheduler.running) == 2

        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 4

        # Loop through until they are all done.
        while len(engine_core.step().outputs) > 0:
            pass

        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 0
        """Test abort cycle."""

        # Basic abort.
        req = make_request()
        request_id = req.request_id

        engine_core.add_request(req)
        assert len(engine_core.scheduler.waiting) == 1
        assert len(engine_core.scheduler.running) == 0
        assert engine_core.scheduler.has_unfinished_requests()
        assert not engine_core.scheduler.has_finished_requests()

        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 1
        assert engine_core.scheduler.has_unfinished_requests()
        assert not engine_core.scheduler.has_finished_requests()

        engine_core.abort_requests([request_id])
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 0
        assert not engine_core.scheduler.has_unfinished_requests()
        assert engine_core.scheduler.has_finished_requests()

        _ = engine_core.step()
        assert not engine_core.scheduler.has_unfinished_requests()
        assert not engine_core.scheduler.has_finished_requests()

        # Add, step, abort 1 of the 3.
        req0 = make_request()
        req1 = make_request()
        req2 = make_request()

        engine_core.add_request(req0)
        engine_core.add_request(req1)
        assert len(engine_core.scheduler.waiting) == 2
        assert len(engine_core.scheduler.running) == 0

        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 2

        engine_core.add_request(req2)
        assert len(engine_core.scheduler.waiting) == 1
        assert len(engine_core.scheduler.running) == 2

        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 3

        # Abort just one.
        engine_core.abort_requests([req1.request_id])
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 2

        _ = engine_core.step()
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
        engine_core.add_request(req0)

        while len(engine_core.step().outputs) > 0:
            pass

        engine_core.add_request(req1)
        while len(engine_core.step().outputs) > 0:
            pass

        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 0


@create_new_process_for_each_test()
def test_engine_core_advanced_sampling(monkeypatch: pytest.MonkeyPatch):
    """
    A basic end-to-end test to verify that the engine functions correctly
    when additional sampling parameters, such as top_p, min_tokens, and
    presence_penalty, are set.
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        """Setup the EngineCore."""
        engine_args = EngineArgs(model=MODEL_NAME)
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)

        engine_core = EngineCore(vllm_config=vllm_config,
                                 executor_class=executor_class,
                                 log_stats=True)
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
        engine_core.add_request(request)

        def _check_engine_state():
            assert len(engine_core.scheduler.waiting) == 1
            assert len(engine_core.scheduler.running) == 0
            # Loop through until they are all done.
            while len(engine_core.step().outputs) > 0:
                pass
            assert len(engine_core.scheduler.waiting) == 0
            assert len(engine_core.scheduler.running) == 0

        _check_engine_state()

        # Second request.
        request2 = make_request()
        request2.sampling_params = SamplingParams(
            top_p=0.99,
            top_k=50,
        )
        engine_core.add_request(request2)
        _check_engine_state()


@create_new_process_for_each_test()
def test_engine_core_concurrent_batches(monkeypatch: pytest.MonkeyPatch):
    """
    Test that the engine can handle multiple concurrent batches.
    """

    def make_request_with_max_tokens(req_id: int,
                                     max_tokens: int) -> EngineCoreRequest:
        request = make_request()
        request.request_id = req_id
        request.sampling_params.max_tokens = max_tokens
        return request

    class DummyExecutor(UniProcExecutor):

        def initialize_from_config(
                self, kv_cache_configs: list[KVCacheConfig]) -> None:
            super().initialize_from_config(kv_cache_configs)

            # Create a thread pool with a single worker
            self.thread_pool = ThreadPoolExecutor(max_workers=1)

        def execute_model(
            self,
            scheduler_output,
        ) -> Future[ModelRunnerOutput]:
            """Make execute_model non-blocking."""

            def _execute():
                output = self.collective_rpc("execute_model",
                                             args=(scheduler_output, ))
                # Make a copy because output[0] may be reused
                # by the next batch.
                return copy.deepcopy(output[0])

            # Use the thread pool instead of creating a new thread
            return self.thread_pool.submit(_execute)

        @property
        def max_concurrent_batches(self) -> int:
            return 2

        def shutdown(self):
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        engine_args = EngineArgs(
            model=MODEL_NAME,
            # To test concurrent batches.
            max_num_seqs=2,
            # Avoid all requests being scheduled once.
            enable_prefix_caching=False,
            max_num_batched_tokens=10,
            # Reduce startup time.
            enforce_eager=True,
        )
        vllm_config = engine_args.create_engine_config()
        engine_core = EngineCore(vllm_config=vllm_config,
                                 log_stats=False,
                                 executor_class=DummyExecutor)
        assert engine_core.batch_queue is not None

        # Add two requests in a row. Each request have 12 prompt tokens.
        req0 = make_request_with_max_tokens(0, 5)
        engine_core.add_request(req0)
        req1 = make_request_with_max_tokens(1, 5)
        engine_core.add_request(req1)

        # Schedule Batch 1: (10, req0)
        assert engine_core.step_with_batch_queue() is None
        assert engine_core.batch_queue.qsize() == 1
        scheduler_output = engine_core.batch_queue.queue[-1][1]
        assert scheduler_output.num_scheduled_tokens[0] == 10
        # num_computed_tokens should have been updated immediately.
        assert engine_core.scheduler.requests[
            req0.request_id].num_computed_tokens == 10

        # Schedule Batch 2: (2, req0), (8, req1)
        assert engine_core.step_with_batch_queue() is None
        assert engine_core.batch_queue.qsize() == 2
        scheduler_output = engine_core.batch_queue.queue[-1][1]
        assert scheduler_output.num_scheduled_tokens[0] == 2
        assert scheduler_output.num_scheduled_tokens[1] == 8
        # num_computed_tokens should have been updated immediately.
        assert engine_core.scheduler.requests[0].num_computed_tokens == 12
        assert engine_core.scheduler.requests[1].num_computed_tokens == 8

        assert engine_core.scheduler.get_num_unfinished_requests() == 2

        # Batch queue is full. Finish Batch 1.
        engine_core.step_with_batch_queue()

        # Schedule Batch 3: (4, req1). Note that req0 cannot be scheduled
        # because it is in the decoding stage now.
        engine_core.step_with_batch_queue()
        assert engine_core.batch_queue.qsize() == 2
        scheduler_output = engine_core.batch_queue.queue[-1][1]
        assert scheduler_output.num_scheduled_tokens[1] == 4

        # Batch queue is full. Finish Batch 2. Get first token of req0.
        output = engine_core.step_with_batch_queue()
        assert output is not None
        assert len(output.outputs) == 1
        assert engine_core.scheduler.requests[req0.request_id].num_tokens == 13

        # Schedule Batch 4: (1, req0).
        engine_core.step_with_batch_queue()
        assert engine_core.batch_queue.qsize() == 2
        scheduler_output = engine_core.batch_queue.queue[-1][1]
        assert scheduler_output.num_scheduled_tokens[0] == 1

        # Batch queue is full. Finish Batch 3. Get first token of req1.
        output = engine_core.step_with_batch_queue()
        assert output is not None
        assert len(output.outputs) == 1
        assert engine_core.scheduler.requests[req1.request_id].num_tokens == 13

        # Schedule Batch 5: (1, req1).
        engine_core.step_with_batch_queue()
        assert engine_core.batch_queue.qsize() == 2
        scheduler_output = engine_core.batch_queue.queue[-1][1]
        assert scheduler_output.num_scheduled_tokens[1] == 1

        # Loop until req0 is finished.
        step = 0
        req_id = 0
        expected_num_tokens = [
            engine_core.scheduler.requests[0].num_tokens + 1,
            engine_core.scheduler.requests[1].num_tokens + 1,
        ]
        while engine_core.scheduler.get_num_unfinished_requests() == 2:
            output = engine_core.step_with_batch_queue()
            if step % 2 == 0:
                # Even steps consumes an output.
                assert output is not None
                assert len(output.outputs) == 1
                if req_id in engine_core.scheduler.requests:
                    assert engine_core.scheduler.requests[
                        req_id].num_tokens == expected_num_tokens[req_id]
                expected_num_tokens[req_id] += 1
                req_id = (req_id + 1) % 2
            else:
                # Odd steps schedules a new batch.
                assert output is None
            step += 1
