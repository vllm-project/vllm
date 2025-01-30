import time
import uuid

import pytest
from transformers import AutoTokenizer

from tests.utils import fork_new_process_for_each_test
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor

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


@fork_new_process_for_each_test
def test_engine_core(monkeypatch):

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        """Setup the EngineCore."""
        engine_args = EngineArgs(model=MODEL_NAME)
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)

        engine_core = EngineCore(vllm_config=vllm_config,
                                 executor_class=executor_class)
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

        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 1

        engine_core.abort_requests([request_id])
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 0

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


@fork_new_process_for_each_test
def test_engine_core_advanced_sampling(monkeypatch):
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
                                 executor_class=executor_class)
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
