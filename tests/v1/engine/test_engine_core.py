import time
import uuid

import pytest
from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore

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
        mm_placeholders=None,
        sampling_params=SamplingParams(),
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
    )


def test_engine_core(monkeypatch):

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        """Setup the EngineCore."""
        engine_args = EngineArgs(model=MODEL_NAME)
        vllm_config = engine_args.create_engine_config(
            usage_context=UsageContext.UNKNOWN_CONTEXT)
        executor_class = AsyncLLM._get_executor_cls(vllm_config)

        engine_core = EngineCore(vllm_config=vllm_config,
                                 executor_class=executor_class,
                                 usage_context=UsageContext.UNKNOWN_CONTEXT)
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
        while len(engine_core.step()) > 0:
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
