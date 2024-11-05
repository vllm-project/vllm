import os
import time
import pytest
import uuid

from transformers import AutoTokenizer, PreTrainedTokenizer

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.usage.usage_lib import UsageContext

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
PROMPT = "Hello my name is Robert and I love quanitzation kernels"
PROMPT_TOKENS = TOKENIZER(PROMPT).input_ids

@pytest.fixture(scope="module")
def engine_core():

    # Set V1 enviornment variable.
    previous = os.getenv("VLLM_USE_V1", None)
    os.environ["VLLM_USE_V1"] = "1"

    engine_args = EngineArgs(model=MODEL_NAME)
    vllm_config = engine_args.create_engine_config()
    executor_class = AsyncLLM._get_executor_cls(vllm_config)

    yield EngineCore(vllm_config=vllm_config,
                      executor_class=executor_class,
                      usage_context=UsageContext.UNKNOWN_CONTEXT)

    # Cleanup V1 enviornment variables.
    if previous is None:
        del os.environ["VLLM_USE_V1"]
    else:
        os.environ["VLLM_USE_V1"] = previous

def make_request() -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=uuid.uuid4(),
        prompt=PROMPT,
        prompt_token_ids=PROMPT_TOKENS,
        sampling_params=SamplingParams(),
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
    )

def test_request_lifecycle(engine_core):
    engine_core.add_request(make_request())
    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 0

    _ = engine_core.step()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 1
    
    while len(engine_core.step()) > 0:
        pass

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 0



    

