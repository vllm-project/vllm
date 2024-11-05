import time
import uuid
from typing import Dict, List

import pytest
from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import EngineCoreClient

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
PROMPT = "Hello my name is Robert and I love quanitzation kernels"
PROMPT_TOKENS = TOKENIZER(PROMPT).input_ids


def make_request(params: SamplingParams) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=str(uuid.uuid4()),
        prompt=PROMPT,
        prompt_token_ids=PROMPT_TOKENS,
        sampling_params=params,
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
    )


@pytest.mark.parametrize("multiprocessing_mode", [True, False])
def test_engine_core_client(monkeypatch, multiprocessing_mode: bool):

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        engine_args = EngineArgs(model=MODEL_NAME)
        vllm_config = engine_args.create_engine_config()
        executor_class = AsyncLLM._get_executor_cls(vllm_config)
        client = EngineCoreClient.make_client(
            vllm_config,
            executor_class,
            UsageContext.UNKNOWN_CONTEXT,
            multiprocess_mode=multiprocessing_mode,
            asyncio_mode=False,
        )

        MAX_TOKENS = 10
        params = SamplingParams(max_tokens=MAX_TOKENS)

        requests = [make_request(params) for _ in range(10)]
        request_ids = [req.request_id for req in requests]
        outputs: Dict[str, List] = {req_id: [] for req_id in request_ids}

        # Add requests to the engine.
        for request in requests:
            client.add_request(request)
            time.sleep(0.1)

        while True:
            engine_core_outputs = client.get_output()

            if len(engine_core_outputs) == 0:
                break

            all_finished = True
            for out in engine_core_outputs:
                outputs[out.request_id].append(out)
                if not out.finished:
                    all_finished = False

            if all_finished:
                break

        for req_id in request_ids:
            assert len(outputs[req_id]) == MAX_TOKENS, (
                f"{outputs[req_id]=}, {MAX_TOKENS=}")

        del client
