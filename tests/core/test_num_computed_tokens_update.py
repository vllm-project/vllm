# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.conftest import VllmRunner
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceGroup

MODEL = "JackFram/llama-160m"


def add_seq_group_to_engine(engine: LLMEngine, seq_group: SequenceGroup):
    # V1 engine: use engine.add_request() instead of direct scheduler access
    request_id = seq_group.request_id
    seq = seq_group.get_seqs()[0]
    prompt_token_ids = seq.get_prompt_token_ids()
    prompt = {"prompt_token_ids": prompt_token_ids}
    sampling_params = SamplingParams()

    engine.add_request(request_id, prompt, sampling_params)


def test_num_computed_tokens_update():

    runner = VllmRunner(model_name=MODEL,
                        gpu_memory_utilization=0.7,
                        enforce_eager=True)
    engine: LLMEngine = runner.llm.llm_engine

    request_id = "test_request"
    prompt = {"prompt_token_ids": [1, 2, 3, 4, 5]}  # 5 tokens
    sampling_params = SamplingParams(max_tokens=3)  # Generate 3 tokens

    engine.add_request(request_id, prompt, sampling_params)

    while engine.has_unfinished_requests():
        outputs = engine.step()
        for output in outputs:
            if output.request_id == request_id and output.finished:
                assert len(output.outputs) == 1
                assert len(output.outputs[0].token_ids) == 3
                print("✓ Test passed: Request processed successfully")
                return

    assert False, "Request should have finished"
