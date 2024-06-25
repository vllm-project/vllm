import torch

from vllm import LLM, SamplingParams
from vllm.model_executor.models.opt import OPTForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata

import os
import pytest

oot_registration_file = os.path.join(os.path.dirname(__file__), "dummy_model.py")

@pytest.parametrize("tensor_parallel_size", [1, 2])
def test_oot_registration(tensor_parallel_size):
    prompts = ["Hello, my name is", "The text does not matter"]
    sampling_params = SamplingParams(temperature=0)
    llm = LLM(model="facebook/opt-125m", worker_init_callback_script=oot_registration_file, tensor_parallel_size=tensor_parallel_size)
    first_token = llm.get_tokenizer().decode(0)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        # make sure only the first token is generated
        rest = generated_text.replace(first_token, "")
        assert rest == ""
