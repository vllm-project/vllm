import multiprocessing
import sys

import torch
from openai import OpenAI

from vllm import LLM, ModelRegistry, SamplingParams
from vllm.model_executor.models.opt import OPTForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata


class MyOPTForCausalLM(OPTForCausalLM):

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        # this dummy model always predicts the first token
        logits = super().compute_logits(hidden_states, sampling_metadata)
        logits.zero_()
        logits[:, 0] += 1.0
        return logits


def test_oot_registration():
    # register our dummy model
    ModelRegistry.register_model("OPTForCausalLM", MyOPTForCausalLM)
    prompts = ["Hello, my name is", "The text does not matter"]
    sampling_params = SamplingParams(temperature=0)
    llm = LLM(model="facebook/opt-125m")
    first_token = llm.get_tokenizer().decode(0)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        # make sure only the first token is generated
        rest = generated_text.replace(first_token, "")
        assert rest == ""


def server_function():
    # register our dummy model
    ModelRegistry.register_model("OPTForCausalLM", MyOPTForCausalLM)
    sys.argv = ["placeholder.py"] + \
        ("--model facebook/opt-125m --dtype"
        " float32 --api-key token-abc123").split()
    import runpy
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')


def test_oot_registration_for_api_server():
    server = multiprocessing.Process(target=server_function)
    server.start()
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )

    completion = client.chat.completions.create(
        model="facebook/opt-125m",
        messages=[{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": "Hello!"
        }],
        temperature=0,
    )

    generated_text = completion.choices[0].message.content
    # make sure only the first token is generated
    rest = generated_text.replace("<s>", "")
    assert rest == ""
