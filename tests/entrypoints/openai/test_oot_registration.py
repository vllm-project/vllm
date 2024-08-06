import sys
import time

import torch
from openai import OpenAI, OpenAIError

from vllm import ModelRegistry
from vllm.model_executor.models.opt import OPTForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.utils import get_open_port


class MyOPTForCausalLM(OPTForCausalLM):

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        # this dummy model always predicts the first token
        logits = super().compute_logits(hidden_states, sampling_metadata)
        logits.zero_()
        logits[:, 0] += 1.0
        return logits


def server_function(port):
    # register our dummy model
    ModelRegistry.register_model("OPTForCausalLM", MyOPTForCausalLM)
    sys.argv = ["placeholder.py"] + \
        ("--model facebook/opt-125m --gpu-memory-utilization 0.10 "
        f"--dtype float32 --api-key token-abc123 --port {port}").split()
    import runpy
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')


def test_oot_registration_for_api_server():
    port = get_open_port()
    ctx = torch.multiprocessing.get_context()
    server = ctx.Process(target=server_function, args=(port, ))
    server.start()
    MAX_SERVER_START_WAIT_S = 60
    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="token-abc123",
    )
    now = time.time()
    while True:
        try:
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
            break
        except OpenAIError as e:
            if "Connection error" in str(e):
                time.sleep(3)
                if time.time() - now > MAX_SERVER_START_WAIT_S:
                    raise RuntimeError("Server did not start in time") from e
            else:
                raise e
    server.kill()
    generated_text = completion.choices[0].message.content
    # make sure only the first token is generated
    rest = generated_text.replace("<s>", "")
    assert rest == ""
