import sys
import time

import torch
from openai import OpenAI, OpenAIError

from vllm import ModelRegistry
from vllm.model_executor.models.opt import OPTForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.utils import get_open_port

from ...utils import VLLM_PATH, RemoteOpenAIServer

chatml_jinja_path = VLLM_PATH / "examples/template_chatml.jinja"
assert chatml_jinja_path.exists()


class MyOPTForCausalLM(OPTForCausalLM):

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        # this dummy model always predicts the first token
        logits = super().compute_logits(hidden_states, sampling_metadata)
        logits.zero_()
        logits[:, 0] += 1.0
        return logits


def server_function(port: int):
    # register our dummy model
    ModelRegistry.register_model("OPTForCausalLM", MyOPTForCausalLM)

    sys.argv = ["placeholder.py"] + [
        "--model",
        "facebook/opt-125m",
        "--gpu-memory-utilization",
        "0.10",
        "--dtype",
        "float32",
        "--api-key",
        "token-abc123",
        "--port",
        str(port),
        "--chat-template",
        str(chatml_jinja_path),
    ]

    import runpy
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')


def test_oot_registration_for_api_server():
    port = get_open_port()
    ctx = torch.multiprocessing.get_context()
    server = ctx.Process(target=server_function, args=(port, ))
    server.start()

    try:
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
                    if time.time() - now > RemoteOpenAIServer.MAX_START_WAIT_S:
                        msg = "Server did not start in time"
                        raise RuntimeError(msg) from e
                else:
                    raise e
    finally:
        server.terminate()

    generated_text = completion.choices[0].message.content
    assert generated_text is not None
    # make sure only the first token is generated
    rest = generated_text.replace("<s>", "")
    assert rest == ""
