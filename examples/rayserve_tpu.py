"""
This example runs a LLM using RayServe and vLLM on TPUs. It can support
single or multi-host inference.

To run this example, run:
  serve run examples.rayserve_tpu:build_app
"""

import json
import logging
from typing import Dict

import ray
from fastapi import FastAPI
from ray import serve
from starlette.requests import Request
from starlette.responses import Response

from vllm import LLM, SamplingParams

logger = logging.getLogger("ray.serve")

app = FastAPI()


@serve.deployment(num_replicas=1, )
@serve.ingress(app)
class VLLMDeployment:

    def __init__(
        self,
        num_tpu_chips,
    ):
        self.llm = LLM(
            model="meta-llama/Meta-Llama-3.1-8B",
            tensor_parallel_size=num_tpu_chips,
            enforce_eager=True,
        )

    @app.post("/v1/generate")
    async def generate(self, request: Request):
        request_dict = await request.json()
        prompts = request_dict.pop("prompt")
        print("Processing prompt ", prompts)
        sampling_params = SamplingParams(temperature=0.7,
                                         top_p=1.0,
                                         n=1,
                                         max_tokens=1000)

        outputs = self.llm.generate(prompts, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = ""
            token_ids = []
            for completion_output in output.outputs:
                generated_text += completion_output.text
                token_ids.extend(list(completion_output.token_ids))

            print("Generated text: ", generated_text)
            ret = {
                "prompt": prompt,
                "text": generated_text,
                "token_ids": token_ids,
            }

        return Response(content=json.dumps(ret))


def get_num_tpu_chips() -> int:
    return int(ray.cluster_resources()["TPU"])


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments."""
    ray.init()
    num_tpu_chips = get_num_tpu_chips()
    pg_resources = []
    pg_resources.append({"CPU": 1})  # for the deployment replica
    for i in range(num_tpu_chips):
        pg_resources.append({"CPU": 1, "TPU": 1})  # for the vLLM actors

    # Use PACK strategy since the deployment may use more than one TPU node.
    return VLLMDeployment.options(
        placement_group_bundles=pg_resources,
        placement_group_strategy="PACK").bind(num_tpu_chips)
