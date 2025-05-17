# SPDX-License-Identifier: Apache-2.0

import random
import uuid

import ray

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams


def generate_prompt():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    return random.choice(prompts)


@ray.remote(num_gpus=1)
class InferenceEngine:

    def __init__(self, **kwargs):
        self.args = AsyncEngineArgs(**kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(self.args)

    async def generate(self, prompt, sampling_params, request_id):
        results_generator = self.engine.generate(prompt, sampling_params,
                                                 request_id)
        final_result = None
        async for result in results_generator:
            final_result = result
        return final_result


if __name__ == "__main__":
    # Create an Async LLM.
    model = "facebook/opt-125m"
    llm = InferenceEngine.remote(model=model)

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    futures = []
    # Generate texts from the prompts.
    for i in range(100):
        prompt = generate_prompt()
        request_id = str(uuid.uuid4().hex)
        result = llm.generate.remote(prompt, sampling_params, request_id)
        futures.append(result)

    # Get the results from the futures.
    # consider using ray.wait() in production
    for future in futures:
        output = ray.get(future)
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
