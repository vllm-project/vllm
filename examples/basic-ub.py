# SPDX-License-Identifier: Apache-2.0

import logging
import os

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Configure logging level for vllm (optional, uses VLLM_LOGGING_LEVEL env var).
logging_level = os.getenv("VLLM_LOGGING_LEVEL", "").upper()
if logging_level:
    logging.basicConfig(level=getattr(logging, logging_level, logging.INFO))

# Create a sampling params object, optionally limiting output tokens via MAX_TOKENS env var.
param_kwargs = {"temperature": 0.8, "top_p": 0.95}
max_tokens_env = os.getenv("MAX_TOKENS")
if max_tokens_env is not None:
    try:
        param_kwargs["max_tokens"] = int(max_tokens_env)
    except ValueError:
        raise ValueError(f"Invalid MAX_TOKENS value: {max_tokens_env}")
sampling_params = SamplingParams(**param_kwargs)


def main():
    # Create an LLM.
    model = "deepseek-ai/DeepSeek-V2-Lite"
    # model = "facebook/opt-125m"
    llm = LLM(model=model, 
              enforce_eager=True,
              compilation_config=2,
              ###############
              trust_remote_code=True,
              max_model_len=1024,
              #load_format="dummy",
              ###############
              #tensor_parallel_size=1,
              data_parallel_size=2,
              enable_expert_parallel=True,
              ###############
              #enable_microbatching=True, 
    )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
