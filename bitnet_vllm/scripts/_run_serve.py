import sys
import bitnet_vllm
bitnet_vllm.register()

from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser

def strip_quant_config(config):
    if hasattr(config, 'quantization_config'):
        delattr(config, 'quantization_config')
    return config

if __name__ == '__main__':
    from vllm import LLM, SamplingParams

    # Use the programmatic LLM API for simplicity
    llm = LLM(
        model='microsoft/bitnet-b1.58-2B-4T-bf16',
        hf_overrides=strip_quant_config,
        dtype='bfloat16',
        max_model_len=2048,
    )

    # Test generation
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "def fibonacci(n):",
    ]
    sampling_params = SamplingParams(temperature=0, max_tokens=64)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"\nPrompt: {prompt!r}")
        print(f"Output: {generated!r}")
