"""Multi-prompt generation test for BitNet in vLLM."""
import bitnet_vllm
bitnet_vllm.register()

from vllm import LLM, SamplingParams


def strip_quant_config(config):
    if hasattr(config, 'quantization_config'):
        delattr(config, 'quantization_config')
    return config


if __name__ == '__main__':
    llm = LLM(
        model='microsoft/bitnet-b1.58-2B-4T-bf16',
        hf_overrides=strip_quant_config,
        dtype='bfloat16',
        max_model_len=512,
        enforce_eager=True,
    )
    params = SamplingParams(temperature=0, max_tokens=64)
    prompts = [
        'Hello, my name is',
        'The capital of France is',
        'def fibonacci(n):',
    ]
    outputs = llm.generate(prompts, params)

    for output in outputs:
        prompt = output.prompt
        text = output.outputs[0].text
        print(f'Prompt: {prompt!r}')
        print(f'Output: {text!r}')
        print()
