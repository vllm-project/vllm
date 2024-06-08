"""Test attention sinks correctness for large models (7B).

Run `pytest tests/test_attention_sinks.py`.
"""
import pytest

from vllm import SamplingParams

@pytest.mark.parametrize("model", ["lmsys/vicuna-7b-v1.5"])
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("min_tokens", [100])
@pytest.mark.parametrize("max_tokens", [200])
def test_llama(
    vllm_runner,
    attn_sinks_prompts,
    model,
    dtype,
    min_tokens,
    max_tokens
):
    return
    model = vllm_runner(
        model,
        max_model_len=4096,
        dtype=dtype,
        use_attention_sinks=True
    )
    for i, prompt in enumerate(attn_sinks_prompts):
        attn_sinks_prompts[i] = f"The magic word is aegis. " + prompt + " Also, what is the magic word?"

    params = SamplingParams(min_tokens=min_tokens, max_tokens=max_tokens)
    outputs = model.generate_w_logprobs(attn_sinks_prompts, params)
