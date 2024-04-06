"""Compare the outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/models/test_chunked_prefill.py`.
"""
import pytest

MODELS = [
    "facebook/opt-125m",
    # "gpt2",
    # "bigcode/tiny_starcoder_py",
    # "EleutherAI/pythia-70m",
    # "bigscience/bloom-560m",
    # "microsoft/phi-2",
    # "stabilityai/stablelm-3b-4e1t",
    # "allenai/OLMo-1B",  # Broken
    # "bigcode/starcoder2-3b",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [96])
@pytest.mark.parametrize("chunked_prefill_token_size", [-1])
@pytest.mark.parametrize("enforce_eager", [True])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    chunked_prefill_token_size: int,
    enforce_eager: bool,
) -> None:
    # To pass the small model tests, we need full precision.
    assert dtype == "float"
    enable_chunked_prefill = False
    max_num_batched_tokens = None
    if chunked_prefill_token_size != -1:
        enable_chunked_prefill = True
        max_num_batched_tokens = chunked_prefill_token_size

    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
    del hf_model

    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=enable_chunked_prefill,
        enforce_eager=enforce_eager,
    )
    vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    del vllm_model

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")
