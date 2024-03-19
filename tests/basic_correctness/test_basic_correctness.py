"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/basic_correctness/test_basic_correctness.py --forked`.
"""
import pytest

MODELS = [
    # "facebook/opt-125m"
    "meta-llama/Llama-2-7b-hf",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [5])
# @pytest.mark.parametrize("enforce_eager", [False, True])
@pytest.mark.parametrize("enforce_eager", [False])
# @pytest.mark.parametrize("max_chunked_prefill_len", [-1, 1, 16])
@pytest.mark.parametrize("max_chunked_prefill_len", [16])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    enforce_eager: bool,
    max_chunked_prefill_len: int,
) -> None:
    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
    del hf_model

    vllm_model = vllm_runner(model,
                             dtype=dtype,
                             enforce_eager=enforce_eager,
                             max_chunked_prefill_len=max_chunked_prefill_len)
    vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    del vllm_model

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")
