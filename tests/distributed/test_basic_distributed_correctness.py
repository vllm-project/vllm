"""Compare the outputs of HF and distributed vLLM when using greedy sampling.

Run `pytest tests/distributed/test_basic_distributed_correctness.py --forked`.
"""
import pytest
import torch

MODELS = [
    "facebook/opt-125m",
    "meta-llama/Llama-2-7b-hf",
]


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [5])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
    del hf_model

    vllm_model = vllm_runner(model, dtype=dtype, tensor_parallel_size=2)
    vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    del vllm_model

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")
