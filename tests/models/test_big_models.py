"""Compare the outputs of HF and vLLM when using greedy sampling.

This tests bigger models and use half precision.

Run `pytest tests/models/test_big_models.py`.
"""
import pytest
import torch

MODELS = [
    "meta-llama/Llama-2-7b-hf",
    # "mistralai/Mistral-7B-v0.1",  # Tested by test_mistral.py
    # "Deci/DeciLM-7b",  # Broken
    # "tiiuae/falcon-7b",  # Broken
    "EleutherAI/gpt-j-6b",
    # "mosaicml/mpt-7b",  # Broken
    # "Qwen/Qwen1.5-0.5B"  # Broken,
]

#TODO: remove this after CPU float16 support ready
target_dtype = "float"
if torch.cuda.is_available():
    target_dtype = "half"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [32])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", [target_dtype])
def test_model_print(
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    with vllm_runner(model, dtype=dtype) as vllm_model:
        # This test is for verifying whether the model's extra_repr
        # can be printed correctly.
        print(vllm_model.model.llm_engine.model_executor.driver_worker.
              model_runner.model)
