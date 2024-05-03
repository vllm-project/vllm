"""Compare the outputs of HF and vLLM when using greedy sampling.

This tests bigger models and use half precision.

Run `pytest tests/models/test_big_models.py`.
"""
import pytest

MODELS = [
    "meta-llama/Llama-2-7b-hf",
    # "mistralai/Mistral-7B-v0.1",  # Broken
    # "Deci/DeciLM-7b",  # Broken
    # "tiiuae/falcon-7b",  # Broken
    "EleutherAI/gpt-j-6b",
    "mosaicml/mpt-7b",
    # "Qwen/Qwen1.5-0.5B"  # Broken,
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
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

    vllm_model = vllm_runner(model, dtype=dtype)
    vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    del vllm_model

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_model_print(
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    vllm_model = vllm_runner(model, dtype=dtype)
    # This test is for verifying whether the model's extra_repr
    # can be printed correctly.
    print(vllm_model.model.llm_engine.model_executor.driver_worker.
          model_runner.model)
    del vllm_model
