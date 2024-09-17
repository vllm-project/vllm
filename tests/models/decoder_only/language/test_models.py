"""Compare the outputs of HF and vLLM when using greedy sampling.

This test only tests small models. Big models such as 7B should be tested from
test_big_models.py because it could use a larger instance to run tests.

Run `pytest tests/models/test_models.py`.
"""
import pytest

from ...utils import check_outputs_equal

MODELS = [
    "facebook/opt-125m",
    "gpt2",
    "bigcode/tiny_starcoder_py",
    "EleutherAI/pythia-70m",
    "bigscience/bloom-560m",  # Testing alibi slopes.
    "microsoft/phi-2",
    "stabilityai/stablelm-3b-4e1t",
    # "allenai/OLMo-1B",  # Broken
    "bigcode/starcoder2-3b",
    "google/gemma-1.1-2b-it",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [96])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    # To pass the small model tests, we need full precision.
    assert dtype == "float"

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
        prompt_embeds = []
        prompt_token_ids = []
        for prompt in example_prompts:
            token_ids = hf_model.tokenizer(prompt,
                                           return_tensors="pt").input_ids.to(
                                               hf_model.model.device)
            prompt_token_ids.append(token_ids)
            prompt_embeds.append(
                hf_model.model.get_input_embeddings()(token_ids).squeeze(0))

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
        vllm_outputs_from_embeds = vllm_model.generate_greedy(
            prompt_embeds, max_tokens)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )

    check_outputs_equal(
        outputs_0_lst=vllm_outputs,
        outputs_1_lst=[
            (prompt_ids.squeeze().tolist() + output_ids, prompt + output_str)
            for (output_ids, output_str), prompt_ids, prompt in zip(
                vllm_outputs_from_embeds, prompt_token_ids, example_prompts)
        ],
        name_0="vllm",
        name_1="vllm_from_embeds",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
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
