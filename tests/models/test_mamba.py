"""Compare the outputs of HF and vLLM when using greedy sampling for Mamba.

Run `pytest tests/models/test_mamba.py`.
"""
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import check_outputs_equal

MODELS = [
    "state-spaces/mamba-370m-hf",
]


# Use lower-level interfaces to create this greedy generator, as mamba will
# choke on the model_kwarg 'attention_mask' if hf_model.generate_greedy is used.
def generate_greedy(model_name, example_prompts, max_tokens):
    # Create a text generation pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Generate texts from the prompts
    outputs = []
    for prompt in example_prompts:
        # Tokenize the input prompt with truncation
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(model.device)

        # Generate text using the model's generate method directly
        generated_ids = model.generate(input_ids, max_new_tokens=max_tokens)
        generated_text = tokenizer.decode(generated_ids[0],
                                          skip_special_tokens=True)

        outputs.append((generated_ids[0].tolist(), generated_text))

    return outputs


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

    hf_outputs = generate_greedy(model, example_prompts, max_tokens)

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
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
