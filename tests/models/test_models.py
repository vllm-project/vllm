"""Compare the outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/models/test_models.py --forked`.
"""
import pytest
from vllm.sampling_params import SamplingParams

MODELS = [
    "facebook/opt-125m",
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
    "tiiuae/falcon-7b",
    "gpt2",
    "bigcode/tiny_starcoder_py",
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-70m",
    "bigscience/bloom-560m",
    "mosaicml/mpt-7b",
    "microsoft/phi-1_5",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
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
@pytest.mark.parametrize("max_tokens", [128])
def test_models_from_prompt_embeds(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    vllm_model = vllm_runner(model, dtype=dtype)
    tokenizer = vllm_model.model.llm_engine.tokenizer
    input_embeddings = vllm_model.model.llm_engine.workers[
        0].model_runner.model.get_input_embeddings()

    prompt_embeds = []
    for prompt in example_prompts:
        token_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        token_embeds = input_embeddings(token_ids)
        prompt_embeds.append(token_embeds[0])

    outputs_from_prompts = vllm_model.model.generate(
        example_prompts,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens),
        prompt_embeds=None)
    outputs_from_embeds = vllm_model.model.generate(
        None,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens),
        prompt_embeds=prompt_embeds,
    )
    del vllm_model

    for output_prompt, output_embed in zip(outputs_from_prompts,
                                           outputs_from_embeds):
        assert output_prompt.outputs[0].token_ids == output_embed.outputs[
            0].token_ids, (
                f"output_prompt: {output_prompt.outputs[0].token_ids}\n",
                f"output_embed: {output_embed.outputs[0].token_ids}",
            )
