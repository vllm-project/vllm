"""Compare the outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/models/test_models.py --forked`.
"""
import pytest

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
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    prompt_embeds = []
    for prompt in example_prompts:
        token_ids = hf_model.tokenizer(
            prompt, return_tensors="pt").input_ids.to("cuda")
        token_embeds = hf_model.model.get_input_embeddings()(token_ids)
        prompt_embeds.append(token_embeds[0])
    del hf_model

    vllm_model = vllm_runner(model, dtype=dtype)
    vllm_outputs_from_prompts = vllm_model.generate_greedy(example_prompts,
                                                           max_tokens,
                                                           prompt_embeds=None)
    vllm_outputs_from_embeds = vllm_model.generate_greedy(
        example_prompts, max_tokens, prompt_embeds=prompt_embeds)
    del vllm_model

    for i in range(len(example_prompts)):
        prompt = example_prompts[i]
        hf_output_str = hf_outputs[i][0]
        vllm_output_str_from_prompts = vllm_outputs_from_prompts[i][0]
        vllm_output_str_from_embeds = vllm_outputs_from_embeds[i][0]

        assert hf_output_str == vllm_output_str_from_prompts, (
            f"Test{i}:\n"
            "HF: {hf_output_str!r}\n"
            "vLLM_prompt: {vllm_output_str_from_prompts!r}")
        assert hf_output_str == vllm_output_str_from_embeds, (
            f"Test{i}:\n"
            "HF: {hf_output_str}\n"
            "vLLM_embeds: {vllm_output_str_from_embeds}")
        assert vllm_output_str_from_prompts == vllm_output_str_from_embeds, (
            f"Test{i}:\n"
            "vLLM_prompt: {vllm_output_str_from_prompts}\n"
            "vLLM_embeds: {vllm_output_str_from_embeds}")
