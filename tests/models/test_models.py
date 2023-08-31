from typing import Optional


def _test_model(
    hf_runner,
    vllm_runner,
    test_prompts,
    model: str,
    tokenizer: Optional[str] = None,
    max_tokens: int = 32,
) -> None:
    hf_model = hf_runner(model, tokenizer)
    hf_outputs = hf_model.generate_greedy(test_prompts, max_tokens)
    del hf_model

    vllm_model = vllm_runner(model, tokenizer)
    vllm_outputs = vllm_model.generate_greedy(test_prompts, max_tokens)
    del vllm_model

    for i in range(len(test_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")


def test_opt(hf_runner, vllm_runner, example_prompts) -> None:
    _test_model(
        hf_runner, vllm_runner, example_prompts, model="facebook/opt-125m")


def test_gpt2(hf_runner, vllm_runner, example_prompts) -> None:
    _test_model(hf_runner, vllm_runner, example_prompts, model="gpt2")


def test_gpt_bigcode(hf_runner, vllm_runner, example_prompts) -> None:
    _test_model(hf_runner, vllm_runner, example_prompts,
                model="bigcode/tiny_starcoder_py")


def test_gpt_j(hf_runner, vllm_runner, example_prompts) -> None:
    _test_model(
        hf_runner, vllm_runner, example_prompts, model="EleutherAI/gpt-j-6b")


def test_gpt_neox(hf_runner, vllm_runner, example_prompts) -> None:
    _test_model(
        hf_runner, vllm_runner, example_prompts, model="EleutherAI/pythia-70m")


def test_bloom(hf_runner, vllm_runner, example_prompts) -> None:
    _test_model(
        hf_runner, vllm_runner, example_prompts, model="bigscience/bloom-560m")


def test_mpt(hf_runner, vllm_runner, example_prompts) -> None:
    _test_model(
        hf_runner, vllm_runner, example_prompts, model="mosaicml/mpt-7b")


def test_falcon(hf_runner, vllm_runner, example_prompts) -> None:
    _test_model(
        hf_runner, vllm_runner, example_prompts, model="tiiuae/falcon-7b")


def test_llama(hf_runner, vllm_runner, example_prompts) -> None:
    _test_model(hf_runner, vllm_runner, example_prompts,
                model="meta-llama/Llama-2-7b-hf")
