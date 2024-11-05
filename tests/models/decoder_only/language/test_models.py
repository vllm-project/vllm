"""Compare the outputs of HF and vLLM when using greedy sampling.

This test only tests small models. Big models such as 7B should be tested from
test_big_models.py because it could use a larger instance to run tests.

Run `pytest tests/models/test_models.py`.
"""
import pytest

from vllm.platforms import current_platform

from ...utils import check_logprobs_close

MODELS = [
    "facebook/opt-125m",  # opt
    "openai-community/gpt2",  # gpt2
    # "Milos/slovak-gpt-j-405M",  # gptj
    # "bigcode/tiny_starcoder_py",  # gpt_bigcode
    # "EleutherAI/pythia-70m",  # gpt_neox
    "bigscience/bloom-560m",  # bloom - testing alibi slopes
    "microsoft/phi-2",  # phi
    # "stabilityai/stablelm-3b-4e1t",  # stablelm
    # "bigcode/starcoder2-3b",  # starcoder2
    "google/gemma-1.1-2b-it",  # gemma
    "Qwen/Qwen2.5-0.5B-Instruct",  # qwen2
    "meta-llama/Llama-3.2-1B-Instruct",  # llama
]

if not current_platform.is_cpu():
    MODELS += [
        # fused_moe which not supported on CPU
        "openbmb/MiniCPM3-4B",
    ]

# TODO: remove this after CPU float16 support ready
target_dtype = "float" if current_platform.is_cpu() else "half"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs)

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)
        # This test is for verifying whether the model's extra_repr
        # can be printed correctly.
        print(vllm_model.model.llm_engine.model_executor.driver_worker.
              model_runner.model)

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
