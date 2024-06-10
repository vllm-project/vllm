"""Compare the outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/models/test_models_logprobs.py --forked`.
"""
import pytest

from tests.models.utils import check_logprobs_close

MODEL_MAX_LEN = 1024

MODELS = [
    "facebook/opt-125m",
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
    "Deci/DeciLM-7b",
    "tiiuae/falcon-7b",
    "gpt2",
    "bigcode/tiny_starcoder_py",
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-1b",
    "bigscience/bloom-1b1",
    "mosaicml/mpt-7b",
    "microsoft/phi-2",
    "stabilityai/stablelm-3b-4e1t",
    "allenai/OLMo-1B",
    "bigcode/starcoder2-3b",
    "Qwen/Qwen1.5-0.5B",
]

SKIPPED_MODELS_ACC = [
    "mosaicml/mpt-7b",
    "allenai/OLMo-1B",
    "bigcode/starcoder2-3b",
]

SKIPPED_MODELS_OOM = [
    "EleutherAI/gpt-j-6b",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [3])
def test_models(
    vllm_runner_nm,
    hf_runner_nm,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    if model in SKIPPED_MODELS_ACC:
        pytest.skip(reason="Low priority models not currently passing. "
                    "We need to re-enable these.")
    if model in SKIPPED_MODELS_OOM:
        pytest.skip(reason="These models cause OOM issue on the CPU"
                    "because it is a fp32 checkpoint.")

    hf_model = hf_runner_nm(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy_logprobs_nm(example_prompts,
                                                      max_tokens, num_logprobs)

    del hf_model

    trust_remote_code = True
    # Falcon fails if trust_remote_code = True
    # https://github.com/vllm-project/vllm/issues/5363
    trust_remote_code = model != "tiiuae/falcon-7b"
    vllm_model = vllm_runner_nm(model,
                                dtype=dtype,
                                trust_remote_code=trust_remote_code,
                                max_model_len=MODEL_MAX_LEN)
    vllm_outputs = vllm_model.generate_greedy_logprobs(example_prompts,
                                                       max_tokens,
                                                       num_logprobs)

    del vllm_model

    # loop through the prompts
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf_model",
        name_1="vllm_model",
    )
