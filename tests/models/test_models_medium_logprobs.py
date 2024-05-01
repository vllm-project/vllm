"""Compares the outputs of hf vs vllm for medium sized models.

There is not bitwise correctness for fp16 inference.
As a result, in this test, we just confirm that the top selected tokens 
of the models are in the top 3 selections of each other.

Run `pytest tests/models/test_models_medium_logprobs.py` --forked.
"""
import pytest

from tests.models.utils import check_logprobs_close

SKIPPED_MODEL_REASON = {
    "THUDM/chatglm3-6b": "Hf side test broken",
    "allenai/OLMo-1B": "Hf side requirement conflict (req torch 2.2)",
    "xverse/XVERSE-7B": "Hf side test broken"
}

MAX_MODEL_LEN = 1024

MODELS = [
    "baichuan-inc/Baichuan2-7B-Chat",
    "bigscience/bloom-560m",
    "THUDM/chatglm3-6b",
    # command-r             -> not tested
    # dbrx                  -> not tested
    "Deci/DeciLM-7B-instruct",
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    "tiiuae/falcon-7b-instruct",
    "google/gemma-1.1-2b-it",
    "gpt2",
    "bigcode/tiny_starcoder_py",
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-1.4b",
    "internlm/internlm2-chat-7b",
    # jais                  -> not tested
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "openbmb/MiniCPM-2B-128k",
    # mixtral               -> not tested
    # mixtral-quant         -> not tested
    "mosaicml/mpt-7b-instruct",
    "allenai/OLMo-1B",
    "facebook/opt-125m",
    # orion                 -> not tested
    "microsoft/phi-2",
    "Qwen/Qwen-1_8B",
    "Qwen/Qwen1.5-1.8B",
    # qwen2 moe             -> not tested
    "stabilityai/stablelm-2-1_6b-chat",
    "bigcode/starcoder2-3b",
    "xverse/XVERSE-7B",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    vllm_runner,
    hf_runner,
    example_prompts,
    model,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    # Skip if explicitly skipped.
    if model in SKIPPED_MODEL_REASON:
        pytest.skip(reason=SKIPPED_MODEL_REASON[model])
    # Run HF.
    hf_model = hf_runner(model_name=model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy_logprobs_limit(
        example_prompts, max_tokens, num_logprobs)
    del hf_model

    # Run vLLM.
    vllm_model = vllm_runner(model_name=model,
                             enforce_eager=True,
                             dtype=dtype,
                             max_model_len=MAX_MODEL_LEN)
    vllm_outputs = vllm_model.generate_greedy_logprobs(example_prompts,
                                                       max_tokens,
                                                       num_logprobs)
    del vllm_model

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
