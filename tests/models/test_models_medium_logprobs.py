"""Compares the outputs of hf vs vllm for medium sized models.

There is not bitwise correctness for fp16 inference.
As a result, in this test, we just confirm that the top selected tokens of the
Marlin/GPTQ models are in the top 3 selections of each other.

Run `pytest tests/models/test_models_medium_logprobs.py`.
"""
import pytest

from tests.models.utils import check_logprobs_close


MAX_MODEL_LEN = 1024

MODELS = [
    "baichuan-inc/Baichuan2-7B-Chat",
    "bigscience/bloom-560m",
    "THUDM/chatglm3-6b",
    # command-r
    # dbrx
    "Deci/DeciLM-7B-instruct",
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    "tiiuae/falcon-7b-instruct",
    "google/gemma-1.1-2b-it",
    "gpt2",
    "bigcode/tiny_starcoder_py",
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-70m",
    "internlm/internlm2-chat-7b",
    # jais
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # llava
    "openbmb/MiniCPM-2B-128k",
    # mixtral
    # mixtral-quant
    "mosaicml/mpt-7b-instruct",
    "allenai/OLMo-1B",
    "facebook/opt-125m",
    # orion
    "microsoft/phi-2",
    "Qwen/Qwen-1_8B",
    "Qwen/Qwen1.5-1.8B",
    # qwen2 moe
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
    # Run HF.
    hf_model = hf_runner(model_name=model,
                         dtype=dtype,
                         max_model_len=MAX_MODEL_LEN,
                         tensor_parallel_size=1)
    hf_outputs = hf_model.generate_greedy_logprobs(
        example_prompts, max_tokens, num_logprobs)
    del hf_model

    # Run vLLM.
    vllm_model = vllm_runner(model_name=model,
                             dtype=dtype,
                             max_model_len=MAX_MODEL_LEN,
                             tensor_parallel_size=1)
    vllm_outputs = vllm_model.generate_greedy_logprobs(
        example_prompts, max_tokens, num_logprobs)
    del gptq_model

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
