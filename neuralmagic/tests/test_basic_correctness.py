import logging

import openai
import pytest
from datasets import load_dataset

from neuralmagic.tools.vllm_server import VllmServer
from tests.conftest import HfRunnerNM
from tests.models.compare_utils import check_logprobs_close


@pytest.fixture(scope="session")
def client():
    client = openai.AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )
    yield client


@pytest.fixture
def hf_runner_nm():
    return HfRunnerNM


@pytest.mark.parametrize(
    "model, max_model_len, sparsity",
    [
        ("mistralai/Mistral-7B-Instruct-v0.2", 4096, None),
        # ("mistralai/Mixtral-8x7B-Instruct-v0.1", 4096, None),
        # ("neuralmagic/zephyr-7b-beta-marlin", 4096, None),
        # ("neuralmagic/OpenHermes-2.5-Mistral-7B-pruned50",
        #  4096, "sparse_w16a16"),
        # "NousResearch/Llama-2-7b-chat-hf",
        # "neuralmagic/TinyLlama-1.1B-Chat-v1.0-marlin",
        # ("neuralmagic/Llama-2-7b-pruned70-retrained-ultrachat",
        #  4096, "--sparsity sparse_w16a16"),
        # "HuggingFaceH4/zephyr-7b-gemma-v0.1",
        # ("Qwen/Qwen1.5-7B-Chat", 4096, None),
        # ("microsoft/phi-2", 2048, None),
        # ("neuralmagic/phi-2-super-marlin", 2048, None),
        # ("neuralmagic/phi-2-pruned50", 2048, "sparse_w16a16"),
        # ("mistralai/Mixtral-8x7B-Instruct-v0.1", 4096, None),
        # ("Qwen/Qwen1.5-MoE-A2.7B-Chat", 4096, None),
        # ("casperhansen/gemma-7b-it-awq", 4096, None),
        # ("TheBloke/Llama-2-7B-Chat-GPTQ", 4096, None),
    ])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [3])
@pytest.mark.asyncio
async def test_models_on_server(
    hf_runner_nm,
    client,
    model: str,
    max_model_len: int,
    sparsity: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:

    ds = load_dataset("nm-testing/qa-chat-prompts", split="train_sft")
    example_prompts = [m[0]["content"] for m in ds["messages"]]
    hf_model = hf_runner_nm(model)
    hf_outputs = hf_model.generate_greedy_logprobs_nm_use_tokens(
        example_prompts, max_tokens, num_logprobs)

    del hf_model

    logger = logging.Logger("vllm_server")
    api_server_args = {
        "--model": model,
        "--max-model-len": max_model_len,
        "--disable-log-requests": None,
    }
    if sparsity:
        api_server_args["--sparsity"] = sparsity

    with VllmServer(api_server_args, logger):
        completion = await client.completions.create(model=model,
                                                     prompt=example_prompts,
                                                     max_tokens=max_tokens,
                                                     temperature=0.0,
                                                     logprobs=num_logprobs)

    vllm_outputs = []
    for req_output in completion.choices:
        output_str = req_output.text
        output_tokens = req_output.logprobs.tokens
        output_logprobs = req_output.logprobs.top_logprobs
        vllm_outputs.append((output_tokens, output_str, output_logprobs))

    # loop through the prompts
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf_model",
        name_1="vllm_model",
    )

    # now repeat using two gpus
    # specifically doing it here, rather than as a pytest param,
    # to avoid repeating the huggingface inference and data collection
    api_server_args["--tensor-parallel-size"] = 2
    with VllmServer(api_server_args, logger):
        completion = await client.completions.create(model=model,
                                                     prompt=example_prompts,
                                                     max_tokens=max_tokens,
                                                     temperature=0.0,
                                                     logprobs=num_logprobs)

    vllm_outputs = []
    for req_output in completion.choices:
        output_str = req_output.text
        output_tokens = req_output.logprobs.tokens
        output_logprobs = req_output.logprobs.top_logprobs
        vllm_outputs.append((output_tokens, output_str, output_logprobs))

    # loop through the prompts
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf_model",
        name_1="vllm_model",
    )
