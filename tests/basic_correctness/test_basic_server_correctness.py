import asyncio
from os import getenv
from typing import Dict, List, Type

import openai
import pytest
import torch
from datasets import load_dataset
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from tests.conftest import HfRunnerNM
from tests.models.compare_utils import check_logprobs_close
from tests.utils.logging import make_logger
from tests.utils.server import ServerContext
from vllm.model_executor.layers.quantization import get_quantization_config


@pytest.fixture(scope="session")
def client():
    client = openai.AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )
    yield client


@pytest.fixture
def hf_runner_nm() -> Type[HfRunnerNM]:
    return HfRunnerNM


async def my_chat(
    client,
    model: str,
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
    num_logprobs: int,
):
    """ submit a single prompt chat and collect results. """
    return await client.chat.completions.create(model=model,
                                                messages=messages,
                                                max_tokens=max_tokens,
                                                temperature=temperature,
                                                logprobs=True,
                                                top_logprobs=num_logprobs)


@pytest.mark.parametrize("model, max_model_len, sparsity, gptq_config", [
    ("mistralai/Mistral-7B-Instruct-v0.2", 4096, None, None),
    ("neuralmagic/OpenHermes-2.5-Mistral-7B-pruned50", 4096, "sparse_w16a16",
     None),
    ("NousResearch/Llama-2-7b-chat-hf", 4096, None, None),
    ("neuralmagic/Llama-2-7b-pruned70-retrained-ultrachat", 4096,
     "sparse_w16a16", None),
    ("Qwen/Qwen1.5-7B-Chat", 4096, None, None),
    ("casperhansen/gemma-7b-it-awq", 4096, None, "gptq_marlin"),
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", 4096, None, None),
    ("Qwen/Qwen1.5-MoE-A2.7B-Chat", 4096, None, None),
])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [3])
@pytest.mark.parametrize("tensor_parallel_size", [None])
# note: repeating the test for 2 values of tensor_parallel_size
#  increases the overall execution time by unnecessarily
#  collecting the HuggingFace runner data twice.
#  Consider refactoring to eliminate that repeat.
def test_models_on_server(
    hf_runner_nm: HfRunnerNM,
    client: AsyncOpenAI,
    model: str,
    max_model_len: int,
    sparsity: str,
    gptq_config: str,
    tensor_parallel_size: int,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    """
    This test compares the output of the vllm OpenAI server against that of
    a HuggingFace transformer.  We expect them to be fairly close.  "Close"
    is measured by checking that the top 3 logprobs for each token includes
    the token of the other inference tool.  The first time that there is no
    exact match, as long as there is a match to one of the top `num_logprobs`
    logprobs, the test will not proceed further, but will pass.

    Parameters to the test identify a model to test, and key arguments
    required for that model (see the `max_model_len`, `sparsity` and
    `gptq_config` params below). The additional parametrizations expand test
    coverage across the functional space of the server.

    :param hf_runner_nm:  fixture for the HfRunnerNM
    :param client: fixture with an openai.AsyncOpenAI client
    :param model:  The Hugginface id for a model to test with
    :param max_model_len: passed to the vllm Server's --max-model-len option
    :param sparsity: passed to the vllm Server's --sparsity option
    :param gptq_config: quantization method id for this model.  default None
        means quantization isn't involved.
    :param tensor_parallel_size: passed to the vllm Server's
        --tensor_parallel_size option
    :param max_tokens: the total number of tokens to consider for closeness
    :param num_logprobs:  the total number of logprobs included when
        calculating closeness
    """
    logger = make_logger("vllm_test")
    # check that the requested gpu count is available in the test env
    gpu_count = torch.cuda.device_count()
    if tensor_parallel_size and gpu_count < tensor_parallel_size:
        pytest.skip(f"gpu count {gpu_count} is insufficient for "
                    f"tensor_parallel_size = {tensor_parallel_size}")

    # skip this model if the current device does not have the required
    # gpu capability.
    device_capability = torch.cuda.get_device_capability()
    capability = device_capability[0] * 10 + device_capability[1]
    if gptq_config and (
            capability <
            get_quantization_config(gptq_config).get_min_capability()):
        pytest.skip("insufficient system GPU device capability "
                    f"({capability}) for this model")

    hf_token = getenv("HF_TOKEN", None)
    logger.info("loading chat prompts for testing.")
    ds = load_dataset("nm-testing/qa-chat-prompts", split="train_sft")
    num_chat_turns = 3
    messages_list = [row["messages"][:num_chat_turns] for row in ds]
    tokenizer = AutoTokenizer.from_pretrained(model)
    chat_prompts = [
        tokenizer.apply_chat_template(messages,
                                      tokenize=False,
                                      add_generation_prompt=True)
        for messages in messages_list
    ]

    logger.info("generating chat responses from HuggingFace runner.")
    hf_model = hf_runner_nm(model, access_token=hf_token)
    hf_outputs = hf_model.generate_greedy_logprobs_nm_use_tokens(
        chat_prompts, max_tokens, num_logprobs, ignore_special_tokens=True)

    del hf_model

    logger.info("generating chat responses from vllm server.")
    api_server_args = {
        "--model": model,
        "--max-model-len": max_model_len,
        "--disable-log-requests": None,
    }
    if sparsity:
        api_server_args["--sparsity"] = sparsity
    if tensor_parallel_size:
        api_server_args["--tensor-parallel-size"] = tensor_parallel_size

    # some devices will require a different `dtype`
    if device_capability[0] < 8:
        api_server_args["--dtype"] = "half"

    asyncio_event_loop = asyncio.get_event_loop()
    temperature = 0.0
    with ServerContext(api_server_args, logger=logger) as _:
        # submit an asynchronous request to the server for each prompt
        chats = [
            my_chat(client, model, messages, max_tokens, temperature,
                    num_logprobs)
            for messages in [query for query in messages_list]
        ]
        # await for all the requests to return, and gather their results
        # in one place
        results = asyncio_event_loop.run_until_complete(asyncio.gather(*chats))

    logger.info("preparing results from vllm server requests to include "
                "tokens and logprobs.")
    vllm_outputs = list()
    for task_result in results:
        for req_output in task_result.choices:
            output_str = req_output.message.content
            output_tokens = req_output.logprobs.model_extra["tokens"]
            output_logprobs = req_output.logprobs.model_extra["top_logprobs"]
            vllm_outputs.append((output_tokens, output_str, output_logprobs))

    logger.info("comparing HuggingFace and vllm Server chat responses")
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf_model",
        name_1="vllm_model",
    )
