import asyncio
import gc
import os
import time
from typing import Dict, List, Type

import openai
import pytest
import torch
from datasets import load_dataset
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from tests.conftest import HfRunnerNM
from tests.models.compare_utils import check_logprobs_close
from tests.nm_utils.logging import make_logger
from tests.nm_utils.server import ServerContext
from tests.nm_utils.utils_skip import should_skip_test_group

if should_skip_test_group(group_name="TEST_MODELS_CORE"):
    pytest.skip("TEST_MODELS_CORE=DISABLE, skipping core model test group",
                allow_module_level=True)

# Silence warning.
os.environ["TOKENIZERS_PARALLELISM"] = "True"

NUM_SAMPLES_TO_RUN = 20
NUM_CHAT_TURNS = 3  # << Should be an odd number.
REQUEST_RATE = 2.5
GPU_COUNT = torch.cuda.device_count()
device_capability = torch.cuda.get_device_capability()
DEVICE_CAPABILITY = device_capability[0] * 10 + device_capability[1]

MODELS = [
    # Llama (8B param variant)
    "meta-llama/Meta-Llama-3-8B-Instruct",
]


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
    num_logprobs: int,
):
    """ submit a single prompt chat and collect results. """
    return await client.chat.completions.create(model=model,
                                                messages=messages,
                                                max_tokens=max_tokens,
                                                temperature=0,
                                                logprobs=True,
                                                top_logprobs=num_logprobs)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("tensor_parallel_size", [1])
def test_models_on_server(
    hf_runner_nm: HfRunnerNM,
    client: AsyncOpenAI,
    model: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
) -> None:
    """
    This test compares the output of the vllm OpenAI server against that of
    a HuggingFace transformer.  We expect them to be fairly close.  "Close"
    is measured by checking that the top N logprobs for each token includes
    the token of the other inference tool.  The first time that there is no
    exact match, as long as there is a match to one of the top `num_logprobs`
    logprobs, the test will not proceed further, but will pass.

    :param hf_runner_nm:  fixture for the HfRunnerNM
    :param client: fixture with an openai.AsyncOpenAI client
    :param model:  The Hugginface id for a model to test with
    :param max_tokens: the maximum number of tokens to generate
    :param num_logprobs: the total number of logprobs checked for "close enough"
    :param tensor_parallel_size: passed to the vllm Server launch
    """
    logger = make_logger("vllm_test")

    # Check that we have enough GPUs to run the test.
    if tensor_parallel_size > 1 and tensor_parallel_size > GPU_COUNT:
        pytest.skip(f"gpu count {GPU_COUNT} is insufficient for "
                    f"tensor_parallel_size = {tensor_parallel_size}")

    # Load dataset.
    logger.info("Loading dataset and converting to chat format.")
    ds = load_dataset("nm-testing/qa-chat-prompts",
                      split="train_sft").select(range(NUM_SAMPLES_TO_RUN))
    messages_list = [row["messages"][:NUM_CHAT_TURNS] for row in ds]
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Note: its very important to tokenize here due to silliness
    # around how the tokenizer works.
    #
    #   The following examples are not equivalent:
    #
    #   -----
    #   prompt = tokenizer.apply_chat_template(message)
    #   -----
    #   prompt = tokenizer.apply_chat_template(
    #       message, tokenize=False)                << adds bos
    #   input_ids = tokenizer(prompt).input_ids     << also adds bos
    #   -----
    input_ids_lst = [
        tokenizer.apply_chat_template(messages,
                                      return_tensors="pt",
                                      add_generation_prompt=True).to("cuda")
        for messages in messages_list
    ]

    logger.info("Generating chat responses from HF transformers.")
    hf_model = hf_runner_nm(model)
    hf_outputs = hf_model.generate_greedy_logprobs_nm_use_tokens(
        input_ids_lst, max_tokens, num_logprobs)
    # Make sure all the memory is cleaned up.
    del hf_model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1.0)

    logger.info("Generating chat responses from vLLM server.")
    api_server_args = {
        "--model": model,
        "--max-model-len": 4096,
        "--tensor-parallel-size": tensor_parallel_size,
    }

    # bfloat16 requires at least Ampere. Set to float16 otherwise.
    if DEVICE_CAPABILITY < 80:
        api_server_args["--dtype"] = "half"

    # TODO: Update this to work like the benchmark script.
    asyncio_event_loop = asyncio.get_event_loop()
    with ServerContext(api_server_args, logger=logger) as _:
        chats = []
        for messages in messages_list:
            chats.append(
                my_chat(client, model, messages, max_tokens, num_logprobs))
        # Gather results.
        results = asyncio_event_loop.run_until_complete(asyncio.gather(*chats))

    logger.info("Processing raw data from vLLM server.")
    vllm_outputs = []

    # See https://platform.openai.com/docs/api-reference/chat/create
    for result in results:
        req_output = result.choices[0]
        output_str = req_output.message.content

        # Unpack from req_output.logprobs.content
        # logprobs.content                      < list of list of token data
        # logprobs.content[i].token             < sampled token
        # logprobs.content[i].top_logprobs      < top logprobs
        # logprobs.content[i].top_logprobs[j].token
        # logprobs.content[i].top_logprobs[j].logprob

        output_tokens = []
        output_logprobs = []
        for token_data in req_output.logprobs.content:
            # Actual sampled token.
            output_tokens.append(token_data.token)
            # Convert TopLogProb --> List[Dict[token, logprob]]
            top_logprobs = {}
            for top_logprob in token_data.top_logprobs:
                top_logprobs[top_logprob.token] = top_logprob.logprob
            output_logprobs.append(top_logprobs)
        vllm_outputs.append((output_tokens, output_str, output_logprobs))

    logger.info("Comparing results.")
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf_model",
        name_1="vllm_model",
    )
