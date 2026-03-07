# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import logging
import os
import random

import pytest
import pytest_asyncio
import yaml

from tests.declarative_tests.approx import approx
from tests.utils import RemoteOpenAIServer

TESTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)),
                 "..")) + "/declarative_tests/"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.fixture
def server_fixture(request):
    model_name = request.node.get_closest_marker("model").args[0]
    ost = request.node.get_closest_marker("output_special_tokens")
    ost = ost is not None and ost.args[0]
    args = [
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--max-num-seqs",
        "128",
    ]
    with RemoteOpenAIServer(model_name, args) as server:
        yield {"server": server, "model": model_name}


@pytest.fixture
def test_cases(request):
    filename = request.node.get_closest_marker("test_case_file").args[0]
    with open(os.path.join(TESTS_DIR, filename)) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


@pytest.mark.model("google/gemma-2-2b-it")
@pytest.mark.test_case_file("test_cases_gemma-2-2b-it.yaml")
@pytest.mark.asyncio
async def test_gemma_2_2b_it(server_fixture, test_cases):
    await run_test_cases_async(server_fixture, test_cases)


@pytest.mark.model("HuggingFaceH4/zephyr-7b-beta")
@pytest.mark.test_case_file("test_cases_zephyr-7b-beta.yaml")
@pytest.mark.asyncio
async def test_zephyr(server_fixture, test_cases):
    await run_test_cases_async(server_fixture, test_cases)


async def run_test_cases_async(server_fixture,
                               test_cases,
                               seq2seq_model=False,
                               sharded=False):
    random.shuffle(test_cases)
    server_fixture["server"].get_async_client(timeout=5)
    for case in test_cases:
        await run_unary_test_case(server_fixture, case)


async def run_unary_test_case(server_fixture, case):
    request = case["request"]
    expected = case.get("response")

    client = server_fixture["server"].get_async_client(timeout=5)

    if request["params"].get("sampling_params") is not None:
        temperature = request["params"].get("sampling_params")["temperature"]
        seed = request["params"].get("sampling_params")["seed"]
    else:
        temperature = 0.0
        seed = 0

    stop = None
    logprobs = False
    if "logprobs" in expected["choices"]:
        logprobs = True
    if request["params"].get("stop") is not None:
        stop = request["params"].get("stop")

    response = await client.chat.completions.create(
        messages=request["requests"],
        model=server_fixture["model"],
        max_tokens=request["params"]["max_tokens"],
        max_completion_tokens=request["params"]["max_completion_tokens"],
        temperature=temperature,
        seed=seed,
        stop=stop,
        logprobs=logprobs,
        top_logprobs=0)

    print_case_details(case)
    logger.info(response)
    dic_value = filter_response_by_expected(response.to_dict(), expected)
    assert dic_value["choices"][0]["message"]["content"] == expected[
        "choices"]["message"]
    assert dic_value["usage"] == expected["usage"]
    assert dic_value["choices"][0]["finish_reason"] == expected["choices"][
        "finish_reason"]
    if logprobs:
        assert get_logprobs(response) == approx(
            expected["choices"]["logprobs"], rel=6e-2, abs=6e-2)


# Extracts log probabilities from chat completion response
# Args:
#   response: Chat completion response object from OpenAI client
# Returns:
#   List of log probabilities for each token in the completion
#   Example: [-1.234, -0.567, -0.890]
def get_logprobs(response):
    contents = response.choices[0].logprobs.content
    logprobs = []
    for token in contents:
        # print(token.logprob)
        logger.info(token.logprob)
        logprobs.append(token.logprob)
    return logprobs


# Recursively filter response dict to keep only keys present in expected dict
def filter_response_by_expected(response, expected):
    if not isinstance(response, dict) or not isinstance(expected, dict):
        return response

    current_keys = list(response.keys())

    for key in current_keys:
        if key not in expected:
            del response[key]
        else:
            val1 = response[key]
            val2 = expected[key]
            if isinstance(val1, dict) and isinstance(val2, dict):
                filter_response_by_expected(val1, val2)

    return response


def print_case_details(case):
    logger.info("----------------------Case details------------------------")
    name = case["name"]
    request = case["request"]
    skip_check = "skip_check" in case
    expected = case.get("response")
    expected_err = case.get("error")
    logger.info("Case name: %s", name)
    logger.info("request: %s", request)
    logger.info("skip_check: %s", skip_check)
    logger.info("expected: %s", expected)
    logger.info("expected_err: %s", expected_err)

    logger.info("----------------------------------------------------------")


# To avoid errors related to event loop shutdown timing
@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()
