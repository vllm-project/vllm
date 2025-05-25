# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from typing import Optional

import openai  # use the official client for correctness check
import pydantic
import pytest
import pytest_asyncio
import regex as re
from httpx import Request, Response
from openai import BadRequestError

from tests.utils import LocalOpenAIServer
from vllm.entrypoints.openai.protocol import (CompletionRequest,
                                              CompletionResponse,
                                              ErrorResponse)
from vllm.transformers_utils.tokenizer import get_tokenizer

# any model with a chat template should work here
MODEL_NAME = "facebook/opt-125m"


@pytest.fixture(scope="module")
def default_server_args():
    return [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager"
    ]


@pytest_asyncio.fixture(scope="module",
                        params=[["--no-enable-prefix-caching"],
                                [
                                    "--no-enable-prefix-caching",
                                    "--disable-frontend-multiprocessing"
                                ]])
async def local_openai_server(default_server_args, request):
    server_args = list(default_server_args)
    if hasattr(request, "param") and request.param:
        server_args.extend(request.param)
    async with LocalOpenAIServer("facebook/opt-125m",
                                 server_args) as local_server:
        yield local_server


@pytest_asyncio.fixture(scope="module")
async def client():
    yield openai.AsyncOpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8001/v1",
        max_retries=0,
    )


@pytest.fixture
def request_handler(local_openai_server):
    return make_handle_request_no_streaming(local_openai_server.server_logic)


@pytest.fixture(autouse=True)
def setup_httpx_mock(httpx_mock, request_handler):
    httpx_mock.add_callback(
        request_handler,
        url="http://localhost:8001/v1/completions",
        method="POST",
        is_reusable=True,
        is_optional=True,
    )


def make_handle_request_no_streaming(server_logic):

    async def handle_request(request: Request) -> Response:
        request_body = json.loads(request.content.decode("utf-8"))
        try:
            completion_request = CompletionRequest(**request_body)
        except pydantic.ValidationError as e:
            return Response(status_code=400,
                            json={"error": {
                                "message": str(e)
                            }})
        generator = await server_logic.create_completion(completion_request)
        if isinstance(generator, ErrorResponse):
            return Response(json=generator.model_dump(),
                            status_code=generator.code)
        elif isinstance(generator, CompletionResponse):
            return Response(json=generator.model_dump(), status_code=200)
        else:
            return Response(
                status_code=400,
                json={
                    "detail":
                    "pytest-httpx mock does not support streaming response."
                },
            )

    return handle_request


def convert_to_json(chunk_str: str):

    def dict_to_obj(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_obj(i) for i in d]
        else:
            return d

    chunk_data = chunk_str[len("data: "):].strip()
    chunk = dict_to_obj(json.loads(chunk_data))
    return chunk


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_single_completion(client: openai.AsyncOpenAI, model_name: str):

    completion = await client.completions.create(model=model_name,
                                                 prompt="Hello, my name is",
                                                 max_tokens=5,
                                                 temperature=0.0)

    assert completion.id is not None
    assert completion.choices is not None and len(completion.choices) == 1

    choice = completion.choices[0]
    assert len(choice.text) >= 5
    assert choice.finish_reason == "length"
    assert completion.usage == openai.types.CompletionUsage(
        completion_tokens=5, prompt_tokens=6, total_tokens=11)

    # test using token IDs
    completion = await client.completions.create(
        model=model_name,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
    )
    assert len(completion.choices[0].text) >= 1
    assert completion.choices[0].prompt_logprobs is None


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_no_logprobs(client: openai.AsyncOpenAI, model_name: str):
    completion = await client.completions.create(
        model=model_name,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
        logprobs=None,
    )
    choice = completion.choices[0]
    assert choice.logprobs is None


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_zero_logprobs(client: openai.AsyncOpenAI, model_name: str):
    # test using token IDs
    completion = await client.completions.create(
        model=model_name,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
        logprobs=0,
    )
    choice = completion.choices[0]
    assert choice.logprobs is not None
    assert choice.logprobs.token_logprobs is not None
    assert choice.logprobs.top_logprobs is not None
    assert len(choice.logprobs.top_logprobs[0]) == 1


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_some_logprobs(client: openai.AsyncOpenAI, model_name: str):
    # test using token IDs
    completion = await client.completions.create(
        model=model_name,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
        logprobs=5,
    )
    choice = completion.choices[0]
    assert choice.logprobs is not None
    assert choice.logprobs.token_logprobs is not None
    assert choice.logprobs.top_logprobs is not None
    assert 5 <= len(choice.logprobs.top_logprobs[0]) <= 6


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize("model_name, prompt_logprobs", [(MODEL_NAME, -1),
                                                         (MODEL_NAME, 0),
                                                         (MODEL_NAME, 1),
                                                         (MODEL_NAME, None)])
async def test_prompt_logprobs_completion(client: openai.AsyncOpenAI,
                                          model_name: str,
                                          prompt_logprobs: Optional[int]):
    params: dict = {
        "prompt": ["A robot may not injure another robot", "My name is"],
        "model": model_name,
    }
    if prompt_logprobs is not None:
        params["extra_body"] = {"prompt_logprobs": prompt_logprobs}

    if prompt_logprobs is not None and prompt_logprobs < 0:
        with pytest.raises(BadRequestError):
            await client.completions.create(**params)
    else:
        completion = await client.completions.create(**params)
        if prompt_logprobs is not None:
            assert completion.choices[0].prompt_logprobs is not None
            assert len(completion.choices[0].prompt_logprobs) > 0

            assert completion.choices[1].prompt_logprobs is not None
            assert len(completion.choices[1].prompt_logprobs) > 0

        else:
            assert completion.choices[0].prompt_logprobs is None


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_parallel_no_streaming(client: openai.AsyncOpenAI,
                                     model_name: str):
    """Parallel sampling without streaming.
    A single request output contains a list of completions.
    """

    prompt = "What is an LLM?"
    n = 3
    max_tokens = 50  # we want some to finish earlier than others

    # High temperature to maximize chance of unique completions.
    completion = await client.completions.create(model=model_name,
                                                 prompt=prompt,
                                                 max_tokens=max_tokens,
                                                 n=n,
                                                 temperature=1.0,
                                                 stream=False,
                                                 logprobs=0,
                                                 seed=42)

    # Assert `n` completions
    num_completions = len(completion.choices)
    assert num_completions == n, (
        f"Num completions {num_completions} but expected {n}.")
    completion_repeats: dict[str, int] = {}
    output_token_lengths = set()
    for idx, choice in enumerate(completion.choices):
        # Assert correct completion index & some finish reason.
        assert choice.index == idx, (
            f"Index {choice.index} but expected {idx}.")
        assert choice.finish_reason is not None, (
            "None finish_reason is invalid.")
        text = choice.text
        completion_repeats[text] = completion_repeats.get(text, 0) + 1
        output_token_lengths.add(len(choice.logprobs.tokens))
    # Assert subrequests finished at different times
    assert len(output_token_lengths) > 1
    # Assert `n` unique completions
    num_unique = len(completion_repeats)
    if num_unique != n:
        repeats = {
            txt: num
            for (txt, num) in completion_repeats.items() if num > 1
        }
        raise AssertionError(
            f"Expected {n} unique completions, got {num_unique};"
            f" repeats: {repeats}.")


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
@pytest.mark.parametrize("logprobs_arg", [1, 0])
async def test_echo_logprob_completion(client: openai.AsyncOpenAI,
                                       model_name: str, logprobs_arg: int):
    tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
    # test using text and token IDs
    for prompt in ("Hello, my name is", [0, 0, 0, 0, 0]):
        completion = await client.completions.create(model=model_name,
                                                     prompt=prompt,
                                                     max_tokens=5,
                                                     temperature=0.0,
                                                     echo=True,
                                                     logprobs=logprobs_arg)

        prompt_text = tokenizer.decode(prompt) if isinstance(prompt,
                                                             list) else prompt
        assert re.search(r"^" + prompt_text, completion.choices[0].text)
        logprobs = completion.choices[0].logprobs
        assert logprobs is not None
        assert len(logprobs.text_offset) > 5
        assert (len(logprobs.token_logprobs) > 5
                and logprobs.token_logprobs[0] is None)
        assert (len(logprobs.top_logprobs) > 5
                and logprobs.top_logprobs[0] is None)
        for top_logprobs in logprobs.top_logprobs[1:]:
            assert max(logprobs_arg,
                       1) <= len(top_logprobs) <= logprobs_arg + 1
        assert len(logprobs.tokens) > 5


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_invalid_json_schema(local_openai_server,
                                   client: openai.AsyncOpenAI,
                                   model_name: str) -> None:
    invalid_json_schema = {
        "$defs": {
            "CarType": {
                "enum": ["sedan", "SUV", "Truck", "Coupe"],
                "title": "CarType",
                "type": "string",
            }
        },
        "properties": {
            "brand": {
                "title": "Brand",
                "type": "string"
            },
            "model": {
                "title": "Model",
                "type": "string"
            },
            "car_type": {
                "$ref": "#/$defs/CarType"
            },
            "foo": "bar",
        },
        "required": ["brand", "model", "car_type"],
        "title": "CarDescription",
        "type": "object",
    }
    prompt = ("Generate a JSON with the brand, model and car_type of"
              "the most iconic car from the 90's")
    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.completions.create(
            model=model_name,
            prompt=prompt,
            extra_body={"guided_json": invalid_json_schema},
        )


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_invalid_regex(local_openai_server, client: openai.AsyncOpenAI,
                             model_name: str):
    prompt = ("Generate an email address for Alan Turing, who works in Enigma."
              "End in .com and new line. Example result:"
              "alan.turing@enigma.com\n")

    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.completions.create(
            model=model_name,
            prompt=prompt,
            extra_body={
                "guided_regex": r"[.*",
                "stop": ["\n"]
            },
        )


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_invalid_grammar(local_openai_server, client: openai.AsyncOpenAI,
                               model_name: str):
    invalid_simplified_sql_grammar = """
        root ::= select_statementinvalidsyntax

        select_statement ::= "SELECT " column " from " table " where " condition

        column ::= "col_1 " | "col_2 "

        table ::= "table_1 " | "table_2 "

        condition ::= column "= " number

        number ::= "1 " | "2 "
    """

    prompt = ("Generate an SQL query to show the 'username' and 'email'"
              "from the 'users' table.")
    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.completions.create(
            model=model_name,
            prompt=prompt,
            extra_body={"guided_grammar": invalid_simplified_sql_grammar},
        )
