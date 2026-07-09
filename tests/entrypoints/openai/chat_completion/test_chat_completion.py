# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

# any model with a chat template defined in tokenizer_config should work here
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def default_server_args():
    return [
        # use half precision for speed and memory savings in CI environment
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
    ]


@pytest.fixture(scope="module")
def server(default_server_args):
    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_invalid_json_schema(client: openai.AsyncOpenAI, model_name: str) -> None:
    invalid_json_schema = {
        "$defs": {
            "CarType": {
                "enum": ["sedan", "SUV", "Truck", "Coupe"],
                "title": "CarType",
                "type": "string",
            }
        },
        "properties": {
            "brand": {"title": "Brand", "type": "string"},
            "model": {"title": "Model", "type": "string"},
            "car_type": {"$ref": "#/$defs/CarType"},
            "foo": "bar",
        },
        "required": ["brand", "model", "car_type"],
        "title": "CarDescription",
        "type": "object",
    }
    prompt = (
        "Generate a JSON with the brand, model and car_type of"
        "the most iconic car from the 90's"
    )
    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            extra_body={"structured_outputs": {"json": invalid_json_schema}},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_invalid_regex(client: openai.AsyncOpenAI, model_name: str):
    prompt = (
        "Generate an email address for Alan Turing, who works in Enigma."
        "End in .com and new line. Example result:"
        "alan.turing@enigma.com\n"
    )

    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            extra_body={"structured_outputs": {"regex": r"[.*"}, "stop": ["\n"]},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_invalid_grammar(client: openai.AsyncOpenAI, model_name: str):
    invalid_simplified_sql_grammar = """
        root ::= select_statementinvalidsyntax

        select_statement ::= "SELECT " column " from " table " where " condition

        column ::= "col_1 " | "col_2 "

        table ::= "table_1 " | "table_2 "

        condition ::= column "= " number

        number ::= "1 " | "2 "
    """

    prompt = (
        "Generate an SQL query to show the 'username' and 'email'"
        "from the 'users' table."
    )
    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            extra_body={
                "structured_outputs": {"grammar": invalid_simplified_sql_grammar}
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_empty_grammar(client: openai.AsyncOpenAI, model_name: str) -> None:
    prompt = "Say hello"
    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            extra_body={"structured_outputs": {"grammar": ""}},
        )


# Pre-tokenized input (token-in, text-out) on the chat completions API.

TOKEN_IN_MESSAGES = [{"role": "user", "content": "Hello, how are you today?"}]

IMAGE_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "http://example.com/x.png"}}
        ],
    }
]


@pytest.mark.asyncio
async def test_prompt_token_ids_round_trip(client: openai.AsyncOpenAI):
    """Chat accepts the templated prompt's own token ids and returns text.

    The engine receives exactly the ids the messages path produced, and the
    output is still detokenized (token-in, text-out). Generated text is not
    compared against the messages path because vLLM greedy decoding is not
    bitwise-reproducible across requests.
    """
    baseline = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=TOKEN_IN_MESSAGES,
        max_completion_tokens=16,
        temperature=0,
        extra_body={"return_token_ids": True},
    )
    prompt_token_ids = baseline.prompt_token_ids
    assert prompt_token_ids

    token_in = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[],
        max_completion_tokens=16,
        temperature=0,
        extra_body={"prompt_token_ids": prompt_token_ids, "return_token_ids": True},
    )

    # token-in feeds the engine exactly the ids from the messages path.
    assert token_in.prompt_token_ids == prompt_token_ids
    # text-out: pre-tokenized input still yields a detokenized message.
    assert token_in.choices[0].message.content


@pytest.mark.asyncio
async def test_prompt_token_ids_streaming(client: openai.AsyncOpenAI):
    """Token-in streams chat-formatted text-out.

    The scale-out derender endpoint is non-streaming, so streaming output from
    pre-tokenized input is only available on the chat completions endpoint.
    """
    baseline = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=TOKEN_IN_MESSAGES,
        max_completion_tokens=16,
        temperature=0,
        extra_body={"return_token_ids": True},
    )
    prompt_token_ids = baseline.prompt_token_ids
    assert prompt_token_ids

    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[],
        max_completion_tokens=16,
        temperature=0,
        stream=True,
        extra_body={"prompt_token_ids": prompt_token_ids, "return_token_ids": True},
    )

    content = ""
    delta_token_ids: list[int] = []
    first_chunk = True
    async for chunk in stream:
        if first_chunk:
            # prompt_token_ids arrives once, on the first chunk.
            assert chunk.prompt_token_ids == prompt_token_ids
            first_chunk = False
        if not chunk.choices:
            continue
        if chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
        if chunk.choices[0].model_dump().get("token_ids"):
            delta_token_ids.extend(chunk.choices[0].token_ids)

    # streamed text-out, reconstructed from deltas, with generated token ids.
    assert content
    assert delta_token_ids


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages, extra_options",
    [
        # A chat-template option cannot apply to pre-tokenized input.
        ([], {"add_generation_prompt": True}),
        # Truncation would desync the max_tokens budget from the real prompt.
        ([], {"truncate_prompt_tokens": 8}),
        # Token ids carry no multimodal features.
        (IMAGE_MESSAGES, {}),
    ],
    ids=["template-option", "truncate-option", "multimodal-content"],
)
async def test_prompt_token_ids_rejects_incompatible_input(
    client: openai.AsyncOpenAI, messages, extra_options
):
    """Options and message content that cannot apply are rejected, not ignored."""
    # Rejection happens during request validation, so the ids need only be
    # non-empty; no need to render real ones.
    with pytest.raises(openai.BadRequestError):
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_completion_tokens=4,
            extra_body={"prompt_token_ids": [1, 2, 3], **extra_options},
        )
