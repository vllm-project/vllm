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


# Pre-tokenized prompts. A router, or the decode stage of disaggregated
# serving, sends the ids it already has so the server skips re-tokenizing:
# in the public ``prompt_token_ids`` field, or in the older
# ``kv_transfer_params`` spelling.

TOKEN_IN_MESSAGES = [{"role": "user", "content": "Hello, how are you today?"}]
DECODE_MESSAGES = [{"role": "user", "content": "unrelated decode-side text"}]


def _pretokenized(spelling: str, ids: list[int]) -> dict:
    if spelling == "prompt_token_ids":
        return {"prompt_token_ids": ids}
    return {"kv_transfer_params": {"prompt_token_ids": ids}}


@pytest.mark.asyncio
@pytest.mark.parametrize("spelling", ["prompt_token_ids", "kv_transfer_params"])
async def test_kv_transfer_prompt_token_ids_round_trip(
    client: openai.AsyncOpenAI, spelling: str
):
    """Supplied ids are used verbatim, skipping tokenize.

    The second request carries different messages, so a response whose
    prompt_token_ids match the supplied ids proves the ids were used rather
    than the request's own messages. Generated text is not compared across
    requests because vLLM greedy decoding is not bitwise-reproducible.
    """
    baseline = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=TOKEN_IN_MESSAGES,
        max_completion_tokens=16,
        temperature=0,
        extra_body={"return_token_ids": True},
    )
    reused_ids = baseline.prompt_token_ids
    assert reused_ids

    decode = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=DECODE_MESSAGES,
        max_completion_tokens=16,
        temperature=0,
        extra_body={**_pretokenized(spelling, reused_ids), "return_token_ids": True},
    )

    # The engine saw the supplied ids, not the request's own messages.
    assert decode.prompt_token_ids == reused_ids
    assert decode.usage.prompt_tokens == len(reused_ids)
    # text-out: reuse still yields a detokenized message.
    assert decode.choices[0].message.content


@pytest.mark.asyncio
@pytest.mark.parametrize("spelling", ["prompt_token_ids", "kv_transfer_params"])
async def test_kv_transfer_prompt_token_ids_streaming(
    client: openai.AsyncOpenAI, spelling: str
):
    """A pre-tokenized prompt streams chat-formatted text-out."""
    baseline = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=TOKEN_IN_MESSAGES,
        max_completion_tokens=16,
        temperature=0,
        extra_body={"return_token_ids": True},
    )
    reused_ids = baseline.prompt_token_ids
    assert reused_ids

    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=DECODE_MESSAGES,
        max_completion_tokens=16,
        temperature=0,
        stream=True,
        extra_body={**_pretokenized(spelling, reused_ids), "return_token_ids": True},
    )

    content = ""
    delta_token_ids: list[int] = []
    first_chunk = True
    async for chunk in stream:
        if first_chunk:
            # prompt_token_ids arrives once, on the first chunk.
            assert chunk.prompt_token_ids == reused_ids
            first_chunk = False
        if not chunk.choices:
            continue
        if chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
        if tids := getattr(chunk.choices[0], "token_ids", None):
            delta_token_ids.extend(tids)

    # streamed text-out, reconstructed from deltas, with generated token ids.
    assert content
    assert delta_token_ids
