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
async def test_invalid_json_schema(client: openai.AsyncOpenAI,
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
        await client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": prompt,
            }],
            extra_body={"guided_json": invalid_json_schema},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_invalid_regex(client: openai.AsyncOpenAI, model_name: str):
    prompt = ("Generate an email address for Alan Turing, who works in Enigma."
              "End in .com and new line. Example result:"
              "alan.turing@enigma.com\n")

    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": prompt,
            }],
            extra_body={
                "guided_regex": r"[.*",
                "stop": ["\n"]
            },
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

    prompt = ("Generate an SQL query to show the 'username' and 'email'"
              "from the 'users' table.")
    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": prompt,
            }],
            extra_body={"guided_grammar": invalid_simplified_sql_grammar},
        )
