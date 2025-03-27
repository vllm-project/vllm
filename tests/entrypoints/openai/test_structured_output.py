# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from typing import NamedTuple
import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
from pydantic import BaseModel, ValidationError

from ...utils import RemoteOpenAIServer

# # any model with a chat template should work here
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


@pytest.fixture(scope="module")
def server():
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "float16",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"


class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType


class TestCase(NamedTuple):
    model_name: str
    structured: bool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(model_name=MODEL_NAME, structured=True),
        TestCase(model_name=MODEL_NAME, structured=False)
    ],
)
async def test_structured_output_with_enum(client: openai.AsyncOpenAI,
                                           test_case: TestCase):
    prompt = ("Generate a JSON with the brand, model and car_type of"
              "the most iconic car from the 90's")
    extra_body = None
    if test_case.structured:
        json_schema = CarDescription.model_json_schema()
        extra_body = {"guided_json": json_schema}
    chat_completion = await client.chat.completions.create(
        model=test_case.model_name,
        messages=[{
            "role": "user",
            "content": prompt,
        }],
        extra_body=extra_body)
    assert chat_completion.id is not None

    choice = chat_completion.choices[0]
    assert choice.finish_reason == "stop"
    message = choice.message
    if test_case.structured:
        CarDescription.model_validate_json(message.content)
    else:
        with pytest.raises(ValidationError):
            CarDescription.model_validate_json(message.content)
    assert message.role == "assistant"
