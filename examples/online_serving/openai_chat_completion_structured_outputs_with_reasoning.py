# SPDX-License-Identifier: Apache-2.0
"""
An example shows how to generate structured outputs from reasoning models
like DeepSeekR1. The thinking process will not be guided by the JSON
schema provided by the user. Only the final output will be structured.

To run this example, you need to start the vLLM server with the reasoning
parser:

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --reasoning-parser deepseek_r1
```

This example demonstrates how to generate chat completions from reasoning models
using the OpenAI Python client library.
"""

from enum import Enum

from openai import OpenAI
from pydantic import BaseModel

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


def print_completion_details(completion):
    print("reasoning_content: ", completion.choices[0].message.reasoning_content)
    print("content: ", completion.choices[0].message.content)


# Guided decoding by Regex
def guided_regex_completion(client: OpenAI, model: str):
    prompt = "What is the capital of France?"

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        extra_body={
            "guided_regex": "(Paris|London)",
        },
    )
    print_completion_details(completion)


class People(BaseModel):
    name: str
    age: int


def guided_json_completion(client: OpenAI, model: str):
    json_schema = People.model_json_schema()

    prompt = "Generate a JSON with the name and age of one random person."
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        extra_body={"guided_json": json_schema},
    )
    print_completion_details(completion)


# Guided decoding by JSON using Pydantic schema
class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"


class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType


def guided_car_json_completion(client: OpenAI, model: str):
    json_schema = CarDescription.model_json_schema()

    prompt = (
        "Generate a JSON with the brand, model and car_type of"
        "the most iconic car from the 90's"
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        extra_body={"guided_json": json_schema},
    )
    print_completion_details(completion)


# Guided decoding by Grammar
def guided_grammar_completion(client: OpenAI, model: str):
    simplified_sql_grammar = """
        root ::= select_statement

        select_statement ::= "SELECT " column " from " table " where " condition

        column ::= "col_1 " | "col_2 "

        table ::= "table_1 " | "table_2 "

        condition ::= column "= " number

        number ::= "1 " | "2 "
    """

    # This may be very slow https://github.com/vllm-project/vllm/issues/12122
    prompt = (
        "Generate an SQL query to show the 'username' and 'email'"
        "from the 'users' table."
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        extra_body={"guided_grammar": simplified_sql_grammar},
    )
    print_completion_details(completion)


def main():
    client: OpenAI = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model: str = models.data[0].id

    print("Guided Regex Completion:")
    guided_regex_completion(client, model)

    print("\nGuided JSON Completion (People):")
    guided_json_completion(client, model)

    print("\nGuided JSON Completion (CarDescription):")
    guided_car_json_completion(client, model)

    print("\nGuided Grammar Completion:")
    guided_grammar_completion(client, model)


if __name__ == "__main__":
    main()
