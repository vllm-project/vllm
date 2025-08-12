# SPDX-License-Identifier: Apache-2.0
"""
An example shows how to generate structured outputs from reasoning models
like DeepSeekR1. The thinking process will not be guided by the JSON
schema provided by the user. Only the final output will be structured.

To run this example, you need to start the vLLM server with the reasoning 
parser:

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
     --enable-reasoning --reasoning-parser deepseek_r1
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

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

# Guided decoding by Regex
prompt = ("What is the capital of France?")

completion = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    extra_body={
        "guided_regex": "(Paris|London)",
    },
)
print("reasoning_content: ", completion.choices[0].message.reasoning_content)
print("content: ", completion.choices[0].message.content)


class People(BaseModel):
    name: str
    age: int


json_schema = People.model_json_schema()

prompt = ("Generate a JSON with the name and age of one random person.")
completion = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    extra_body={"guided_json": json_schema},
)
print("reasoning_content: ", completion.choices[0].message.reasoning_content)
print("content: ", completion.choices[0].message.content)


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


json_schema = CarDescription.model_json_schema()

prompt = ("Generate a JSON with the brand, model and car_type of"
          "the most iconic car from the 90's")
completion = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    extra_body={"guided_json": json_schema},
)
print("reasoning_content: ", completion.choices[0].message.reasoning_content)
print("content: ", completion.choices[0].message.content)

# Guided decoding by Grammar
simplified_sql_grammar = """
    ?start: select_statement

    ?select_statement: "SELECT " column_list " FROM " table_name

    ?column_list: column_name ("," column_name)*

    ?table_name: identifier

    ?column_name: identifier

    ?identifier: /[a-zA-Z_][a-zA-Z0-9_]*/
"""

# This may be very slow https://github.com/vllm-project/vllm/issues/12122
prompt = ("Generate an SQL query to show the 'username' and 'email'"
          "from the 'users' table.")
completion = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    extra_body={"guided_grammar": simplified_sql_grammar},
)
print("reasoning_content: ", completion.choices[0].message.reasoning_content)
print("content: ", completion.choices[0].message.content)
