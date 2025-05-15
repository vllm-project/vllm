# ruff: noqa: E501
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

__doc__ = f"""
This script demonstrates various structured output capabilities of vLLM's OpenAI-compatible server.
It can run individual constraint types or all of them.
It supports both streaming responses and concurrent non-streaming requests.

To use this example, you must start an vLLM server with any model of your choice.

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct
```

To serve a reasoning model, you can use the following command:
```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --reasoning-parser deepseek_r1
```

Examples:

Run all constraints, non-streaming:
python {__file__}

Run only regex constraint, streaming:
python {__file__} --constraint regex --stream

Run json and choice constraints, non-streaming:
python {__file__} --constraint json choice
"""  # noqa: E501

import argparse
import asyncio
import os
from enum import Enum
from typing import Any, Literal, get_args

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel


async def print_stream_response(
    stream_response: AsyncStream[ChatCompletionChunk],
    title: str,
):
    print(f"\n{title} (Streaming):")
    full_response = ""
    async for chunk in stream_response:
        content = chunk.choices[0].delta.content or ""
        print(content, end="", flush=True)
        full_response += content
    print()
    return full_response


class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"


class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType


ConstraintsFormat = Literal["choice", "regex", "json", "grammar",
                            "structural_tag"]

PARAMS: dict[ConstraintsFormat, Any] = {
    "choice": {
        "messages": [{
            "role": "user",
            "content": "Classify this sentiment: vLLM is wonderful!"
        }],
        "extra_body": {
            "guided_choice": ["positive", "negative"]
        },
    },
    "regex": {
        "messages": [{
            "role":
            "user",
            "content":
            "Generate an email address for Alan Turing, who works in Enigma. End in .com and new line. Example result: 'alan.turing@enigma.com\\n'"
        }],
        "extra_body": {
            "guided_regex": r"\\w+@\\w+\\.com\\n",
            "stop": ["\\n"],
        },
    },
    "json": {
        "messages": [{
            "role":
            "user",
            "content":
            "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's"
        }],
        "response_format": {
            "type": "json_schema",
            "json_schema": CarDescription.model_json_schema(),
        },
    },
    "grammar": {
        "messages": [{
            "role":
            "user",
            "content":
            "Generate an SQL query to show the 'username' and 'email'from the 'users' table."
        }],
        "extra_body": {
            "guided_grammar": """
root ::= select_statement

select_statement ::= "SELECT " column " from " table " where " condition

column ::= "col_1 " | "col_2 "

table ::= "table_1 " | "table_2 "

condition ::= column "= " number

number ::= "1 " | "2 "
""",
            "guided_decoding_disable_any_whitespace": True,
        }
    },
    "structural_tag": {
        "messages": [
            {
                "role":
                "user",
                "content":
                """
You have access to the following function to retrieve the weather in a city:

{
    "name": "get_weather",
    "parameters": {
        "city": {
            "param_type": "string",
            "description": "The city to get the weather for",
            "required": True
        }
    }
}

If a you choose to call a function ONLY reply in the following format:
<{start_tag}={function_name}>{parameters}{end_tag}
where

start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function
              argument value as value.
end_tag => `</function>`

Here is an example,
<function=example_function_name>{"example_name": "example_value"}</function>

Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- Always add your sources when using search results to answer the user query

You are a helpful assistant.

Given the previous instructions, what is the weather in New York City, Boston,
and San Francisco?""",
            },
        ],
        "response_format": {
            "type":
            "structural_tag",
            "structures": [{
                "begin": "<function=get_weather>",
                "schema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string"
                        }
                    },
                    "required": ["city"],
                },
                "end": "</function>"
            }],
            "triggers": ["<function="]
        },
    },
}


async def main():
    parser = argparse.ArgumentParser(
        description=
        "Run OpenAI Chat Completion with various structured outputs capabilities",
    )
    _ = parser.add_argument(
        "--constraint",
        type=str,
        nargs="+",
        choices=[*get_args(ConstraintsFormat), "all"],
        default=["all"],
        help="Specify which constraint(s) to run.",
    )
    _ = parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable streaming output",
    )
    args = parser.parse_args()

    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    model = (await client.models.list()).data[0].id

    constraints: list[ConstraintsFormat] = list(get_args(ConstraintsFormat)) \
        if "all" in args.constraint \
        else list(set(args.constraint))

    results = await asyncio.gather(
        *[
            client.chat.completions.create(
                model=model,
                stream=args.stream,
                **PARAMS[name],
            ) for name in constraints
        ],
        return_exceptions=True,
    )

    if args.stream:
        for constraint_name, stream_or_exc in zip(constraints, results):
            if isinstance(stream_or_exc, Exception):
                print(f"Error for {constraint_name}: {stream_or_exc}\n")
            else:
                print_stream_response(
                    stream_or_exc,
                    title=constraint_name,
                )
    else:
        for constraint_name, response_or_exc in zip(constraints, results):
            print(f"Constraint: {constraint_name}\n")
            if isinstance(response_or_exc, Exception):
                print(f"Output:\n  Error: {response_or_exc}\n")
            else:
                assert isinstance(response_or_exc, ChatCompletion)
                # response_or_exc is a ChatCompletion object
                print(
                    f"Output:\n  {response_or_exc.choices[0].message.content!r}\n"
                )


if __name__ == "__main__":
    asyncio.run(main())
