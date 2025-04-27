# SPDX-License-Identifier: Apache-2.0
from openai import OpenAI

# This example demonstrates the `structural_tag` response format.
# It can be used to specify a structured output format that occurs between
# specific tags in the response. This example shows how it could be used
# to enforce the format of a tool call response, but it could be used for
# any structured output within a subset of the response.


def main():
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="-",
    )

    messages = [{
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
and San Francisco?
"""
    }]

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages,
        response_format={
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
                    }
                },
                "end": "</function>"
            }],
            "triggers": ["<function="]
        })
    print(response)


if __name__ == "__main__":
    main()
