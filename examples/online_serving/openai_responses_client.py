# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Set up this example by starting a vLLM OpenAI-compatible server with tool call
options enabled.
Reasoning models can be used through the Responses API as seen here
https://platform.openai.com/docs/api-reference/responses
For example:
vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3

"""

from openai import OpenAI

input_messages = [{"role": "user", "content": "What model are you?"}]


def main():
    base_url = "http://0.0.0.0:8000/v1"
    client = OpenAI(base_url=base_url, api_key="empty")
    model = "Qwen/Qwen3-1.7B"  # get_first_model(client)
    response = client.responses.create(
        model=model,
        input=input_messages,
    )

    for out in response.output:
        if out.type == "function_call":
            print("Function call:", out.name, out.arguments)
            tool_call = out

    import fbvscode

    fbvscode.set_trace()

    input_messages.append(tool_call)  # append model's function call message
    input_messages.append(
        {  # append result message
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": str(result),
        }
    )
    response_2 = client.responses.create(
        model=model,
        input=input_messages,
        tools=tools,
    )
    print(response_2.output_text)


if __name__ == "__main__":
    main()
