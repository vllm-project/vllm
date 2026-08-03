# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Set up this example by starting a vLLM OpenAI-compatible server.
Reasoning models can be used through the Responses API as seen here
https://platform.openai.com/docs/api-reference/responses
For example:
vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3

"""

from openai import OpenAI

input_messages = [{"role": "user", "content": "What model are you?"}]


def main():
    base_url = "http://localhost:8000/v1"
    client = OpenAI(base_url=base_url, api_key="empty")
    model = "Qwen/Qwen3-8B"  # get_first_model(client)
    response = client.responses.create(
        model=model,
        input=input_messages,
    )

    for message in response.output:
        if message.type == "reasoning":
            # append reasoning message
            input_messages.append(message)

    response_2 = client.responses.create(
        model=model,
        input=input_messages,
    )
    print(response_2.output_text)
    # I am Qwen, a large language model developed by Alibaba Cloud.
    # I am designed to assist with a wide range of tasks, including
    # answering questions, creating content, coding, and engaging in
    # conversations. I can help with various topics and provide
    # information or support in multiple languages. How can I assist you today?


if __name__ == "__main__":
    main()
