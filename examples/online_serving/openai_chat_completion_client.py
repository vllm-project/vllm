# SPDX-License-Identifier: Apache-2.0
"""Example Python client for OpenAI Chat Completion using vLLM API server
NOTE: start a supported chat completion model server with `vllm serve`, e.g.
    vllm serve meta-llama/Llama-2-7b-chat-hf
"""
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

messages = [{
    "role": "system",
    "content": "You are a helpful assistant."
}, {
    "role": "user",
    "content": "Who won the world series in 2020?"
}, {
    "role": "assistant",
    "content": "The Los Angeles Dodgers won the World Series in 2020."
}, {
    "role": "user",
    "content": "Where was it played?"
}]


def main():
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )

    print("-" * 50)
    print("Chat completion results:")
    print(chat_completion)
    print("-" * 50)


if __name__ == "__main__":
    main()
