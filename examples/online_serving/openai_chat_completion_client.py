# SPDX-License-Identifier: Apache-2.0

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

chat_completion = client.chat.completions.create(messages=[{
    "role":
    "system",
    "content":
    "You are a helpful assistant."
}, {
    "role": "user",
    "content": "Say Hi"
}],
                                                 model=model,
                                                 stream=True)

print("Chat completion results:")
# print(chat_completion)
for chunk in chat_completion:
    print(chunk, '\n\n')
    # print(chunk.choices[0].delta.content)
