# SPDX-License-Identifier: Apache-2.0
"""
An example shows how to generate chat completions from reasoning models
like DeepSeekR1.

To run this example, you need to start the vLLM server with the reasoning 
parser:

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
     --enable-reasoning --reasoning-parser deepseek_r1
```

This example demonstrates how to generate chat completions from reasoning models
using the OpenAI Python client library.
"""

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

# Round 1
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = client.chat.completions.create(model=model, messages=messages)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

print("reasoning_content for Round 1:", reasoning_content)
print("content for Round 1:", content)

# Round 2
messages.append({"role": "assistant", "content": content})
messages.append({
    "role": "user",
    "content": "How many Rs are there in the word 'strawberry'?",
})
response = client.chat.completions.create(model=model, messages=messages)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

print("reasoning_content for Round 2:", reasoning_content)
print("content for Round 2:", content)
