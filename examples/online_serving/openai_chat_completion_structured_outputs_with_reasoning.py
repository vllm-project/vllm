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
