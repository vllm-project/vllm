# Crusoe Managed Inference

[Crusoe](https://crusoe.ai) offers a managed inference service with an OpenAI-compatible API, powered by vLLM.

## Usage

Set your `CRUSOE_API_KEY` environment variable and point the OpenAI client at the Crusoe endpoint:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://managed-inference-api-proxy.crusoecloud.com/v1/",
    api_key=os.environ["CRUSOE_API_KEY"],
)

response = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    messages=[
        {
            "role": "user",
            "content": "Hello, how are you?",
        }
    ],
    temperature=1,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
)

print(response.to_json())
```

For a full list of available models and API documentation, see the [Crusoe Managed Inference docs](https://docs.crusoe.ai/managed-inference/overview).
