# Crusoe

[Crusoe](https://crusoe.ai/) provides Managed Inference, an OpenAI-compatible API for open-weight models, powered by vLLM. Because the service speaks the OpenAI API, code written against a self-hosted vLLM server works against Crusoe endpoints without changes.

## Prerequisites

- A Crusoe account
- An Inference API key, created in the [Crusoe Console](https://console.crusoe.ai/) under **Security > Inference API Key**

Set the key as an environment variable:

```bash
export CRUSOE_API_KEY="your-api-key"
```

## Using the OpenAI SDK

Point the OpenAI client at the Crusoe endpoint:

```python
import os

from openai import OpenAI

client = OpenAI(
    base_url="https://api.inference.crusoecloud.com/v1",
    api_key=os.environ["CRUSOE_API_KEY"],
)

response = client.chat.completions.create(
    model="zai/GLM-5.2",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
)

print(response.choices[0].message.content)
```

## Verifying with curl

!!! console "Command"

    ```bash
    curl https://api.inference.crusoecloud.com/v1/chat/completions \
        -H "Authorization: Bearer $CRUSOE_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "zai/GLM-5.2",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "max_tokens": 50
        }'
    ```

## Available models

List the current model catalog:

```bash
curl https://api.inference.crusoecloud.com/v1/models \
    -H "Authorization: Bearer $CRUSOE_API_KEY"
```

For the full list of available models and API details, see the [Crusoe Managed Inference docs](https://docs.crusoecloud.com/managed-inference/overview).
