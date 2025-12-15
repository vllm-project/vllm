# AI Badgr

[AI Badgr](https://aibadgr.com) is a budget-focused, OpenAI-compatible LLM provider offering tier-based model access.

AI Badgr can connect to vLLM as a backend to serve local models through its OpenAI-compatible API interface.

## Prerequisites

Set up the vLLM environment:

```bash
pip install vllm
```

## Deploy

### Using vLLM as AI Badgr's Backend

1. Start the vLLM server with a supported chat completion model, e.g.:

    ```bash
    vllm serve meta-llama/Meta-Llama-3-8B-Instruct --host 0.0.0.0 --port 8000
    ```

2. Configure AI Badgr to point to your vLLM endpoint:

    - **Base URL**: `http://{your-vllm-host}:8000/v1`
    - **API Key**: Use any non-empty string (vLLM doesn't require authentication by default)
    - **Model**: Use the model name from your vLLM deployment (e.g., `meta-llama/Llama-3.2-3B-Instruct`)

### Connecting to AI Badgr's API

If you want to connect to AI Badgr's hosted service instead, you can use any OpenAI-compatible client:

```python
import os
from openai import OpenAI

# AI Badgr configuration using environment variables
client = OpenAI(
    base_url=os.getenv("AIBADGR_BASE_URL", "https://aibadgr.com/api/v1"),
    api_key=os.getenv("AIBADGR_API_KEY")
)

# AI Badgr uses tier-based model names
# Tier models: basic, normal, premium (recommended)
response = client.chat.completions.create(
    model="premium",  # or "normal", "basic"
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    temperature=0.7,
    max_tokens=150
)

print(response.choices[0].message.content)
```

### Environment Variables

When using AI Badgr's hosted service, you can set these environment variables:

- `AIBADGR_API_KEY`: Your AI Badgr API key
- `AIBADGR_BASE_URL`: Base URL for AI Badgr API (default: `https://aibadgr.com/api/v1`)

!!! note "Budget/Utility Provider"
    AI Badgr is positioned as a budget-focused utility provider. The tier-based pricing model (basic/normal/premium) allows cost optimization based on your use case.

## Model Tiers

AI Badgr offers tier-based model access:

- **basic**: Budget-tier models for simple tasks
- **normal**: Balanced performance and cost
- **premium**: Recommended default for most applications

Advanced users can also use specific model names for reproducibility or when they need to target exact model versions:

- `phi-3-mini` (maps to basic tier)
- `mistral-7b` (maps to normal tier)
- `llama3-8b-instruct` (maps to premium tier)

OpenAI model names (e.g., `gpt-3.5-turbo`) are also accepted and mapped automatically by AI Badgr to equivalent tier models.
