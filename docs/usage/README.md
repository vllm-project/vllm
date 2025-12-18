# Using vLLM

First, vLLM must be [installed](../getting_started/installation/README.md) for your chosen device in either a Python or Docker environment.

Then, vLLM supports the following usage patterns:

- [Inference and Serving](../serving/offline_inference.md): Run a single instance of a model.
- [Deployment](../deployment/docker.md): Scale up model instances for production.
- [Training](../training/rlhf.md): Train or fine-tune a model.

## OpenAI-Compatible API Usage

vLLM provides an OpenAI-compatible API for serving models. When using
multimodal **classification** models such as Qwen2.5-VL, requests must use
structured multimodal input.

### Qwen2.5-VL Classification Example

When serving a Qwen2.5-VL classification model, passing raw text or media URLs
directly may result in a `400 Bad Request` error. Instead, use structured input
with the OpenAI SDK.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

response = client.responses.create(
    model="Qwen2.5-VL-7B",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Classify this video"},
                {
                    "type": "video_url",
                    "video_url": {"url": "https://example.com/video.mp4"},
                },
            ],
        }
    ],
)

print(response)
```