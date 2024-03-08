# OpenAI Compatbile Server

vLLM provides an HTTP server that implements OpenAI's [Completions](https://platform.openai.com/docs/api-reference/completions) and [Chat](https://platform.openai.com/docs/api-reference/chat) API.

You can start the server using Python, or using [Docker](deploying_with_docker)
```bash
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf --dtype float32 --api-key token-abc123
```

To call the server, you can use the official OpenAI Python client library, or any other HTTP client.
```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

completion = client.chat.completions.create(
  model="meta-llama/Llama-2-7b-hf",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
```

## API Reference
Please see the [OpenAI API Reference](https://platform.openai.com/docs/api-reference) for more information on the API. Here we list the parameters that are supported by the server.

## Extra Parameters
vLLM supports a super set

