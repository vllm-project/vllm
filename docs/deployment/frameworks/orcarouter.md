# OrcaRouter

[OrcaRouter](https://www.orcarouter.ai) is an OpenAI-compatible model routing gateway. Point your application at OrcaRouter and it forwards requests to the best upstream — including a self-hosted vLLM server registered as a custom endpoint. It also runs gateway-level, zero-trust security for AI agents on the same endpoint — screening every prompt/response and governing every tool call on a default-deny basis, with no application code changes.

Routing requests through OrcaRouter keeps a single OpenAI-compatible base URL for all of your applications: instead of configuring each one with the address of your vLLM server, they all talk to OrcaRouter, which sends traffic to your vLLM deployment (and can fall back to other upstreams when it is unavailable).

## Prerequisites

Set up the vLLM and OrcaRouter environment:

```bash
pip install vllm openai
```

You also need an [OrcaRouter API key](https://www.orcarouter.ai).

## Deploy

### 1. Start the vLLM server

Serve the model that will be the upstream, e.g.

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

### 2. Register your vLLM server as a custom endpoint

In the OrcaRouter console, register your vLLM endpoint (`http://{your-vllm-server-host}:{your-vllm-server-port}/v1`) as a custom endpoint and assign it a model id, e.g. `my-org/vllm-llama-3-1-8b`. Requests to that model id are routed to your vLLM server.

### 3. Call it through OrcaRouter

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.orcarouter.ai/v1", # OrcaRouter's OpenAI-compatible endpoint
    api_key="sk-orca-...", # your OrcaRouter API key
)

response = client.chat.completions.create(
    model="my-org/vllm-llama-3-1-8b", # the id you assigned to your vLLM endpoint
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    temperature=0.2,
    max_tokens=80,
)

print(response.choices[0].message.content)
```

You can also use the router model `orcarouter/auto`, which picks an upstream automatically for the request.

For details, see the [OrcaRouter documentation](https://docs.orcarouter.ai).
