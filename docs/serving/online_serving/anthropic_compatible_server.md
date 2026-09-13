# Anthropic-Compatible Server

vLLM provides an HTTP server that implements a subset of the
[Anthropic Messages API](https://docs.anthropic.com/en/api/messages). The
Anthropic-compatible endpoints are available from the regular `vllm serve`
command alongside the OpenAI-compatible endpoints:

- `POST /v1/messages` generates a message.
- `POST /v1/messages/count_tokens` counts the tokens in a message before
  generation.

The server translates Messages API requests into vLLM chat requests. The model
therefore needs a [chat template](./README.md#chat-template), and model-specific
features such as tool calling and reasoning require the corresponding vLLM
parser configuration.

## Start the Server

The following example serves a small instruction-tuned model under the name
`local-model` and enables API key authentication:

```bash
vllm serve Qwen/Qwen3-0.6B \
    --served-model-name local-model \
    --api-key token-abc123
```

The `model` field in each request must match the model name exposed by the
server. Using `--served-model-name` is useful when a client expects a model name
that differs from the Hugging Face repository name.

!!! warning "API key authentication does not protect every endpoint"
    The `--api-key` option only authenticates selected API path prefixes. See
    [API Key Authentication
    Limitations](../../usage/security.md#api-key-authentication-limitations)
    before exposing a vLLM server to an untrusted network.

## Create a Message

The [official Anthropic Python
client](https://github.com/anthropics/anthropic-sdk-python) can call a vLLM
server by overriding its base URL. Pass the vLLM API key as an authentication
token so the client sends an `Authorization: Bearer` header:

```bash
uv pip install anthropic
```

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8000",
    auth_token="token-abc123",
)

message = client.messages.create(
    model="local-model",
    max_tokens=128,
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)
print(message.content[0].text)
```

The equivalent HTTP request is:

```bash
curl http://localhost:8000/v1/messages \
    -H "Authorization: Bearer token-abc123" \
    -H "Content-Type: application/json" \
    -H "anthropic-version: 2023-06-01" \
    -d '{
        "model": "local-model",
        "max_tokens": 128,
        "messages": [
            {"role": "user", "content": "Why is the sky blue?"}
        ]
    }'
```

## Stream a Message

Set `stream=True` to receive Anthropic-compatible server-sent events:

```python
with client.messages.stream(
    model="local-model",
    max_tokens=128,
    messages=[{"role": "user", "content": "Count from one to five."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

## Count Input Tokens

The token-counting endpoint applies the model's chat template and counts the
resulting input without running generation:

```python
count = client.messages.count_tokens(
    model="local-model",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)
print(count.input_tokens)
```

Include `system`, `tools`, and `tool_choice` in a token-counting request when
they are also present in the subsequent message request. This keeps the count
consistent with the rendered prompt.

## Supported Features

The Messages API adapter supports:

- Text and system messages.
- Streaming responses.
- URL and base64 image content for compatible [multimodal
  models](../../features/multimodal_inputs.md).
- `tools`, `tool_choice`, `tool_use`, and `tool_result` content blocks.
- JSON schema output through `output_config.format`.
- Thinking content blocks when a reasoning parser is configured.
- The vLLM-specific `cache_salt`, `chat_template_kwargs`, `vllm_xargs`,
  `kv_transfer_params`, and `ec_transfer_params` request fields.

Tool calls depend on the model's output format. Start the server with
`--enable-auto-tool-choice` and the parser recommended for the model:

```bash
vllm serve <model> \
    --served-model-name local-model \
    --enable-auto-tool-choice \
    --tool-call-parser <parser>
```

See [Tool Calling](../../features/tool_calling.md) for supported models and
parser values. Similarly, see [Reasoning Outputs](../../features/reasoning_outputs.md)
before enabling `--reasoning-parser`.

## Compatibility Scope

This interface is intended for clients that use the Messages API request and
response formats; it does not make an open-weight model behave like an
Anthropic-hosted model. The adapter covers the two upstream Messages API
endpoints listed above, not the complete Anthropic API surface, and not every
field or content block is available. Model output quality, tool selection, and
reasoning behavior remain dependent on the served model and its chat template
and parsers.

For a ready-to-use coding-agent integration, see the
[Claude Code guide](../integrations/claude_code.md).
