# Derenderer APIs

The derenderer API is the post processing counterpart to the [Renderer APIs](renderer.md). Where `/render` turns a request into token ID (preprocessing), `/derender` turns generated token IDs back into a fully formed OpenAI compatible response (detokenization, reasoning parsing, tool call parsing), all without a GPU.

This closes the loop for a token-in / token-out engine in disaggregated serving:

- **GPU less post processing**: Detokenization, reasoning parsing, and tool call parsing run on the same GPU less frontend that hosts `/render`
- **Parser parity**: The derenderer reuses vLLM's tool and reasoning parsers, so a disaggregated deployment produces the same `content`/`reasoning`/ `tool_calls` split as a standard `vllm serve` server
- **Non-streaming**: The endpoints expect a complete `GenerateResponse` with all token IDs present and perform one-shot parsing. Streaming derender would require a separate endpoint design and is not currently supported but is in the pipeline

Both endpoints are hosted by the GPU less rendering server started with [`vllm launch render`](../../cli/launch/render.md), alongside the `/render`
endpoints.

## Pipeline

```text
                render                 generate                derender
  request  ───────────────▶  token_ids  ─────────▶  token_ids  ──────────▶  response
 (chat /            (GPU less)          (token-in /            (GPU less)   (OpenAI
 completion)                            token-out engine)                   compatible)
```

## API Reference

- Chat Completions Derender API (`/v1/chat/completions/derender`)
    - Post process a single `GenerateResponse` into a `ChatCompletionResponse`
- Completions Derender API (`/v1/completions/derender`)
    - Post process a list of `GenerateResponse` objects (one per prompt) into a `CompletionResponse`

## Request format

Each request wraps the engine's `GenerateResponse`(s) together with the caller metadata needed to reconstruct the final response without a GPU.

`/v1/chat/completions/derender`:

| Field | Type | Description |
| --- | --- | --- |
| `model` | `str` | Served model name |
| `generate_response` | `GenerateResponse` | The complete token-in / token-out engine response to derender |
| `prompt_tokens` | `int \| None` | Prompt token count for usage accounting. Defaults to `0` when omitted. The caller already has this from the `/render` step |
| `chat_request` | `ChatCompletionRequest \| None` | The original (post `adjust_request`) request from `/render`. Required for tool/reasoning parsing so parsers receive `tools`, `tool_choice` and related context. When omitted, the endpoint falls back to plain detokenization |

`/v1/completions/derender`:

| Field | Type | Description |
| --- | --- | --- |
| `model` | `str` | Served model name |
| `generate_responses` | `list[GenerateResponse]` | One response per prompt, parallel to the `list[GenerateRequest]` returned by `/v1/completions/render` |
| `prompt_tokens` | `list[int] \| None` | One prompt token count per response. Each defaults to `0`. If provided, its length must equal `len(generate_responses)` |
| `completion_request` | `CompletionRequest \| None` | The original (post `adjust_request`) request from `/render`, mirroring `chat_request` |

The caller supplied token structures are validated against server resource bounds (`VLLM_MAX_N_SEQUENCES`, `max_model_len`, `max_logprobs`) before any `tokenizer.decode()` or parser runs, so oversized payloads are rejected with a `400` rather than exhausting CPU/memory.

## Example

The example below drives the full `render → generate → derender` round trip for a chat request against a GPU less render server (`/render`, `/derender`) and a token-in / token-out engine (`/inference/v1/generate`).

```python
import httpx

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
RENDER = "http://localhost:8100"  # vllm launch render ...
ENGINE = "http://localhost:8200"  # token-in / token-out engine

chat_request = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 32,
}

with httpx.Client(timeout=60.0) as client:
    # 1. Render: request -> token IDs (GPU less)
    generate_request = client.post(
        f"{RENDER}/v1/chat/completions/render", json=chat_request
    ).json()
    prompt_tokens = len(generate_request["token_ids"])

    # 2. Generate: token IDs -> token IDs (token-in / token-out engine)
    generate_response = client.post(
        f"{ENGINE}/inference/v1/generate", json=generate_request
    ).json()

    # 3. Derender: token IDs -> ChatCompletionResponse (GPU less)
    response = client.post(
        f"{RENDER}/v1/chat/completions/derender",
        json={
            "model": MODEL,
            "generate_response": generate_response,
            "prompt_tokens": prompt_tokens,
            "chat_request": chat_request,
        },
    ).json()

print(response["choices"][0]["message"]["content"])
```

Passing `chat_request` lets the derenderer run the configured tool and reasoning parsers. This means `response["choices"][0]["message"]` carries the same `content` / `reasoning` / `tool_calls` split a `vllm serve` server would produce. Omit `chat_request` for plain detokenization only.
