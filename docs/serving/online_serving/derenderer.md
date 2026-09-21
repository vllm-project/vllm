# Derenderer APIs

The derenderer API is the post processing counterpart to the [Renderer APIs](renderer.md). Where `/render` turns a request into token ID (preprocessing), `/derender` turns generated token IDs back into a fully formed OpenAI compatible response (detokenization, reasoning parsing, tool call parsing), all without a GPU.

This closes the loop for a token-in / token-out engine in disaggregated serving:

- **GPU less post processing**: Detokenization, reasoning parsing, and tool call parsing run on the same GPU less frontend that hosts `/render`
- **Parser parity**: The derenderer reuses vLLM's tool and reasoning parsers, so a disaggregated deployment produces the same `content`/`reasoning`/ `tool_calls` split as a standard `vllm serve` server
- **Streaming and one-shot parsing**: Non-streaming calls send a complete `GenerateResponse` with all token IDs present and get one-shot parsing. Both endpoints also accept `stream: true`, taking one `GenerateStreamResponse` delta plus a client carried `stream_state` and returning `{chunk, stream_state}`. The chat endpoint's streaming path supports reasoning and tool call parsing while emitting the same `reasoning`/`content`/`tool_calls` deltas the generate streaming path would for the same token IDs (see [Streaming](#streaming))

Both endpoints are hosted by the GPU less rendering server started with [`vllm launch render`](../../cli/launch/render.md), alongside the `/render`
endpoints.

## Pipeline

```text
                render                 generate                derender
  request  ───────────────▶  token_ids  ─────────▶  token_ids  ──────────▶  response
 (chat /            (GPU less)          (token-in /            (GPU less)   (OpenAI
 completion)            │               token-out engine)          ▲        compatible)
                        └─────────────── request + prompt_tokens ──┘
```

The derender step needs more than the engine's `token_ids`. It also consumes the original `chat_request`/`completion_request` and `prompt_tokens` carried over from the render step (see [Request format](#request-format)) so the tool and reasoning parsers have the context they need. The chat streaming path additionally needs `prompt_token_ids` (see [Streaming cost](#streaming-cost)) when a parser is configured.

## API Reference

- Chat Completions Derender API (`/v1/chat/completions/derender`)
    - Post process a single `GenerateResponse` into a `ChatCompletionResponse`
    - With `stream: true`, post process one `GenerateStreamResponse` chunk into a `ChatCompletionStreamResponse` chunk
- Completions Derender API (`/v1/completions/derender`)
    - Post process a list of `GenerateResponse` objects (one per prompt) into a `CompletionResponse`
    - With `stream: true`, post process one `GenerateStreamResponse` chunk into a `CompletionStreamResponse` chunk

## Request format

Each request wraps the engine's `GenerateResponse`(s) together with the caller metadata needed to reconstruct the final response without a GPU.

`/v1/chat/completions/derender`:

??? code

    ```python
    --8<-- "vllm/entrypoints/scale_out/token_in_token_out/protocol.py:derender-chat-request"
    ```

`/v1/completions/derender`:

??? code

    ```python
    --8<-- "vllm/entrypoints/scale_out/token_in_token_out/protocol.py:derender-completion-request"
    ```

Streaming requests set `stream: true` and carry one generate chunk (`generate_chunk`) plus `stream_state` instead of a complete response.

`/v1/chat/completions/derender` with `stream: true`:

??? code

    ```python
    --8<-- "vllm/entrypoints/scale_out/token_in_token_out/protocol.py:derender-chat-stream-request"
    ```

`/v1/completions/derender` with `stream: true`:

??? code

    ```python
    --8<-- "vllm/entrypoints/scale_out/token_in_token_out/protocol.py:derender-completion-stream-request"
    ```

Both return `{"chunk": ..., "stream_state": ...}`. `chunk` is a `ChatCompletionStreamResponse` or `CompletionStreamResponse` and `stream_state` goes with the next call.

Oversized payloads are rejected with a `400` before any `tokenizer.decode()` or parser runs.

## Parser configuration

The derenderer builds its tool and reasoning parsers from its own server flags plus the `chat_request` sent with each call. It can't see how the request was rendered or served anywhere else, so for its output to match a `vllm serve` server:

- Start the render server with the same `--tool-call-parser`, `--reasoning-parser`, `--enable-auto-tool-choice`, `--chat-template` and `--default-chat-template-kwargs` you'd give `vllm serve` for this model. If `/render` and `/derender` run on different servers, give both the same values.
- Send the full `chat_request` that went to `/render`, not just `messages` and `tools`. Fields like `chat_template_kwargs`, `reasoning_effort`, `tool_choice` and `include_reasoning` change how the output is parsed.

A mismatch doesn't fail. The parser just splits `reasoning`, `content` and `tool_calls` differently from what `vllm serve` would return.

`/v1/completions/derender` only detokenizes. It never runs tool or reasoning parsers, the same as `/v1/completions` on `vllm serve`. It only reads `skip_special_tokens` from `completion_request`.

## Streaming

Streaming derender mirrors `/inference/v1/generate`. Set `stream: true` in the body on the same path and the endpoint takes one generate stream chunk instead of a complete response. It answers with JSON, not SSE. Each call turns one generate chunk into one derendered chunk and the client forwards that to its own caller.

The server keeps no state between calls. Everything the next call needs is in `stream_state` which the client carries:

- **Echo `stream_state` back.** Leave it out (or send `null`) on the first call, then send the `stream_state` from each response with the next request, unchanged. State that doesn't add up (e.g. `output_chunk_lens` that don't sum to the number of output tokens) is rejected with a `400`.
- **One choice per chunk.** `/inference/v1/generate` sends one choice per SSE event and each derender call accepts at most one. More than one is rejected with a `400`.
- **`n > 1` is N streams.** Generate interleaves chunks for different choices and tags each with its `index`. Keep one `stream_state` per index and send each chunk with the state for its index. Each index gets its own `role` delta on its first chunk.
- **Send every chunk through, including the last ones.** The finish reason usually arrives with the final tokens. A chunk with a `finish_reason` and no tokens is accepted too and flushes any buffered tool call arguments. With `stream_options: {"include_usage": true}`, generate ends with a usage only chunk (`choices: []`). Send it through derender like any other chunk to get the usage chunk for the response. `usage.prompt_tokens` is the request's `prompt_tokens` if set, otherwise the generate chunk's.
- **Send the same context on every call.** `chat_request` and `prompt_token_ids` aren't kept between calls, so they go with every chunk including the usage chunk. Both are required when a tool or reasoning parser is configured. `prompt_token_ids` is the `token_ids` of the `GenerateRequest` returned by `/render`.
- **Don't forward `[DONE]`.** It marks the end of the generate stream and isn't a chunk.

Streaming chunks don't carry logprobs yet, so `logprobs` on a generate chunk are dropped. The non-streaming endpoints do resolve them, including `token_id:N` placeholders.

## Streaming cost

Streaming derender threads a client carried `stream_state` across per-chunk calls instead of keeping session state on the server.

Plain detokenization (no parser configured) carries only a small, bounded incremental decode window in `stream_state` independent of generation length.

When a tool or reasoning parser is configured, parser internal state (buffered markup, reasoning/tool phase) can't be serialized. `stream_state` therefore instead carries the full `output_token_ids` seen so far plus `output_chunk_lens`, the token count of each chunk they arrived in. Each chunk rebuilds a fresh parser and replays that history through `parse_delta`, one call per original chunk, before processing the new tokens for real. For the parser path only, this means:

- **Transport**: `output_token_ids` and `output_chunk_lens` round-trip in full in both directions on every call. `output_chunk_lens` has one entry per chunk, which is one per token without speculative decoding. This means O(n) bytes per chunk, O(n²) bytes over a full generation. Bounded by `max_model_len`. `prompt_token_ids` is sent in full on every call too and it isn't trimmed as `output_token_ids` grows. This means that for most of a stream it dominates the per chunk payload. A 100k token prompt with 1k tokens of output means `prompt_token_ids` is ~99% of the request body on every chunk.
- **Compute**: replay is O(n) `parse_delta` calls per chunk (O(n²) per generation). `parse_delta` itself is O(n) for parsers that re-scan accumulated text (e.g. Hermes tool-call JSON, DeepSeek-R1 reasoning). The per-generation cost is O(n³) character work, not O(n²). This is a deliberately minimal first implementation with no caching layer.
- The parser path also requires `prompt_token_ids` so `parse_delta` can settle whether the prompt left reasoning open or not. Since parser state can't be carried across calls, it re-scans the full prompt once per chunk.
- Replay runs off the event loop on the renderer's executor (`renderer_num_workers`, default `1`). Size it for the expected number of concurrent parser configured streams.

`output_token_ids` and `prompt_token_ids` are both bounded by `max_model_len` but callers streaming long reasoning traces through a parser configured model should expect materially more state transport and CPU cost than the plain detokenization path.

## Example

The example below drives the full `render → generate → derender` round trip for a chat request against a GPU less render server (`/render`, `/derender`) and a token-in / token-out engine (`/inference/v1/generate`).

Launch the two servers first:

```bash
vllm launch render meta-llama/Llama-3.2-1B-Instruct --port 8100
vllm serve meta-llama/Llama-3.2-1B-Instruct --tokens-only --port 8200
```

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

Passing `chat_request` lets the derenderer run the configured tool and reasoning parsers. This means `response["choices"][0]["message"]` carries the same `content` / `reasoning` / `tool_calls` split a `vllm serve` server would produce. `chat_request` can only be omitted for a model with no tool or reasoning parser configured. A  parser configured model rejects a missing `chat_request` with a 400 rather than silently falling back to plain detokenization.

### Streaming example

The same round trip, streamed. `/render` keeps `stream` and `stream_options` on the `GenerateRequest` it returns, so the generate call streams too. Each generate chunk goes through derender with the `stream_state` from the previous call (see [Streaming](#streaming)).

```python
import json

import httpx

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
RENDER = "http://localhost:8100"  # vllm launch render ...
ENGINE = "http://localhost:8200"  # token-in / token-out engine

chat_request = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 32,
    "stream": True,
    "stream_options": {"include_usage": True},
}

with httpx.Client(timeout=60.0) as client:
    # 1. Render: request -> token IDs (GPU less)
    generate_request = client.post(
        f"{RENDER}/v1/chat/completions/render", json=chat_request
    ).json()
    prompt_token_ids = generate_request["token_ids"]

    # 2. Generate: stream token IDs (token-in / token-out engine)
    stream_state = None  # one per choice index when n > 1
    with client.stream(
        "POST", f"{ENGINE}/inference/v1/generate", json=generate_request
    ) as generate_stream:
        for line in generate_stream.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue

            # 3. Derender: one generate chunk -> one chat completion chunk
            derendered = client.post(
                f"{RENDER}/v1/chat/completions/derender",
                json={
                    "stream": True,
                    "model": MODEL,
                    "generate_chunk": json.loads(line[len("data: ") :]),
                    "stream_state": stream_state,
                    "prompt_tokens": len(prompt_token_ids),
                    "prompt_token_ids": prompt_token_ids,
                    "chat_request": chat_request,
                },
            ).json()
            stream_state = derendered["stream_state"]

            chunk = derendered["chunk"]
            for choice in chunk["choices"]:
                print(choice["delta"].get("content") or "", end="", flush=True)
            if chunk.get("usage"):
                print(f"\n{chunk['usage']}")
```

With a tool or reasoning parser configured, `delta` also carries `reasoning` and `tool_calls`, the same deltas `/v1/chat/completions` streams for those tokens.
