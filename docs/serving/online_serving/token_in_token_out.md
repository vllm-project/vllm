# Tokens In <> Tokens Out API

`/inference/v1/generate` takes prompt token IDs and returns generated token IDs. It is the generation step of the `render → generate → derender` pipeline used in disaggregated serving (see the [Renderer APIs](renderer.md) and [Derenderer APIs](derenderer.md)).

The endpoint is registered on `vllm serve` with `--enable-scale-out`, and always with `--tokens-only`. `--tokens-only` also skips loading the tokenizer, so the server can't turn token IDs back into text.

## Output modes

The request field `output_mode` selects how much of the postprocessing the server does for the caller.

| `output_mode` | Response adds | Server needs |
| --- | --- | --- |
| `tokens` (default) | nothing, token IDs only | nothing |
| `text` | `text` on every choice and decoded tokens with `bytes` in `logprobs` | a tokenizer |

`text` returns text and token IDs in a single call, without a separate [derender](derenderer.md) hop per chunk. This suits latency sensitive streaming, where the derender hop lands on every output token.

Every response and every stream chunk echoes `output_mode`. Servers that predate the field ignore it and return token IDs only with a 200, so check that the `output_mode` in the response matches the one you sent. Older servers don't return it.

```python
import httpx

response = httpx.post(
    "http://localhost:8000/inference/v1/generate",
    json={
        "token_ids": [151644, 872, 198],
        "sampling_params": {"max_tokens": 32},
        "output_mode": "text",
    },
).json()

assert response["output_mode"] == "text"
print(response["choices"][0]["text"])
print(response["choices"][0]["token_ids"])
```

### Text

- `text` comes from the request's own detokenizer, so `skip_special_tokens`, `spaces_between_special_tokens`, `stop` and `include_stop_str_in_output` behave as they do on `/v1/completions`.
- A matched stop string is cut from `text` unless `include_stop_str_in_output` is set. `token_ids` keeps every generated token, so decoding `token_ids` yourself brings the stop string back.
- `/derender` only accepts `output_mode: "tokens"` responses and returns a 400 for text responses which are already detokenized.
- With `stream: true`, each choice's `text` is the delta since the previous chunk. A chunk is sent whenever the engine output carries new text or a `finish_reason`, even with no new token IDs. That covers text held back for stop string matching and the final output after an abort.

### Logprobs

With `output_mode: "tokens"`, logprobs are `GenerateLogProbs` with an integer `token_id` and `rank` per entry and no `token` or `bytes` (see [Generate Output Logprobs](renderer.md#generate-output-logprobs)). With `output_mode: "text"`, they are `ChatCompletionLogProbs` carrying the decoded token strings and their UTF-8 `bytes` for the sampled token and every entry in `top_logprobs`; a server started with `--return-tokens-as-token-ids` returns `token_id:N` placeholders there instead, as `/v1/completions` does.

### Errors

The server returns a 400 for:

- `output_mode: "text"` on a server without a tokenizer (`--tokens-only` or `--skip-tokenizer-init`).
- `output_mode: "text"` with `sampling_params.detokenize: false`.
- An unsupported `output_mode` value.

For using `output_mode` with separate prefill and decode pools, see [Disaggregated Prefilling](../../features/disagg_prefill.md#generate-api-output-modes).

## Multimodal inputs

A request can carry media in two ways, and can use both at once:

- `content_parts`: raw media (`image_url`, `audio_url` or `video_url`, each with an optional `uuid`). `token_ids` holds one placeholder token per item, and the server expands it.
- `features`: items whose placeholders are already expanded in `token_ids`, at the ranges given in `mm_placeholders`. Pass the processed tensors in `kwargs_data`, or omit them to look the items up in the processor cache by `mm_hashes`.

When both are set, every `content_parts` placeholder must come after the last `features` range. Only the tokens after that range are processed, so the expanded runs before it are left as they are.

This fits multi-turn rollouts. Each turn resends the previous prompt with its images already expanded, adds the new turn with one placeholder per new image, and sends the earlier images as `features`:

```python
import httpx

url = "http://localhost:8000/inference/v1/generate"

# Turn 1: turn1_ids holds one <|image_pad|> for image A.
turn1 = httpx.post(url, json={
    "token_ids": turn1_ids,
    "content_parts": [{"type": "image_url", "url": url_a, "uuid": "img-a"}],
    "sampling_params": {"max_tokens": 256},
    "return_token_ids": True,
}).json()

# Turn 2: image A is expanded in turn1["prompt_token_ids"];
# turn2_ids appends one <|image_pad|> for image B.
turn2 = httpx.post(url, json={
    "token_ids": (
        turn1["prompt_token_ids"] + turn1["choices"][0]["token_ids"] + turn2_ids
    ),
    "features": {
        "mm_hashes": {"image": ["img-a"]},
        "mm_placeholders": turn1["mm_placeholders"],
    },
    "content_parts": [{"type": "image_url", "url": url_b, "uuid": "img-b"}],
    "sampling_params": {"max_tokens": 256},
    "return_token_ids": True,
}).json()
```

Here `"img-a"` finds image A in the processor cache because turn 1 sent it with that `uuid`. Two cases need a different approach:

- If the server sets `--mm-processor-kwargs` or `--media-io-kwargs`, the `uuid` is hashed with them and is no longer the cache key. Take `features` from a [render](renderer.md) response instead.
- If the item has been evicted from the cache, the request fails. Resend the item with `kwargs_data`.

## Aborting requests

`POST /inference/v1/abort_requests` aborts in-flight requests. It is registered wherever `/inference/v1/generate` is and requires the API key when `--api-key` is set, like `/inference/v1/generate`.

```bash
curl -X POST http://localhost:8000/inference/v1/abort_requests \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"request_ids": ["generate-tokens-42"]}'
```

The request ID is the `request_id` the server returned: the `X-Request-Id` header or body `request_id` you sent, prefixed with `generate-tokens-`. A missing `request_ids` returns a 400. The response is empty and the abort finishes in the background.

With `--tokens-only`, the same handler is also served at `POST /abort_requests` for existing deployments. That path doesn't require the API key, even when `--api-key` is set. See [Security](../../usage/security.md#unprotected-endpoints-no-api-key-required).
