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
