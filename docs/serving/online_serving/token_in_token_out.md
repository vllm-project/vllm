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
- A matched stop string is cut from `text` unless `include_stop_str_in_output` is set. `token_ids` keeps every generated token, so decoding `token_ids` yourself, or through `/derender`, brings the stop string back.
- With `stream: true`, each choice's `text` is the delta since the previous chunk. A chunk is sent whenever the engine output carries new text or a `finish_reason`, even with no new token IDs. That covers text held back for stop string matching and the final output after `/abort_requests`.

### Logprobs

At `tokens` the logprob entries carry `token_id:N` placeholders. At `text` they carry the decoded token strings and their UTF-8 `bytes`, for the sampled token and every entry in `top_logprobs`. A server started with `--return-tokens-as-token-ids` keeps the placeholders at every `output_mode`, as `/v1/completions` does.

On a server with a tokenizer, the engine already decodes every top-k token whenever `sampling_params.detokenize` is true, at every `output_mode`. Wide logprobs (for example `top_logprobs=20`) are the expensive case for the API server CPU. If you want them without that cost on a server that has a tokenizer, either:

- send `output_mode: "tokens"` with `sampling_params.detokenize: false`, which rules out `stop` strings, or
- send the request to a `--tokens-only` server.

### Validation

These requests fail with a 400 instead of returning empty or misleading output:

| Request | Why |
| --- | --- |
| `output_mode` other than `tokens` on a server without a tokenizer (`--tokens-only` or `--skip-tokenizer-init`) | Without a tokenizer, the text would be empty |
| `output_mode: "text"` with `sampling_params.detokenize: false` | The request asks for text and forbids producing it |
| An unsupported `output_mode` value | The response shape depends on it |

## Prefill and decode pools

`output_mode` has no behavior tied to the server's role or to `kv_transfer_params`. Only the output of the last leg reaches the client, so set `output_mode` on that leg only. A proxy that reuses the client's request body for the prefill leg has to reset `output_mode` to `tokens` there. A tokenizer free prefill pool rejects anything else with a 400, which surfaces the misconfiguration.

A decode pool that returns text runs `--enable-scale-out` with a tokenizer and without `--tokens-only`. Prefill pools can stay `--tokens-only`.

## Aborting requests

`POST /abort_requests` aborts in-flight requests. It is registered wherever `/inference/v1/generate` is, so it does not need `--tokens-only`.

```bash
curl -X POST http://localhost:8000/abort_requests \
  -H "Content-Type: application/json" \
  -d '{"request_ids": ["generate-tokens-42"]}'
```

The request ID is the `request_id` the server returned: the `X-Request-Id` header or body `request_id` you sent, prefixed with `generate-tokens-`. A missing `request_ids` returns a 400. The response is empty and the abort finishes in the background.

`/abort_requests` doesn't require the API key, even when `--api-key` is set. See [Security](../../usage/security.md#unprotected-endpoints-no-api-key-required).
