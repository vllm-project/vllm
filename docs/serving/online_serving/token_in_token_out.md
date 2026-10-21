# Tokens In <> Tokens Out API

`POST /inference/v1/generate` is the token-in/token-out (TITO) HTTP endpoint: it takes prompt token IDs and returns generated token IDs, logprobs keyed by integer token ID, and `finish_reason`/`stop_reason`, with no detokenization. Use it when the caller does its own tokenization and chat templating, such as an RL rollout worker. It is also the generation step of the `render → generate → derender` pipeline used in disaggregated serving (see the [Renderer APIs](renderer.md) and [Derenderer APIs](derenderer.md)).

## Enabling the endpoint (`--enable-scale-out`, `--tokens-only`)

- `--enable-scale-out` registers it, with `/render`, `/derender` and `/inference/v1/abort_requests`.
- `--tokens-only` always registers it, skips loading the tokenizer and forces `detokenize=False`. Stop strings (`sampling_params.stop`) are then rejected with a 400; use `stop_token_ids`.
- With neither flag, it returns 403 `Scale-out endpoints are disabled. Restart vllm serve with --enable-scale-out to enable them.`

```bash
vllm serve Qwen/Qwen3-0.6B --enable-scale-out

curl -X POST http://localhost:8000/inference/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"token_ids": [151644, 872, 198], "sampling_params": {"max_tokens": 32, "logprobs": 1}}'
```

A full client is in [examples/scale_out/token_generation_client.py](../../../examples/scale_out/token_generation_client.py).

## Request fields

| Field | Notes |
| --- | --- |
| `token_ids` | Required `list[int]`: the already-templated prompt. Non-empty, no negative IDs. |
| `sampling_params` | Required. Any [SamplingParams](../../api/README.md#inference-parameters) field, e.g. `max_tokens`, `temperature`, `logprobs`, `stop_token_ids`. Without `max_tokens`, the server applies its own default (as on the OpenAI endpoints), not 16. |
| `output_mode` | `"tokens"` (default) or `"text"`, see [Output modes](#output-modes). |
| `stream`, `stream_options` | SSE chunks ending with `data: [DONE]`. |
| `return_token_ids` | Echo the final prompt IDs as `prompt_token_ids`. |
| `features` / `content_parts` | Multimodal input, mutually exclusive. |
| `request_id`, `model`, `cache_salt`, `priority`, `kv_transfer_params`, `ec_transfer_params` | Optional. |

## Response fields

| `choices[i]` field | Notes |
| --- | --- |
| `token_ids` | Generated token IDs. |
| `finish_reason` | `stop`, `length` or `abort`. |
| `stop_reason` | Stop token ID or string; `null` on EOS, length and abort. |
| `logprobs` | `{"content": [{"token_id", "logprob", "rank", "top_logprobs": [...]}]}`, one entry per token, sampled token first. `token_id` is an integer, not the older `"token_id:<id>"` string. Needs `sampling_params.logprobs`. |
| `weight_versions`, `routed_experts`, `sampling_mask` | Optional, see [Weight versions](#weight-versions); the last two need `--enable-return-routed-experts` / `--return-sampling-mask`. |

The top level adds `request_id`, `usage`, `prompt_token_ids` and `prompt_logprobs`. Stream chunks carry the same per-choice fields for the new tokens.

## Targeting a data-parallel rank (`X-data-parallel-rank`)

Send the header `X-data-parallel-rank: <int>` to pin a request to one data-parallel (DP) engine rank; there is no body field. Without the header, or with a non-integer value, the request is load balanced. A rank outside `[0, data_parallel_size)` is rejected.

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

## Weight versions

The final output of every choice carries `weight_versions`, the label of the weights that sampled each range of its output tokens. Ranges are half-open over output-token indexes (prompt excluded), contiguous and in generation order, so a request that crossed one weight update gets two:

```json
"weight_versions": [
  {"version": "v7", "start": 0, "end": 57},
  {"version": "v8", "start": 57, "end": 128}
]
```

- With `stream: true`, only the last chunk of each choice carries the list, including the last chunk after an abort. Its offsets cover the whole output. Earlier chunks carry `null` or omit the field.
- An output with no tokens, such as a request aborted before its first token, gets `[]`.
- The labels are the ones you set with `/update_weight_version` or `/finish_weight_update`, see [Async Reinforcement Learning](../../training/async_rl.md#weight-versions-in-outputs). vLLM doesn't order or check them, and the label is `"default"` until you set one.

## Multimodal inputs

A request can carry media in two ways, and can use both at once:

- `content_parts`: raw media (`image_url`, `audio_url` or `video_url`, each with an optional `uuid`). `token_ids` holds one placeholder token per item, and the server expands it.
- `features`: items whose placeholders are already expanded in `token_ids`, at the ranges given in `mm_placeholders`. Pass the processed tensors in `kwargs_data`, or omit them to look the items up in the processor cache by `mm_hashes`.

When both are set, every `content_parts` placeholder must come after the last `features` range. Only the tokens after that range are processed, so the expanded runs before it are left as they are. Models that insert media at the start of the prompt (such as PaliGemma, BLIP-2 and Molmo) would place a new item right after that range without an error, so don't combine the two fields for them.

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
