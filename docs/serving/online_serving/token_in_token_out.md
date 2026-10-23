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
| `derender` | the parsed assistant `message` (`content`, `reasoning`, `tool_calls`) on every choice, plus `logprobs` shaped like coupled chat's | a tokenizer and the parser flags you'd give `vllm serve` |

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

### Derender

`derender` returns the assistant `message` a coupled `/v1/chat/completions` would, without a separate [derender](derenderer.md) call. `/v1/chat/completions/render` and `/v1/messages/render` return `parse_context` on the `GenerateRequest`. Send it back with `output_mode: "derender"`. It carries the tools, `tool_choice`, `include_reasoning`, resolved JSON only `chat_template_kwargs` and the history tool call count. A `parse_context` field this server doesn't know fails a `derender` request and is ignored at the other levels.

- The parser reads the engine's `text`, so it sees the stop string cut and the post `adjust_request` decode flags. That makes it match coupled chat for parsers whose `adjust_request` changes decoding which `/derender` can't.
- A server with a tool or reasoning parser needs `parse_context`. Without a parser, `message.content` is the plain text.
- `parse_context` is ignored at other levels, so a rendered request can go to a prefill leg unchanged.
- `logprobs` follow coupled chat. `top_logprobs` is cut to `sampling_params.logprobs`, `logprob_token_ids` is honored. When a parser hides the reasoning with `include_reasoning: false`, `logprobs`, `token_ids` and `sampling_mask` are dropped so they don't expose it.
- `finish_reason` follows coupled chat too. It's `tool_calls` when the message has tool calls except with a named `tool_choice` or a `length` cut.
- The message `role` is the generate server's `--response-role`. `parse_context` doesn't carry `messages`, `add_generation_prompt` or `echo`, so the last message's role and `echo` aren't applied. Use [`/derender`](derenderer.md) for those.
- `stream: true` isn't supported yet.
- Cohere grounding citations aren't returned. The parser strips the citation spans from `content`, and the document map can't go over JSON in `parse_context`.

### Logprobs

With `output_mode: "tokens"`, logprobs are `GenerateLogProbs` with an integer `token_id` and `rank` per entry and no `token` or `bytes` (see [Generate Output Logprobs](renderer.md#generate-output-logprobs)). With `output_mode: "text"`, they are `ChatCompletionLogProbs` carrying the decoded token strings and their UTF-8 `bytes` for the sampled token and every entry in `top_logprobs`; a server started with `--return-tokens-as-token-ids` returns `token_id:N` placeholders there instead, as `/v1/completions` does.

### Errors

The server returns a 400 for:

- `output_mode: "text"` or `"derender"` on a server without a tokenizer (`--tokens-only` or `--skip-tokenizer-init`).
- `output_mode: "text"` or `"derender"` with `sampling_params.detokenize: false`.
- `output_mode: "derender"` with `stream: true`.
- `output_mode: "derender"` without `parse_context` on a server with a tool or reasoning parser or with a named `tool_choice` in `parse_context` that matches none of its `tools`. This is checked before generating.
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
