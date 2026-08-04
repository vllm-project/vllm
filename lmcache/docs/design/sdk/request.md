# Request Stream SDK

> Stateful, single-request orchestration over the [SDK context](context.md).

## Goal

`LMCacheSDKContext` is stateless — its `retrieve`/`store` move KV by token ids
and forget. A token dropping request spans several passes (prefill, modify the
cached KV, decode), which are encoded as different requests. 
`LMCacheRequestStream` binds all inference passes belonging to one request
instead of leaving the management to the user.

```py
import lmcache.sdk as lmc_sdk

kind = lmc_sdk.LMCacheSDKCacheKind.KV
ctx = lmc_sdk.connect(kind=kind, url=..., http_url=..., model_name=...)
request = lmc_sdk.request.create_request(
    [ctx], post_completion, prompt_token_ids=source_tokens  #contexts: iterable
)

request.generate({"max_tokens": 1})    # prefill -> offload the prompt KV
request.modify_kv(drop_tokens)         # drop_tokens: retrieve -> edit -> store
request.generate({"max_tokens": 256})  # replay the uncached tail + decode
```

`post_completion(prompt_token_ids, sampling_params, cache_salt)` calls the
engine (e.g. vLLM `/v1/completions`, streaming) and yields one
`TokenEvent(token_id, text)` per token. See the 
[token-dropping example](../../../examples/token_dropping/).

RequestStream is not used on its own. `LMCacheBatchedStream` wraps it for
submitting batched requests. See [batch.md](batch.md).

## State model

The full logical sequence is `tokens` + `_suffix_tokens`. The token-bearing
fields mean different things:

- **`tokens`** — sequence backing the stored KV; sent as the next prompt.
  Replaced when the KV is modified via `update`.
- **`_suffix_tokens`** — tokens *not* in the stored KV: the non-chunk-aligned
  tail left after `modify_kv`. Managed by the next `generate`.
- **`_decoded` / `_text_parts`** — cumulative generated count / text across all
  `generate`s (`decoded_tokens`, `output_text`); unaffected by compaction.
- **`done`** — True once a `generate` yields `< max_tokens` (EOS). Reset when
  `update` is called.

## Suffix contract

`retrieve` returns only a **chunk-aligned** prefix, so after `modify_kv`
edits it the remainder (sub-chunk tail + not-yet-offloaded tokens) has no
stored KV. The stream carries it across the edit:

- **`modify_kv`** records the uncached tail `tokens[cached_len:]` after KV
  modification (e.g. compaction).
- **`generate`** prepends `_suffix_tokens` (plus any caller `suffix_tokens`)
  to `tokens` for request submission, then clears it.

## Public API

- **`LMCacheRequestStream(contexts, post_completion,
  prompt_token_ids, cache_salt="")`** or `create_request(contexts, ...)`;
  `contexts` is an iterable (e.g. `[kv_ctx]` or `[kv_ctx, q_ctx]`).
- **`generate(sampling_params, suffix_tokens=())`** returns `StreamPerfMetrics`,
  the performance metrics of one request.
- **`modify_kv(fn, timeout=30.0, poll_interval=0.2)`**: calls `retrieve`, passing
  the KV and tokens so far to 
  `fn(Mapping[LMCacheSDKCacheKind, torch.Tensor], tokens[:cached_len])` which
  returns the new pair `(new_kv, new_tokens)`, then replaces the original KV via 
  `update`.
- **`retrieve(kind, timeout=30.0, poll_interval=0.2)`** and 
  **`update(kind, kv, tokens)`**:
  wrap the context's `retrieve()` / `store()`.
- Accessor **properties**: `request_stream_id`, `suffix_tokens`, `decoded_tokens`,
  `output_text`, `output_tokens`, `is_done`.
- **`StreamPerfMetrics`** (per call metric, times in **seconds**).
- **`TokenEvent(token_id, text)`**, **`PostCompletion`** (Protocol),
  **`LMCacheRequestStreamError`**.
