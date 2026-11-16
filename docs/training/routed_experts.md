# Routed Experts Return (MoE Routing Replay / R3)

`--enable-return-routed-experts` makes vLLM return the routed experts (the expert IDs each MoE layer's router selected) for every token of a request. RL trainers use it for MoE routing replay (R3): the training forward pass reuses the rollout's routing.

```bash
vllm serve Qwen/Qwen3-30B-A3B --enable-return-routed-experts
# offline: LLM("Qwen/Qwen3-30B-A3B", enable_return_routed_experts=True)
```

The flag sets `AuxOutputConfig.enable_return_routed_experts` (full form: `--aux-output-config '{"enable_return_routed_experts": true, "max_bytes": N}'`). There is no per-request on/off switch.

## Requirements

Startup fails with a `ValueError` unless:

- the runner is `generate` and the model is **MoE**;
- **prefix caching is enabled** (the default). `--no-enable-prefix-caching` fails; the old v0.21 advice to disable it is wrong.

Rejected at startup: adaptive speculative verification, DCP/PCP > 1, and the `NixlConnector`, `NixlPullConnector`, `NixlPushConnector`, `MoRIIOConnector`, `MooncakeConnector` KV connectors. Regular speculative decoding works.

### How it works: aux output connector

Routing is stored by the aux output connector (`vllm/distributed/aux_output_connector/`, configured via `--aux-output-config`), keyed by KV block hash and shared with prefix caching, which is why prefix caching is required. `max_bytes` is its host-memory LRU capacity (default: derived from KV cache capacity). If a prefix-cache hit needs evicted rows, the worker raises `BlockObjectStoreError`; increase `max_bytes`.

### Backend row restoration

A backend's `MoERunner` or `RoutedExpertsCaptureSource` can optionally implement
`restore_routed_experts_rows(topk_ids, ctx) -> Tensor`. During binding, the capturer
selects the first such module in `model.modules()` order and uses its callback
for every captured layer. Without a provider, the existing DP/TP recovery is used.
Existing capture sources need no changes.

The callback must restore this DP rank's token-ordered logical expert IDs for all
captured layers, using the supplied `ForwardContext` rather than the provider's
layer-specific state. It must preserve the input, device, dtype and top-k width,
and keep collective participation consistent across ranks, including empty
ranks. Operations must support graph replay. Models requiring different per-layer
restorers cannot use this hook. Repeated layer IDs do not select a different
restorer; the existing buffer-write semantics are unchanged.

## Output format

| API | Field |
| --- | --- |
| Offline (`LLM`, `AsyncLLM`) | `CompletionOutput.routed_experts`: `np.ndarray` or `None` |
| `/v1/completions`, `/v1/chat/completions` | `choices[i].routed_experts`: base64 `.npy`, **non-streaming only** |
| `/inference/v1/generate` | `routed_experts` per choice (base64 `.npy`); with `stream: true`, on the final chunk |

- **Shape:** `(num_tokens - 1 - routed_experts_prompt_start, num_moe_layers, num_experts_per_tok)`, `num_tokens` = prompt + output. The last sampled token has no row.
- **Rows:** row `i` is position `routed_experts_prompt_start + i` of `prompt_token_ids + output_token_ids`.
- **dtype:** `uint8` for at most 256 experts, else `uint16`; logical expert IDs.
- Filled only when the request finishes. `None` if the flag is off or the request aborted before any forward pass.

```python
import base64, io
import numpy as np

experts = np.load(io.BytesIO(base64.b64decode(choice["routed_experts"])))
```

## `routed_experts_prompt_start`

Per-request, default `0`: drops the first N prompt-token rows (e.g. routing already received in earlier turns). Must be `0 <= value <= prompt_len`, else HTTP 400. Pass it top-level on `/v1/*completions`, inside `sampling_params` on `/inference/v1/generate`, or as `SamplingParams(routed_experts_prompt_start=...)`.

Example: [examples/rl/routed_experts_e2e.py](../../examples/rl/routed_experts_e2e.py).

For sparse-attention selections, see [Indexer Replay](indexer_replay.md).
Indexer and routed-expert return can be enabled independently or together.

## Sessions: delivering rows through a shared object store

`routed_experts_prompt_start` keeps earlier turns out of a response, but the client must then hold every row it has received, and each sibling request or trainer worker that needs the same prefix must be handed them. A session stores the rows once in a shared object store instead: any request of the session, on any replica, and any process that can reach the store, reads them by key.

Start the server with `--aux-output-config '{"enable_return_routed_experts": true, "object_key_prefix": "<run>"}'` (and/or `"enable_return_indexer_topk": true`) and send `aux_session` inside `sampling_params` on `/inference/v1/generate`. Requests without a session keep the inline behavior.

- **Store.** The Mooncake client is configured by `MOONCAKE_CONFIG_PATH`, as for the Mooncake KV connector, and must use `"mode": "standalone-store"`: objects placed in the memory of an embedded client disappear when that client exits. Objects are hard-pinned and never evicted, so size the pool for everything written between two removals. A failed write (for example a full pool) is logged and stops the engine at its next step.
- **Key prefix.** `object_key_prefix` enables sessions. It is one path segment without `/`. Objects are immutable, so never reuse a prefix: choose a new one for every start of a deployment, including a restart of the same job, and a different one for every deployment that shares the store. Replicas of one deployment that share a configuration use the same prefix: replicas that compute the same positions of a session at the same weight versions write the same keys. A session under another prefix shares no KV with it.
- **Requests.** `aux_session` is a non-empty string without `/`. It is rejected without `object_key_prefix`, together with `stop` strings (the frontend ends such a request before the engine can return its keys; use `stop_token_ids`), and on streaming-input requests. `routed_experts_prompt_start` and `indexer_topk_prompt_start` are ignored for a session request. Only the Python API server accepts `aux_session`.
- **Weight versions.** The weight version of the KV of every position is part of the block hashes of a session request (per-segment cache salts), so every KV cache tier and KV connector keyed by block hashes reuses KV across a weight update only where its versions are the ones the request states, with no connector-specific code. Pass the previous response's `aux_version_segments` with the session's next request: `[[start, version], ..., [end, null]]`, starts increasing from 0 and adjacent versions differing. The engine salts the prompt positions below `end` (rounded down to an object, and to the last object that ends before the prompt's last token) with these versions and later positions with the current weight version (`update_weight_version`, `default` before the first call). A request without them computes its prompt as if new to the session. When the weight version changes, positions of running session requests that are neither computed nor hashed yet are salted with the new version; the positions a waiting session request computes, from its prefix-cache hit or the end of its loaded KV on, are salted with the version current when it is scheduled. Change the version only while no step is in flight, that is while the engine is paused.
- **Objects.** The engine writes one immutable object per block of the full-attention KV cache group, holding one array per enabled output (`routed_experts`, `indexer_topk`), named `<object_key_prefix>/<aux_session>/<hash of the block's last hash block>`, once the block's last position is computed. The rows after the last listed block are one object named `<object_key_prefix>/<aux_session>/tail/<hash>`, where the hash covers the preceding block, the tokens, the versions of those positions and the request's other block-hash extra keys (LoRA, multimodal inputs, prompt embeddings), so tails of different branches, lengths or versions never share a key. With pipeline parallelism each stage writes the layers it holds to `<name>/pp<stage>` (routed experts only: `enable_return_indexer_topk` requires a pipeline-parallel size of 1); responses list the names without the suffix.
- **Hidden states.** `"enable_return_hidden_states": true` also stores, as field `hidden_states` with shape `[tokens, hidden_size]`, the hidden states that the model's forward pass returns for every position. For most models these are the input of the LM head; a model that applies its final normalization inside `compute_logits` (Step3.5) returns the states before it. NumPy has no bfloat16, so a bfloat16 model's rows are int16 bit patterns: `torch.from_numpy(rows).view(torch.bfloat16)`. With pipeline parallelism only the objects of the last stage hold them, and `read_aux_rows` reads them from there.
- **Top-k logprobs.** `"return_topk_logprobs": k` stores, for every position `p`, the `k` most likely next tokens and their log-probabilities (the model's logits from `compute_logits`, log-softmax in float32, before any sampling parameter) as fields `topk_token_ids` (int32) and `topk_logprobs` (float32), shape `[tokens, k]`, in the objects of the last pipeline stage. Unlike `prompt_logprobs`, which disables prefix-cache hits, a hit keeps the rows of the request that computed the position, so a teacher can score long multi-turn prompts incrementally. The engine computes them for the scheduled tokens of session requests only, in chunks of at most 1024 rows and, as for `prompt_logprobs`, at most 64Mi / (k + 1) rows. Neither the float32 logits of a chunk nor the step's results are part of the memory profile: leave headroom in `gpu_memory_utilization` for about 1024 * vocabulary size * 10 bytes plus 8 * k bytes per scheduled session token of a step. `k` must not exceed the vocabulary size, and the engine refuses to start with LoRA.
- **Without routed experts.** With `object_key_prefix` and another output enabled, `enable_return_routed_experts` is optional: objects, full-block and tail alike, then carry only the other enabled fields, and requests without a session return no rows. This also works on dense models, for example a dense teacher with `return_topk_logprobs`.
- **Response.** The final output of a stopped request (the last chunk when streaming) lists in `aux_output_keys` the keys of the objects holding rows `[0, num_tokens - 1)`, in order, and returns `aux_version_segments` for the session's next request, up to the last full object computed after its block hashes existed: with hash look-ahead, the object that the last sampled token completes is left to the next request, which computes it again under its own version. Every turn lists all its keys, about 100 bytes per KV cache block of the request. A request that stops on an output in flight while it is preempted, or is aborted, lists no keys. Read the rows with `object_store.read_aux_rows(object_store.MooncakeAuxObjectStore.create(), keys, "routed_experts", pipeline_parallel_size)` (or `"indexer_topk"`) from `vllm.distributed.aux_output_connector`, which concatenates the stages' layers, with the same `MOONCAKE_CONFIG_PATH`; it waits up to 60 seconds for objects that are still being written. Create the store client in a process that does not start an engine afterwards: a Mooncake client created before the engine breaks CUDA initialization in the engine's workers.
- **KV cache layout.** The engine refuses to start with `object_key_prefix` without prefix caching, if the target model has no plain full-attention KV cache group, if prefix-cache hits may end inside a KV block, if block hashes look ahead more than one token, or with prefix replay (disable `CacheConfig.swa_bounded_replay` for models that use it). For models with sliding-window KV cache groups (DeepSeek-V4 class), set `prefix_cache_retention_interval` to `null` or the block size so that a later turn hits the outputs of earlier turns.

The engine only writes objects; the consumer deletes them with a Mooncake client (`mooncake.store.MooncakeDistributedStore`) connected to the same store: `remove_by_regex(object_store.aux_session_pattern(object_key_prefix, aux_session), True)` for a session after reading the objects of its last turn, or `remove_by_regex("^" + re.escape(object_key_prefix + "/"), True)` for everything under the prefix. Release by session, not by the keys a response lists: a session also holds the objects of blocks hashed when a request stopped, which that response lists in its tail and the next turn lists by name, and the objects of every pipeline-parallel stage. Delete a session only after its requests have finished, and a prefix only after the engines that write under it have shut down or stopped writing to it: Mooncake skips objects that are still being written, even with force, and they stay pinned. Do not send another request with a released `aux_session`: the engine still lets it hit the KV of the removed objects. Continue under a new session name, which recomputes the prefix.

### Limits

- A key names the tokens and the weight versions a request states for them, and the store keeps the first object written under it. A position that a running request computes again keeps the name it was scheduled with and so the earlier rows: a recompute after a failed synchronous KV load. Positions recomputed for any other reason (KV evicted from every tier, tiers cleared at a reset, client `aux_version_segments` from a replica whose KV is not here, a trailing block that EAGLE-style drafting computes again, `prompt_logprobs` or `skip_reading_prefix_cache`) are salted with the current version and named by it. Provision KV storage so session prefixes stay cached: an object whose KV is gone is written again under a new name.
- A block is hashed when its tokens exist, not when it is computed, and its hash never changes. A partly prefilled request, and a decoding request whose last sampled token completes a block, name the positions hashed before a weight update and computed after it by the earlier version; so do siblings with the same prompt whose prefill progress differs at the update.
- Draft KV that multi-module MTP computes again after a weight update carries the target positions' versions; the target rows are not affected.
- Same name means same rows only with batch invariance.
- Configure KV connectors to store the blocks that decoding completes, for example `save_decode_cache: true` for `MooncakeStoreConnector`: otherwise the last block of every turn is missing from the store, and after a weight update the next turn computes it again under its earlier-version name.
- Replicas that share the object store must use the same pipeline-parallel size and enabled outputs.
- Replicas that share KV stores or the object store must hash identically: the same hash algorithm (`sha256`, or the same `PYTHONHASHSEED`), hash block size and look-ahead.
- Do not use a Mooncake master for the object pool that also serves a KV store that is reset (`remove_all(force=True)`).
- Prefill/decode disaggregation (NIXL, Mooncake P2P, Moriio) is not supported: the decode side does not receive the prefill's segments.
