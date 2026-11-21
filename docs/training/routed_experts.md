# Routed Experts Return (MoE Routing Replay / R3)

`--enable-return-routed-experts` makes vLLM return the routed experts (the expert IDs each MoE layer's router selected) for every token of a request. RL trainers use it for MoE routing replay (R3): the training forward pass reuses the rollout's routing.

```bash
vllm serve Qwen/Qwen3-30B-A3B --enable-return-routed-experts
# offline: LLM("Qwen/Qwen3-30B-A3B", enable_return_routed_experts=True)
```

The flag sets `AuxOutputConfig.enable_return_routed_experts` (full form: `--aux-output-config '{"enable_return_routed_experts": true, "max_bytes": N}'`). Requests can opt out with [`return_aux_outputs`](#return_aux_outputs).

## Requirements

Startup fails with a `ValueError` unless:

- the runner is `generate` and the model is **MoE**;
- **prefix caching is enabled** (the default). `--no-enable-prefix-caching` fails; the old v0.21 advice to disable it is wrong.

Rejected at startup: adaptive speculative verification, PP > 1, DCP/PCP > 1, and the `NixlConnector`, `NixlPullConnector`, `NixlPushConnector`, `MoRIIOConnector`, `MooncakeConnector` KV connectors. Regular speculative decoding works.

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

- **Shape:** `(num_tokens - 1 - aux_prompt_start, num_moe_layers, num_experts_per_tok)`, `num_tokens` = prompt + output. The last sampled token has no row.
- **Rows:** row `i` is position `aux_prompt_start + i` of `prompt_token_ids + output_token_ids`.
- **dtype:** `uint8` for at most 256 experts, else `uint16`; logical expert IDs.
- Filled only when the request finishes. `None` if the flag is off, the request set `return_aux_outputs=False`, or the request aborted before any forward pass.

```python
import base64, io
import numpy as np

experts = np.load(io.BytesIO(base64.b64decode(choice["routed_experts"])))
```

## `aux_prompt_start`

Per-request, default `0`: drops the rows of the first N prompt positions (e.g. routing already received in earlier turns). Must be `0 <= value <= prompt_len`, else HTTP 400. Pass it inside `sampling_params` on `/inference/v1/generate` or as `SamplingParams(aux_prompt_start=...)`. `routed_experts_prompt_start` is a deprecated alias, also accepted top-level on `/v1/*completions`: it logs a warning once, and a request whose two values disagree (0 counts as unset) is rejected.

## `return_aux_outputs`

Per-request, default `true`. Set it to `false` for requests that don't need RL auxiliary outputs, such as evaluation traffic on a server that also serves rollouts. Such a request gets no `routed_experts`, no `sampling_mask` and no sample `logprobs` or `logprob_token_ids`, and vLLM does not store its routing. `prompt_logprobs` and the `top_k` sampling parameter are unaffected. Pass it inside `sampling_params` on `/inference/v1/generate` or as `SamplingParams(return_aux_outputs=False)`.

Requests that opt out get their own prefix-cache namespace, because their KV blocks have no stored routing. They share cached prefixes with each other but not with requests that record routing.

Routing capture and mask computation run for the whole batch, so an opted-out request still pays that part of the cost while the server flags are on, and on a `--return-sampling-mask` server it still needs `temperature > 0` and `top_k > 0`. A streaming-input session keeps the value of its first chunk; a later chunk that changes it is rejected. A chunk without its own `sampling_params` uses the ones passed to `generate()`.

Example: [examples/rl/routed_experts_e2e.py](../../examples/rl/routed_experts_e2e.py).

## Namespaces: delivering routed experts through a shared object store

With inline return the client must hold every row it has received and hand them to each request or trainer worker that needs the same prefix. A namespace stores them once in a shared object store instead: the response lists keys, and any process that can reach the store reads them. The design is specified in [Aux Outputs for RL](../design/rl_aux_output/README.md).

Start the server with `--aux-output-config '{"enable_return_routed_experts": true, "object_key_prefix": "<run>"}'` and send `aux_namespace` inside `sampling_params` on `/inference/v1/generate`, or as `SamplingParams(aux_namespace=...)` offline. Requests without a namespace keep the inline behavior.

- **Requests.** `aux_namespace` is at most 256 characters: non-empty segments of `[A-Za-z0-9._-]` separated by `/`, none of them `tail`. `aux_prompt_start` says how many leading positions the client already holds rows for. A request with `aux_namespace` is rejected without `object_key_prefix`, with `return_aux_outputs` false, with `n > 1` (send each sample as its own request), with `stop` strings (the frontend ends such a request before the engine can return its listing; use `stop_token_ids`), and on streaming-input requests. Only `POST /inference/v1/generate` and the offline `LLM` API accept `aux_namespace`; the OpenAI completion and chat routes take none and return no listing.
- **Granularity.** Requests of one namespace share KV and objects; requests of different namespaces, nested or not, share nothing. Use the GRPO group id to share a group's prompt across its samples (stored once, released once per group), or `<group>/<sample>` to give every sample its own computation (the prompt computed and stored per sample, released per sample). Routing all requests of a namespace to one replica is recommended: it keeps the hits and, under group granularity, keeps two replicas from computing one block of a group at different weight versions.
- **Objects.** The engine writes one immutable object per block of the full-attention KV cache group, named `<object_key_prefix>/<aux_namespace>/<hash of the block's last hash block>`, once the step that computed the block's last position has settled. The positions after a request's last full block are one object named `<object_key_prefix>/<aux_namespace>/tail/<random name>`, a UUID4 in hex that the listing carries. An object is a safetensors file with one array, `routed_experts`, one entry per position. The key prefix and the namespace are part of the cache salt, and the weight version is not: a block computed again gets the same name, and the store keeps the first object.
- **Response.** The final output of a stopped or aborted request (the last chunk when streaming) carries `aux_objects`, `[key, start, end]` entries in position order: positions `[start, end)` are the first `end - start` entries of object `key`, whose first position is `start`. They cover `[aux_prompt_start, num_tokens - 1)` for a stopped request, the blocks the request took from the prefix cache included, so the first entry can start before `aux_prompt_start`. An aborted request lists the positions its returned tokens cover, from `aux_prompt_start`, or nothing. A preempted request's tails come before the objects that computed their positions again, so entries can overlap; keep the earlier one. A key is listed even if its write is pending or was dropped.
- **Turns.** Send `aux_prompt_start = min(positions held, longest common token prefix of the new prompt and the held tokens, prompt length - 1)`, and splice each listing onto the trajectory with the earlier copy winning: use each entry for positions `[max(start, end held), end)` ([consumer contract](../design/rl_aux_output/consumer.md#splicing-a-trajectory)).
- **Reading.** `object_store.read_aux_rows(object_store.MooncakeAuxObjectStore.create(mooncake_config_path), keys, "routed_experts")` from `vllm.distributed.aux_output_connector` reads objects in order and concatenates their rows; it waits up to 600 s for objects still being written. A key still missing after a pause, started after the requests stopped, of every engine that served their namespaces is lost: mask the samples that need it.
- **Release.** The engine only writes. Remove a namespace with a Mooncake client (`mooncake.store.MooncakeDistributedStore`) connected to the object store: `remove_by_regex(object_store.aux_namespace_pattern(object_key_prefix, aux_namespace), True)`, which also removes the namespaces nested under it, once every sample it expects is settled, no request of it runs or will be sent, and every engine that served it has completed a pause since (`pause_generation` in any mode). Do not send another request with a released namespace.
- **Store.** The Mooncake client is configured by the file `"mooncake_config_path"` names, or by `MOONCAKE_CONFIG_PATH`, and must use `"mode": "standalone-store"`, with a master and pool separate from any Mooncake KV store. By default objects are not pinned and the store keeps them by offloading to SSD on eviction; for a memory-only store set `"object_hard_pin": true`. Use a new `object_key_prefix` whenever the engines start.
- **Store failures.** A write the store refuses is retried every 0.2 s. When none has landed for `"object_put_stall_timeout_s"` (default 600 s, capped at half of `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`), the engine drops writes until one lands again; it keeps running, and responses still list the dropped objects, which are lost.
- **KV cache layout.** The engine refuses to start with `object_key_prefix` if the model has no plain full-attention KV cache group, if prefix-cache hits may end inside a KV block, if block hashes look ahead more than one token, or with prefix replay (disable `CacheConfig.swa_bounded_replay`). For models with sliding-window KV cache groups, set `prefix_cache_retention_interval` to `null` or the block size so that a later turn hits the outputs of earlier turns.
