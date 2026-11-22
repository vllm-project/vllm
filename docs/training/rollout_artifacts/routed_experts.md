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

Rejected at startup: adaptive speculative verification, PP > 1 and DCP/PCP > 1. Regular speculative decoding works.

KV connectors are not rejected, including the prefill/decode ones (`NixlConnector`, `MoRIIOConnector`, `MooncakeConnector`, ...). In a prefill/decode deployment, however, the decode engine holds no rows for the KV the prefill engine computed. It accepts that KV only up to `routed_experts_prompt_start` and computes the rest of the prompt again, so the output is correct but the prefill work past that position is lost.

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

## Row storage

The aux output connector (`vllm/distributed/aux_output_connector/`) stores routing rows per engine, keyed by KV block hash, in the host memory of the engine's first tensor-parallel worker. Each row exists once, in the engine that computed it; there is no separate object store to deploy. Every data-parallel rank is a separate engine with its own rows.

The scheduler holds the rows of every block it keeps in the GPU prefix cache, and, when a KV connector is configured, of blocks the connector may load back, up to the store capacity. It reuses a cached block only while it holds that block's rows and otherwise computes the block again, producing new rows. Positions before `routed_experts_prompt_start` need no rows: a request reuses cached KV up to that position even when their rows are gone.

## Replica affinity

A later turn of a session reuses its prefix only on the engine that holds the prefix's KV and rows. Sent to any other engine, the request computes the whole prefix again:

- time to first token and throughput vary with where each turn lands, and multi-turn rollouts pay for their context once per turn;
- the prefix-cache hit rate falls as replicas or load grow and is not reproducible across runs;
- a KV store shared between replicas does not fill the gap: the scheduler refuses an external KV hit that reaches past `routed_experts_prompt_start` unless this engine holds the rows of every loaded block.

Route every request of a session (a multi-turn rollout, or a group of samples sharing a prompt) to the same engine:

- Put a session ID on each request and have the router map it to an engine with consistent hashing (a hash ring or rendezvous hashing), so adding or removing an engine remaps only its share of sessions.
- Fall back to another engine only when the target is down or saturated, and accept that the next turn of that session recomputes its prefix.
- Set `routed_experts_prompt_start` to the number of tokens whose rows the client already holds. A hit then needs rows only for the positions it returns, and a shared KV store can serve the earlier turns on another engine.

Sessions whose rows are written as objects to a shared Mooncake store, readable from every replica and released per session, are developed on the `aux-pr/` branches of the [aoshen02/vllm](https://github.com/aoshen02/vllm/tree/aux-pr/7-rename-recomputed-positions) fork. Switch to them when requests of a session cannot be pinned to one engine or replicas share KV through a store.

## Weight updates and cache resets

Any reset of the prefix cache drops every row of the engine. That includes `sleep` at level 1 or 2, `pause_generation(clear_cache=True)` and `reset_prefix_cache`. A turn sent after the reset has rows only for what it computes again.

What happens to the KV is your choice, through `reset_connector`:

| Call | GPU prefix cache | KV connector tier (e.g. CPU offload) | Rows |
| --- | --- | --- | --- |
| `pause_generation(clear_cache=False)`, weights updated in place, `resume_generation()` | kept | kept | kept |
| `sleep(level=1, reset_connector=False)` | cleared | kept | cleared |
| `sleep(level=1)` (default `reset_connector=True`) | cleared | cleared | cleared |

- **No reset.** Updating weights while paused with `clear_cache=False` keeps the cache and the rows, so later turns of a session keep their hits. The reused KV and rows were computed by the weights in place when each position was first computed.
- **Sleep, keeping the connector.** With `reset_connector=False`, the KV of earlier turns can still be loaded back from the connector tier, but its rows are gone. Set `routed_experts_prompt_start` to the number of tokens whose rows the client already holds: positions before it need no rows, so they load from the tier and only the new part of the prompt is computed. Without it, the hit reaches past `routed_experts_prompt_start` into blocks whose rows this engine no longer holds, so it is refused and the prompt is computed again.
- **Sleep with the default.** Every session computes its whole prefix again on the next turn.

Keeping the cache or the connector tier across an update means later turns reuse KV, and rows, computed by the previous weights. This is a training decision, not a correctness requirement of vLLM: keep them if your algorithm accepts that staleness, reset with the defaults if it does not. The defaults reset because vLLM cannot know which one you want.

## Sizing `max_bytes`

Rows per token take `num_hidden_layers × num_experts_per_tok × itemsize` bytes, with `itemsize` 1 for at most 256 experts and 2 above; every hidden layer counts, dense ones included.

- **Default (`max_bytes` unset).** One block of rows per KV block: the KV cache token capacity, logged at startup as `KV cache size: N tokens`, times the bytes per token, doubled when a KV connector is configured. The arena is anonymous memory, committed as rows are written.
- **A smaller `max_bytes` does not save memory.** Rows of blocks in the GPU prefix cache are never evicted. When they outgrow `max_bytes`, the store copies its whole arena into one at least twice as large and logs a warning, a latency spike for the steps that wait on the store. The LRU drops only rows of blocks the scheduler no longer holds, so the rows of a session last as long as its KV does.
- **Size the KV cache for the sessions.** Later turns hit only while `sessions per engine × average context ≤ KV cache token capacity`, with headroom for the peaks; past that, KV evicts and turns recompute whatever `max_bytes` is.
- **Set `max_bytes` only with a KV offload tier larger than the GPU cache.** Use `(GPU KV tokens + offload tier tokens) × bytes per token × 1.25`, so rows stay for every block the tier can load back.

| Model | Bytes per token | Sessions × context per engine | Rows |
| --- | --- | --- | --- |
| Qwen3-30B-A3B (48 layers, top-8 of 128) | 48 × 8 × 1 = 384 B | 32 × 16K = 524K tokens | 201 MB |
| DeepSeek-V4.1-Flash (40 layers, top-6 of 384) | 40 × 6 × 2 = 480 B | 51 × 200K = 10.2M tokens | 4.9 GB |
| DeepSeek-V4.1-Flash | 480 B | 51 × 1M = 51M tokens | 24.5 GB |

For Qwen3-30B-A3B the rows are 0.4% of the bfloat16 KV of the same tokens (96 KiB per token), so host memory is never the limit; the same holds for every row of the table on a node with hundreds of GB of host memory. What limits reuse is the KV cache capacity and replica affinity, not the row store.

Example: [examples/rl/routed_experts_e2e.py](../../examples/rl/routed_experts_e2e.py).
