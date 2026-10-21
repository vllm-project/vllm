# Routed Experts Return (MoE Routing Replay / R3)

`--enable-return-routed-experts` makes vLLM return the routed experts (the expert IDs each MoE layer's router selected) for every token of a request. RL trainers use it for MoE routing replay (R3): the training forward pass reuses the rollout's routing.

```bash
vllm serve Qwen/Qwen3-30B-A3B --enable-return-routed-experts
# offline: LLM("Qwen/Qwen3-30B-A3B", enable_return_routed_experts=True)
```

The flag sets `AuxOutputConfig.enable_return_routed_experts` (full form: `--aux-output-config '{"enable_return_routed_experts": true, "max_bytes": N}'`). Requests can opt out with [`return_routed_experts`](#return_routed_experts).

## Requirements

Startup fails with a `ValueError` unless:

- **Model Runner V2** is used (`VLLM_USE_V2_MODEL_RUNNER=1` if vLLM falls back to V1);
- the runner is `generate` and the model is **MoE**;
- **prefix caching is enabled** (the default). `--no-enable-prefix-caching` fails; the old v0.21 advice to disable it is wrong.

Rejected at startup: adaptive speculative verification, PP > 1, DCP/PCP > 1, and the `NixlConnector`, `NixlPullConnector`, `NixlPushConnector`, `MoRIIOConnector`, `MooncakeConnector` KV connectors. Regular speculative decoding works.

### How it works: aux output connector

Routing is stored by the aux output connector (`vllm/distributed/aux_output_connector/`, configured via `--aux-output-config`), keyed by KV block hash and shared with prefix caching, which is why prefix caching is required. `max_bytes` is its host-memory LRU capacity (default: derived from KV cache capacity). If a prefix-cache hit needs evicted rows, the worker raises `BlockObjectStoreError`; increase `max_bytes`.

## Output format

| API | Field |
| --- | --- |
| Offline (`LLM`, `AsyncLLM`) | `CompletionOutput.routed_experts`: `np.ndarray` or `None` |
| `/v1/completions`, `/v1/chat/completions` | `choices[i].routed_experts`: base64 `.npy`, **non-streaming only** |
| `/inference/v1/generate` | `routed_experts` per choice (base64 `.npy`); with `stream: true`, on the final chunk |

- **Shape:** `(num_tokens - 1 - routed_experts_prompt_start, num_moe_layers, num_experts_per_tok)`, `num_tokens` = prompt + output. The last sampled token has no row.
- **Rows:** row `i` is position `routed_experts_prompt_start + i` of `prompt_token_ids + output_token_ids`.
- **dtype:** `uint8` for at most 256 experts, else `uint16`; logical expert IDs.
- Filled only when the request finishes. `None` if the flag is off, the request set `return_routed_experts=False`, or the request aborted before any forward pass.

```python
import base64, io
import numpy as np

experts = np.load(io.BytesIO(base64.b64decode(choice["routed_experts"])))
```

## `routed_experts_prompt_start`

Per-request, default `0`: drops the first N prompt-token rows (e.g. routing already received in earlier turns). Must be `0 <= value <= prompt_len`, else HTTP 400. Pass it top-level on `/v1/*completions`, inside `sampling_params` on `/inference/v1/generate`, or as `SamplingParams(routed_experts_prompt_start=...)`.

## `return_routed_experts`

Per-request, default `true`. Set it to `false` for requests that don't need routing, such as evaluation traffic on a server that also serves RL rollouts. vLLM then neither stores nor returns routing for the request. A step whose requests all opt out also skips the device-to-host copy. Pass it the same way as `routed_experts_prompt_start`. It is ignored when the server does not record routed experts.

Requests that opt out get their own prefix-cache namespace, because their KV blocks have no stored routing. They share cached prefixes with each other but not with requests that record routing.

## Limitations

- The Rust HTTP frontend returns `routed_experts` only on `/inference/v1/generate`. Its completions and chat completions routes ignore `routed_experts_prompt_start` and `return_routed_experts` and return no routing.

Example: [examples/rl/routed_experts_e2e.py](../../examples/rl/routed_experts_e2e.py).
