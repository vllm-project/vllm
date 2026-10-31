# Routed Experts Return (MoE Routing Replay / R3)

`--enable-return-routed-experts` makes vLLM return the routed experts (the expert IDs each MoE layer's router selected) for every token of a request. RL trainers use it for MoE routing replay (R3): the training forward pass reuses the rollout's routing.

```bash
vllm serve Qwen/Qwen3-30B-A3B --enable-return-routed-experts
# offline: LLM("Qwen/Qwen3-30B-A3B", enable_return_routed_experts=True)
```

The flag sets `AuxOutputConfig.enable_return_routed_experts` (full form: `--aux-output-config '{"enable_return_routed_experts": true, "max_bytes": N}'`). There is no per-request on/off switch.

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
- Filled only when the request finishes. `None` if the flag is off or the request aborted before any forward pass.

```python
import base64, io
import numpy as np

experts = np.load(io.BytesIO(base64.b64decode(choice["routed_experts"])))
```

## `routed_experts_prompt_start`

Per-request, default `0`: drops the first N prompt-token rows (e.g. routing already received in earlier turns). Must be `0 <= value <= prompt_len`, else HTTP 400. Pass it top-level on `/v1/*completions`, inside `sampling_params` on `/inference/v1/generate`, or as `SamplingParams(routed_experts_prompt_start=...)`.

## Limitations

- The Rust HTTP frontend never returns `routed_experts` and always sends `routed_experts_prompt_start = 0`. Use the Python API server.

Example: [examples/rl/routed_experts_e2e.py](../../examples/rl/routed_experts_e2e.py).

## Layer-owned row restoration

Every `MoERunner` or `RoutedExpertsCaptureSource` used for capture implements
`restore_routed_experts_rows(topk_ids) -> torch.Tensor`. Capture binding registers
that module for its layer. The capturer calls it directly, then writes the result
at the current microbatch offset. Router and capture signatures are unchanged.

The built-in MoE runner uses its SP configuration and dispatch path to restore
rows, delegating the gather/slice operations to `collect_local_routing_rows()`.
Monolithic prepare/finalize implementations specify whether they gather rows
before routing; native MegaMoE uses the input SP size supplied by its model.
A `MoERunner` subclass can override restoration for a different communication
layout, without replacing the capturer or using a Platform hook.

Return a 2-D integer tensor on the input device, with the same top-k width,
containing this DP rank's token rows in local order, without padding. Do not
modify the input or apply EPLB mapping. Empty tensors are valid; there is no
`None` fallback. A direct capture source without this method fails at binding.

Only TP0 publishes the restored rows. For local SP input with zero or one
token, TP0 already owns every valid row, so the built-in restorer skips the
collective. Non-publishing TP ranks may retain padding in this case.

Read current forward/microbatch metadata at invocation. Any collective must
include every peer, including ranks with no local output rows, and use
rank-consistent communication decisions. Device work must use the current stream
without host reads of device values. During graph replay, Python is not rerun:
the recorded operations, addresses and row selection must remain valid.
