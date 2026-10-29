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

## Backend capture layouts

A model module can implement `get_routed_experts_capture_layouts(num_tokens_across_dp, dp_rank)` to return a mapping from layer IDs to `RoutedExpertsCaptureLayout`. `bind_routed_experts_capturer()` registers the provider modules alongside the capture callbacks; the layout provider can be an enclosing module rather than the module producing expert IDs. Rebind after replacing registered modules. Ordinary MoE uses `MoERunner` as its provider. Native MegaMoE uses `DeepGemmMegaMoEExperts`; callers declare the input SP partition count with its `input_sp_size` constructor argument (default: `1`, unsharded).

Before each forward, activate its `ForwardContext` and finish preparing backend communication metadata. Then call `prepare_routed_experts_capture_layouts(model, num_tokens, num_tokens_across_dp, dp_rank)` and assign its result to `get_forward_context().routed_experts_capture_layouts`. The counts describe the execution batch, including execution padding. Providers can read backend metadata from that active context. Each microbatch or nested forward needs its own layouts.

These types and helpers live in `vllm.model_executor.layers.fused_moe.routed_experts_capture_layout`. The built-in `make_routed_experts_capture_layout()` helper describes DP-major, evenly padded SP placement. Other backends construct layouts directly, without replacing the capturer or changing router signatures. For example, two physical shards of seven rows with three and two valid rows are described by:

```python
layout = RoutedExpertsCaptureLayout(
    input_rows=7,
    gather_group=tp_group,
    shard_sizes=(7, 7),
    source_ranges=((0, 3), (7, 2)),
)
```

`input_rows` is this rank's row count at capture. `shard_sizes` follows group-rank order; `source_ranges` gives ordered `(start, length)` spans after the optional gather. The capturer gathers through `GroupCoordinator.all_gatherv()`, selects those spans, and writes logical expert IDs in local token order. Without a gather, leave `gather_group=None` and `shard_sizes=()`.

Providers only describe layouts; they do not run collectives. Every bound capture layer needs a layout, and providers must not declare the same layer twice. All group members must agree on shard sizes and participate even when their selected spans are empty; the communicator must support the declared sizes. Layout metadata must remain fixed across graph replay. This extension does not remove the feature restrictions listed above.
