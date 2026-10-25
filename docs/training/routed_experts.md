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

## Host-memory admission

For `N` GPU KV blocks, `B` tokens/block and `R` routed-expert bytes/token, the
default SHM arena needs `N * B * R` bytes per output worker. This check also
requires an explicit `aux_output_config.max_bytes` to cover every KV block;
an undersized arena is rejected instead of silently reducing replay coverage.

Before creating the Store, startup queries system available RAM and the remaining
container cgroup memory, using the smaller value. The Store uses anonymous shared
`mmap`, so `/dev/shm` mount free space is not its capacity limit; see the
[Linux tmpfs documentation](https://www.kernel.org/doc/html/latest/filesystems/tmpfs.html).
Schedulers that enforce their memory request outside cgroups can export the budget
in `VLLM_AUX_OUTPUT_HOST_MEMORY_BYTES` from their job wrapper; startup then uses
the smallest of that value, the cgroup headroom and available RAM.
The check reserves runtime headroom, accounts for staging and estimated metadata,
and divides the remaining budget across local DP output workers. It then rounds
down to complete SHM objects and infers the maximum GPU KV block count.

If the arenas cannot fit or a configured `aux_output_config.max_bytes` cannot
cover all KV blocks, startup fails with required/available SHM and a block limit.
Use the reported `--num-gpu-blocks-override N` (Python:
`num_gpu_blocks_override=N`) or a smaller positive count. Unset
`aux_output_config.max_bytes` so the arena follows the chosen KV block count, or
resize it within the reported host budget. A smaller `--max-model-len` may be
needed to fit the resulting KV capacity.

Alternatively, reduce `--kv-cache-memory-bytes` or `--gpu-memory-utilization`.
An explicit KV memory byte budget takes precedence over utilization; no linear
conversion from CPU RAM to a GPU utilization percentage is assumed. If staging
alone exhausts the budget, the error asks for more RAM/container memory or smaller
`--max-num-batched-tokens`/`--max-num-seqs` instead of suggesting zero blocks.
This is a startup estimate, not an atomic host-memory reservation; concurrent
allocations and later output accumulation can change the available memory.

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
