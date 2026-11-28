# Rollout Artifacts

During RL rollouts and on-policy distillation, the trainer often needs auxiliary per-token outputs from the inference pass—beyond just generated token IDs—to eliminate train–inference mismatch or compute distillation targets.

## Quick Routing Guide

| Artifact | Document | Use Case | Key Flag / Parameter |
| --- | --- | --- | --- |
| **Routed Experts (R3)** | [Routed Experts Return](routed_experts.md) | MoE routing replay: reuse rollout router expert IDs during the training forward pass | `--enable-return-routed-experts` |
| **Sampling Mask** | [Sampling Mask](sampling_mask.md) | Distribution replay: return the exact token IDs that survived top-k / top-p / min-p filtering | `--return-sampling-mask`, `--logprobs-mode processed_logprobs` |
| **Prompt Token ID Logprobs** | [Prompt Token ID Logprobs](prompt_token_id_logprobs.md) | On-policy distillation (teacher scoring): compute teacher logprobs for exact student top-K candidate tokens | `prompt_logprob_token_ids` |
| **Hidden States** | [Hidden State Extraction](../../features/speculative_decoding/extract_hidden_states.md) | Export hidden states via connector for training speculative-decoding draft models (e.g., EAGLE) | `kv_transfer_params` / hidden states connector |

## Auxiliary Output Connector (`aux_output_connector`)

For large per-token artifacts such as MoE routed experts, vLLM uses the **aux output connector** (`vllm/distributed/aux_output_connector/`) to store artifact rows per engine in host memory, keyed by KV block hash, on the engine's first tensor-parallel worker. See [Routed Experts Return — Row storage](routed_experts.md#row-storage) for storage layout, replica affinity, cache invalidation on weight updates, and `max_bytes` sizing.

To deliver the rows through a shared object store instead of the response, see [Aux Outputs for RL](aux_outputs.md).
