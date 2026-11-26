# Training and Reinforcement Learning (RL)

vLLM is widely used as the high-throughput rollout and teacher-scoring engine for post-training workflows including RLHF, GRPO, Online DPO, and on-policy distillation.

## Quick Routing Guide (Task → Doc)

| What you want to do | Document | Key Flags / APIs |
| --- | --- | --- |
| Synchronize updated trainer weights into a running vLLM engine (IPC, NCCL, M2N, Sharded RDT, ModelExpress) | [Weight Updates](weight_updates/README.md) | `--weight-transfer-config`, `VLLM_RELOAD_MODE` |
| Sync weights over HTTP without installing or importing vLLM in the trainer process | [Trainer Without Importing vLLM (HTTP-Only)](weight_updates/README.md#trainer-without-importing-vllm-http-only) | `VLLM_SERVER_DEV_MODE=1`, `/init_weight_transfer_engine`, `/update_weights` |
| Understand in-place weight reload vs. layerwise reload & online quantization (QeRL) | [Direct Weight Reload](weight_updates/direct_reload.md) / [Layerwise (Re)loading](weight_updates/layerwise.md) | `VLLM_RELOAD_MODE=auto\|direct\|layerwise` |
| Verify that a weight update landed identically on every rank | [Weight Checker](weight_updates/weight_checker.md) | SHA-256 parameter digest check |
| Export rollout artifacts to the trainer (MoE routing, sampling masks, teacher top-K logprobs) | [Rollout Artifacts](rollout_artifacts/README.md) | `--enable-return-routed-experts`, `prompt_logprob_token_ids` |
| Overlap rollout generation and training (async RL / one-off pipelining) | [Async Reinforcement Learning](async_rl.md) | `pause_generation`, `resume_generation`, `weight_version` |
| Integrate vLLM with Hugging Face TRL | [Transformers Reinforcement Learning (TRL)](trl.md) | `use_vllm=True`, `vllm_mode` |

## Related Features Across vLLM

These capabilities live under `docs/features/` because they also serve general inference, but are frequently used in RL and post-training setups:

- **[Sleep Mode](../features/sleep_mode.md)**: Free GPU memory (weights and KV cache) between rollout and training phases in colocated RL.
- **[Batch Invariance](../features/batch_invariance.md)**: Reduce train–inference numerical mismatch by making rollout outputs invariant to batching.
- **[Online Quantization](../features/quantization/online.md)**: Quantize full-precision trainer weights on the fly during [direct](weight_updates/direct_reload.md#online-quantization) or [layerwise](weight_updates/layerwise.md#layerwise-loading-with-online-quantization) reload.
- **[Hidden State Extraction](../features/speculative_decoding/extract_hidden_states.md)**: Export per-token hidden states via a connector for training speculative-decoding draft models (e.g., EAGLE).

## Open-Source RL Libraries Using vLLM

The following open-source RL libraries use vLLM for fast rollouts (sorted alphabetically and non-exhaustive):

- [Cosmos-RL](https://github.com/nvidia-cosmos/cosmos-rl)
- [ms-swift](https://github.com/modelscope/ms-swift/tree/main)
- [NeMo-RL](https://github.com/NVIDIA-NeMo/RL)
- [Open Instruct](https://github.com/allenai/open-instruct)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [PipelineRL](https://github.com/ServiceNow/PipelineRL)
- [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl)
- [SkyRL](https://github.com/NovaSky-AI/SkyRL)
- [TRL](https://github.com/huggingface/trl) (see [TRL integration guide](trl.md))
- [Unsloth](https://github.com/unslothai/unsloth)
- [verl](https://github.com/volcengine/verl)
