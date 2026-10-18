# Reinforcement Learning from Human Feedback

Reinforcement Learning from Human Feedback (RLHF) is a technique that fine-tunes language models using human-generated preference data to align model outputs with desired behaviors. vLLM can be used to generate the completions for RLHF.

The following open-source RL libraries use vLLM for fast rollouts (sorted alphabetically and non-exhaustive):

- [Cosmos-RL](https://github.com/nvidia-cosmos/cosmos-rl)
- [ms-swift](https://github.com/modelscope/ms-swift/tree/main)
- [NeMo-RL](https://github.com/NVIDIA-NeMo/RL)
- [Open Instruct](https://github.com/allenai/open-instruct)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [PipelineRL](https://github.com/ServiceNow/PipelineRL)
- [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl)
- [SkyRL](https://github.com/NovaSky-AI/SkyRL)
- [TRL](https://github.com/huggingface/trl)
- [Unsloth](https://github.com/unslothai/unsloth)
- [verl](https://github.com/volcengine/verl)

For weight synchronization between training and inference, see the [Weight Transfer](weight_transfer/README.md) documentation, which covers the pluggable backend system with [NCCL](weight_transfer/nccl.md) (multi-GPU) and [IPC](weight_transfer/ipc.md) (same-GPU) engines. To verify that an update landed on every rank, see the [Weight Checker](weight_checker.md).

For MoE policies, see [Routed Experts Return](routed_experts.md) (`--enable-return-routed-experts`) to get the per-token expert routing of each rollout for routing replay (R3) in the trainer.

For pipelining generation and training to improve GPU utilization and throughput, see the [Async Reinforcement Learning](async_rl.md) guide, which covers the pause/resume API for safely updating weights mid-flight.

## Routed-expert outputs in Mooncake

`--aux-output-config '{"enable_return_routed_experts":true,"backend":"mooncake"}'`
stores routed-expert (R3) data in Mooncake and returns ordered `aux_output_keys`
on the terminal `/inference/v1/generate` response. Configure the backend with
`VLLM_AUX_OUTPUT_MOONCAKE_CONFIG_PATH`; it is independent of KV transfer configuration.

- Share a namespace only between instances using the same weights, R3 profile
  (layers, experts per token and dtype), block size, hash settings and reset
  generation. Generation starts at zero again after a process restart; it does
  not identify a checkpoint. Use a separate Store or a separate Mooncake tenant
  for each model/checkpoint/profile cohort, and rotate the tenant when restarting
  with different weights. P and D must use the same cohort.
- Tenant isolation requires **both** `tenant_id` in the client JSON and
  `--enable_multi_tenants=true` on the Mooncake master with a registered tenant
  quota. Without that master setting, different client tenant IDs share the
  default namespace. See [Mooncake tenant configuration](https://kvcache-ai.github.io/Mooncake/deployment/mooncake-store-deployment-guide.html#tenant-quota-management).
- Retention is owned by the deployment. Size the pool for outstanding outputs
  and all consumers' read delays. An embedded client's contributed memory lasts
  only as long as that storage participant; use separately managed Store nodes
  when outputs must outlive vLLM processes.
- Response-tail keys (`r3-tail:`) are outside the `vllm-artifact/` full-block
  namespace. Include both prefixes in ACL, quota and cleanup policies. There is
  no automatic tail TTL or consumer reference count. After **all** readers,
  including any P/D handoff, finish, the owner may reclaim its tail keys with
  the native client's `remove(key)`. Do not delete shared full-block keys when
  one request or trainer finishes. Without explicit cleanup, tails remain until
  Store eviction or loss of their storage replicas.

See the following notebooks showing how to use vLLM for GRPO:

- [Efficient Online Training with GRPO and vLLM in TRL](https://huggingface.co/learn/cookbook/grpo_vllm_online_training)
- [Qwen-3 4B GRPO using Unsloth + vLLM](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)
