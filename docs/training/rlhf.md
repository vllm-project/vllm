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

See the following basic examples to get started if you don't want to use an existing library:

- [Training and inference processes are located on separate GPUs](../examples/offline_inference/weight_syncing.md#rlhf)
- [Training and inference processes communicate via IPC](../examples/offline_inference/weight_syncing.md#rlhf-ipc)

See the following notebooks showing how to use vLLM for GRPO:

- [Efficient Online Training with GRPO and vLLM in TRL](https://huggingface.co/learn/cookbook/grpo_vllm_online_training)
- [Qwen-3 4B GRPO using Unsloth + vLLM](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)
