# Reinforcement Learning from Human Feedback

Reinforcement Learning from Human Feedback (RLHF) is a technique that fine-tunes language models using human-generated preference data to align model outputs with desired behaviors.

vLLM can be used to generate the completions for RLHF. Some ways to do this include using libraries like [TRL](https://github.com/huggingface/trl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [verl](https://github.com/volcengine/verl) and [unsloth](https://github.com/unslothai/unsloth).

See the following basic examples to get started if you don't want to use an existing library:

- [Training and inference processes are located on separate GPUs (inspired by OpenRLHF)](../examples/offline_inference/rlhf.md)
- [Training and inference processes are colocated on the same GPUs using Ray](../examples/offline_inference/rlhf_colocate.md)
- [Utilities for performing RLHF with vLLM](../examples/offline_inference/rlhf_utils.md)

See the following notebooks showing how to use vLLM for GRPO:

- [Qwen-3 4B GRPO using Unsloth + vLLM](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)
