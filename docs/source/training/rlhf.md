# Reinforcement Learning from Human Feedback

Reinforcement Learning from Human Feedback (RLHF) is a technique that fine-tunes language models using human-generated preference data to align model outputs with desired behaviours.

vLLM can be used to generate the completions for RLHF. We have the following very basic examples you can use to get started:

- [Training and inference processes are located on separate GPUs (inspired by OpenRLHF)](https://docs.vllm.ai/en/latest/getting_started/examples/rlhf.html)
- [Training and inference processes are colocated on the same GPUs using Ray](https://docs.vllm.ai/en/latest/getting_started/examples/rlhf_colocate.html)
- [Utilities for performing RLHF with vLLM](https://docs.vllm.ai/en/latest/getting_started/examples/rlhf_utils.html)
