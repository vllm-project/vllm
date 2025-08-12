# Reinforcement Learning from Human Feedback

Reinforcement Learning from Human Feedback (RLHF) is a technique that fine-tunes language models using human-generated preference data to align model outputs with desired behaviors.

vLLM can be used to generate the completions for RLHF. The best way to do this is with libraries like [TRL](https://github.com/huggingface/trl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) and [verl](https://github.com/volcengine/verl).

See the following basic examples to get started if you don't want to use an existing library:

- [Training and inference processes are located on separate GPUs (inspired by OpenRLHF)](https://docs.vllm.ai/en/latest/getting_started/examples/rlhf.html)
- [Training and inference processes are colocated on the same GPUs using Ray](https://docs.vllm.ai/en/latest/getting_started/examples/rlhf_colocate.html)
- [Utilities for performing RLHF with vLLM](https://docs.vllm.ai/en/latest/getting_started/examples/rlhf_utils.html)
