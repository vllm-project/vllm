# Transformers Reinforcement Learning

[Transformers Reinforcement Learning](https://huggingface.co/docs/trl) (TRL) is a full stack library that provides a set of tools to train transformer language models with methods like Supervised Fine-Tuning (SFT), Group Relative Policy Optimization (GRPO), Direct Preference Optimization (DPO), Reward Modeling, and more. The library is integrated with 🤗 transformers.

Online methods such as GRPO or Online DPO require the model to generate completions. vLLM can be used to generate these completions!

See the [vLLM integration guide](https://huggingface.co/docs/trl/main/en/vllm_integration) in the TRL documentation for more information.

TRL currently supports the following online trainers with vLLM:

- [GRPO](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [Online DPO](https://huggingface.co/docs/trl/main/en/online_dpo_trainer)
- [RLOO](https://huggingface.co/docs/trl/main/en/rloo_trainer)
- [Nash-MD](https://huggingface.co/docs/trl/main/en/nash_md_trainer)
- [XPO](https://huggingface.co/docs/trl/main/en/xpo_trainer)

To enable vLLM in TRL, set the `use_vllm` flag in the trainer configuration to `True`. You can control how vLLM operates during training with the `vllm_mode` parameter, which supports two modes:

1. **Server mode**
2. **Colocate mode**

Some trainers also support **vLLM sleep mode**, which offloads parameters and caches to GPU RAM during training, helping reduce memory usage. Learn more in the [memory optimization docs](https://huggingface.co/docs/trl/main/en/reducing_memory_usage#vllm-sleep-mode).


!!! info
    For more information on the `use_vllm` flag you can provide to the configs of these online methods, see:

    - [`trl.GRPOConfig.use_vllm`](https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOConfig.use_vllm)
    - [`trl.OnlineDPOConfig.use_vllm`](https://huggingface.co/docs/trl/main/en/online_dpo_trainer#trl.OnlineDPOConfig.use_vllm)
    - [`trl.RLOOConfig.use_vllm`](https://huggingface.co/docs/trl/main/en/rloo_trainer#trl.RLOOConfig.use_vllm)

