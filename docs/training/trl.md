# Transformers Reinforcement Learning

Transformers Reinforcement Learning (TRL) is a full stack library that provides a set of tools to train transformer language models with methods like Supervised Fine-Tuning (SFT), Group Relative Policy Optimization (GRPO), Direct Preference Optimization (DPO), Reward Modeling, and more. The library is integrated with 🤗 transformers.

Online methods such as GRPO or Online DPO require the model to generate completions. vLLM can be used to generate these completions!

See the guide [vLLM for fast generation in online methods](https://huggingface.co/docs/trl/main/en/speeding_up_training#vllm-for-fast-generation-in-online-methods) in the TRL documentation for more information.

!!! info
    For more information on the `use_vllm` flag you can provide to the configs of these online methods, see:
    - [`trl.GRPOConfig.use_vllm`](https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOConfig.use_vllm)
    - [`trl.OnlineDPOConfig.use_vllm`](https://huggingface.co/docs/trl/main/en/online_dpo_trainer#trl.OnlineDPOConfig.use_vllm)
