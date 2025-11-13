[RFC]: vLLM Support for Generic Model Definitions #28326

---

### Motivation.

As users experiment with model architectures they often need a fast inference path to explore post-training (with reinforcement learning).  Since training in this domain is inference dominated, vLLM should increase support for research by providing utilities to construct fast-inference models directly from training code with low cognitive overhead.

Recent exploration in [bitwise-exact reinforcement learning](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/deterministic_vllm_rl) with vLLM has shown that leveraging the features of vLLM are crucial to ensure stability in training runs.  Further, vLLM is already used in many RL frameworks (such as VERL and TorchForge), and by simplifying integration, we can continue to help grow the vLLM impact and community.

### Proposed Change.

The goal is to support generic training/etc. ⇔ inference compatibility.  We propose a formal **specification** and **helper utilities** for use of `ModelRegistry.register_model` with training and other workloads.

- A specification of the minimum set of features needed to use vLLM with a user-defined model
- A wrapper that will enable simplistic “training-only” model specifications (by shimming interfaces or features for use with `register_model`)
- A set of vLLM-provided `nn.Module`s to support both training and efficient inference, such as `Attention`
- Flexibility to support user-defined parallelism (specifically exposing process group instantiation to model registration)


### Feedback Period.

1-2 weeks

### CC List.

@zhuohan123 @tianyu-l @teja-rao

### Any Other Things.

Current very-close interface (that requires invasive changes without utilities):
https://docs.vllm.ai/en/latest/contributing/model/basic/

Additional features (module boilerplate) here: 
https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py

Additional backwards passes for attention could be added here:
https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.
