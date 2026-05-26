# Summary

!!! important
    Many decoder language models can now be automatically loaded using the [Transformers modeling backend](../../models/supported_models.md#transformers) without having to implement them in vLLM. See if `vllm serve <model>` works first!

vLLM models are specialized [PyTorch](https://pytorch.org/) models that take advantage of various [features](../../features/README.md#compatibility-matrix) to optimize their performance.

The complexity of integrating a model into vLLM depends heavily on the model's architecture.
The process is considerably straightforward if the model shares a similar architecture with an existing model in vLLM.
However, this can be more complex for models that include new operators (e.g., a new attention mechanism).

Read through these pages for a step-by-step guide:

- [Basic Model](basic.md)
- [Registering a Model](registration.md)
- [Unit Testing](tests.md)
- [Multi-Modal Support](multimodal.md)
- [Speech-to-Text Support](transcription.md)

!!! tip
    If you are encountering issues while integrating your model into vLLM, feel free to open a [GitHub issue](https://github.com/vllm-project/vllm/issues)
    or ask on our [developer slack](https://slack.vllm.ai).
    We will be happy to help you out!
