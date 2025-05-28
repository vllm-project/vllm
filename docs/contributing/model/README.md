---
title: Adding a New Model
---
[](){ #new-model }

This section provides more information on how to integrate a [PyTorch](https://pytorch.org/) model into vLLM.

Contents:

- [Basic](basic.md)
- [Registration](registration.md)
- [Tests](tests.md)
- [Multimodal](multimodal.md)

!!! note
    The complexity of adding a new model depends heavily on the model's architecture.
    The process is considerably straightforward if the model shares a similar architecture with an existing model in vLLM.
    However, for models that include new operators (e.g., a new attention mechanism), the process can be a bit more complex.

!!! tip
    If you are encountering issues while integrating your model into vLLM, feel free to open a [GitHub issue](https://github.com/vllm-project/vllm/issues)
    or ask on our [developer slack](https://slack.vllm.ai).
    We will be happy to help you out!
