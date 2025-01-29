(multi-modality)=

# Multi-Modality

vLLM provides experimental support for multi-modal models through the {mod}`vllm.multimodal` package.

Multi-modal inputs can be passed alongside text and token prompts to [supported models](#supported-mm-models)
via the `multi_modal_data` field in {class}`vllm.inputs.PromptType`.

Looking to add your own multi-modal model? Please follow the instructions listed [here](#supports-multimodal).

## Module Contents

```{eval-rst}
.. autodata:: vllm.multimodal.MULTIMODAL_REGISTRY
```

## Submodules

:::{toctree}
:maxdepth: 1

inputs
parse
processing
profiling
registry
:::
