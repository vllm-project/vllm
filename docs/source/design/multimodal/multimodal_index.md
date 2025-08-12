(multi-modality)=

# Multi-Modality

```{eval-rst}
.. currentmodule:: vllm.multimodal
```

vLLM provides experimental support for multi-modal models through the {mod}`vllm.multimodal` package.

Multi-modal inputs can be passed alongside text and token prompts to [supported models](#supported-mm-models)
via the `multi_modal_data` field in {class}`vllm.inputs.PromptType`.

Currently, vLLM only has built-in support for image data. You can extend vLLM to process additional modalities
by following [this guide](#adding-multimodal-plugin).

Looking to add your own multi-modal model? Please follow the instructions listed [here](#enabling-multimodal-inputs).

## Guides

```{toctree}
:maxdepth: 1

adding_multimodal_plugin
```

## Module Contents

```{eval-rst}
.. automodule:: vllm.multimodal
```

### Registry

```{eval-rst}
.. autodata:: vllm.multimodal.MULTIMODAL_REGISTRY
```

```{eval-rst}
.. autoclass:: vllm.multimodal.MultiModalRegistry
    :members:
    :show-inheritance:
```

### Base Classes

```{eval-rst}
.. autodata:: vllm.multimodal.NestedTensors
```

```{eval-rst}
.. autodata:: vllm.multimodal.BatchedTensorInputs
```

```{eval-rst}
.. autoclass:: vllm.multimodal.MultiModalDataBuiltins
    :members:
    :show-inheritance:
```

```{eval-rst}
.. autodata:: vllm.multimodal.MultiModalDataDict
```

```{eval-rst}
.. autoclass:: vllm.multimodal.MultiModalKwargs
    :members:
    :show-inheritance:
```

```{eval-rst}
.. autoclass:: vllm.multimodal.MultiModalPlugin
    :members:
    :show-inheritance:
```

### Image Classes

```{eval-rst}
.. automodule:: vllm.multimodal.image
    :members:
    :show-inheritance:
```
