(input-processing)=

# Input Processing

```{eval-rst}
.. currentmodule:: vllm.inputs
```

Each model can override parts of vLLM's [input processing pipeline](#input-processing-pipeline) via
{data}`~vllm.inputs.INPUT_REGISTRY` and {data}`~vllm.multimodal.MULTIMODAL_REGISTRY`.

Currently, this mechanism is only utilized in [multi-modal](#multi-modality) models for preprocessing multi-modal input
data in addition to input prompt, but it can be extended to text-only language models when needed.

## Guides

```{toctree}
:maxdepth: 1

input_processing_pipeline
```

## Module Contents

### LLM Engine Inputs

```{eval-rst}
.. autoclass:: vllm.inputs.DecoderOnlyInputs
    :members:
    :show-inheritance:
```

### Registry

```{eval-rst}
.. autodata:: vllm.inputs.INPUT_REGISTRY
```

```{eval-rst}
.. automodule:: vllm.inputs.registry
    :members:
    :show-inheritance:
```
