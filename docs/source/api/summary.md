# Summary

(configuration)=

## Configuration

API documentation for vLLM's configuration classes.

```{autodoc2-summary}
    vllm.config.ModelConfig
    vllm.config.CacheConfig
    vllm.config.TokenizerPoolConfig
    vllm.config.LoadConfig
    vllm.config.ParallelConfig
    vllm.config.SchedulerConfig
    vllm.config.DeviceConfig
    vllm.config.SpeculativeConfig
    vllm.config.LoRAConfig
    vllm.config.PromptAdapterConfig
    vllm.config.MultiModalConfig
    vllm.config.PoolerConfig
    vllm.config.DecodingConfig
    vllm.config.ObservabilityConfig
    vllm.config.KVTransferConfig
    vllm.config.CompilationConfig
    vllm.config.VllmConfig
```

(offline-inference-api)=

## Offline Inference

LLM Class.

```{autodoc2-summary}
    vllm.LLM
```

LLM Inputs.

```{autodoc2-summary}
    vllm.inputs.PromptType
    vllm.inputs.TextPrompt
    vllm.inputs.TokensPrompt
```

## vLLM Engines

Engine classes for offline and online inference.

```{autodoc2-summary}
    vllm.LLMEngine
    vllm.AsyncLLMEngine
```

## Inference Parameters

Inference parameters for vLLM APIs.

(sampling-params)=
(pooling-params)=

```{autodoc2-summary}
    vllm.SamplingParams
    vllm.PoolingParams
```

(multi-modality)=

## Multi-Modality

vLLM provides experimental support for multi-modal models through the {mod}`vllm.multimodal` package.

Multi-modal inputs can be passed alongside text and token prompts to [supported models](#supported-mm-models)
via the `multi_modal_data` field in {class}`vllm.inputs.PromptType`.

Looking to add your own multi-modal model? Please follow the instructions listed [here](#supports-multimodal).

```{autodoc2-summary}
    vllm.multimodal.MULTIMODAL_REGISTRY
```

### Inputs

User-facing inputs.

```{autodoc2-summary}
    vllm.multimodal.inputs.MultiModalDataDict
```

Internal data structures.

```{autodoc2-summary}
    vllm.multimodal.inputs.PlaceholderRange
    vllm.multimodal.inputs.NestedTensors
    vllm.multimodal.inputs.MultiModalFieldElem
    vllm.multimodal.inputs.MultiModalFieldConfig
    vllm.multimodal.inputs.MultiModalKwargsItem
    vllm.multimodal.inputs.MultiModalKwargs
    vllm.multimodal.inputs.MultiModalInputs
```

### Data Parsing

```{autodoc2-summary}
    vllm.multimodal.parse
```

### Data Processing

```{autodoc2-summary}
    vllm.multimodal.processing
```

### Memory Profiling

```{autodoc2-summary}
    vllm.multimodal.profiling
```

### Registry

```{autodoc2-summary}
    vllm.multimodal.registry
```

## Model Development

```{autodoc2-summary}
    vllm.model_executor.models.interfaces_base
    vllm.model_executor.models.interfaces
    vllm.model_executor.models.adapters
```
