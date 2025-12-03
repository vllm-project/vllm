# Summary

## Configuration

API documentation for vLLM's configuration classes.

- [vllm.config.ModelConfig][]
- [vllm.config.CacheConfig][]
- [vllm.config.LoadConfig][]
- [vllm.config.ParallelConfig][]
- [vllm.config.SchedulerConfig][]
- [vllm.config.DeviceConfig][]
- [vllm.config.SpeculativeConfig][]
- [vllm.config.LoRAConfig][]
- [vllm.config.MultiModalConfig][]
- [vllm.config.PoolerConfig][]
- [vllm.config.StructuredOutputsConfig][]
- [vllm.config.ProfilerConfig][]
- [vllm.config.ObservabilityConfig][]
- [vllm.config.KVTransferConfig][]
- [vllm.config.CompilationConfig][]
- [vllm.config.VllmConfig][]

## Offline Inference

LLM Class.

- [vllm.LLM][]

LLM Inputs.

- [vllm.inputs.PromptType][]
- [vllm.inputs.TextPrompt][]
- [vllm.inputs.TokensPrompt][]

## vLLM Engines

Engine classes for offline and online inference.

- [vllm.LLMEngine][]
- [vllm.AsyncLLMEngine][]

## Inference Parameters

Inference parameters for vLLM APIs.

- [vllm.SamplingParams][]
- [vllm.PoolingParams][]

## Multi-Modality

vLLM provides experimental support for multi-modal models through the [vllm.multimodal][] package.

Multi-modal inputs can be passed alongside text and token prompts to [supported models](../models/supported_models.md#list-of-multimodal-language-models)
via the `multi_modal_data` field in [vllm.inputs.PromptType][].

Looking to add your own multi-modal model? Please follow the instructions listed [here](../contributing/model/multimodal.md).

- [vllm.multimodal.MULTIMODAL_REGISTRY][]

### Inputs

User-facing inputs.

- [vllm.multimodal.inputs.MultiModalDataDict][]

Internal data structures.

- [vllm.multimodal.inputs.PlaceholderRange][]
- [vllm.multimodal.inputs.NestedTensors][]
- [vllm.multimodal.inputs.MultiModalFieldElem][]
- [vllm.multimodal.inputs.MultiModalFieldConfig][]
- [vllm.multimodal.inputs.MultiModalKwargsItem][]
- [vllm.multimodal.inputs.MultiModalKwargsItems][]
- [vllm.multimodal.inputs.MultiModalKwargs][]
- [vllm.multimodal.inputs.MultiModalInputs][]

### Data Parsing

- [vllm.multimodal.parse][]

### Data Processing

- [vllm.multimodal.processing][]

### Memory Profiling

- [vllm.multimodal.profiling][]

### Registry

- [vllm.multimodal.registry][]

## Model Development

- [vllm.model_executor.models.interfaces_base][]
- [vllm.model_executor.models.interfaces][]
- [vllm.model_executor.models.adapters][]
