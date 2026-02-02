# Kimi-Audio: vLLM Patterns for External Multimodal Components (Investigation)

This note summarizes relevant vLLM multimodal integration patterns for models that include external components (vision/audio towers, connectors, processors) which may not map 1:1 to the language-model checkpoint.

## Key vLLM Mechanism: `_mark_tower_model()` / `_mark_language_model()`

vLLM provides context managers on multimodal model base classes (see `vllm/model_executor/models/interfaces.py`) to mark which submodules belong to the language model vs. tower model components.

### What it does

- Tracks which children were assigned inside the context (`collect_children`).
- When a modality is disabled via `--limit-mm-per-prompt` (limit set to 0), vLLM wraps initialization using `no_init_weights(...)` and a `StageMissingLayer` placeholder.
- This prevents unnecessary parameter initialization for disabled modalities and avoids loading/expecting weights for those tower components.

This is the primary pattern vLLM uses to keep multimodal tower components logically separate from the language model.

Reference: `vllm/model_executor/models/interfaces.py` (`_mark_tower_model`, `_mark_language_model`).

## Observed Model Implementations

### 1) Voxtral (audio tower + adapter)

File: `vllm/model_executor/models/voxtral.py`

Pattern:
- Instantiate the language model under `_mark_language_model(vllm_config)`.
- Instantiate the audio encoder tower + connector under `_mark_tower_model(vllm_config, "audio")`.

Key takeaway:
- The audio tower is a first-class model submodule, but it is explicitly marked as tower model. This enables correct behavior when audio is disabled (e.g., startup in text-only mode).

### 2) FunAudioChat (continuous + discrete audio towers)

File: `vllm/model_executor/models/funaudiochat.py`

Pattern:
- Instantiate one or more audio towers under `_mark_tower_model(vllm_config, "audio")`.
- Instantiate the language model under `_mark_language_model(vllm_config)`.
- Use a registered multimodal processor to create placeholders and perform prompt replacement.

Key takeaway:
- Multiple tower modules can be grouped under a single modality mark.
- Prompt-replacement logic is used to map an audio placeholder token into the appropriate number of “audio tokens” for the model.

### 3) General multimodal processor registration

Many multimodal models register processors via:

```python
@MULTIMODAL_REGISTRY.register_processor(
    SomeMultiModalProcessor,
    info=SomeProcessingInfo,
    dummy_inputs=SomeDummyInputsBuilder,
)
```

This defines:
- how request-time multimodal data is parsed and converted into model kwargs
- how dummy multimodal inputs are generated for profiling / warmup

## Practical Implications for Kimi-Audio

### A) Tower components that are not in the checkpoint

If a tower module contains parameters that do not exist in the HF checkpoint (or are generated at runtime), vLLM’s default weight-loading will fail in V1 multiprocessing unless the module is either:

1. **not registered as a model parameter/submodule**, or
2. **explicitly marked and conditionally skipped** when the corresponding modality is disabled (`--limit-mm-per-prompt`), or
3. **its weights are treated as optional** by custom weight-loading logic.

In vLLM’s existing implementations, (2) is achieved via `_mark_tower_model()` plus disabling modalities when needed.

### B) Modality contract: `audio` == waveform arrays

vLLM’s core multimodal parser (`vllm/multimodal/parse.py`) treats the `audio` modality as raw audio arrays/tensors (AudioItem):
- `np.ndarray`, `torch.Tensor`, `list[float]`, or `(array, sample_rate)`

It does **not** accept dict-of-tensors such as `{ "whisper_input_features": ... }` under the `audio` key.

Therefore, Kimi-Audio must either:

- **align with vLLM’s `audio` modality** by accepting waveform inputs and converting to Kimi features inside the Kimi multimodal processor/model, or
- use a different modality key for precomputed features (embedding-only path) and provide a translation layer from waveform to that modality.

## Recommended Pattern to Apply

For Kimi-Audio, the vLLM-aligned approach is:

1. Treat the request-time modality `audio` as raw waveform AudioItem.
2. Convert waveform → Kimi-specific tensors (e.g., whisper features) inside Kimi’s multimodal processor/model.
3. Keep any audio tower/connector parameters either:
   - marked with `_mark_tower_model(vllm_config, "audio")` if they are checkpoint-backed, or
   - runtime-only (not registered as parameters) if they are not checkpoint-backed.

This mirrors existing vLLM audio multimodal models and is compatible with V1 multiprocessing + dummy-input warmups.
