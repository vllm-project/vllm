# Kimi-Audio: Audio tower architecture placement (Decision)

## Problem

Kimi-Audio integration needs an audio processing component ("audio tower") to
transform request audio into model-consumable tensors.

Two constraints drive the architecture decision:

1. **Checkpoint compatibility / V1 multiprocessing**
   vLLM V1 uses subprocesses and a strict weight-loading check via
   `DefaultModelLoader.load_weights()`. Any parameters registered on the model
   that are not present in the checkpoint can trigger:

   `ValueError: Following weights were not initialized from checkpoint: {...}`

2. **vLLM multimodal modality contract**
   vLLM treats the `audio` modality as a raw waveform `AudioItem`
   (ndarray/tensor/(array,sr)), not a dict-of-features.

## Options considered

### Option A: Audio tower as a model submodule (tower component)

- Instantiate the audio tower inside `__init__` and register it as a child
  module.
- Wrap tower initialization with `with self._mark_tower_model(vllm_config, "audio"):`
  to separate it from the language model.

**Pros**
- Matches many vLLM multimodal implementations (e.g. Voxtral, FunAudioChat).
- Can be conditionally skipped when audio is disabled via `--limit-mm-per-prompt`.

**Cons**
- If the audio tower introduces parameters not present in the checkpoint,
  V1's strict weight-loading check can fail in multiprocessing.
- Requires either checkpoint-backed parameters only, or optional-weight logic.

### Option B: Audio tower as a runtime-only component (not registered)

- Keep the audio tower out of the model's parameter space.
- Instantiate lazily at runtime and store it in a non-`nn.Module` field or in a
  registry/context that avoids `model.named_parameters()`.

**Pros**
- Avoids V1 checkpoint strictness for non-checkpoint parameters.
- Works for components that are purely functional or that load weights from a
  different source.

**Cons**
- Must be carefully managed for device placement and lifecycle.
- Harder to apply LoRA/quantization hooks; less consistent with vLLM "tower"
  model patterns.

## Decision

**Choose a hybrid approach:**

1. **Use vLLM's `audio` modality as waveform input** (align with vLLM parser).
2. **Convert waveform → Kimi feature tensors inside the multimodal processor/model.**
3. **Place the audio tower depending on weight provenance:**
   - If the tower is checkpoint-backed (weights exist in the HF checkpoint),
     implement it as a proper model submodule under
     `with self._mark_tower_model(vllm_config, "audio"):`.
   - If the tower contains non-checkpoint parameters (or uses external libraries
     to compute features), keep it runtime-only and avoid registering its
     parameters as part of the model.

This preserves functionality while keeping V1 multiprocessing weight loading
stable.

## Required changes (implementation plan)

### Data flow

- Request audio (OpenAI `/v1/audio/transcriptions`) → vLLM provides waveform as
  `mm_data["audio"]`.
- Kimi multimodal processor parses waveform `AudioItem`.
- Processor/model computes Kimi-specific tensors (e.g. `whisper_input_features`,
  masks/ids) and passes them as passthrough kwargs.
- Model consumes these tensors in `forward()` / `embed_input_ids()`.

### Code areas to update

- `vllm/model_executor/models/kimi_audio_asr.py`
  - Ensure the multimodal parser for Kimi-Audio accepts waveform `AudioItem`.
  - Ensure dummy inputs for `audio` modality are waveform AudioItems.
  - Ensure no non-checkpoint parameters are registered on the model unless
    checkpoint-backed.

- `vllm/model_executor/models/interfaces.py`
  - Use `_mark_tower_model(vllm_config, "audio")` if introducing a true tower
    module.

- (Optional) Add tests
  - V1 mp startup should not raise missing-weights errors.
  - Audio transcription should not fail in `_get_audio_with_sr`.

## Compatibility notes

- **V1 engine**: strict weight loading + mp subprocess behavior requires
  checkpoint cleanliness for registered parameters.
- **V2 engine**: the same principles apply; keeping non-checkpoint parameters
  out of `named_parameters()` avoids surprises.

