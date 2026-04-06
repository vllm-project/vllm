# Kimi-Audio

`KimiAudioForConditionalGeneration` is supported in vLLM for the verified
`text-output` subset of `moonshotai/Kimi-Audio-7B-Instruct`.

## Supported Behavior

- Audio-to-text transcription with text prompts.
- Audio understanding with text-only responses.
- Batched inference for the same `text-output` subset.
- Offline inference through [`LLM.generate`](../../examples/offline_inference/audio_language.py).

The current implementation keeps Kimi-Audio's model-specific prompt packing and
dual-stream input construction inside the Kimi-Audio model and processor
modules. This avoids changing vLLM's general multimodal runtime behavior for
other models.

## Loading And Runtime Defaults

- Load the model through the Hugging Face repo id
  `moonshotai/Kimi-Audio-7B-Instruct`.
- The discrete speech tokenizer is resolved from
  `THUDM/glm-4-voice-tokenizer` by default.
- An optional local path override for the speech tokenizer is available
  through `KIMI_AUDIO_SPEECH_TOKENIZER_PATH`.
- An optional device override for the speech tokenizer is available through
  `KIMI_AUDIO_SPEECH_TOKENIZER_DEVICE`.

## Required Prompt Shape

Kimi-Audio requests must use the official Kimi text-output prompt format with
the audio placeholder sequence:

```text
<|im_media_begin|><|im_kimia_text_blank|><|im_media_end|><|im_kimia_speech_ct_id|>
```

The example in [`examples/offline_inference/audio_language.py`](../../examples/offline_inference/audio_language.py)
shows the recommended pattern:

- Build the prompt with
  `vllm.model_executor.models.kimi_audio_prompt.KimiAudioPromptBuilder`.
- Pass structured `messages` through `mm_processor_kwargs`.
- Stop on `kimia_text_eos` before falling back to generic tokenizer EOS ids.

## Current Scope

The validated support boundary is:

- User text plus user audio inputs.
- Assistant text outputs.
- Text-output prompting for transcription and audio question answering.

The following capabilities are not part of the current support scope:

- Audio output generation.
- `output_type="both"`.
- Assistant history containing `audio-text` turns.
- The official Kimi-Audio detokenizer and vocoder path.

## Validation Status

The current implementation has been aligned against the official Kimi-Audio
repository for:

- English transcription on single-request and batched requests.
- Chinese transcription on the official ASR sample.
- Audio question answering with text-only output.

These checks confirm alignment for the current `text-output` subset, but they
do not imply support for the unsupported capabilities listed above.
