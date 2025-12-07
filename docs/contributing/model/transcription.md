# Speech-to-Text (Transcription/Translation) Support

This document walks you through the steps to add support for speech-to-text (ASR) models to vLLM’s transcription and translation APIs by implementing [SupportsTranscription][vllm.model_executor.models.interfaces.SupportsTranscription].
Please refer to the [supported models](../../models/supported_models.md#transcription) for further guidance.

## Update the base vLLM model

It is assumed you have already implemented your model in vLLM according to the basic model guide. Extend your model with the [SupportsTranscription][vllm.model_executor.models.interfaces.SupportsTranscription] interface and implement the following class attributes and methods.

### `supported_languages` and `supports_transcription_only`

Declare supported languages and capabilities:

- The `supported_languages` mapping is validated at init time.
- Set `supports_transcription_only=True` if the model should not serve text generation (eg Whisper).

??? code "supported_languages and supports_transcription_only"

    ```python
    from typing import ClassVar, Mapping, Literal
    import numpy as np
    import torch
    from torch import nn

    from vllm.config import RendererConfig, SpeechToTextConfig
    from vllm.inputs.data import PromptType
    from vllm.model_executor.models.interfaces import SupportsTranscription
    
    class YourASRModel(nn.Module, SupportsTranscription):
        # Map of ISO 639-1 language codes to language names
        supported_languages: ClassVar[Mapping[str, str]] = {
            "en": "English",
            "it": "Italian",
            # ... add more as needed
        }
        
        # If your model only supports audio-conditioned generation
        # (no text-only generation), enable this flag.
        supports_transcription_only: ClassVar[bool] = True
    ```

Provide an ASR configuration via [get_speech_to_text_config][vllm.model_executor.models.interfaces.SupportsTranscription.get_speech_to_text_config].

This is for controlling general behavior of the API when serving your model:

??? code "get_speech_to_text_config()"

    ```python
    class YourASRModel(nn.Module, SupportsTranscription):
        ...

        @classmethod
        def get_speech_to_text_config(
            cls,
            renderer_config: RendererConfig,
            task_type: Literal["transcribe", "translate"],
        ) -> SpeechToTextConfig:
            return SpeechToTextConfig(
                sample_rate=16_000,
                max_audio_clip_s=30,
                # Set to None to disable server-side chunking if your
                # model/processor handles it already
                min_energy_split_window_size=None,
            )
    ```

See [Audio preprocessing and chunking](#audio-preprocessing-and-chunking) for what each field controls.

Implement the prompt construction via [get_generation_prompt][vllm.model_executor.models.interfaces.SupportsTranscription.get_generation_prompt]. The server passes you the resampled waveform and task parameters; you return a valid [PromptType][vllm.inputs.data.PromptType]. There are two common patterns:

#### Multimodal LLM with audio embeddings (e.g., Voxtral, Gemma3n)

Return a dict containing `multi_modal_data` with the audio, and either a `prompt` string or `prompt_token_ids`:

??? code "get_generation_prompt()"

    ```python
    class YourASRModel(nn.Module, SupportsTranscription):
        ...

        @classmethod
        def get_generation_prompt(
            cls,
            audio: np.ndarray,
            stt_config: SpeechToTextConfig,
            renderer_config: RendererConfig,
            language: str | None,
            task_type: Literal["transcribe", "translate"],
            request_prompt: str,
            to_language: str | None,
        ) -> PromptType:
            # Example with a free-form instruction prompt
            task_word = "Transcribe" if task_type == "transcribe" else "Translate"
            prompt = (
                "<start_of_turn>user\n"
                f"{task_word} this audio: <audio_soft_token>"
                "<end_of_turn>\n<start_of_turn>model\n"
            )

            return {
                "multi_modal_data": {"audio": (audio, stt_config.sample_rate)},
                "prompt": prompt,
            }
    ```

    For further clarification on multi modal inputs, please refer to [Multi-Modal Inputs](../../features/multimodal_inputs.md).

#### Encoder–decoder audio-only (e.g., Whisper)

Return a dict with separate `encoder_prompt` and `decoder_prompt` entries:

??? code "get_generation_prompt()"

    ```python
    class YourASRModel(nn.Module, SupportsTranscription):
        ...

        @classmethod
        def get_generation_prompt(
            cls,
            audio: np.ndarray,
            stt_config: SpeechToTextConfig,
            renderer_config: RendererConfig,
            language: str | None,
            task_type: Literal["transcribe", "translate"],
            request_prompt: str,
            to_language: str | None,
        ) -> PromptType:
            if language is None:
                raise ValueError("Language must be specified")

            prompt = {
                "encoder_prompt": {
                    "prompt": "",
                    "multi_modal_data": {
                        "audio": (audio, stt_config.sample_rate),
                    },
                },
                "decoder_prompt": (
                    (f"<|prev|>{request_prompt}" if request_prompt else "")
                    + f"<|startoftranscript|><|{language}|>"
                    + f"<|{task_type}|><|notimestamps|>"
                ),
            }
            return cast(PromptType, prompt)
    ```

### `validate_language` (optional)

Language validation via [validate_language][vllm.model_executor.models.interfaces.SupportsTranscription.validate_language]

If your model requires a language and you want a default, override this method (see Whisper):

??? code "validate_language()"

    ```python
    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        if language is None:
            logger.warning(
                "Defaulting to language='en'. If you wish to transcribe "
                "audio in a different language, pass the `language` field "
                "in the TranscriptionRequest."
            )
            language = "en"
        return super().validate_language(language)
    ```

### `get_num_audio_tokens` (optional)

Token accounting for streaming via [get_num_audio_tokens][vllm.model_executor.models.interfaces.SupportsTranscription.get_num_audio_tokens]

Provide a fast duration→token estimate to improve streaming usage statistics:

??? code "get_num_audio_tokens()"

    ```python
    class YourASRModel(nn.Module, SupportsTranscription):
        ...

        @classmethod
        def get_num_audio_tokens(
            cls,
            audio_duration_s: float,
            stt_config: SpeechToTextConfig,
            renderer_config: RendererConfig,
        ) -> int | None:
            # Return None if unknown; otherwise return an estimate.
            return int(audio_duration_s * stt_config.sample_rate // 320)  # example
    ```

## Audio preprocessing and chunking

The API server takes care of basic audio I/O and optional chunking before building prompts:

- Resampling: Input audio is resampled to `SpeechToTextConfig.sample_rate` using `librosa`.
- Chunking: If `SpeechToTextConfig.allow_audio_chunking` is True and the duration exceeds `max_audio_clip_s`, the server splits the audio into overlapping chunks and generates a prompt per chunk. Overlap is controlled by `overlap_chunk_second`.
- Energy-aware splitting: When `min_energy_split_window_size` is set, the server finds low-energy regions to minimize cutting within words.

Relevant server logic:

??? code "_preprocess_speech_to_text()"

    ```python
    # vllm/entrypoints/openai/speech_to_text.py
    async def _preprocess_speech_to_text(...):
        language = self.model_cls.validate_language(request.language)
        ...
        y, sr = librosa.load(bytes_, sr=self.asr_config.sample_rate)
        duration = librosa.get_duration(y=y, sr=sr)
        do_split_audio = (self.asr_config.allow_audio_chunking
                        and duration > self.asr_config.max_audio_clip_s)
        chunks = [y] if not do_split_audio else self._split_audio(y, int(sr))
        prompts = []
        for chunk in chunks:
            prompt = self.model_cls.get_generation_prompt(
                audio=chunk,
                stt_config=self.asr_config,
                renderer_config=self.renderer_config,
                language=language,
                task_type=self.task_type,
                request_prompt=request.prompt,
                to_language=to_language,
            )
            prompts.append(prompt)
        return prompts, duration
    ```

## Exposing tasks automatically

vLLM automatically advertises transcription support if your model implements the interface:

```python
if supports_transcription(model):
    if model.supports_transcription_only:
        return ["transcription"]
    supported_tasks.append("transcription")
```

When enabled, the server initializes the transcription and translation handlers:

```python
state.openai_serving_transcription = OpenAIServingTranscription(...) if "transcription" in supported_tasks else None
state.openai_serving_translation = OpenAIServingTranslation(...) if "transcription" in supported_tasks else None
```

No extra registration is required beyond having your model class available via the model registry and implementing `SupportsTranscription`.

## Examples in-tree

- Whisper encoder–decoder (audio-only): [vllm/model_executor/models/whisper.py](../../../vllm/model_executor/models/whisper.py)
- Voxtral decoder-only (audio embeddings + LLM): [vllm/model_executor/models/voxtral.py](../../../vllm/model_executor/models/voxtral.py). Make sure to have installed `mistral-common[audio]`.
- Gemma3n decoder-only with fixed instruction prompt: [vllm/model_executor/models/gemma3n_mm.py](../../../vllm/model_executor/models/gemma3n_mm.py)

## Test with the API

Once your model implements `SupportsTranscription`, you can test the endpoints (API mimics OpenAI):

- Transcription (ASR):

    ```bash
    curl -s -X POST \
      -H "Authorization: Bearer $VLLM_API_KEY" \
      -H "Content-Type: multipart/form-data" \
      -F "file=@/path/to/audio.wav" \
      -F "model=$MODEL_ID" \
      http://localhost:8000/v1/audio/transcriptions
    ```

- Translation (source → English unless otherwise supported):

    ```bash
    curl -s -X POST \
      -H "Authorization: Bearer $VLLM_API_KEY" \
      -H "Content-Type: multipart/form-data" \
      -F "file=@/path/to/audio.wav" \
      -F "model=$MODEL_ID" \
      http://localhost:8000/v1/audio/translations
    ```

Or check out more examples in [examples/online_serving](../../../examples/online_serving).

!!! note
    - If your model handles chunking internally (e.g., via its processor or encoder), set `min_energy_split_window_size=None` in the returned `SpeechToTextConfig` to disable server-side chunking.
    - Implementing `get_num_audio_tokens` improves accuracy of streaming usage metrics (`prompt_tokens`) without an extra forward pass.
    - For multilingual behavior, keep `supported_languages` aligned with actual model capabilities.
