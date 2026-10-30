# Speech to Text APIs

## Transcriptions API

Our Transcriptions API is compatible with [OpenAI's Transcriptions API](https://platform.openai.com/docs/api-reference/audio/createTranscription);
you can use the [official OpenAI Python client](https://github.com/openai/openai-python) to interact with it.

!!! note
    To use the Transcriptions API, please install with extra audio dependencies using `pip install vllm[audio]`.

Code example: [examples/speech_to_text/openai/openai_transcription_client.py](../../../examples/speech_to_text/openai/openai_transcription_client.py)

NOTE: beam search is currently supported in the transcriptions endpoint for encoder-decoder multimodal models, e.g., whisper, but highly inefficient as work for handling the encoder/decoder cache is actively ongoing. This is an active point of ongoing optimization and will be handled properly in the very near future.

### API Enforced Limits

Set the maximum audio file size (in MB) that VLLM will accept, via the
`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` environment variable. Default is 25 MB.

### Uploading Audio Files

The Transcriptions API supports uploading audio files in various formats including FLAC, MP3, MP4, MPEG, MPGA, M4A, OGG, WAV, and WEBM.

**Using OpenAI Python Client:**

??? code

    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )

    # Upload audio file from disk
    with open("audio.mp3", "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="openai/whisper-large-v3-turbo",
            file=audio_file,
            language="en",
            response_format="verbose_json",
        )

    print(transcription.text)
    ```

**Using curl with multipart/form-data:**

??? code

    ```bash
    curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
      -H "Authorization: Bearer token-abc123" \
      -F "file=@audio.mp3" \
      -F "model=openai/whisper-large-v3-turbo" \
      -F "language=en" \
      -F "response_format=verbose_json"
    ```

**Supported Parameters:**

- `file`: The audio file to transcribe (required)
- `model`: The model to use for transcription (required)
- `language`: The language code (e.g., "en", "zh") (optional)
- `prompt`: Optional text to guide the transcription style (optional)
- `response_format`: Format of the response ("json", "text", "verbose_json", or
  "diarized_json") (optional)
- `temperature`: Sampling temperature between 0 and 1 (optional)

For the complete list of supported parameters including sampling parameters and vLLM extensions, see the [protocol definitions](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/speech_to_text/transcription/protocol.py).

**Response Format:**

For `verbose_json` response format:

??? code

    ```json
    {
      "text": "Hello, this is a transcription of the audio file.",
      "language": "en",
      "duration": 5.42,
      "segments": [
        {
          "id": 0,
          "seek": 0,
          "start": 0.0,
          "end": 2.5,
          "text": "Hello, this is a transcription",
          "tokens": [50364, 938, 428, 307, 275, 28347],
          "temperature": 0.0,
          "avg_logprob": -0.245,
          "compression_ratio": 1.235,
          "no_speech_prob": 0.012
        }
      ]
    }
    ```
Currently “verbose_json” response format doesn’t support no_speech_prob.

For models with hotword support, `hotwords` takes a comma-separated list of terms
(names, jargon) to bias recognition toward. Currently, this is supported by
`FunAudioLLM/Fun-ASR-Nano-2512-vllm` and `OpenMOSS-Team/MOSS-Transcribe-Diarize`.
Keep the list short and relevant to the audio: long or unrelated lists can
degrade accuracy.

For models with diarization support, `diarized_json` returns OpenAI-compatible
speaker segments. Currently, this is supported by
`OpenMOSS-Team/MOSS-Transcribe-Diarize`.

```json
{
  "task": "transcribe",
  "duration": 6.1,
  "text": "Hello. Hi, how are you?",
  "segments": [
    {
      "type": "transcript.text.segment",
      "id": "segment_0",
      "start": 0.0,
      "end": 2.8,
      "text": "Hello.",
      "speaker": "S01"
    }
  ],
  "usage": {"type": "duration", "seconds": 7}
}
```

### Extra Parameters

The following [sampling parameters](../../api/README.md#inference-parameters) are supported.

??? code

    ```python
    --8<-- "vllm/entrypoints/speech_to_text/transcription/protocol.py:transcription-sampling-params"
    ```

The following extra parameters are supported:

??? code

    ```python
    --8<-- "vllm/entrypoints/speech_to_text/transcription/protocol.py:transcription-extra-params"
    ```

## Translations API

Our Translation API is compatible with [OpenAI's Translations API](https://platform.openai.com/docs/api-reference/audio/createTranslation);
you can use the [official OpenAI Python client](https://github.com/openai/openai-python) to interact with it.
Whisper models can translate audio from one of the 55 non-English supported languages into English.
Please mind that the popular `openai/whisper-large-v3-turbo` model does not support translating.

!!! note
    To use the Translation API, please install with extra audio dependencies using `pip install vllm[audio]`.

Code example: [examples/speech_to_text/openai/openai_translation_client.py](../../../examples/speech_to_text/openai/openai_translation_client.py)

### Extra Parameters

The following [sampling parameters](../../api/README.md#inference-parameters) are supported.

```python
--8<-- "vllm/entrypoints/speech_to_text/translation/protocol.py:translation-sampling-params"
```

The following extra parameters are supported:

```python
--8<-- "vllm/entrypoints/speech_to_text/translation/protocol.py:translation-extra-params"
```

## Realtime API

The Realtime API provides WebSocket-based streaming audio transcription, allowing real-time speech-to-text as audio is being recorded.

!!! note
    To use the Realtime API, please install with extra audio dependencies using `uv pip install vllm[audio]`.

### Audio Format

Audio must be sent as base64-encoded PCM16 audio at 16kHz sample rate, mono channel.

### Protocol Overview

1. Client connects to `ws://host/v1/realtime`
2. Server sends `session.created` event
3. Client optionally sends `session.update` with model/params
4. Client sends `input_audio_buffer.commit` when ready
5. Client sends `input_audio_buffer.append` events with base64 PCM16 chunks
6. Server sends `transcription.delta` events with incremental text
7. Server sends `transcription.done` with final text + usage
8. Repeat from step 5 for next utterance
9. Optionally, client sends input_audio_buffer.commit with final=True
    to signal audio input is finished. Useful when streaming audio files

### Client → Server Events

| Event | Description |
| ----- | ----------- |
| `input_audio_buffer.append` | Send base64-encoded audio chunk: `{"type": "input_audio_buffer.append", "audio": "<base64>"}` |
| `input_audio_buffer.commit` | Trigger transcription processing or end: `{"type": "input_audio_buffer.commit", "final": bool}` |
| `session.update` | Configure session: `{"type": "session.update", "model": "model-name"}` |

### Server → Client Events

| Event | Description |
| ----- | ----------- |
| `session.created` | Connection established with session ID and timestamp |
| `transcription.delta` | Incremental transcription text: `{"type": "transcription.delta", "delta": "text"}` |
| `transcription.done` | Final transcription with usage stats |
| `error` | Error notification with message and optional code |

#### Example Clients

- [openai_realtime_client.py](https://github.com/vllm-project/vllm/tree/main/examples/speech_to_text/realtime/openai_realtime_client.py) - Upload and transcribe an audio file
- [openai_realtime_microphone_client.py](https://github.com/vllm-project/vllm/tree/main/examples/speech_to_text/realtime/openai_realtime_microphone_client.py) - Gradio demo for live microphone transcription

### Serving Voxtral realtime

`Voxtral-Mini-4B-Realtime-2602` uses a sliding window in every attention layer,
in the text decoder and in the audio encoder. The KV cache of a session stops
growing once both windows are full.

**Windows.** Both are set with `--hf-overrides`:

```bash
vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 --tokenizer-mode mistral \
  --hf-overrides '{"text_config":{"sliding_window":1024},"audio_config":{"sliding_window":256}}'
```

| Window | Key | Unit | Default |
| ------ | --- | ---- | ------- |
| Text (decoder) | `text_config.sliding_window` | token (80 ms of audio) | 8192 (~11 min) |
| Audio (encoder) | `audio_config.sliding_window` | frame (20 ms of audio) | 750 (15 s) |

Smaller windows use less KV cache per session, so more sessions fit in the same
memory.

!!! warning
    Keep the text window at 1024 or more. In our tests on long read speech, with a
    256 text window the model dropped punctuation and capitals for up to 3 min at a
    time, and over 8 streams in 90 min it output nothing over real speech for 20 min
    in total ([#47614](https://github.com/vllm-project/vllm/issues/47614)). At 1024,
    punctuation came back within 50 s, as with the default window, and 1.3 min was
    lost. Changing the audio window (256 or 750) made no difference.

**Session length.** By default, `--max-model-len` caps a session: one text token
per 80 ms of audio, so about 2 h 55 at 131072 (the default) and 11 min at 8192. At
the cap, the server ends the session with `transcription.done`. The client has to
reconnect, and the model starts again without context.

`--enable-unbounded-streaming-input` removes the cap. Before a session reaches
`--max-model-len`, its positions are shifted down and its cached keys are
re-rotated to match, so the attention scores do not change. The KV cache of the
session stays bounded by the windows.

- Requirements, checked at startup: every attention layer uses a sliding window
  and RoPE (plain, Llama 3 or YaRN), the KV cache is not quantized, prefix caching
  is off (`--no-enable-prefix-caching`), and the platform is CUDA. Nothing is
  specific to Voxtral: any streaming-input session on such a model works the same
  way, for example on Mistral-7B-v0.1.
- `--unbounded-streaming-input-margin-tokens` (default 4096): the shift happens this many tokens
  before `--max-model-len`. `max_model_len - margin - text window` must be at
  least one KV cache block. It is also roughly the number of tokens between two
  shifts: with 8192, 4096 and 1024, 3072 tokens, about 4 min.

**16 GiB GPU.** This command starts on an RTX 4090 Laptop (16 GiB):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 --tokenizer-mode mistral \
  --hf-overrides '{"text_config":{"sliding_window":1024},"audio_config":{"sliding_window":256}}' \
  --max-model-len 8192 --max-num-batched-tokens 512 --max-num-seqs 12 \
  --gpu-memory-utilization 0.82 --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --enable-unbounded-streaming-input
```

CUDA graphs are captured for batches of up to 2 x `--max-num-seqs` tokens, so set
`--max-num-seqs` to the number of sessions you need. `expandable_segments` reduces
memory fragmentation and leaves about 1 GiB more for the KV cache on this card.
