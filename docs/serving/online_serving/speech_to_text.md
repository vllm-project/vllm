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
memory. A session holds about 32 MiB + 0.125 MiB per text token + 0.25 MiB per
audio frame: 224 MiB with 1024 and 256, 1.2 GiB with the defaults.

!!! warning
    Keep the text window at 1024 or more. On long read speech, with a 256 text window
    the model sometimes output nothing over real speech for minutes
    ([#47614](https://github.com/vllm-project/vllm/issues/47614)). A 90 min run going
    from 2 to 8 streams lost 20 min of text this way. It did not happen at 1024.
    Stretches without punctuation or capitals reached 3 min at 256, 77 s at 1024 with
    or without shifts, and about 30 s from 2048 up. The audio window (256 or 750) made
    no difference.

**Session length.** By default, `--max-model-len` caps a session: one text token
per 80 ms of audio, so about 2 h 55 at 131072 (the default) and 11 min at 8192. At
the cap, the server ends the session with `transcription.done`. To go on, the client
sends a new `input_audio_buffer.commit` or reconnects, and the model starts again
without context.

`--enable-unbounded-streaming-input` (off by default) removes the cap. Before a
session reaches `--max-model-len`, its positions are shifted down and its cached
keys are re-rotated to match, so the attention scores stay the same apart from
rounding. The KV cache of the session stays bounded by the windows.

- Requirements, checked at startup: every attention layer uses a sliding window
  and RoPE (plain, Llama 3 or YaRN) over its whole key, the KV cache is not
  quantized, prefix caching is off (`--disable-prefix-caching`), and the platform
  is CUDA. KV connectors, KV offloading, encoder cache connectors, speculative
  decoding, pipeline parallelism and prefill context parallelism are refused. Any
  streaming-input session on such a model works the same way.
- `--max-model-len` must be at least the text window plus 2 KV cache blocks (256
  tokens on Voxtral), so `--max-model-len 8192` with the default text window of
  8192 is refused. It no longer limits the session, and a large value costs CPU:
  every token resends the session since the last shift. 8192 is enough with a 1024
  text window.
- `--unbounded-streaming-input-margin-tokens` (default: half of
  `max_model_len - text window`): the shift happens this many tokens before
  `--max-model-len`. The rest of the room is roughly the distance between two
  shifts: with 8192 and 1024, a shift every 3456 tokens, about 4.5 min.
- A shift only happens when a new input chunk arrives. A chunk can hold up to
  `max_model_len - text window - 1 block` input tokens. If it generates more
  tokens than the margin, the turn ends early with `finish_reason` `length` and
  the session continues with the next chunk. Voxtral sends and generates one token
  per chunk, so this only concerns other models.

**16 GiB GPU.** This command starts on an RTX 4090 Laptop (16 GiB):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 --tokenizer-mode mistral \
  --hf-overrides '{"text_config":{"sliding_window":1024},"audio_config":{"sliding_window":256}}' \
  --max-model-len 8192 --max-num-batched-tokens 512 --max-num-seqs 12 \
  --gpu-memory-utilization 0.82 --disable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --enable-unbounded-streaming-input
```

CUDA graphs are captured for batches of up to 2 x `--max-num-seqs` tokens, so keep
`--max-num-seqs` close to the number of sessions you need. `expandable_segments`
reduces memory fragmentation.

`--max-num-seqs` must also stay under what fits in the KV cache: `Available KV cache
memory` in the startup log divided by the KV cache of a session. On this card that
line says 3.4 to 4.2 GiB, so 15 to 19 sessions fit with windows of 1024 and 256.
Sessions above `--max-num-seqs` wait for a free slot. If more sessions run than fit,
all of them stall ([#60137](https://github.com/vllm-project/vllm/issues/60137)).
`Maximum concurrency` in the startup log does not limit sessions: it is the worst
case for a request of `--max-model-len` tokens with full batches in flight. The
startup check needs that worst case free, so a large `--max-num-batched-tokens` can
stop the server from starting.

Each session needs 12.5 tokens/s (`Avg generation throughput` in the log). If the
GPU does not deliver that, all sessions fall behind real time.
