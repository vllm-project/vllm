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
- `response_format`: Format of the response ("json", "text") (optional)
- `temperature`: Sampling temperature between 0 and 1 (optional)

For the complete list of supported parameters including sampling parameters and vLLM extensions, see the [protocol definitions](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/protocol.py#L2182).

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

### Serving a sliding-window realtime model (Voxtral)

For a sliding-window realtime model such as `Voxtral-Mini-4B-Realtime-2602`, per-stream
KV memory is bounded by the attention window, not by `--max-model-len`. This gives two
operator levers.

**Concurrency: narrow the window.** The decoder's sliding window sets per-stream KV cost,
so narrowing it lets more streams fit in KV:

```bash
vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 --tokenizer-mode mistral \
  --hf-overrides '{"text_config":{"sliding_window":512},"audio_config":{"sliding_window":256}}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
```

Per-stream KV (in blocks) plateaus at, per KV group (decoder + audio encoder):

```text
blocks_per_stream = cdiv(min(sliding_window - 1 + max_num_batched_tokens, max_model_len), block_size) + 1
```

so the number of streams that fit in KV is `num_gpu_blocks / Σ blocks_per_stream`, and
`num_gpu_blocks` scales with VRAM. vLLM logs this at startup as
`Maximum concurrency for ... tokens per request: Nx`.

!!! note
    That `Nx` is a KV-admission ceiling (how many streams *fit* in KV), not measured
    throughput, and it is computed for requests that fill `max_model_len` -- a realtime
    session's KV plateaus at the sliding window instead, so for streaming the `Nx`
    understates stream capacity and the per-stream formula above is the number to size
    against. Real capacity is `min(KV at the window, compute, max_num_seqs)`; on small
    GPUs compute saturates first (real-time falls behind) while VRAM stays flat.
    Narrowing the window trades a little transcription fidelity for KV headroom, so
    measure on your audio.

!!! note
    Narrowing the window also shrinks the encoder cache budget, so a server configured
    this way rejects long **one-shot** clips on `/v1/audio/transcriptions` with a 400
    (`exceeds the pre-allocated encoder cache size`) that the default config would accept.
    Streaming sessions are unaffected (audio arrives in small chunks). Serve long offline
    clips from a default (non-overridden) server, or raise the window.

**Duration: `--max-model-len` and unbounded streaming.** For realtime, `--max-model-len`
also acts as a duration cap (about 1 text token per 80 ms of audio, so the 131072 default is ~2 h 55).
A session that reaches it is finished gracefully (`FINISHED_LENGTH_CAPPED`) so the client can
reconnect. To run **indefinitely at constant VRAM**, enable RoPE re-anchoring:

```bash
  --enable-realtime-unbounded --realtime-reanchor-margin-tokens 4096
```

- `--enable-realtime-unbounded` (default off): periodically re-anchors the RoPE position clock
  so the absolute counter never reaches `max_model_len`. Requires a **non-fp8** KV cache, a
  sliding-window decoder, **prefix caching off** (`--no-enable-prefix-caching`), CUDA, and
  plain RoPE with `rope_theta=1e6` (Voxtral/Ministral); rejected at startup otherwise.
- `--realtime-reanchor-margin-tokens` (default 4096): re-anchor this many tokens before the cap.
  Must be smaller than `max_model_len - sliding_window`.

**Fitting on a 16 GiB GPU.** The model plus a full `PIECEWISE` cudagraph capture leaves little
room for KV on 16 GiB. The capture set scales with `--max-num-seqs`, so the default (256)
captures large batch graphs you will never use and can OOM at startup. Set `--max-num-seqs` to
your real concurrency (for example 16). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` also
helps: it switches the allocator to growable segments, cuts fragmentation, and frees roughly
1 GiB on a 16 GiB card, with no effect on results. A working unbounded config on a 16 GiB
RTX 4090 Laptop, validated at 4 concurrent real-time streams (wall/audio 1.00, flat VRAM
~14.3 GiB, transcripts complete and identical across streams):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 \
  --tokenizer-mode mistral \
  --hf-overrides '{"text_config":{"sliding_window":512},"audio_config":{"sliding_window":256}}' \
  --max-model-len 4096 --no-enable-prefix-caching --max-num-seqs 16 \
  --enable-realtime-unbounded --realtime-reanchor-margin-tokens 2048 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
```

**Silence-rut mitigation: the blank-run penalty.** Sliding-window realtime models that
re-ingest their own output can fall into a self-sustained decoding rut: on marginal audio the
model starts emitting its blank/silence token every frame and keeps doing so over real speech,
for minutes, while the engine looks perfectly healthy (steady 1 token/frame, empty deltas).
Temperature does not help, the distribution genuinely collapses on the blank token. The
mitigation penalizes the blank token progressively once a request has emitted it more than K
consecutive times (`penalty = min(cap, alpha * (run - K))`):

```bash
  --realtime-blank-run-k 200
```

- `--realtime-blank-run-k` (default 0 = off): consecutive blank frames before the penalty
  engages. Set it well above the longest natural inter-sentence silence run of the model
  (Voxtral realtime: natural runs reach ~165 frames, i.e. 13 s at 12.5 tok/s; 200 is a
  validated value).
- `--realtime-blank-penalty` (default 0.5) and `--realtime-blank-penalty-cap` (default 7.0):
  slope and ceiling. The defaults are calibrated from measured logit margins on Voxtral
  realtime (the blank token wins by +3.5 to +6.6 logits inside a rut, by +8.5 to +17 on
  genuinely silent audio), so a saturated penalty breaks a rut but never flips real silence
  into invented text.

The blank token id is declared by the model class (`SupportsRealtime.realtime_blank_token_id`,
32 for Voxtral realtime); models that do not declare one ignore the flags. The run length is
accumulated in the model runner keyed by request id, because the realtime streaming path
recycles the engine request per audio chunk and clears its output-token list, which makes
history-based logits processors blind on this path.

Related sizing note: one realtime encoder chunk is ~350-400 tokens. A
`--max-num-batched-tokens` budget smaller than that splits every chunk across scheduler steps,
and on marginal audio the splitting alone can seed the rut, even for a single stream. Size the
budget so chunks are not split (at least 512 for a single stream, ideally N_streams x 400).
