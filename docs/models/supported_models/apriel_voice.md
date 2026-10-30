# Apriel Voice

[Apriel Voice 27B Preview](https://huggingface.co/ServiceNow-AI/Apriel-Voice-27B-preview)
combines a Whisper large-v3-turbo speech encoder and a learned projector with a
Gemma 4 26B-A4B backbone. It accepts audio, text, and images, and generates text
responses and tool calls. Speech synthesis and turn-taking are handled by the
application.

The registered architecture is `AprielVoiceModel`, with configuration model type
`aprielvoice`.

## Serving

The following configuration is adapted from the model card's standard vLLM
serving example for four GPUs. Install vLLM with audio dependencies and Apriel
Voice model support before running it.

```bash
VLLM_MAX_AUDIO_DECODE_DURATION_S=1800 vllm serve \
    ServiceNow-AI/Apriel-Voice-27B-preview \
    --trust-remote-code \
    --served-model-name Apriel-Voice-27B \
    --dtype bfloat16 \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 262144 \
    --max-num-batched-tokens 16384 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --speculative-config '{"model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":3}' \
    --default-chat-template-kwargs '{"enable_thinking":true}' \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4
```

`--trust-remote-code` loads the checkpoint's custom Hugging Face processor. The
model implementation is native to vLLM. Reasoning and tool calling use the
existing `gemma4` parsers and the checkpoint's chat template.

The speculative decoding option requires access to the Gemma 4 assistant
checkpoint and additional memory. For a minimal deployment, omit
`--speculative-config`. GPU memory requirements also depend on context length
and concurrency.

## Pre-tool speech

When the conversation contains audio and the request supplies tools, the
checkpoint's chat template enables pre-tool speech by default
(`enable_pretool_speech=true`). It prompts the model to produce a short
natural-language sentence before a tool call. The response is text; the
application handles speech synthesis.

To disable the automatic pre-tool speech instruction for all requests, replace
the serving command's chat-template defaults with:

```bash
--default-chat-template-kwargs '{"enable_thinking":true,"enable_pretool_speech":false}'
```

For an individual request, pass `chat_template_kwargs` in the OpenAI Python
client's `extra_body`:

```python
extra_body={"chat_template_kwargs": {"enable_pretool_speech": False}}
```

In a raw HTTP request, use `"chat_template_kwargs": {"enable_pretool_speech": false}`
in the JSON body. Request values override the server defaults. Set the value to
`true` to enable the instruction for a request when the server default is disabled.

## Multimodal input

For offline inference, include `<|audio|>` in the prompt and provide the waveform
and sample rate in `multi_modal_data["audio"]`. The processor converts audio to
mono at 16 kHz. Audio shorter than two seconds receives three seconds of leading
silence, matching the model's preprocessing. Longer recordings are split into
30-second chunks.

Use `<|image|>` in offline prompts with `multi_modal_data["image"]`. Image support
requires the checkpoint's Gemma 4 vision tower. Precomputed image embeddings are
unsupported.

Audio chunks share the context budget with text and image tokens. The serving
example admits audio up to 1,800 seconds. For one-hour input, the model card
suggests `VLLM_MAX_AUDIO_DECODE_DURATION_S=3600` and
`--max-num-batched-tokens 32768`, subject to context and memory limits. These
settings control admission and capacity, not transcription quality.

See [Multimodal Inputs](../../features/multimodal_inputs.md) for the input APIs.
