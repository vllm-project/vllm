# Bounded streaming sessions

The experimental streaming session API accepts one image at a time and returns
one text reply per image while retaining a bounded history in KV cache. Enable
it with `--enable-streaming`; it is disabled by default. The ordinary chat and
completion APIs keep their existing behavior.

```bash
vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --enable-streaming \
    --enforce-eager \
    --no-async-scheduling \
    --api-server-count 1 \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image": 4}'
```

This API currently supports one active session per Python API server, one image
in flight per session, and dense causal models with one full-attention KV cache
group. Tensor parallelism is supported. Pipeline, data, and context parallelism,
speculative decoding, encoder-decoder models, multimodal pruning, KV/encoder
transfer connectors, and prompt embeddings are unsupported. Streaming retention
requires eager execution and synchronous scheduling. Unsupported configurations
are rejected instead of running with inconsistent cache state.

## Open, push, and close

Create a session with a task prompt and retention budgets:

```bash
curl http://localhost:8000/v1/streaming/sessions \
    -H 'Content-Type: application/json' \
    -d '{
        "system_prompt": "Describe the most recent image in one short sentence.",
        "retention": {
            "max_video_segments": 2,
            "max_session_tokens": 4000
        },
        "sampling": {"temperature": 0, "max_tokens": 32}
    }'
```

The response includes a `session_id`. Push an encoded JPEG or PNG as the raw
request body, replacing `SESSION_ID` with that value:

```bash
curl http://localhost:8000/v1/streaming/sessions/SESSION_ID/frame \
    -H 'Content-Type: image/jpeg' \
    --data-binary @frame.jpg
```

Each response contains `frame_index` (starting at zero), `text`, `finish_reason`,
`token_count`, `ttft_s`, and `latency_s`. Wait for the reply before sending the
next image. A concurrent push returns HTTP 409. Encoded images are limited to
8 MiB and decoded images to 4,194,304 pixels.

```bash
curl -X DELETE http://localhost:8000/v1/streaming/sessions/SESSION_ID
```

Closing releases the engine request. An interrupted frame request, a reply
timeout, or an engine error closes the session; create a new session after such
a failure. Sessions idle for 15 minutes are closed automatically. Creating a
second session while one is active returns HTTP 503; unknown sessions return
HTTP 404. `GET /v1/streaming/config` returns the served model name and default
retention and sampling settings.

To constrain every reply, set `sampling.structured_outputs`, for example
`{"choice": ["red", "blue"]}`. This uses the same constraint settings as the
[structured outputs API](structured_outputs.md), with a fresh grammar per image.

For a client that sends image files sequentially and always closes its session,
see [the streaming client example](../examples/features/streaming_sessions.md).

## Retention and positions

The first prompt provides a pinned attention sink. Later image turns and
generated replies can be evicted. During consolidation, the HTTP API replaces
the first image with the oldest retained image while keeping the original task
prompt. Old context is discarded, so this is a lossy policy: choose budgets and
evaluate output quality for the application's own workload.

| Setting | Default | Meaning |
| --- | --- | --- |
| `max_video_segments` | `30` | Target number of retained image turns after the pinned first turn. |
| `max_text_tokens` | `null` | Optional budget for unpinned text and previous replies. |
| `max_session_tokens` | `7000` | Positive total retained prompt token budget, including the pinned prompt. |
| `reprefill_threshold` | `0.7` | Fraction of the model's position range at which the retained context is replayed from fresh positions. Must be between zero and one. |

Keep the total prompt budget plus each reply at or below `--max-model-len`.
The pinned first prompt and each newest turn must fit within the prompt budget;
a session whose newest turn cannot fit is terminated. Retention evicts whole KV
blocks, so a segment may leave a partial block until enough adjacent old context
can be removed. Image-count and text-token targets are therefore best effort;
the total session token limit is enforced even when those targets cannot be met.

Eviction keeps surviving keys at their original RoPE positions. Before positions
reach the model's limit, consolidation releases that session's KV and re-prefills
the surviving context from fresh positions. Consolidation removes incomplete
turns and replies to discarded frames before replay. It introduces an occasional
longer frame latency and also handles histories that cannot be removed in place
without splitting an image across KV blocks. Such layouts may trigger replay
before the position threshold is reached. Retained image features are replayed
through the normal encoder cache lifecycle during consolidation; ordinary
encoder cache sizing is unchanged.

Bounded sessions bypass prefix sharing and send full processed image features on
each chunk so retained images remain available for replay. Other requests retain
the configured prefix and multimodal processor caches. Feature payloads retained
for replay also consume CPU memory, bounded by the retained context.

The engine mechanism also accepts text-only `StreamingInput` sessions by setting
`SamplingParams.output_kind=RequestOutputKind.DELTA` and
`SamplingParams.extra_args["streaming_retention"]` to
`vllm.v1.streaming.retention.StreamingRetentionParams(...)`. The HTTP frame API
described here accepts images; video capture, decoding, and sampling remain with
the client.

Use one output (`n=1`) without stop strings. Keep detokenization, sampled-output
logprob, and structured-output settings fixed across chunks of a session;
generation length and sampling distribution may change between chunks. Prompt
scoring (`prompt_logprobs` or `prompt_logprob_token_ids`) is unsupported.

For bounded sessions, `RequestOutput.prompt` and `prompt_token_ids` describe the
current input chunk. Decoded output and token deltas also belong to that chunk;
collect a complete history in the client if needed.

For image inputs supplied directly to `AsyncLLM`, render each chunk with
`engine.renderer.render_chat_async(..., skip_mm_cache=True)` and wrap the returned
engine input in `StreamingInput`. Bounded sessions require full processed image
payloads for consolidation; data-less cache hits or shared-memory address
descriptors are rejected. The HTTP API sets this option automatically.
Generic SDK image turns keep the original first image during consolidation;
only known image-only turns can replace it without losing per-frame annotations.
