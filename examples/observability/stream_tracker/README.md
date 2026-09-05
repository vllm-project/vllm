# Stream tracker middleware

A self-contained ASGI middleware for the OpenAI-compatible server that answers
`GET /streams` with a live JSON snapshot of every in-flight completion request:
id, state, age, prompt preview, token count so far, and current tokens/s.

`/metrics` and `/load` are aggregate by design (per-request metric labels are a
cardinality anti-pattern), and OpenTelemetry tracing reconstructs requests
after the fact. This example fills the remaining gap — "what is every stream
doing *right now*" — as opt-in user code loaded through the server's
`--middleware` flag, with no server changes and no new metric series.

The middleware observes request and response frames without buffering or
modifying them. Response bytes are forwarded verbatim, and every tracking step
is wrapped so an internal failure can never affect the request being served.
Token counts are derived from SSE events (approximately one per token for
streamed completions); non-streaming requests are tracked for lifecycle only.

## Run it

1. Start the server with the middleware on the import path:

   ```bash
   cd examples/observability/stream_tracker
   PYTHONPATH=. vllm serve Qwen/Qwen2.5-1.5B-Instruct \
       --middleware stream_tracker.StreamTracker
   ```

2. Send a few streaming requests:

   ```bash
   for i in 1 2 3 4; do
     curl -s -N http://localhost:8000/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d "{\"model\": \"Qwen/Qwen2.5-1.5B-Instruct\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Story $i\"}],
            \"stream\": true, \"max_tokens\": 200}" > /dev/null &
   done
   ```

3. Watch the snapshot while they run:

   ```bash
   curl -s http://localhost:8000/streams | python3 -m json.tool
   ```

   ```json
   {
       "active": [
           {
               "id": "4",
               "state": "streaming",
               "model": "Qwen/Qwen2.5-1.5B-Instruct",
               "preview": "Story 4",
               "preview_truncated": false,
               "stream": true,
               "tokens": 63,
               "tok_s": 42.1,
               "idle_s": 0.02,
               "age_s": 1.55,
               "status": 200
           }
       ],
       "recent": []
   }
   ```

Finished requests move to `recent` (a ring of the last 50) with an `e2e_s`
duration and whole-stream average `tok_s`. Request states are `pending`
(accepted, no first token yet), `streaming`, and the terminal `done`,
`error` (HTTP >= 400 or an application exception), or `aborted` (client
disconnected before completion).

## Notes

- The middleware runs outside the server's CORS layer, so browser dashboards
  should fetch `/streams` through a same-origin proxy rather than directly.
- Prompt previews expose request content on an unauthenticated route. Keep
  the port private, or delete the preview lines in `_parse_body` if you serve
  untrusted networks.
- Tracking state lives in the API server process. With multiple API server
  processes, each answers `/streams` for the requests it handled.
