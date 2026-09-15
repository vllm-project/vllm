# Responses API

vLLM's [Responses API](https://platform.openai.com/docs/api-reference/responses)
(`/v1/responses`) is the multi-turn, stateful counterpart to the Chat Completions
API and is the primary interface for the **gpt-oss / Harmony** model family.
This document shows how to configure the server and call every supported endpoint
with `curl`.

## Launch the Server

```bash
# Generic text-generation model
vllm serve Qwen/Qwen3-8B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --api-key "token-abc123"

# gpt-oss / Harmony model (detected automatically; always enables tool use)
vllm serve <gpt-oss-model-path> --api-key "token-abc123"

# Enable the in-memory response store (required for background mode, retrieval,
# and continuation via previous_response_id).  Responses are never evicted, so
# enable only when you control the request volume.
VLLM_ENABLE_RESPONSES_API_STORE=1 vllm serve <model> --api-key "token-abc123"
```

Set shell variables used throughout the examples:

```bash
BASE_URL="http://localhost:8000"
MODEL="Qwen/Qwen3-8B"        # replace with your model
API_KEY="token-abc123"        # omit -H "Authorization: …" if no key
```

---

## Basic Text Input (Synchronous)

```bash
curl "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{
    "model": "'"$MODEL"'",
    "input": "What is the capital of France?"
  }'
```

The response is a `ResponsesResponse` object:

```json
{
  "id": "resp_<uuid>",
  "object": "response",
  "created_at": 1700000000,
  "model": "Qwen/Qwen3-8B",
  "status": "completed",
  "output": [
    {
      "type": "message",
      "id": "msg_<uuid>",
      "role": "assistant",
      "status": "completed",
      "content": [{"type": "output_text", "text": "Paris.", "annotations": []}]
    }
  ],
  "usage": {
    "input_tokens": 12,
    "output_tokens": 4,
    "total_tokens": 16,
    "input_tokens_details": {
      "cached_tokens": 0,
      "input_tokens_per_turn": [12],
      "cached_tokens_per_turn": [0]
    },
    "output_tokens_details": {
      "reasoning_tokens": 0,
      "tool_output_tokens": 0,
      "output_tokens_per_turn": [4],
      "tool_output_tokens_per_turn": [0]
    }
  },
  ...
}
```

---

## Structured Message / Item Input

Instead of a plain string you may pass a list of input items.  Supported item
types are any `ResponseInputItemParam` variants from the OpenAI SDK, including
`EasyInputMessageParam` (role + string content), rich content-part arrays, and
prior output items for multi-turn continuation.

```bash
curl "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{
    "model": "'"$MODEL"'",
    "input": [
      {
        "type": "message",
        "role": "user",
        "content": "Explain quantum entanglement in one sentence."
      }
    ]
  }'
```

### Multimodal content parts

Multimodal inputs (images, audio) are accepted when the served model and the
vLLM configuration support them.  Pass content parts in the `content` array:

```bash
curl "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{
    "model": "'"$MODEL"'",
    "input": [
      {
        "type": "message",
        "role": "user",
        "content": [
          {"type": "input_text", "text": "Describe this image."},
          {"type": "input_image", "image_url": "https://example.com/photo.jpg",
           "detail": "auto"}
        ]
      }
    ]
  }'
```

> **Note**: Multimodal support depends on the model and the vLLM media-IO
> connectors configured at server startup.

---

## System Instructions

Use the `instructions` field to inject a system-level prompt:

```bash
curl "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{
    "model": "'"$MODEL"'",
    "instructions": "You are a concise assistant. Reply in one sentence.",
    "input": "What is 2 + 2?"
  }'
```

For gpt-oss / Harmony models, `instructions` is extracted and placed in the
Harmony preamble automatically.

---

## Generation Controls

All of the following fields are supported in `ResponsesRequest`:

| Field | Default | Notes |
|---|---|---|
| `temperature` | 1.0 | Sampling temperature |
| `top_p` | 1.0 | Nucleus sampling |
| `top_k` | 0 | Top-k (vLLM extension) |
| `max_output_tokens` | model `max_model_len` | Cap on generated tokens |
| `max_tool_calls` | `null` | Cap on tool-call turns |
| `presence_penalty` | 0.0 | Per-token presence penalty |
| `frequency_penalty` | 0.0 | Per-token frequency penalty |
| `repetition_penalty` | 1.0 | vLLM extension |
| `seed` | `null` | Reproducibility |
| `stop` | `[]` | Stop strings |
| `truncation` | `"disabled"` | `"auto"` truncates prompt to fit |
| `parallel_tool_calls` | `true` | Allow multiple tool calls per turn |
| `tool_choice` | `"auto"` | `"auto"`, `"none"`, `"required"`, or `{"type":"function","name":"…"}` |

```bash
curl "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{
    "model": "'"$MODEL"'",
    "input": "Write a haiku about autumn.",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_output_tokens": 60,
    "seed": 42
  }'
```

---

## SSE Streaming

Add `"stream": true`.  Use `curl -N` (no buffering) to receive events as they
arrive:

```bash
curl -N "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{
    "model": "'"$MODEL"'",
    "input": "Count to five.",
    "stream": true
  }'
```

The server emits newline-delimited `data:` lines.  A complete response
produces the following event sequence (sequence numbers increase monotonically):

```
data: {"type":"response.created","sequence_number":0,"response":{...,"status":"in_progress"}}
data: {"type":"response.in_progress","sequence_number":1,"response":{...}}
data: {"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{...}}
data: {"type":"response.content_part.added","sequence_number":3,...}
data: {"type":"response.output_text.delta","sequence_number":4,"delta":"One"}
data: {"type":"response.output_text.delta","sequence_number":5,"delta":", two"}
...
data: {"type":"response.output_text.done","sequence_number":N,"text":"One, two, three, four, five."}
data: {"type":"response.content_part.done","sequence_number":N+1,...}
data: {"type":"response.output_item.done","sequence_number":N+2,...}
data: {"type":"response.completed","sequence_number":N+3,"response":{...,"status":"completed"}}
data: [DONE]
```

---

## Function Tools

### Define tools and call the model

```bash
curl "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{
    "model": "'"$MODEL"'",
    "input": "What is the weather in Paris?",
    "tools": [
      {
        "type": "function",
        "name": "get_weather",
        "description": "Return current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "City name"}
          },
          "required": ["city"],
          "additionalProperties": false
        },
        "strict": true
      }
    ],
    "tool_choice": "auto",
    "store": true
  }'
```

The model may respond with a `function_call` output item:

```json
{
  "output": [
    {
      "type": "function_call",
      "id": "fc_<uuid>",
      "call_id": "call_<uuid>",
      "name": "get_weather",
      "arguments": "{\"city\":\"Paris\"}"
    }
  ],
  "status": "completed",
  ...
}
```

> **Parser requirement**: Function tool calling requires either `--enable-auto-tool-choice`
> with a compatible `--tool-call-parser`, or a gpt-oss / Harmony model.

### Send the tool result in a follow-up turn

Use the `id` of the response above as `previous_response_id` and include a
`function_call_output` input item containing the tool's result.

> **Prerequisite**: `VLLM_ENABLE_RESPONSES_API_STORE=1` must be set so that the
> server can look up the prior response.

```bash
PREV_RESP_ID="resp_<uuid from above>"

curl "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{
    "model": "'"$MODEL"'",
    "previous_response_id": "'"$PREV_RESP_ID"'",
    "input": [
      {
        "type": "function_call_output",
        "call_id": "call_<uuid>",
        "output": "{\"temperature\": 18, \"unit\": \"C\", \"condition\": \"Cloudy\"}"
      }
    ],
    "store": true
  }'
```

The model receives the full conversation history (original input → function
call → tool output) and generates a final answer.

### `tool_choice` variants

| Value | Behaviour |
|---|---|
| `"auto"` (default) | Model decides whether to call a tool |
| `"none"` | Tool calling disabled (coerced from `"auto"` when no tools provided) |
| `"required"` | Model must call at least one tool |
| `{"type":"function","name":"get_weather"}` | Model must call the named function |

---

## Built-in Tools

vLLM exposes three built-in tool types that the server must be configured to
support.  They are only available when:
1. The corresponding plugin is registered (`--tool-server …`), **and**
2. The request explicitly includes the matching tool entry.

### `web_search_preview` (browser)

```bash
# Server must be launched with a browser tool plugin:
# vllm serve <model> --tool-server <browser-server-url>

curl "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{
    "model": "'"$MODEL"'",
    "input": "Search for the latest news about vLLM.",
    "tools": [{"type": "web_search_preview"}]
  }'
```

### `code_interpreter` (Python)

```bash
# Server must be launched with a python tool plugin:
# vllm serve <model> --tool-server <python-server-url>

curl "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{
    "model": "'"$MODEL"'",
    "input": "Compute the factorial of 10 in Python.",
    "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}]
  }'
```

### Container tool

```bash
curl "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{
    "model": "'"$MODEL"'",
    "input": "Run a shell command.",
    "tools": [{"type": "container", "container": {"type": "auto"}}]
  }'
```

> **Restriction**: Built-in tools are filtered so that tools registered on the
> server are only available when the request explicitly includes the matching
> tool entry.  Requesting `web_search_preview` without a `browser` plugin
> registered on the server results in the tool being silently omitted.

---

## MCP Tools

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers can be
connected by including an `mcp` tool entry in `tools`.

### `allowed_tools` semantics

| `allowed_tools` value | Effect |
|---|---|
| Omitted / `null` | All tools on the MCP server are permitted |
| `"*"` (or list containing `"*"`) | Normalised to `null`; all tools permitted |
| `["tool1", "tool2"]` | Only the listed tools are available |

```bash
curl "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{
    "model": "'"$MODEL"'",
    "input": "List files in the repository.",
    "tools": [
      {
        "type": "mcp",
        "server_label": "repo_browser",
        "server_url": "https://mcp.example.com/repo",
        "allowed_tools": ["list_files", "read_file"]
      }
    ]
  }'
```

Multiple MCP servers may be listed in the same `tools` array.

---

## Reasoning Output

Models with a reasoning parser (e.g., Qwen3 with `--reasoning-parser qwen3`) or
gpt-oss / Harmony models produce `reasoning` output items containing the model's
internal thinking.

```bash
curl "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{
    "model": "'"$MODEL"'",
    "input": "Prove that the square root of 2 is irrational.",
    "reasoning": {"effort": "high"},
    "include_reasoning": true
  }'
```

Set `"include_reasoning": false` to keep reasoning tokens generating on the
server but suppress them in the response (reduces bandwidth).

> **Requirements**:
> - For non-Harmony models: `--reasoning-parser <name>` must be set at launch.
> - For gpt-oss / Harmony: reasoning is automatic.
> - `include_reasoning` is a vLLM extension; it defaults to `true`.

---

## Output Logprobs

Pass `"include": ["message.output_text.logprobs"]` and `"top_logprobs": N` (0–20):

```bash
curl "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{
    "model": "'"$MODEL"'",
    "input": "Name a color.",
    "include": ["message.output_text.logprobs"],
    "top_logprobs": 5
  }'
```

Each `output_text` content part will include a `logprobs` array.

> **Restriction**: Logprobs are **not** supported for gpt-oss / Harmony models
> (`model_type == "gpt_oss"`).

---

## Multi-turn Continuation with `previous_response_id`

> **Prerequisite**: `VLLM_ENABLE_RESPONSES_API_STORE=1` must be set so that the
> server stores responses and can look up the prior one.

```bash
# Turn 1
RESP=$(curl -s "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{
    "model": "'"$MODEL"'",
    "input": "My name is Alice.",
    "store": true
  }')

PREV_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

# Turn 2 — model remembers the prior conversation
curl "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{
    "model": "'"$MODEL"'",
    "previous_response_id": "'"$PREV_ID"'",
    "input": "What is my name?",
    "store": true
  }'
```

---

## `previous_input_messages`

An alternative to `previous_response_id` for supplying prior conversation
history in-band (as Harmony-formatted messages).  **Mutually exclusive** with
`previous_response_id`.

```bash
curl "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{
    "model": "'"$MODEL"'",
    "previous_input_messages": [
      {"role": "user", "content": [{"type": "text", "text": "My name is Bob."}]},
      {"role": "assistant", "content": [{"type": "text", "text": "Hello Bob!"}]}
    ],
    "input": "What is my name?"
  }'
```

---

## `store` and `background`

### Store flag

By default, vLLM does **not** store responses.  Even if a client sends
`"store": true`, the store is silently ignored unless
`VLLM_ENABLE_RESPONSES_API_STORE=1` is set on the server.

### Background mode

Background mode lets the client submit a request and retrieve the result
later.  `background` requires `store: true` and the store to be enabled.

```bash
# Submit in background
RESP=$(curl -s "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{
    "model": "'"$MODEL"'",
    "input": "Summarise the history of computing.",
    "store": true,
    "background": true
  }')

RESP_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "Submitted: $RESP_ID (status=$(echo $RESP | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])"))"
```

---

## Retrieve a Response

### Synchronous retrieval

```bash
curl "$BASE_URL/v1/responses/$RESP_ID" \
  -H "Authorization: ******"
```

Returns the stored `ResponsesResponse`, e.g. `{"status": "completed", ...}`.

### Streaming retrieval with `starting_after`

Replay the SSE event stream starting from a specific event index:

```bash
curl -N "$BASE_URL/v1/responses/$RESP_ID?stream=true&starting_after=0" \
  -H "Authorization: ******"
```

Events are emitted from the event store starting at the given sequence number.
The stream completes when the `response.completed` event has been emitted.

### Error: response not found

If `$RESP_ID` does not exist in the store, the server returns HTTP 404:

```json
{"object": "error", "message": "Response with id 'resp_xxx' not found.", "type": "invalid_request_error"}
```

---

## Cancel a Background Response

```bash
curl -X POST "$BASE_URL/v1/responses/$RESP_ID/cancel" \
  -H "Authorization: ******"
```

Only responses in `"queued"` or `"in_progress"` status can be cancelled.
Attempting to cancel a synchronous (non-background) response returns an error:

```json
{"object": "error", "message": "Cannot cancel a synchronous response.", "type": "invalid_request_error"}
```

---

## Background Streaming

You can also stream a background response as it generates:

```bash
# Submit background + streaming
RESP=$(curl -s "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: ******" \
  -d '{
    "model": "'"$MODEL"'",
    "input": "Write a poem about the ocean.",
    "store": true,
    "background": true,
    "stream": true
  }')
```

The server immediately returns a queued response.  Use the streaming retrieval
endpoint (with `stream=true`) to follow the event stream.

---

## Response Usage Fields

Every completed response includes a `usage` object:

```json
{
  "usage": {
    "input_tokens": 42,
    "output_tokens": 18,
    "total_tokens": 60,
    "input_tokens_details": {
      "cached_tokens": 8,
      "input_tokens_per_turn": [24, 18],
      "cached_tokens_per_turn": [8, 0]
    },
    "output_tokens_details": {
      "reasoning_tokens": 10,
      "tool_output_tokens": 5,
      "output_tokens_per_turn": [10, 8],
      "tool_output_tokens_per_turn": [5, 0]
    }
  }
}
```

`*_per_turn` arrays contain one entry per generation turn (including tool-call
turns).

---

## Validation Failures and Limitations

| Scenario | Error |
|---|---|
| `background: true` without `store: true` | 400 `invalid_request_error` |
| `background: true` but store not enabled on server | 400 `invalid_request_error` |
| Both `previous_response_id` and `previous_input_messages` set | 400 `invalid_request_error` |
| `previous_response_id` references an unknown ID | 404 `invalid_request_error` |
| Prompt length ≥ `max_model_len` | 400 `invalid_request_error` |
| Logprobs requested on a gpt-oss / Harmony model | 400 `invalid_request_error` |
| `tool_choice: "required"` with no tools | 422 Pydantic validation error |
| Named `tool_choice` that is not in `tools` | 422 Pydantic validation error |
| `prompt` field set (not yet implemented) | 422 Pydantic validation error |
| Cancelling a non-background / non-cancellable response | 400 `invalid_request_error` |
| `store: true` without `VLLM_ENABLE_RESPONSES_API_STORE=1` | Store silently disabled; request proceeds |

---

## vLLM-Specific Extra Parameters

The following fields extend the standard OpenAI `ResponsesRequest`:

```python
--8<-- "vllm/entrypoints/openai/responses/protocol.py:responses-extra-params"
```

The following vLLM-specific fields are also returned in `ResponsesResponse`:

```python
--8<-- "vllm/entrypoints/openai/responses/protocol.py:responses-response-extra-params"
```

---

## Testing

### Layout

Integration and unit tests for the Responses API live under
`tests/entrypoints/openai/responses/`.  Shared fixtures and helper utilities
are defined in `conftest.py`; each test file covers a distinct concern:

| File | Covers |
|---|---|
| `test_basic.py` | Basic text input, instructions, multi-turn chat, structured input |
| `test_simple.py` | Streaming consistency, logprobs, reasoning tokens, max-tokens |
| `test_errors.py` | Validation failures and error responses |
| `test_stateful.py` | Store lifecycle, `previous_response_id`, background mode |
| `test_function_call.py` | Function tool invocation and follow-up turns |
| `test_structured_output.py` | JSON schema / structured-output requests |
| `test_sampling_params.py` | Extra sampling parameters |
| `test_streaming_events.py` | `SimpleStreamingEventProcessor` / `split_delta` unit tests |
| `test_serving_responses.py` | `OpenAIServingResponses` internal unit tests |
| `test_serving_responses_extra.py` | Validation failures, store lifecycle, cancel/retrieve routes, usage accounting, streaming sequence numbers |
| `test_harmony.py` | Harmony (gpt-oss) model end-to-end |
| `test_harmony_utils.py` | `harmony_to_response_output` / `response_previous_input_to_harmony` unit tests |
| `test_mcp_tools.py` | MCP tool integration |
| `test_protocol.py` | Protocol / schema correctness |

### Running the tests

```bash
# 1. Create and activate the virtual environment
uv venv --python 3.12
source .venv/bin/activate

# 2. Install vLLM (Python-only changes only need the precompiled wheel)
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# 3. Install test dependencies
uv pip install -r requirements/test/cuda.in

# 4. Run the full suite
.venv/bin/python -m pytest tests/entrypoints/openai/responses/ -v

# 5. Run a single file
.venv/bin/python -m pytest tests/entrypoints/openai/responses/test_basic.py -v
```

Most tests that exercise the server start a `RemoteOpenAIServer` and therefore
require a GPU.  The pure unit tests (e.g. `test_harmony_utils.py`,
`test_streaming_events.py`, `test_serving_responses_extra.py`) run without a
GPU and finish in seconds.

### Fixtures provided by `conftest.py`

| Fixture | Scope | Description |
|---|---|---|
| `pairs_of_event_types` | function | `dict[str, str]` mapping every `done` event type to its matching `start` event type (e.g. `"response.completed" → "response.created"`). Pass to `validate_streaming_event_stack`. |
| `default_server_args` | module | Default `vllm serve` flags for Qwen3 tests: `--max-model-len 18192`, `--enforce-eager`, `--enable-auto-tool-choice`, xgrammar backend, `hermes` tool-call parser, `qwen3` reasoning parser. |
| `server_with_store` | module | `RemoteOpenAIServer` started with `default_server_args`, `VLLM_ENABLE_RESPONSES_API_STORE=1`, and `VLLM_SERVER_DEV_MODE=1`. |
| `client` | function | Async OpenAI client connected to `server_with_store`. |

Individual test files may define their own `server` and `client` fixtures that
override the module-scoped defaults (e.g. to use a different model or omit the
store).

### Helper utilities

**`validate_streaming_event_stack(events, pairs_of_event_types)`**

Validates three aspects of a collected SSE event list in one call:

1. *Pairing* — every `start` event is closed by its matching `done` event
   (stack-based; derived automatically from `pairs_of_event_types`).
2. *Ordering* — `response.created` is first, `response.completed` is last,
   `response.in_progress` is the second event if present, and there is exactly
   one `created` / one `completed`.
3. *Field consistency* — `item_id`, `output_index`, and `content_index` are
   consistent across all events within each output-item lifecycle.

```python
events = []
async for event in stream:
    events.append(event)
validate_streaming_event_stack(events, pairs_of_event_types)
```

**`retry_for_tool_call(client, *, model, expected_tool_type, max_retries=3, **create_kwargs)`**

Calls `client.responses.create` up to `max_retries` times and returns the
first response that contains an output item of `expected_tool_type`.  Returns
the last response if none match (so assertions still fire with a clear
diagnostic).

```python
response = await retry_for_tool_call(
    client,
    model=MODEL,
    expected_tool_type="function_call",
    input="What is the weather in Paris?",
    tools=[...],
)
assert has_output_type(response, "function_call")
```

**`retry_streaming_for(client, *, model, validate_events, max_retries=3, **create_kwargs)`**

Calls `client.responses.create(stream=True)` up to `max_retries` times,
returning the first event list for which `validate_events(events)` returns
`True`.

```python
events = await retry_streaming_for(
    client,
    model=MODEL,
    validate_events=lambda evts: events_contain_type(evts, "function_call"),
    input="What is the weather in Paris?",
    tools=[...],
)
```

**`has_output_type(response, type_name)`**

Returns `True` if `response.output` contains at least one item whose `.type`
equals `type_name`.

**`events_contain_type(events, type_substring)`**

Returns `True` if any event's `.type` contains `type_substring`.

**`log_response_diagnostics(response, *, label="Response Diagnostics")`**

Logs reasoning text, tool-call attempts, MCP items, and output types at
`INFO` level using Python's standard `logging` module.  Pass `--log-cli-level
INFO` to pytest (or run with `pytest -s`) to see the output.  Returns the
extracted `dict` so callers can make additional assertions.

```python
diagnostics = log_response_diagnostics(response, label="tool call test")
assert diagnostics["model_attempted_tool_calls"]
```

### Writing a new test

1. **Create a new file** (or add to an existing one) under
   `tests/entrypoints/openai/responses/`.
2. **Reuse the shared fixtures** — import from `conftest` or let pytest
   inject them automatically:

   ```python
   import pytest
   from .conftest import (
       validate_streaming_event_stack,
       has_output_type,
       log_response_diagnostics,
   )

   @pytest.mark.asyncio
   async def test_my_feature(
       client,
       pairs_of_event_types,
   ):
       response = await client.responses.create(
           model="Qwen/Qwen3-1.7B",
           input="Hello!",
       )
       assert response.status == "completed"
       assert has_output_type(response, "message")
   ```

3. **For streaming tests**, collect events and call
   `validate_streaming_event_stack`:

   ```python
   @pytest.mark.asyncio
   async def test_streaming(client, pairs_of_event_types):
       stream = await client.responses.create(
           model="Qwen/Qwen3-1.7B",
           input="Count to three.",
           stream=True,
       )
       events = [e async for e in stream]
       validate_streaming_event_stack(events, pairs_of_event_types)
   ```

4. **For tests that need the store**, use `server_with_store` / `client`
   directly (both are in scope via `conftest.py`) — no extra setup needed.

5. **For pure unit tests** that mock the engine, follow the pattern in
   `test_serving_responses_extra.py`: create a `_make_serving()` helper that
   returns an `OpenAIServingResponses` backed by `MagicMock` objects so that
   no GPU is required.

---

## No Full OpenAI Parity

The current implementation does **not** support:

- `prompt` template field (raises a validation error).
- File/image uploads via the Files API.
- Remote function execution (tools run locally via MCP or the built-in tool server).
- Response deletion.
- Response listing (`GET /v1/responses`).
- Persistent storage — the in-memory store (`VLLM_ENABLE_RESPONSES_API_STORE=1`)
  is non-evicting and is lost when the server restarts.
- Full content-filter `incomplete_details` reason (only `max_output_tokens` is
  currently populated).
