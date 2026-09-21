---
name: check-api-compat
description: Determine the correct behavior of vLLM's OpenAI-compatible or Anthropic-compatible API by checking the official upstream spec, instead of guessing from vLLM's own code. Use for any bug, issue, PR, review, or question touching /v1/chat/completions, /v1/completions, /v1/responses, /v1/embeddings, /v1/messages, or their request/response models — including "is this a bug or intended", "what should this field default to", "does upstream allow this", and any change that adds, removes, or renames a field on those models. Do not use for vLLM-internal APIs (engine, scheduler, LLM class) or for endpoints with no upstream counterpart.
---

# Checking vLLM API compatibility against the upstream spec

vLLM's OpenAI and Anthropic server surfaces are *reimplementations* of someone
else's spec. vLLM's own code, tests, and docstrings are not evidence of correct
behavior — they are the thing under test. Establish what upstream says first,
then compare.

Never answer "what should this field do" from memory. Field names, types, and
especially defaults drift between spec versions.

## 1. Pin down the question

Name three things before looking anything up:

- **Which surface** — OpenAI-compatible or Anthropic-compatible. They are separate
  code paths in vLLM and separate specs.
- **Which endpoint**, by path.
- **Which field or behavior** — a request field, a response field, a streaming
  event shape, an error code, or a status code.

## 2. Read vLLM's implementation

In a vLLM checkout, the request/response models live beside each endpoint:

| surface | path |
| --- | --- |
| OpenAI chat | `vllm/entrypoints/openai/chat_completion/` |
| OpenAI completions | `vllm/entrypoints/openai/completion/` |
| OpenAI responses | `vllm/entrypoints/openai/responses/` |
| OpenAI embeddings | `vllm/entrypoints/pooling/embed/` |
| OpenAI shared base models, errors, usage | `vllm/entrypoints/serve/engine/protocol.py` |
| Anthropic messages | `vllm/entrypoints/anthropic/` |
| audio | `vllm/entrypoints/speech_to_text/*/protocol.py` |

Each directory holds `protocol.py` (the Pydantic models), `serving.py` (behavior),
and `api_router.py` (routing and status codes). Read the field declaration *and*
the code that consumes it — a field can be declared faithfully and then ignored,
which is still an incompatibility.

## 3. Fetch the upstream spec

Prefer a source that can actually be read as text, in this order.

### OpenAI

1. The OpenAPI spec — authoritative, machine-readable, and fetches cleanly:
   `https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml`
2. The installed `openai` Python SDK's typed params, which track the spec closely.
   Locate them with `python -c "import openai,os;print(os.path.dirname(openai.__file__))"`,
   then read e.g. `types/chat/completion_create_params.py`,
   `types/chat/chat_completion.py`, `types/chat/chat_completion_chunk.py`,
   `types/responses/`. Note the SDK version — an old SDK is an old spec.
3. The prose reference at `https://platform.openai.com/docs/api-reference/...`
   for semantics the types do not carry. It is script-heavy and often fetches
   poorly; treat a failed fetch as a fetch failure, not as absence of a field.

### Anthropic

1. `https://docs.claude.com/en/api/messages` and
   `https://docs.claude.com/en/api/messages-streaming` — these render as clean
   text and are the primary source for both fields and streaming event order.
2. The installed `anthropic` SDK's `types/message_create_params.py`,
   `types/message.py`, `types/raw_message_stream_event.py`.

Say which source and which version answered the question, so the finding can be
re-checked later.

## 4. Compare on all four axes

A field is compatible only when **name**, **type**, **default**, and **semantics**
all match. Most real bugs hide in the last two: a field that exists with the right
type but the wrong default, or one that is accepted and silently dropped.

Also check, when relevant:

- **Streaming** — chunk shape, the order of events, and the terminal event
  (`data: [DONE]` for OpenAI; `message_stop` for Anthropic).
- **Errors** — HTTP status *and* the error body's `type`/`code` strings.
- **Omitted vs null** — upstream often distinguishes an absent field from an
  explicit `null`. Pydantic defaults erase that distinction unless the model is
  written to preserve it.

## 5. Classify before calling it a bug

Three outcomes, and they are not the same:

- **Matches upstream** — no action.
- **vLLM extension** — a field upstream has no name for at all (`vllm_xargs`,
  `chat_template_kwargs`, sampling knobs like `top_k`). These are intentional and
  allowed. Do not report them as violations.
- **Incompatibility** — a field that *shares a name with upstream* but differs in
  type, default, or behavior, or an upstream field that is missing, ignored, or
  malformed. This is the reportable case.

An unsupported-but-declared field is its own subcase: decide whether vLLM rejects
it explicitly or accepts and ignores it, and report which, since silently ignoring
a documented field is the more damaging failure.

## 6. Report

State: the endpoint and field, what upstream specifies (with the source), what
vLLM does (with `file:line`), which of the three outcomes it is, and the smallest
fix. When behavior turns on a spec detail that is easy to disbelieve, quote the
upstream line rather than paraphrasing it.
