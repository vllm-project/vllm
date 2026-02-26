# Responses + Harmony Contribution Plan (Core GPT-OSS Focus)

## Purpose
This repo is intended as a focused working area for contributions in core vLLM, GPT-OSS models, and Harmony/Responses API features. The list below captures ten concrete tasks grouped by impact.

## 1) Concrete contribution list (priority order)

### 1. Complete refusal and non-text handling in Harmony input conversion
**Why**: Open-ended chat/tool payloads are currently flattened to text-only assumptions in some branches, which drops refusal and non-text content.
**Scope**:
- [vllm/entrypoints/openai/parser/harmony_utils.py](../vllm/entrypoints/openai/parser/harmony_utils.py)
- [vllm/entrypoints/openai/responses/harmony.py](../vllm/entrypoints/openai/responses/harmony.py)
**Tasks**:
- Add support for `refusal` fields and non-text content blocks in `_parse_chat_format_message`/`_parse_harmony_format_message` and equivalent response parsing paths.
- Ensure round-trips preserve typed content in `Message.from_*` and serializer paths.

### 2. Wire tool-call detection through chat-output parsing
**Why**: `parse_chat_output()` currently always reports no tool call in the tuple return value, even when tool calls are present.
**Scope**: [vllm/entrypoints/openai/parser/harmony_utils.py](../vllm/entrypoints/openai/parser/harmony_utils.py)
**Tasks**:
- Detect commentary/recipient-marked tool call messages while parsing and set `is_tool_call` correctly.
- Add unit coverage for models that emit tool calls and partial tool calls.

### 3. Preserve MCP/tool-call context instead of forcing `mcp_call` fallback shape
**Why**: When converting parser state to response output, MCP metadata is currently overwritten and error handling is missing.
**Scope**: [vllm/entrypoints/openai/parser/responses_parser.py](../vllm/entrypoints/openai/parser/responses_parser.py)
**Tasks**:
- Store and emit tool-server label data when converting `ResponseFunctionToolCallOutputItem` to `McpCall`.
- Add support for error outputs in function-call result conversion rather than silently dropping metadata.

### 4. Add output annotations/logprob propagation to fallback output builders
**Why**: Parser fallback paths currently emit placeholder values for `annotations`/`logprobs`, which loses observability and parity with non-harmony parsing.
**Scope**:
- [vllm/entrypoints/openai/parser/responses_parser.py](../vllm/entrypoints/openai/parser/responses_parser.py)
- [vllm/entrypoints/openai/responses/harmony.py](../vllm/entrypoints/openai/responses/harmony.py)
**Tasks**:
- Thread through available token-level logprob structures where requested.
- Keep annotation structure stable and non-null in final output items.

### 5. Include tool output messages in streaming harmony message history
**Why**: `StreamingHarmonyContext.append_tool_output()` still has a TODO to add tool output messages; without it, state reconstruction can omit tool-result content in streamed runs.
**Scope**: [vllm/entrypoints/openai/responses/context.py](../vllm/entrypoints/openai/responses/context.py)
**Tasks**:
- Append parsed tool-result `Message` objects into `_messages`.
- Validate that returned stream event order matches non-streaming output item conversion.

### 6. Clarify and fix previous-turn reconstruction in harmony continuation
**Why**: The slice-delete/reappend block for previous-response continuation appears intentionally redundant and can mis-handle turn boundaries.
**Scope**: [vllm/entrypoints/openai/responses/serving.py](../vllm/entrypoints/openai/responses/serving.py)
**Tasks**:
- Remove no-op behavior and implement explicit final-channel turn trimming policy.
- Add regression tests for multi-turn conversations where last message is `analysis`/`final`.

### 7. Add robust stateful response persistence and cleanup
**Why**: response/message stores are explicit in-memory hacks with known leak risks.
**Scope**:
- [vllm/entrypoints/openai/responses/serving.py](../vllm/entrypoints/openai/responses/serving.py)
- [vllm/entrypoints/openai/responses/context.py](../vllm/entrypoints/openai/responses/context.py)
**Tasks**:
- Track TTL/size limits or explicit pruning for `response_store`, `msg_store`, and `event_store`.
- Ensure state used for `previous_response_id` survives normal use while preventing unbounded growth.

### 8. Harden stateful tool execution contract for streaming
**Why**: Tool execution + streaming paths still have known quirks (disconnect handling and per-request session behavior) not fully codified.
**Scope**: [vllm/entrypoints/openai/responses/serving.py](../vllm/entrypoints/openai/responses/serving.py)
**Tasks**:
- Address `TODO` around disconnect handling in stream generator.
- Add/extend tests around `previous_response_id` when `background=True`, including stream replay (`starting_after`).

### 9. Improve parser compatibility for nested JSON tool arguments
**Why**: Nested JSON tool arguments are known to fail in one parser path and are xfailed in streaming mode.
**Scope**: [tests/entrypoints/openai/tool_parsers/test_hunyuan_a13b_tool_parser.py](../tests/entrypoints/openai/tool_parsers/test_hunyuan_a13b_tool_parser.py), [vllm/entrypoints/openai/tool_parsers/hunyuan_a13b_tool_parser.py](../vllm/entrypoints/openai/tool_parsers/hunyuan_a13b_tool_parser.py)
**Tasks**:
- Implement nested-object parsing in tool-parser extraction.
- Remove remaining skip/xfail behavior and add focused regression tests.

### 10. Fix GPT-OSS MoE Triton routing-weight path
**Why**: `apply_router_weight_on_input` is currently ignored in the custom MOE kernel path, which can alter behavior versus reference implementation.
**Scope**: [vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py](../vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py)
**Tasks**:
- Thread/consume `apply_router_weight_on_input` in `OAITritonExperts.apply`.
- Add kernel parity tests in [tests/kernels/moe/test_modular_oai_triton_moe.py](../tests/kernels/moe/test_modular_oai_triton_moe.py).

## 2) How Responses API statefulness currently works

### Core behavior
1. On request, `create_responses()` optionally loads prior response with `previous_response_id`.
2. `previous_response_id` is loaded from `self.response_store`; if missing, the request returns `invalid_request_error`.
3. For non-Harmony models, `construct_input_messages()` prepends previous chat messages (`msg_store`) and previous assistant outputs (`prev_response.output`) before appending current input.
4. For Harmony/GPT-OSS, `_construct_input_messages_with_harmony()` loads previous harmony messages from `msg_store[prev_response.id]` and appends parsed new input.
5. `msg_store`/`response_store` are only reliably populated when `store=True`.

### Practical caveats
- `msg_store` is in-memory only and has no eviction policy (`FIXME` comments mark this as a memory leak risk).
- `response_store` and `event_store` are also in-memory hacks with no retention policy.
- Request-level state is per `response_id`; streaming replay (`starting_after`) reads from `event_store` rather than recomputing.
- In Harmony, `context.messages` includes init state + generated messages; stateful reconstruction depends on correct `_construct_input_messages_with_harmony()` behavior.

## 3) TODOs affecting stateful correctness

### Direct TODOs in responses/harmony flow
- `vllm/entrypoints/openai/responses/harmony.py` and `.../parser/harmony_utils.py`: refusal/non-text support gaps.
- `.../parser/harmony_utils.py`: `parse_chat_output()` does not report `is_tool_call` yet.
- `.../responses/context.py`: add tool output messages in streaming harmony history.
- `.../responses/serving.py`: known streaming bug around tool session initialization/streaming path, plus disconnect TODO.
- `.../responses/serving.py`: previous-response continuation block is currently redundant and needs explicit final-turn handling.
- `.../responses/serving.py`: store/event maps include explicit FIXME about unbounded memory use.
- `.../protocol.py`: incomplete-details only covers max tokens; content_filter reason is still TODO.
- `.../protocol.py`: non-harmony previous_input message support is marked TODO.

### Parser/Output conversion TODOs that impact state replay
- `.../parser/responses_parser.py`: MCP server label and error-output conversion are incomplete.
- `.../parser/responses_parser.py` and `.../responses/harmony.py`: annotations/logprobs are placeholders in several conversion paths.
- `.../streaming_events.py`: TODOs around logprobs and web-search URL/ids in emitted events (impacting debugability and stream consumers).

## 4) Suggested near-term execution order
1. Address parser/harmony correctness items (1, 2, 4) together to stabilize message/output semantics.
2. Fix streaming tool output and state reconstruction (5, 6) to avoid multi-turn drift.
3. Add robust state persistence policy + regression tests (7, 8).
4. Improve tool-call robustness and kernel parity work (9, 10).
