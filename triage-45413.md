# Triage: Issues & PRs Related to PR #45413

PR #45413 introduces a streaming parser engine with a declarative state
machine for reasoning/content/tool-call transitions, plus a new Qwen3
parser.  This file tracks which open issues and PRs it fixes, partially
addresses, or supersedes.

---

## Tier 1: Likely Fixed — Reproduce and link

These map directly to parser engine fixes.  Reproduce each, then link to
the PR and (if confirmed) close.

- [ ] [#45440](https://github.com/vllm-project/vllm/issues/45440) — Second tool call dropped when `</tool_call>` and `<tool_call>` share a delta
  - **Engine fix:** Multi-token delta splitting + TOOL_BETWEEN→TOOL_START transition

- [ ] [#45256](https://github.com/vllm-project/vllm/issues/45256) — Incomplete JSON args with stream_interval > 1 (missing closing `}`)
  - **Engine fix:** Brace-depth tracking + sequential token processing

- [ ] [#43221](https://github.com/vllm-project/vllm/issues/43221) — `</think>` + `<tool_call>` in same MTP delta truncates reasoning
  - **Engine fix:** TokenIDScanner splits them; state machine emits REASONING_END before TOOL_CALL_START

- [ ] [#42021](https://github.com/vllm-project/vllm/issues/42021) — Qwen3.5 tool calls returned as raw XML in content with enable_thinking + spec decode
  - **Engine fix:** REASONING→TOOL transition emits implicit REASONING_END

- [ ] [#39056](https://github.com/vllm-project/vllm/issues/39056) — Tool calls inside `<think>` block are lost
  - **Engine fix:** `<tool_call>` from REASONING state emits REASONING_END + TOOL_CALL_START

- [ ] [#39771](https://github.com/vllm-project/vllm/issues/39771) — ValueError crash on malformed `<parameter=...>`
  - **Engine fix:** Regex-based arg_converter; no `.index()` calls

- [ ] [#38789](https://github.com/vllm-project/vllm/issues/38789) — `</think>` leaks into content with stop sequences
  - **Engine fix:** Token ID-based detection + detokenizer hold-back recovery

- [ ] [#35266](https://github.com/vllm-project/vllm/issues/35266) — Missing opening `{` in streaming tool call args
  - **Engine fix:** arg_converter builds JSON from `<parameter=>` elements; brace always emitted

- [ ] [#35221](https://github.com/vllm-project/vllm/issues/35221) — Reasoning-only output misclassified as content
  - **Engine fix:** Initial state = REASONING; all text before `</think>` is reasoning

- [ ] [#34684](https://github.com/vllm-project/vllm/issues/34684) — Entire reasoning + `</think>` leaks into content
  - **Engine fix:** Token ID detection + terminal consumed by lexer, never emitted

- [ ] [#34322](https://github.com/vllm-project/vllm/issues/34322) — IndexError in streaming tool calls
  - **Engine fix:** Slots pre-allocated on TOOL_CALL_START; no OOB access

---

## Tier 2: Partially Addressed — Mention in PR

- [ ] [#40875](https://github.com/vllm-project/vllm/issues/40875) — ngram spec decode corrupts tool output
  - Parser-level corruption fixed; upstream generation corruption (rejection sampling) is separate

- [ ] [#43713](https://github.com/vllm-project/vllm/issues/43713) — qwen3_xml emits invalid JSON for multiple `<function=>` blocks
  - Engine handles consecutive tools, but only if qwen3_xml uses the engine parser

- [ ] [#39584](https://github.com/vllm-project/vllm/issues/39584) — "Multiple tool calls in one delta" assertion crash (Responses API)
  - Engine splits deltas (reducing trigger), but assertion in Responses API serving layer is separate

- [ ] [#36116](https://github.com/vllm-project/vllm/issues/36116) — Qwen3-Coder pseudo-streaming (args buffered, not streamed)
  - Qwen3 side fixed; MiniMax-M2 side not covered

---

## Tier 3: Not Related — Skip

| # | Why |
|---|-----|
| [#44676](https://github.com/vllm-project/vllm/issues/44676) | Sampling-layer: ThinkingBudgetStateHolder injects tokens mid-generation |
| [#39697](https://github.com/vllm-project/vllm/issues/39697) | Same sampling-layer budget bug as #44676 |
| [#39573](https://github.com/vllm-project/vllm/issues/39573) | Thinking token budget enforcement in V1 scheduler |
| [#34650](https://github.com/vllm-project/vllm/issues/34650) | Structured output manager timing bug in V1 scheduler |
| [#31501](https://github.com/vllm-project/vllm/issues/31501) | Stream-interval buffering in output_processor.py (MiniMax/gpt-oss) |
| [#43338](https://github.com/vllm-project/vllm/issues/43338) | gpt-oss parser multi-token boundary (not ported) |
| [#36730](https://github.com/vllm-project/vllm/issues/36730) | API field naming (`reasoning` vs `reasoning_content`) |
| [#40528](https://github.com/vllm-project/vllm/issues/40528) | Serving layer assert crash (GLM/DeepSeek) |
| [#36435](https://github.com/vllm-project/vllm/issues/36435) | Responses API event emission layer bug |
| [#42210](https://github.com/vllm-project/vllm/issues/42210) | Stop-string/parser interaction (Llama-3.1) |
| [#38603](https://github.com/vllm-project/vllm/issues/38603) | Final chunk serialization emits empty tool_calls fields |
| [#44104](https://github.com/vllm-project/vllm/issues/44104) | Empty `tool_calls: []` in response serialization |
| [#37167](https://github.com/vllm-project/vllm/issues/37167) | Responses API message-conversion layer |
| [#40192](https://github.com/vllm-project/vllm/issues/40192) | Server hangs (not parser related) |
| [#38885](https://github.com/vllm-project/vllm/issues/38885) | Chat template "None" vs "null" (needs arg_converter fix) |

---

## Cross-Model Issues (engine framework helps, model not yet ported)

These motivate porting additional models to the engine.

- [ ] [#43933](https://github.com/vllm-project/vllm/issues/43933) — DeepSeek: reasoning + content in same chunk
- [ ] [#40911](https://github.com/vllm-project/vllm/issues/40911) — Gemma4: tool call leaks into content
- [ ] [#39043](https://github.com/vllm-project/vllm/issues/39043) — Gemma4 + Claude Code tool calling broken
- [ ] [#42400](https://github.com/vllm-project/vllm/issues/42400) — GLM-5.1 + Claude Code tool parsing fails

---

## Superseded PRs (12)

These fix Qwen3-specific bugs that the parser engine handles by design.
Can be closed once #45413 merges.

- [ ] [#45365](https://github.com/vllm-project/vllm/pull/45365) — Close Qwen3 Coder args on same delta
- [ ] [#45335](https://github.com/vllm-project/vllm/pull/45335) — Emit closing brace when param + function end same delta
- [ ] [#45299](https://github.com/vllm-project/vllm/pull/45299) — Qwen3 reasoning-to-content in short responses
- [ ] [#45164](https://github.com/vllm-project/vllm/pull/45164) — Qwen3 `</think>` split across batches
- [ ] [#44141](https://github.com/vllm-project/vllm/pull/44141) — Append (not prepend) promoted tool calls
- [ ] [#40861](https://github.com/vllm-project/vllm/pull/40861) — Qwen3 XML/Coder streaming regressions (4600+ lines)
- [ ] [#40783](https://github.com/vllm-project/vllm/pull/40783) — Qwen3 reasoning parser multi-bug fix
- [ ] [#39055](https://github.com/vllm-project/vllm/pull/39055) — Tool calls inside `<think>` non-streaming
- [ ] [#38864](https://github.com/vllm-project/vllm/pull/38864) — Qwen3 `</think>` leak with stop sequences
- [ ] [#34495](https://github.com/vllm-project/vllm/pull/34495) — IndexError in Qwen3Coder streaming
- [ ] [#33866](https://github.com/vllm-project/vllm/pull/33866) — Qwen3 prompt prefix `<think>` format
- [ ] [#32538](https://github.com/vllm-project/vllm/pull/32538) — Whitespace between reasoning and tool_call

---

## Overlapping PRs (3)

Qwen3 parts superseded; other model/infra parts still needed.

- [ ] [#40348](https://github.com/vllm-project/vllm/pull/40348) — Strip grouped think markers (base parser + KimiK2 parts still needed)
- [ ] [#39598](https://github.com/vllm-project/vllm/pull/39598) — MTP empty fields (Pydantic null serialization fix in serving.py still needed)
- [ ] [#39044](https://github.com/vllm-project/vllm/pull/39044) — End-token desync (DeepSeek/Ernie/Step3p5 parts still needed)

---

## Complementary PRs (2)

No overlap; address different concerns.

- [#45389](https://github.com/vllm-project/vllm/pull/45389) — Brace handling in required tool streaming (infrastructure)
- [#38996](https://github.com/vllm-project/vllm/pull/38996) — Qwen3.5 "None"→null normalization (may need in engine arg_converter too)

---

## Design / Feature Issues

- [ ] [#44873](https://github.com/vllm-project/vllm/issues/44873) — Streaming parser engine feature request — **directly addressed by #45413**
- [ ] [#43267](https://github.com/vllm-project/vllm/issues/43267) — Streaming tool_calls arguments — **directly addressed** (core engine feature)
- [ ] [#32713](https://github.com/vllm-project/vllm/issues/32713) — RFC: Unified Parser class — **complementary** (engine is the streaming backend)
- [ ] [#34857](https://github.com/vllm-project/vllm/issues/34857) — H1 2026 tool calling roadmap — **complementary** (engine is one slice)
