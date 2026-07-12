# Issue #48207 — request validation for tool calling

Issue #48207 is filed as a broad architectural proposal ("add a hard-interception
request validation/normalization module upstream of vLLM"). Most of what it asks for
already exists, so this change fixes the one concrete hole in the existing validation
rather than adding a new module. See "Scope" below.

## 1. Root cause

vLLM already validates tool-calling requests in two places:

- `ChatCompletionRequest.check_tool_usage` (`vllm/entrypoints/openai/chat_completion/protocol.py`)
  rejects `tool_choice` without `tools`, a named `tool_choice` that matches no tool, an
  empty `tools` array, etc.
- `OnlineRenderer.render_chat` (`vllm/renderers/online_renderer.py:117`) rejects
  `tool_choice="required"` and named tool choice when the server has no tool-call parser,
  because both depend on the tool parser to work at all.

That second guard had an exemption that is not sound:

```python
tool_parsing_unavailable = (
    tool_parser is None
    and not is_mistral_tokenizer(tokenizer)
    and not self.use_harmony          # <-- gpt-oss skipped the check entirely
)
```

`use_harmony` is `model_config.hf_config.model_type == "gpt_oss"`. The exemption assumes
Harmony models handle tool calls natively and therefore need no tool parser. That is false:

- `HarmonyParser.parse` only emits tool calls when a tool parser is loaded —
  `case _SegmentType.TOOL if self.tool_parser:` (`vllm/parser/harmony.py:192`). With no
  tool parser, Harmony tool segments are parsed and then **silently dropped**.
- The tool JSON grammar is installed by `ToolParser.adjust_request`
  (`vllm/tool_parsers/abstract_tool_parser.py:137`), which is likewise only reached when a
  tool parser exists. Without one, `structured_outputs` stays `None`, so generation is
  **completely unconstrained** — nothing forces the model to emit a tool call at all.
- `ParserManager.get_tool_parser` returns `None` unless `--enable-auto-tool-choice` is
  passed (`vllm/parser/parser_manager.py:35`), so a gpt-oss server started with only
  `--reasoning-parser` (or with no parser flags) has no tool parser.

Net effect for gpt-oss without a tool parser: a request with `tool_choice="required"` or a
named tool choice is **accepted**, generates unconstrained, and comes back with an empty
`tool_calls` list — with the model's raw tool syntax either left in `content` or dropped,
and, for `required`, `finish_reason` mislabeled as `"tool_calls"`
(`vllm/entrypoints/openai/chat_completion/serving.py:960`). That is exactly the
"garbled output / empty responses" class of symptom the issue describes. Every other model
family already returned a clean 400 here; gpt-oss silently returned a wrong answer.

Verified before the fix (`structured_outputs=None`, no error raised):

```
model_type=any      tool_choice=required -> 400: tool_choice="required" requires --tool-call-parser to be set
model_type=any      tool_choice=named    -> 400: ...
model_type=gpt_oss  tool_choice=required -> ACCEPTED (no error). structured_outputs=None
model_type=gpt_oss  tool_choice=named    -> ACCEPTED (no error). structured_outputs=None
```

## 2. The fix and why

Narrow the Harmony exemption so it applies only to `tool_choice="auto"`, not to named /
`"required"`.

The distinction is the API contract. `"auto"` is best-effort — the model *may* call a tool,
and degrading to a plain text completion is a legitimate outcome, so Harmony (which renders
tool syntax natively) can keep accepting it. Named and `"required"` tool choice *guarantee*
the caller a tool call. vLLM cannot honor that guarantee without a tool parser, so it now
fails closed with the same 400 that every non-Harmony model already returned, instead of
returning an empty `tool_calls` list.

This is the "reject irreparable requests with 400" behavior the issue asks for, applied to
the one path where it was actually missing. No new flag and no new module: the validation
already existed and was reused; only its scope was wrong.

## 3. Files changed

- `vllm/renderers/online_renderer.py` — restrict the `use_harmony` exemption in
  `render_chat` to the `"auto"` branch; named/`"required"` now hit the existing
  "requires --tool-call-parser to be set" 400 on Harmony models too.
- `tests/entrypoints/openai/chat_completion/test_serving_chat.py` — add
  `test_tool_choice_validation_without_parser_harmony`, mirroring the existing
  `test_tool_choice_validation_without_parser` (which covers non-Harmony models). It asserts
  gpt-oss + `required`/named without a tool parser returns a 400 mentioning
  `--tool-call-parser`, and that `"auto"` is still accepted.
- `NOTES.md` — this file.

## 4. Risk / uncertainty

- **Behavior change, intentional**: gpt-oss servers with no tool-call parser that today send
  `tool_choice="required"` or a named tool choice will now get a 400 instead of a 200. Those
  responses were already unusable (empty `tool_calls`, unconstrained generation), so this
  turns a silent wrong answer into a loud, actionable one — but it *is* a
  200→400 change and should be called out in review. The error message names the exact flags
  to add. Correctly configured servers (`--enable-auto-tool-choice --tool-call-parser ...`)
  are unaffected.
- **Deliberately not touched**: the `is_mistral_tokenizer` exemption on the same condition
  looks like it may have the same hole, but Mistral has its own grammar/renderer path
  (`is_mistral_grammar_eligible`, `MistralParser`) that I did not fully trace, and confirming
  it needs a real Mistral model. Left alone to keep this scoped to one issue.
- `"auto"` behavior is unchanged everywhere, including Harmony — verified by test.
- The issue also asks for normalization/content-filtering and an
  `--enable-request-validation` flag. Not implemented: normalization would silently rewrite
  user requests, and the `tools`/`tool_choice`/`response_format` conflicts the issue lists
  are already rejected by `check_tool_usage` and `check_structured_outputs_count`. Making
  correct validation opt-in behind a flag would also leave the default path broken.

## 5. How I verified it

The environment had no vLLM build, so I layered a venv over an existing CPU torch install
(`torch 2.12.0+cpu`) plus `requirements/common.txt`, which is enough to run these
mock-engine entrypoint tests (they never touch a GPU).

- **Reproduced the bug** with a direct probe of `render_chat` across the
  `model_type` × `tool_choice` matrix — gpt-oss + `required`/named was accepted with
  `structured_outputs=None` (output quoted in §1).
- **Confirmed the fix closes it and nothing else moves.** Same probe against real
  `OnlineRenderer` instances, with and without a tool parser:

  | model | server config | `auto` | `required` | named |
  |---|---|---|---|---|
  | non-harmony | no parser | 400 | 400 | 400 |
  | non-harmony | `--tool-call-parser openai --enable-auto-tool-choice` | accepted | accepted | accepted |
  | gpt-oss | no parser | accepted | **400 (fixed)** | **400 (fixed)** |
  | gpt-oss | `--tool-call-parser openai --enable-auto-tool-choice` | accepted | accepted | accepted |

- **New test fails without the fix, passes with it.** Reverting only
  `online_renderer.py` makes `test_tool_choice_validation_without_parser_harmony` fail —
  the request is not rejected and instead runs on to an unrelated downstream error,
  which is the acceptance bug itself.
- **No regressions.** `tests/entrypoints/openai/chat_completion/test_serving_chat.py`
  (excluding the GPU-server classes): 33 passed at baseline → 34 passed with the change
  (the 34th is the new test), with the same 2 pre-existing errors from GPU-only fixtures
  that cannot start here. Also ran `test_serving_tokenization.py`,
  `test_lora_resolvers.py`, and `test_tool_choice_content_none.py` (12 passed), which
  cover the other users of `OnlineRenderer`.
- **Lint**: `ruff check` and `ruff format --check` clean on both changed files.

Not run: the GPU-backed gpt-oss server tests (`TestGPTOSSChat`,
`TestGPTOSSSpeculativeChat`) and any model-eval suite — no GPU in this environment. The
change only rejects requests that previously could not produce a valid tool call, so it
does not affect generated output for any working configuration.
