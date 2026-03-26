# [Bugfix][Tool Parser] Fix Kimi-K2 argument truncation when tool_call_end + tool_call_begin in same delta

## Bug

When speculative decoding (or multi-step scheduling) bundles multiple tokens into a single delta, `<|tool_call_end|>` of tool N and `<|tool_call_begin|>` of tool N+1 can arrive in the same streaming delta. When this happens, the "starting a new tool call" branch (the `cur_tool_start_count > cur_tool_end_count` condition) fires and overwrites `tool_call_portion` with the new tool's data, then increments `current_tool_id` — without ever finalizing tool N's remaining argument tokens.

**Result**: tool N's streamed arguments are truncated. For example, `{"city": "San Francisco", "units": "fahrenheit"}` becomes `{"city": "San Francisco", "units": "` — invalid JSON missing the final value and closing brace.

## Fix

Before advancing `current_tool_id` in the "starting a new tool" branch, check whether `tool_call_end_token_id` is also in `delta_token_ids`. If so, extract the closing tool's full arguments from `current_text`, compute the un-streamed diff against what was already emitted, and return that diff as a `DeltaMessage`. The new tool's name will be sent on the next streaming iteration.

## Test

`test_tool_end_and_next_begin_same_delta` — tokenizes a two-tool-call sequence with the real Kimi-K2 tokenizer, then merges the last 4 argument tokens of tool 0 together with `<|tool_call_end|>` and `<|tool_call_begin|>` into a single delta (simulating speculative decoding). Asserts both tools produce complete, valid JSON arguments.

## Commands

```bash
pytest tests/tool_parsers/test_kimi_k2_tool_parser.py -xvs
# 27 passed
```
