# Recover Qwen3 XML Tool Calls Emitted Inside `<think>`

## Problem Statement

This change addresses a **parsing compatibility bug** for the following setup:

- `--reasoning-parser qwen3`
- `--tool-call-parser qwen3_coder`

Qwen3/Qwen3.5 models can emit XML tool-call markup such as:

```text
<tool_call>
<function=Finish>
<parameter=answer>
204
</parameter>
</function>
</tool_call>
```

inside the reasoning region delimited by `<think> ... </think>`.

The issue is **not** that vLLM causes the model to generate tool calls inside
`<think>`. That is model output behavior.

The actual bug is that, when this output happens, vLLM loses the tool call
during non-streaming parsing.

## Root Cause

In the affected path:

1. `qwen3_reasoning_parser` extracts everything before `</think>` into
   `reasoning`.
2. downstream tool parsing inspects only `content`.
3. any `<tool_call>...</tool_call>` block that remains inside `reasoning`
   never reaches `qwen3_coder`.

As a result, the OpenAI-compatible response can contain:

- populated `reasoning`
- empty `tool_calls`

even though the model actually produced a valid XML tool call.

## Minimal Fix

The patch changes only:

- `vllm/reasoning/qwen3_reasoning_parser.py`

During non-streaming reasoning extraction:

1. detect XML tool-call blocks embedded in the extracted reasoning text
2. remove those blocks from the returned `reasoning`
3. prepend them to the returned `content`

This allows the existing `qwen3_coder` tool parser to parse them normally,
without changing the generic OpenAI serving pipeline.

## Why This Is The Right Scope

This patch intentionally fixes **recovery/parsing**, not model generation.

It does not try to force Qwen3.5 models to stop emitting tool calls inside
`<think>`. Instead, it makes vLLM robust when that output pattern appears.

This is the smallest change that fixes the observed benchmark failure while
keeping the rest of the tool-calling stack unchanged.

## Validation Scope

The added tests validate that:

- normal reasoning extraction remains unchanged
- embedded tool calls are promoted from `reasoning` into `content`
- promoted content remains parseable by `qwen3_coder`
- truncated reasoning without `</think>` still recovers embedded tool calls
- post-`</think>` content is preserved

## Limitation

This patch fixes the non-streaming path.

Streaming recovery is not addressed here, because the streaming path would
require additional stateful changes in the serving layer to forward reasoning
delivered tool markup into the tool parser before `</think>` is observed.
