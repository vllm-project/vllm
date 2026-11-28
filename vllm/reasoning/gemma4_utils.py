# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.

"""Gemma4 thinking/reasoning output parsing utilities for offline inference.

Standalone functions that parse decoded model text to extract structured
thinking content from Gemma4 models. These are pure-Python utilities with
zero heavy dependencies — they work on raw decoded strings from any
inference backend (vLLM, HuggingFace, TGI, etc.).

For the OpenAI-compatible API reasoning parser (streaming +
non-streaming), see `vllm.reasoning.gemma4_reasoning_parser`.
For tool call parsing, see `vllm.tool_parsers.gemma4_utils`.

Usage with vLLM offline inference::

    from vllm import LLM, SamplingParams
    from vllm.reasoning.gemma4_utils import parse_thinking_output

    llm = LLM(model="google/gemma-4-it")
    outputs = llm.generate(prompt, SamplingParams(...))
    text = tokenizer.decode(outputs[0].outputs[0].token_ids, skip_special_tokens=False)

    # Extract thinking / answer (works with or without enable_thinking)
    result = parse_thinking_output(text)
    print(result["thinking"])  # chain-of-thought or None
    print(result["answer"])  # final answer

Ported from `transformers.models.gemma4.utils_gemma4` so that vLLM users
do not need a transformers dependency for output parsing.
"""

# ---- Thinking Mode Utility ----

# Thinking delimiter tokens as they appear in decoded text.
# Gemma4 uses <|channel> (start) and <channel|> (end) as thinking delimiters.
_THINKING_START_TAG = "<|channel>"
_THINKING_END_TAG = "<channel|>"
_THOUGHT_ROLE = "thought"
_THOUGHT_PREFIX = "thought\n"
_THOUGHT_PREFIX_CRLF = "thought\r\n"

# Sentinel tokens that may appear in decoded output.
_TURN_END_TAG = "<turn|>"
_TRAILING_SENTINELS = (_TURN_END_TAG, "<eos>")


def parse_thinking_output(text: str) -> dict[str, str | None]:
    """Parse decoded Gemma4 model output.

    Use this on **all** Gemma4 output regardless of whether thinking mode
    was enabled. It handles four cases:

    1. **Thinking enabled, closed tags present** — splits on `<|channel>`/
       `<channel|>` to separate chain-of-thought from the answer and
       strips the `thought\\n` role label.
    2. **Thinking enabled, truncated mid-thought** — when output ends inside
       an unclosed `<|channel>thought\\n...` block (or a truncated `<|channel>`
       / `<|channel>thought` header), extracts the partial thought into
       `"thinking"` and keeps it out of `"answer"`.
    3. **Thinking disabled, spurious label** — strips the bare
       `thought\\n` prefix that some Gemma4 models emit even
       without thinking mode.
    4. **Clean output** — returns the text unchanged.

    Both the answer text and any unfinished thought are cleaned of trailing
    sentinel tokens (`<turn|>`, `<eos>`, etc.).

    Args:
        text: Decoded model output text (from `tokenizer.decode(...)`).

    Returns:
        A dict with keys:
            - `"thinking"`: The chain-of-thought text, or `None` if no
              thinking delimiters were found.
            - `"answer"`: The final answer text.

    Example::

        >>> from vllm.reasoning.gemma4_utils import parse_thinking_output
        >>> output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        >>> result = parse_thinking_output(output_text)
        >>> print(result["thinking"])  # chain-of-thought reasoning or None
        >>> print(result["answer"])    # final answer

    """
    thinking_parts: list[str] = []
    answer_parts: list[str] = []
    has_thinking = False
    remaining = text

    while _THINKING_END_TAG in remaining:
        thinking_block, after_end = remaining.split(_THINKING_END_TAG, 1)
        if has_thinking and _THINKING_START_TAG not in thinking_block:
            break
        has_thinking = True
        remaining = after_end

        if _THINKING_START_TAG in thinking_block:
            prefix, raw_thinking = thinking_block.split(_THINKING_START_TAG, 1)
            if prefix.strip():
                answer_parts.append(prefix)
        else:
            raw_thinking = thinking_block

        cleaned = _clean_thinking(raw_thinking)
        if cleaned:
            thinking_parts.append(cleaned)

    unfinished = _find_unfinished_thought(remaining)
    if unfinished is not None:
        has_thinking = True
        prefix, unfinished_thinking = unfinished
        if prefix.strip():
            answer_parts.append(prefix)
        if unfinished_thinking:
            thinking_parts.append(unfinished_thinking)
    elif has_thinking:
        answer_parts.append(remaining)

    if has_thinking:
        return {
            "thinking": "\n".join(thinking_parts),
            "answer": _clean_answer("".join(answer_parts)),
        }

    # No thinking delimiters found.
    # Strip spurious "thought\n" role label that some Gemma4 models sometimes
    # emit even without thinking mode enabled, then clean trailing tokens.
    answer = _strip_thought_label(text)
    answer = _clean_answer(answer)
    return {"thinking": None, "answer": answer}


def _clean_thinking(text: str) -> str:
    """Strip the `thought\\n` role label and surrounding whitespace inside a block."""
    text = text.lstrip()
    if text.startswith(_THOUGHT_PREFIX):
        return text[len(_THOUGHT_PREFIX) :].strip()
    if text.startswith(_THOUGHT_PREFIX_CRLF):
        return text[len(_THOUGHT_PREFIX_CRLF) :].strip()
    stripped = text.strip()
    if stripped == _THOUGHT_ROLE:
        return ""
    return stripped


def _find_unfinished_thought(text: str) -> tuple[str, str] | None:
    """Locate an unclosed Gemma4 thought block at the tail of `text`.

    Matches either `<|channel>thought\\n...` (with optional trailing
    sentinels) or a truncated `<|channel>` / `<|channel>thought` header at
    the end of output, so plain-text `<|channel>` mentions are not split.
    """
    search_from = 0
    while True:
        idx = text.find(_THINKING_START_TAG, search_from)
        if idx == -1:
            return None
        after = text[idx + len(_THINKING_START_TAG) :]
        header = _strip_trailing_sentinels(after).lstrip()
        if header.startswith(_THOUGHT_PREFIX):
            return text[:idx], header[len(_THOUGHT_PREFIX) :].strip()
        if header.startswith(_THOUGHT_PREFIX_CRLF):
            return text[:idx], header[len(_THOUGHT_PREFIX_CRLF) :].strip()
        if _THOUGHT_ROLE.startswith(header):
            return text[:idx], ""
        search_from = idx + len(_THINKING_START_TAG)


def _strip_thought_label(text: str) -> str:
    """Strip the spurious `thought\\n` label from the start of text.

    Only strips when `thought` appears as the very first word followed by
    a newline — preserving the word `thought` in any other context.
    """
    if text.startswith(_THOUGHT_PREFIX):
        return text[len(_THOUGHT_PREFIX) :]
    if text.startswith(_THOUGHT_PREFIX_CRLF):
        return text[len(_THOUGHT_PREFIX_CRLF) :]
    return text


def _strip_trailing_sentinels(text: str) -> str:
    """Strip repeated trailing `<turn|>` and `<eos>` tokens and trailing spaces."""
    text = text.rstrip()
    while True:
        for sentinel in _TRAILING_SENTINELS:
            if text.endswith(sentinel):
                text = text[: -len(sentinel)].rstrip()
                break
        else:
            return text


def _clean_answer(text: str) -> str:
    """Clean trailing sentinel tokens from the answer text.

    Strips `<turn|>`, `<eos>`, and surrounding whitespace that the
    model appends at the end of its response.
    """
    return _strip_trailing_sentinels(text.strip())
