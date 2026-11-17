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

# Sentinel tokens that may appear in decoded output.
_TURN_END_TAG = "<turn|>"
_EOS_TAG = "<eos>"
_TRAILING_SENTINELS = (_TURN_END_TAG, _EOS_TAG)

_THOUGHT_LABEL = "thought"
_THOUGHT_PREFIX = f"{_THOUGHT_LABEL}\n"


def parse_thinking_output(text: str) -> dict[str, str | None]:
    """Parse decoded Gemma4 model output.

    Use this on **all** Gemma4 output regardless of whether thinking mode
    was enabled.  It handles three cases:

    1. **Thinking enabled, tags present** — splits on `<|channel>`/
       `<channel|>` to separate chain-of-thought from the answer and
       strips the `thought\\n` role label.
    2. **Thinking disabled, spurious label** — strips the bare
       `thought\\n` prefix that some Gemma4 models emit even
       without thinking mode.
    3. **Clean output** — returns the text unchanged.

    The answer text is always cleaned of trailing sentinel tokens
    (`<turn|>`, `<eos>`, etc.).

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
    remaining = text
    found_thinking = False

    while _THINKING_END_TAG in remaining:
        found_thinking = True
        thinking_block, remaining = remaining.split(_THINKING_END_TAG, 1)
        if _THINKING_START_TAG in thinking_block:
            before, thinking = thinking_block.split(_THINKING_START_TAG, 1)
            if before:
                answer_parts.append(before)
        else:
            thinking = thinking_block

        cleaned = _clean_thinking(thinking)
        if cleaned:
            thinking_parts.append(cleaned)

    unclosed = _find_unfinished_thought(remaining)
    if unclosed is not None:
        found_thinking = True
        before_start, partial_thinking = unclosed
        if before_start:
            answer_parts.append(before_start)
        cleaned = _clean_answer(partial_thinking)
        if cleaned:
            thinking_parts.append(cleaned)
        remaining = ""

    if found_thinking:
        if remaining:
            answer_parts.append(remaining)
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


def _find_unfinished_thought(text: str) -> tuple[str, str] | None:
    """Find the first unclosed Gemma4 `thought` channel block, if any.

    Matches either `<|channel>thought\n...` or a truncated `<|channel>` /
    `<|channel>thought` header at the end of the text (ignoring trailing
    whitespace and sentinel tokens), so plain-text mentions like
    `"answer mentions <|channel> literally"` are not split falsely.
    """
    search_from = 0
    while True:
        idx = text.find(_THINKING_START_TAG, search_from)
        if idx == -1:
            return None
        after = text[idx + len(_THINKING_START_TAG) :]
        if after.startswith(_THOUGHT_PREFIX):
            return text[:idx], after[len(_THOUGHT_PREFIX) :]
        if _THOUGHT_LABEL.startswith(_strip_trailing_sentinels(after)):
            return text[:idx], ""
        search_from = idx + len(_THINKING_START_TAG)


def _clean_thinking(text: str) -> str:
    """Strip the `thought\\n` channel role label and surrounding whitespace."""
    # Gemma4 emits `thought\n` inside `<|channel>thought\n...<channel|>`
    # (analogous to `user\n` in `<|turn>user\n...<turn|>`). Only strip leading
    # whitespace before checking the label so `<|channel>thought\n` keeps its
    # trailing newline intact when stripping `_THOUGHT_PREFIX`.
    stripped = text.lstrip()
    if stripped.startswith(_THOUGHT_PREFIX):
        return stripped[len(_THOUGHT_PREFIX) :].strip()
    if stripped.rstrip() == _THOUGHT_LABEL:
        return ""
    return stripped.strip()


def _strip_thought_label(text: str) -> str:
    """Strip the spurious `thought\\n` label from the start of text.

    Only strips when `thought` appears as the very first word followed by
    a newline — preserving the word `thought` in any other context.
    """
    if text.startswith(_THOUGHT_PREFIX):
        return text[len(_THOUGHT_PREFIX) :]
    return text


def _strip_trailing_sentinels(text: str) -> str:
    """Strip trailing sentinel tokens (`<turn|>`, `<eos>`) and whitespace."""
    text = text.rstrip()
    while True:
        for sentinel in _TRAILING_SENTINELS:
            if text.endswith(sentinel):
                text = text[: -len(sentinel)].rstrip()
                break
        else:
            break
    return text


def _clean_answer(text: str) -> str:
    """Clean trailing sentinel tokens from the answer text.

    Strips `<turn|>`, `<eos>`, and surrounding whitespace that the
    model appends at the end of its response.
    """
    return _strip_trailing_sentinels(text.lstrip())
