# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.

"""Gemma4 output parsing utilities for offline inference.

Standalone functions that parse decoded model text to extract structured
thinking content and tool calls from Gemma4 models. These are pure-Python
utilities with zero heavy dependencies — they work on raw decoded strings
from any inference backend (vLLM, HuggingFace, TGI, etc.).

Usage with vLLM offline inference::

    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.gemma4_utils import (
        parse_output,
        parse_tool_calls,
    )

    llm = LLM(model="google/gemma-4-it")
    outputs = llm.generate(prompt, SamplingParams(...))
    text = tokenizer.decode(outputs[0].outputs[0].token_ids, skip_special_tokens=False)

    # Extract thinking / answer (works with or without enable_thinking)
    result = parse_output(text)
    print(result["thinking"])  # chain-of-thought or None
    print(result["answer"])  # final answer

    # Extract tool calls
    tool_calls = parse_tool_calls(text)
    for tc in tool_calls:
        print(f"{tc['name']}({tc['arguments']})")

Ported from ``transformers.models.gemma4.utils_gemma4`` so that vLLM users
do not need a transformers dependency for output parsing.
"""

import json

import regex as re

# ---- Thinking Mode Utility ----

# Thinking delimiter tokens as they appear in decoded text.
# Gemma4 uses <|channel> (start) and <channel|> (end) as thinking delimiters.
_THINKING_START_TAG = "<|channel>"
_THINKING_END_TAG = "<channel|>"

# Sentinel tokens that may appear in decoded output.
_TURN_END_TAG = "<turn|>"


def parse_thinking_output(text: str) -> dict[str, str | None]:
    """Parse decoded Gemma4 model output.

    Use this on **all** Gemma4 output regardless of whether thinking mode
    was enabled.  It handles three cases:

    1. **Thinking enabled, tags present** — splits on ``<|channel>``/
       ``<channel|>`` to separate chain-of-thought from the answer and
       strips the ``thought\\n`` role label.
    2. **Thinking disabled, spurious label** — strips the bare
       ``thought\\n`` prefix that some Gemma4 models emit even
       without thinking mode.
    3. **Clean output** — returns the text unchanged.

    The answer text is always cleaned of trailing sentinel tokens
    (``<turn|>``, ``<eos>``, etc.).

    Args:
        text: Decoded model output text (from ``tokenizer.decode(...)``).

    Returns:
        A dict with keys:
            - ``"thinking"``: The chain-of-thought text, or ``None`` if no
              thinking delimiters were found.
            - ``"answer"``: The final answer text.

    Example::

        >>> from vllm.model_executor.models.gemma4_utils import parse_thinking_output
        >>> output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        >>> result = parse_thinking_output(output_text)
        >>> print(result["thinking"])  # chain-of-thought reasoning or None
        >>> print(result["answer"])    # final answer
    """
    if _THINKING_END_TAG in text:
        parts = text.split(_THINKING_END_TAG, 1)
        thinking_block = parts[0]
        answer = _clean_answer(parts[1])

        # Extract thinking content: strip the start tag if present
        if _THINKING_START_TAG in thinking_block:
            thinking = thinking_block.split(_THINKING_START_TAG, 1)[1]
        else:
            thinking = thinking_block

        # Strip the "thought\n" channel role label the model emits inside
        # <|channel>thought\n...<channel|> (analogous to "user\n" in
        # <|turn>user\n...<turn|>).
        thinking = _strip_thought_label(thinking.strip())
        thinking = thinking.strip()

        return {"thinking": thinking, "answer": answer}

    # No thinking delimiters found.
    # Strip spurious "thought\n" role label that some Gemma4 models sometimes
    # emit even without thinking mode enabled, then clean trailing tokens.
    answer = _strip_thought_label(text)
    answer = _clean_answer(answer)
    return {"thinking": None, "answer": answer}


def _strip_thought_label(text: str) -> str:
    """Strip the spurious ``thought\\n`` label from the start of text.

    Only strips when ``thought`` appears as the very first word followed by
    a newline — preserving the word ``thought`` in any other context.
    """
    if text.startswith("thought\n"):
        return text[len("thought\n") :]
    return text


def _clean_answer(text: str) -> str:
    """Clean trailing sentinel tokens from the answer text.

    Strips ``<turn|>``, ``<eos>``, and surrounding whitespace that the
    model appends at the end of its response.
    """
    text = text.strip()
    # Strip trailing <turn|> (Gemma4 turn-end marker)
    if text.endswith(_TURN_END_TAG):
        text = text[: -len(_TURN_END_TAG)].rstrip()
    # Strip trailing <eos> if present
    if text.endswith("<eos>"):
        text = text[:-5].rstrip()
    return text


# ---- Tool Call Parsing Utility ----
#
# NOTE: For the OpenAI-compatible API server tool parser (streaming +
# non-streaming), see vllm/tool_parsers/gemma4_tool_parser.py.
# This module provides offline inference utilities for direct user import.

# Tool call delimiter tokens as they appear in decoded text.
# Standard format: <|tool_call>call:name{args}<tool_call|>
_TOOL_CALL_START_TAG = "<|tool_call>"
_TOOL_CALL_END_TAG = "<tool_call|>"
_TOOL_RESPONSE_START_TAG = "<|tool_response>"

# Gemma4 escape token as it appears in decoded text.
_ESCAPE_TOKEN = '<|"|>'


def _parse_tool_arguments(args_str: str) -> dict[str, str]:
    """Parse tool call arguments from the Gemma4 compact format.

    Handles the ``key:<|"|>value<|"|>`` format used by Gemma4, with fallback
    to heuristic key-value extraction. Also tolerates the slightly different
    ``key: "value"`` format (space + plain quotes) that some chat templates
    produce.

    Args:
        args_str: Raw argument string from inside ``call:name{...}``.

    Returns:
        Dictionary of argument name → value.
    """
    if not args_str or not args_str.strip():
        return {}

    # Replace Gemma4 escape tokens with standard quotes.
    cleaned = args_str.replace(_ESCAPE_TOKEN, '"')

    # Try JSON parsing first (handles nested values, arrays, etc.).
    try:
        parsed = json.loads("{" + cleaned + "}")
        # Ensure all values are strings for consistency.
        return {k: str(v) if not isinstance(v, str) else v for k, v in parsed.items()}
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: extract key:"value" pairs (allow optional space after colon).
    arguments = {}
    for key, value in re.findall(r'(\w+):\s*"([^"]*)"', cleaned):
        arguments[key] = value

    if not arguments:
        # Last resort: extract key:value pairs (unquoted).
        for key, value in re.findall(r"(\w+):\s*([^,}]+)", args_str):
            arguments[key] = value.strip().strip('"').replace(_ESCAPE_TOKEN, "")

    return arguments


def parse_tool_calls(text: str, *, strict: bool = False) -> list[dict]:
    """Parse tool calls from decoded Gemma4 model output.

    Uses a tiered parsing strategy to handle known output variations in
    Gemma4 models, which may emit
    non-standard tool call formats.

    Parsing tiers:
        1. **Standard**: ``<|tool_call>call:name{args}<tool_call|>``
           (special token IDs 48/49 in decoded text)
        2. **Fallback** (when ``strict=False``): bare ``call:name{args}``
           patterns, including ``<call>name{args}`` (fragmented tokens from
           multimodal inputs)

    Args:
        text: Decoded model output text (from ``tokenizer.decode(...,
            skip_special_tokens=False)``).
        strict: If ``True``, only match the standard ``<|tool_call>`` format.
            If ``False`` (default), also try fallback patterns for
            known Gemma4 output variations.

    Returns:
        A list of dicts, each with keys:
            - ``"name"``: The tool function name (e.g. ``"get_weather"``).
            - ``"arguments"``: A dict of argument name → value.

    Example::

        >>> from vllm.model_executor.models.gemma4_utils import (
        ...     parse_tool_calls
        ... )
        >>> output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        >>> tool_calls = parse_tool_calls(output)
        >>> for tc in tool_calls:
        ...     print(f"Call: {tc['name']}({tc['arguments']})")
    """
    results = []

    # Tier 1: Standard format with special tokens.
    # <|tool_call>call:name{args}<tool_call|>
    # Note: Some Gemma4 models emit <turn|> instead of <tool_call|>.
    standard_pattern = r"<\|tool_call\>call:(\w+)\{(.*?)\}(?:<tool_call\|>|<turn\|>)"
    for match in re.finditer(standard_pattern, text, re.DOTALL):
        name, args_str = match.group(1), match.group(2)
        results.append(
            {
                "name": name,
                "arguments": _parse_tool_arguments(args_str),
            }
        )

    if results or strict:
        return results

    # Tier 2: Fallback for known Gemma4 output variations.
    # Matches: <call>name{args}, call:name{args}, or bare call:name{args}<eos>
    fallback_pattern = r"(?:<call>|(?:^|\s)call:)(\w+)\{(.*?)\}"
    for match in re.finditer(fallback_pattern, text, re.DOTALL):
        name, args_str = match.group(1), match.group(2)
        results.append(
            {
                "name": name,
                "arguments": _parse_tool_arguments(args_str),
            }
        )

    return results


def has_tool_response_tag(text: str) -> bool:
    """Check if model output properly ends with a tool response tag.

    Some Gemma4 models sometimes emit ``<eos>`` instead of
    ``<|tool_response>`` after a tool call. This helper detects
    whether the model used the proper termination, so callers can
    decide whether to inject ``<|tool_response>`` into the next prompt.

    Args:
        text: Decoded model output text.

    Returns:
        ``True`` if the output ends with ``<|tool_response>``
        (proper behavior), ``False`` otherwise.

    Example::

        >>> from vllm.model_executor.models.gemma4_utils import (
        ...     has_tool_response_tag
        ... )
        >>> if not has_tool_response_tag(model_output):
        ...     # Model used <eos> instead — inject <|tool_response> manually
        ...     next_prompt = "<|tool_response>" + tool_result
    """
    stripped = text.rstrip()
    return stripped.endswith(_TOOL_RESPONSE_START_TAG)
