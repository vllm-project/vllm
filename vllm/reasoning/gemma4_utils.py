# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.

"""Gemma4 thinking/reasoning output parsing utilities for offline inference.

Standalone functions that parse decoded model text to extract structured
thinking content from Gemma4 models. These are pure-Python utilities with
zero heavy dependencies — they work on raw decoded strings from any
inference backend (vLLM, HuggingFace, TGI, etc.).

For the OpenAI-compatible API reasoning parser (streaming +
non-streaming), see ``vllm.reasoning.gemma4_reasoning_parser``.
For tool call parsing, see ``vllm.tool_parsers.gemma4_utils``.

Usage with vLLM offline inference::

    from vllm import LLM, SamplingParams
    from vllm.reasoning.gemma4_utils import parse_thinking_output

    llm = LLM(model="google/gemma-4-it")
    outputs = llm.generate(prompt, SamplingParams(...))
    text = tokenizer.decode(outputs[0].outputs[0].token_ids,
                            skip_special_tokens=False)

    # Extract thinking / answer (works with or without enable_thinking)
    result = parse_thinking_output(text)
    print(result["thinking"])  # chain-of-thought or None
    print(result["answer"])    # final answer

Ported from ``transformers.models.gemma4.utils_gemma4`` so that vLLM users
do not need a transformers dependency for output parsing.
"""

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

        >>> from vllm.reasoning.gemma4_utils import parse_thinking_output
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
        return text[len("thought\n"):]
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
