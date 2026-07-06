# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared streaming simulation helpers for parser engine tests."""

from __future__ import annotations

from typing import Any

from vllm.entrypoints.openai.engine.protocol import DeltaMessage


def _build_token_id_map(parser) -> dict[str, int]:
    """Map special token text to token IDs from the parser's config."""
    token_id_map: dict[str, int] = {}
    cfg = getattr(parser, "parser_engine_config", None)
    vocab = getattr(parser, "vocab", None)
    if cfg is not None and vocab is not None:
        for text in (cfg.token_id_terminals or {}).values():
            tid = vocab.get(text)
            if tid is not None:
                token_id_map[text] = tid
    return token_id_map


def simulate_tool_streaming(
    parser,
    request,
    chunks: list[str],
) -> list[tuple[DeltaMessage | None, str]]:
    """Feed text chunks through ``extract_tool_calls_streaming()``."""
    token_id_map = _build_token_id_map(parser)

    results: list[tuple[Any, str]] = []
    previous_text = ""
    previous_token_ids: list[int] = []

    for chunk in chunks:
        current_text = previous_text + chunk

        delta_token_ids: list[int] = [
            tid for text, tid in token_id_map.items() if text in chunk
        ]

        current_token_ids = previous_token_ids + delta_token_ids

        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=tuple(previous_token_ids),
            current_token_ids=tuple(current_token_ids),
            delta_token_ids=tuple(delta_token_ids),
            request=request,
        )
        results.append((delta, current_text))
        previous_text = current_text
        previous_token_ids = list(current_token_ids)

    return results


def collect_tool_arguments(
    results: list[tuple[DeltaMessage | None, str]],
) -> str:
    """Concatenate all streamed argument fragments."""
    args_text = ""
    for delta, _ in results:
        if delta and delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.function and tc.function.arguments:
                    args_text += tc.function.arguments
    return args_text


def collect_content(
    results: list[tuple[DeltaMessage | None, str]],
) -> str:
    """Concatenate all streamed content parts."""
    parts: list[str] = []
    for delta, _ in results:
        if delta and delta.content:
            parts.append(delta.content)
    return "".join(parts)


def collect_function_name(
    results: list[tuple[DeltaMessage | None, str]],
) -> str | None:
    """Return first function name from deltas."""
    for delta, _ in results:
        if delta and delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.function and tc.function.name:
                    return tc.function.name
    return None


def simulate_reasoning_streaming(
    parser,
    chunks: list[str],
    delta_token_ids_per_chunk: list[tuple[int, ...]] | None = None,
) -> tuple[str, str]:
    """Feed chunks through ``extract_reasoning_streaming()``.

    Returns ``(reasoning_text, content_text)`` tuple.
    """
    token_id_map = (
        _build_token_id_map(parser) if delta_token_ids_per_chunk is None else {}
    )

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    prev_text = ""
    prev_ids: list[int] = []
    for i, chunk in enumerate(chunks):
        cur_text = prev_text + chunk
        if delta_token_ids_per_chunk is not None:
            d_ids = delta_token_ids_per_chunk[i]
        else:
            d_ids = tuple(tid for text, tid in token_id_map.items() if text in chunk)
        cur_ids = prev_ids + list(d_ids)
        delta = parser.extract_reasoning_streaming(
            previous_text=prev_text,
            current_text=cur_text,
            delta_text=chunk,
            previous_token_ids=tuple(prev_ids),
            current_token_ids=tuple(cur_ids),
            delta_token_ids=d_ids,
        )
        if delta:
            if delta.reasoning:
                reasoning_parts.append(delta.reasoning)
            if delta.content:
                content_parts.append(delta.content)
        prev_text = cur_text
        prev_ids = list(cur_ids)
    return "".join(reasoning_parts), "".join(content_parts)
