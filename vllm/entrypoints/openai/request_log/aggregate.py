# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate streaming SSE chunks back into a single response dict.

The OpenAI streaming protocol emits a sequence of ``data: {json}\\n\\n``
frames terminated by ``data: [DONE]\\n\\n``. For request logging we want
one structured record per request, so we re-assemble those frames into a
shape that mirrors the corresponding non-streaming response object.

The aggregators are best-effort: they preserve content / reasoning /
tool-call text and the final ``finish_reason`` and ``usage``, which is
what offline inspection actually needs. They are not a perfect
round-trip of every vendor-specific field.
"""

from __future__ import annotations

import json
from typing import Any


def _iter_data_frames(chunks: list[str]):
    """Yield decoded JSON dicts from a list of raw SSE chunk strings.

    Skips the terminating ``[DONE]`` frame and any malformed entries.
    """
    for chunk in chunks:
        if not chunk:
            continue
        # A single yield from the server can carry multiple frames if
        # buffered, so split on the SSE frame separator first.
        for frame in chunk.split("\n\n"):
            frame = frame.strip()
            if not frame:
                continue
            # Strip optional leading ``event: ...`` lines and grab the
            # ``data: ...`` payload.
            data_line = None
            for line in frame.split("\n"):
                if line.startswith("data:"):
                    data_line = line[len("data:") :].strip()
                    break
            if data_line is None:
                continue
            if data_line == "[DONE]":
                continue
            try:
                yield json.loads(data_line)
            except json.JSONDecodeError:
                continue


def aggregate_chat_stream(chunks: list[str]) -> dict[str, Any]:
    """Reconstruct a ``ChatCompletionResponse``-shaped dict from chat SSE.

    Per-choice we accumulate ``delta.content`` and ``delta.reasoning``
    text, capture the latest ``finish_reason``, and merge tool-call
    deltas keyed by their ``index``. The last non-null ``usage`` wins.
    """
    head: dict[str, Any] = {}
    # choice index -> aggregated message state
    choice_state: dict[int, dict[str, Any]] = {}
    usage: Any = None

    for frame in _iter_data_frames(chunks):
        if not isinstance(frame, dict):
            continue
        # First frame seen wins for top-level identifiers.
        for k in ("id", "model", "created", "system_fingerprint", "service_tier"):
            if k not in head and frame.get(k) is not None:
                head[k] = frame[k]
        if frame.get("usage") is not None:
            usage = frame["usage"]

        for choice in frame.get("choices") or []:
            idx = choice.get("index", 0)
            state = choice_state.setdefault(
                idx,
                {
                    "index": idx,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning": "",
                        "tool_calls": {},  # tool-index -> partial dict
                    },
                    "finish_reason": None,
                    "stop_reason": None,
                    "logprobs": None,
                    "_has_content": False,
                    "_has_reasoning": False,
                },
            )
            delta = choice.get("delta") or {}
            if delta.get("role"):
                state["message"]["role"] = delta["role"]
            if delta.get("content"):
                state["message"]["content"] += delta["content"]
                state["_has_content"] = True
            if delta.get("reasoning"):
                state["message"]["reasoning"] += delta["reasoning"]
                state["_has_reasoning"] = True
            for tc_delta in delta.get("tool_calls") or []:
                tc_idx = tc_delta.get("index", 0)
                tc_state = state["message"]["tool_calls"].setdefault(
                    tc_idx,
                    {
                        "id": None,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if tc_delta.get("id"):
                    tc_state["id"] = tc_delta["id"]
                if tc_delta.get("type"):
                    tc_state["type"] = tc_delta["type"]
                fn = tc_delta.get("function") or {}
                if fn.get("name"):
                    tc_state["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    tc_state["function"]["arguments"] += fn["arguments"]

            if choice.get("finish_reason") is not None:
                state["finish_reason"] = choice["finish_reason"]
            if choice.get("stop_reason") is not None:
                state["stop_reason"] = choice["stop_reason"]
            if choice.get("logprobs") is not None:
                state["logprobs"] = choice["logprobs"]

    aggregated_choices = []
    for idx in sorted(choice_state.keys()):
        s = choice_state[idx]
        msg = s["message"]
        # Materialize tool_calls dict-of-index → list ordered by index.
        tool_calls_raw = msg.pop("tool_calls")
        tool_calls = [tool_calls_raw[k] for k in sorted(tool_calls_raw.keys())]
        # Drop empty optional fields for cleaner output.
        if not s.pop("_has_content"):
            msg["content"] = None
        if not s.pop("_has_reasoning"):
            msg.pop("reasoning", None)
        else:
            # Keep "reasoning" only if non-empty.
            if not msg["reasoning"]:
                msg.pop("reasoning", None)
        msg["tool_calls"] = tool_calls
        aggregated_choices.append(
            {
                "index": idx,
                "message": msg,
                "logprobs": s["logprobs"],
                "finish_reason": s["finish_reason"],
                "stop_reason": s["stop_reason"],
            }
        )

    out: dict[str, Any] = {
        "id": head.get("id"),
        "object": "chat.completion",
        "created": head.get("created"),
        "model": head.get("model"),
        "choices": aggregated_choices,
        "usage": usage,
    }
    if "system_fingerprint" in head:
        out["system_fingerprint"] = head["system_fingerprint"]
    if "service_tier" in head:
        out["service_tier"] = head["service_tier"]
    return out


def aggregate_completion_stream(chunks: list[str]) -> dict[str, Any]:
    """Reconstruct a ``CompletionResponse``-shaped dict from completion SSE."""
    head: dict[str, Any] = {}
    choice_state: dict[int, dict[str, Any]] = {}
    usage: Any = None

    for frame in _iter_data_frames(chunks):
        if not isinstance(frame, dict):
            continue
        for k in ("id", "model", "created", "system_fingerprint", "service_tier"):
            if k not in head and frame.get(k) is not None:
                head[k] = frame[k]
        if frame.get("usage") is not None:
            usage = frame["usage"]
        for choice in frame.get("choices") or []:
            idx = choice.get("index", 0)
            state = choice_state.setdefault(
                idx,
                {
                    "index": idx,
                    "text": "",
                    "logprobs": None,
                    "finish_reason": None,
                    "stop_reason": None,
                },
            )
            if choice.get("text"):
                state["text"] += choice["text"]
            if choice.get("finish_reason") is not None:
                state["finish_reason"] = choice["finish_reason"]
            if choice.get("stop_reason") is not None:
                state["stop_reason"] = choice["stop_reason"]
            if choice.get("logprobs") is not None:
                # logprobs in completion stream are per-chunk lists; merge
                # naively by extending the parallel arrays.
                cur = state["logprobs"]
                lp = choice["logprobs"]
                if cur is None:
                    state["logprobs"] = {
                        "text_offset": list(lp.get("text_offset") or []),
                        "token_logprobs": list(lp.get("token_logprobs") or []),
                        "tokens": list(lp.get("tokens") or []),
                        "top_logprobs": list(lp.get("top_logprobs") or []),
                    }
                else:
                    for k in (
                        "text_offset",
                        "token_logprobs",
                        "tokens",
                        "top_logprobs",
                    ):
                        cur[k].extend(lp.get(k) or [])

    choices_out = [choice_state[k] for k in sorted(choice_state.keys())]
    return {
        "id": head.get("id"),
        "object": "text_completion",
        "created": head.get("created"),
        "model": head.get("model"),
        "choices": choices_out,
        "usage": usage,
    }


def aggregate_responses_stream(chunks: list[str]) -> dict[str, Any]:
    """For ``/v1/responses`` we just preserve the raw event list.

    The Responses streaming format is event-typed and considerably more
    complex than the chat/completion streams; doing a faithful aggregation
    is out of scope for the initial implementation. Saving the raw events
    keeps the record useful and lossless.
    """
    return {"events": [chunk for chunk in chunks if chunk]}


def aggregate_anthropic_messages_stream(chunks: list[str]) -> dict[str, Any]:
    """Reconstruct an ``AnthropicMessagesResponse``-shaped dict from /v1/messages SSE.

    Walks the event stream (``message_start``, ``content_block_start``,
    ``content_block_delta``, ``content_block_stop``, ``message_delta``,
    ``message_stop``) and rebuilds the final response in the same shape
    that the non-streaming endpoint would return. Tool-use blocks have
    their ``input`` parsed from the concatenated ``input_json_delta``
    pieces; if parsing fails the raw partial JSON string is preserved
    under ``input_partial_json`` so the record is still useful.
    """
    out: dict[str, Any] = {
        "id": None,
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": None,
        "stop_reason": None,
        "stop_sequence": None,
        "usage": None,
    }
    # index -> partial block dict (mirrors AnthropicContentBlock fields)
    blocks: dict[int, dict[str, Any]] = {}
    # index -> raw partial_json string (only for tool_use blocks)
    tool_args_buf: dict[int, str] = {}
    error: dict[str, Any] | None = None

    for frame in _iter_data_frames(chunks):
        if not isinstance(frame, dict):
            continue
        etype = frame.get("type")

        if etype == "message_start":
            msg = frame.get("message") or {}
            if msg.get("id"):
                out["id"] = msg["id"]
            if msg.get("model"):
                out["model"] = msg["model"]
            if msg.get("usage"):
                out["usage"] = dict(msg["usage"])

        elif etype == "content_block_start":
            idx = frame.get("index", 0)
            cb = dict(frame.get("content_block") or {})
            blocks[idx] = cb
            if cb.get("type") == "tool_use":
                tool_args_buf.setdefault(idx, "")

        elif etype == "content_block_delta":
            idx = frame.get("index", 0)
            block = blocks.setdefault(idx, {"type": None})
            delta = frame.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta" and delta.get("text"):
                block["type"] = block.get("type") or "text"
                block["text"] = (block.get("text") or "") + delta["text"]
            elif dtype == "thinking_delta" and delta.get("thinking"):
                block["type"] = block.get("type") or "thinking"
                block["thinking"] = (block.get("thinking") or "") + delta["thinking"]
            elif dtype == "signature_delta" and delta.get("signature"):
                block["signature"] = (block.get("signature") or "") + delta["signature"]
            elif dtype == "input_json_delta":
                if delta.get("partial_json"):
                    tool_args_buf[idx] = (
                        tool_args_buf.get(idx, "") + delta["partial_json"]
                    )

        elif etype == "content_block_stop":
            idx = frame.get("index", 0)
            existing = blocks.get(idx)
            if existing is not None and existing.get("type") == "tool_use":
                raw = tool_args_buf.get(idx, "")
                if raw:
                    try:
                        existing["input"] = json.loads(raw)
                    except json.JSONDecodeError:
                        existing["input"] = None
                        existing["input_partial_json"] = raw

        elif etype == "message_delta":
            delta = frame.get("delta") or {}
            if delta.get("stop_reason") is not None:
                out["stop_reason"] = delta["stop_reason"]
            if delta.get("stop_sequence") is not None:
                out["stop_sequence"] = delta["stop_sequence"]
            usage = frame.get("usage")
            if usage:
                # message_delta carries the final token totals; merge into
                # whatever message_start gave us.
                merged = dict(out["usage"] or {})
                merged.update(usage)
                out["usage"] = merged

        elif etype == "error":
            error = frame.get("error")

        # message_stop / ping / unknown -> nothing to do

    out["content"] = [blocks[i] for i in sorted(blocks.keys())]
    if error is not None:
        out["error"] = error
    return out
