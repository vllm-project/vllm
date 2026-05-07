# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Server-side agentic primitives for Codec streaming responses.

Two pieces, layered cleanly on top of the wire format defined in
codec_frame.py:

  - ToolWatcher: a uint32-compare state machine that detects delimited
    regions (tool calls, reasoning blocks, vision spans, sandbox runs)
    in the output token stream without ever decoding. Mirrors the
    libcodec / @codecai/web / codecai / Codec.Net implementations
    bit-identically — same edge cases, same buffering, same nested-
    start handling.

  - parse_tool_call: when a region completes, render its body through
    the tokenizer, parse as JSON (the convention every chat-tuned
    model in current use follows), and surface name + arguments_json
    on the next frame.

Why server-side: orchestrators don't have to detokenize on every
frame just to scan for marker text. The server already has the
tokenizer. The server already has the IDs. This PR exposes the
detection result directly in the Codec wire format so clients get
structured tool_call data alongside the raw token stream.

Disabled by default. Activated per-request via the `tool_watcher`
field on ChatCompletionRequest / CompletionRequest.

No new external dependencies — only stdlib + the codec_frame module
in this same package.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Tool-call data model (mirrors openai-style { id, name, arguments } shape)
# ---------------------------------------------------------------------------


@dataclass
class ToolCallEvent:
    """One tool call detected in the model's output stream.

    `arguments_json` is the raw JSON string between the start/end markers.
    `name` is parsed from that JSON when the model uses the standard
    `{"name": "...", "arguments": {...}}` shape; otherwise None.
    """

    name: Optional[str]
    arguments_json: str
    id: Optional[str] = None  # server-generated, e.g. "tc_<uuid>"

    def to_wire_dict(self) -> dict:
        """Serialise to the dict shape encoded into msgpack frames and
        the protobuf ToolCall message."""
        out: dict = {"arguments_json": self.arguments_json}
        if self.name is not None:
            out["name"] = self.name
        if self.id is not None:
            out["id"] = self.id
        return out


# ---------------------------------------------------------------------------
# Watcher state machine
# ---------------------------------------------------------------------------


@dataclass
class _WatcherState:
    """Minimal per-request state. Cheap to instantiate; one per stream."""

    start_id: int
    end_id: int
    inside: bool = False
    region_ids: List[int] = field(default_factory=list)


class ToolWatcher:
    """Stateful detector for delimited regions in a token-ID stream.

    The hot path is `feed(ids)` — a single linear pass that:
      - emits passthrough IDs (everything outside a region) untouched
      - on region close, returns the buffered body IDs for downstream
        parsing
      - never invokes the tokenizer

    This mirrors codec_tool_watcher (libcodec) and ToolWatcher
    (@codecai/web, codecai, Codec.Net). Same state-machine semantics
    so client-side and server-side detection produce identical results.

    Edge cases (matched bit-for-bit with the other implementations):
      - stray end marker: passes through as a regular ID
      - nested start marker: inner ignored, outer end closes the region
      - region split across feeds: body buffered, emitted on close
    """

    def __init__(self, start_id: int, end_id: int) -> None:
        self._st = _WatcherState(start_id=start_id, end_id=end_id)

    @property
    def inside(self) -> bool:
        return self._st.inside

    def reset(self) -> None:
        self._st.inside = False
        self._st.region_ids = []

    def feed(self, ids: List[int]) -> Tuple[List[int], List[List[int]]]:
        """Process a batch of newly-emitted token IDs.

        Returns:
          passthrough_ids: IDs that should be forwarded as the frame's
            `ids` field (markers consumed; region body IDs withheld
            until the region closes).
          completed_regions: list of region bodies (each a list of
            uint32s, markers excluded) that closed during this feed.

        Both are returned per-feed so the caller can attach completed
        regions to the same frame whose passthrough IDs come from this
        feed — keeps tool-call surfaces aligned with their stream
        position.
        """
        out_ids: List[int] = []
        completed: List[List[int]] = []
        st = self._st
        for tok in ids:
            if not st.inside:
                if tok == st.start_id:
                    st.inside = True
                    st.region_ids = []
                    # Marker itself is NOT forwarded — orchestrators
                    # don't want the "begin tool call" token in the
                    # outbound stream.
                else:
                    out_ids.append(tok)
            else:
                if tok == st.end_id:
                    completed.append(list(st.region_ids))
                    st.region_ids = []
                    st.inside = False
                    # End marker also withheld.
                elif tok == st.start_id:
                    # Nested start — ignore (same as the other ports).
                    pass
                else:
                    st.region_ids.append(tok)
        return out_ids, completed


# ---------------------------------------------------------------------------
# Body → ToolCallEvent
# ---------------------------------------------------------------------------


def parse_tool_call(
    region_body_text: str, *, call_id: Optional[str] = None
) -> ToolCallEvent:
    """Parse the body of a tool-call region (already detokenized) into
    a structured event.

    The convention every chat-tuned model in current use follows:
        { "name": "<function>", "arguments": { ... } }

    We accept both pretty-printed and compact JSON. If parsing fails
    (malformed body, partial JSON, etc.) we still return an event with
    name=None and arguments_json set to the raw body — the caller can
    surface that to the client so it can return a "invalid_arguments"
    error to the model.

    Empty / whitespace-only bodies produce an event with name=None
    and arguments_json="" — same shape, distinguishable downstream.
    """
    body = region_body_text.strip()
    if not body:
        return ToolCallEvent(name=None, arguments_json="", id=call_id)

    name: Optional[str] = None
    try:
        parsed: Any = json.loads(body)
        if isinstance(parsed, dict):
            n = parsed.get("name")
            if isinstance(n, str):
                name = n
    except json.JSONDecodeError:
        # Keep the raw body so the caller can decide how to handle it.
        pass

    return ToolCallEvent(name=name, arguments_json=body, id=call_id)


# ---------------------------------------------------------------------------
# Helpers for the serving layer
# ---------------------------------------------------------------------------


def detokenize_region(tokenizer, region_ids: List[int]) -> str:
    """Convenience wrapper around the tokenizer's batch decode that
    skips special tokens — tool-call body text is pure JSON, no chat
    template chrome.

    Tokenizer compatibility: works with any tokenizer exposing a
    .decode(ids, skip_special_tokens=bool) method (HF AutoTokenizer,
    vLLM's AnyTokenizer, the MistralTokenizer wrapper). We don't import
    transformers directly to keep this module dependency-free —
    duck-typing on .decode() is enough.
    """
    return tokenizer.decode(region_ids, skip_special_tokens=True)


def make_call_id(seq_no: int) -> str:
    """Server-generated tool call id. Stable shape; sequence-numbered
    rather than UUID so test fixtures stay deterministic."""
    return f"tc_{seq_no:08x}"


# ---------------------------------------------------------------------------
# Marker resolution helpers (vLLM-specific, not present in sglang)
# ---------------------------------------------------------------------------


def resolve_marker_id(tokenizer, marker: str) -> Optional[int]:
    """Resolve a special-token string like ``<tool_call>`` to its single
    integer ID in the loaded tokenizer's vocab.

    Returns None if the marker doesn't exist as a single token — the
    caller should disable the watcher in that case (the model can't
    emit a single-token boundary, so ID-level detection is impossible).

    Tries three lookup paths in order, since vLLM ships several
    tokenizer flavours and they don't all expose the same surface:
      1. ``added_tokens_encoder`` (HF fast / slow)
      2. ``get_vocab()`` returning a dict-like mapping
      3. ``encode(marker, add_special_tokens=False)`` returning a list
         whose length must be 1
    """
    enc = getattr(tokenizer, "added_tokens_encoder", None)
    if isinstance(enc, dict) and marker in enc:
        return int(enc[marker])

    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            vocab = get_vocab()
            if marker in vocab:
                return int(vocab[marker])
        except Exception:
            pass

    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            ids = encode(marker, add_special_tokens=False)
            # Some tokenizers return tensors; coerce to list.
            ids = list(ids) if not isinstance(ids, list) else ids
            if len(ids) == 1:
                return int(ids[0])
        except Exception:
            pass

    return None
