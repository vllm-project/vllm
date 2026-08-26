# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-text channel segmentation utilities for MuseGlimmer output."""

from __future__ import annotations

from collections.abc import Iterator

import regex as re

EOM = "<|eom|>"
EOT = "<|eot|>"
ASSISTANT_TURN_OPEN = "<|start|>assistant"
FUNCTION_CALLS_OPEN = "<atem:function_calls>"
REASONING_RECIPIENT = "self"
USER_RECIPIENT = "user"

# All parts except <|message|> are optional. The bare form is used for public
# chain-of-thought or untagged content.
MSG_HEADER_RE = re.compile(
    r"(?:<\|start\|>\s*assistant)?[^\S\n]*"
    r"(?:to=(?P<recipient>[A-Za-z0-9_.\-]+))?<\|message\|>"
)
MSG_END_RE = re.compile(r"<\|eom\|>|<\|eot\|>")

# Whitespace handling must match MSG_HEADER_RE exactly. If this pattern were
# stricter, a header it rejected but MSG_HEADER_RE accepted would be recognised
# as a header yet not as a body boundary: iter_messages would then cut the
# preceding body at the stray <|start|> and skip past the header, silently
# dropping that message's body.
FRAMED_HEADER_PATTERN = (
    r"<\|start\|>\s*assistant[^\S\n]*"
    r"(?:to=[A-Za-z0-9_.\-]+)?<\|message\|>"
)
BARE_HEADER_WITH_ATEM_PATTERN = (
    r"to=(?!(?:self|user)<\|message\|>)[A-Za-z0-9_.\-]+<\|message\|>"
    r"(?=\s*<atem:(?:function_calls>|invoke(?:\s|>)))"
)
BODY_BOUNDARY_PATTERN = rf"(?:{FRAMED_HEADER_PATTERN}|{BARE_HEADER_WITH_ATEM_PATTERN})"
BODY_BOUNDARY_RE = re.compile(BODY_BOUNDARY_PATTERN)

_STRUCTURAL_MARKERS = (EOM, EOT, "<|start|>", "<|message|>")
_MAX_MARKER_LEN = max(len(marker) for marker in _STRUCTURAL_MARKERS)
_OPEN_TAIL_HEADER_RE = re.compile(r"[^\S\n]+(?:t|to|to=[A-Za-z0-9_.\-]*)$")


def current_assistant_turn(text: str) -> str:
    """Return the text following the latest framed assistant marker."""
    index = text.rfind(ASSISTANT_TURN_OPEN)
    if index == -1:
        return text
    return text[index + len(ASSISTANT_TURN_OPEN) :]


def iter_messages(text: str) -> Iterator[tuple[str | None, str, bool]]:
    """Yield ``(recipient, body, closed)`` for each MuseGlimmer message.

    A body ends at an explicit end marker, a fully framed assistant header, or
    a bare recipient header immediately followed by ATEM tool-call markup.
    """
    pos = 0
    while pos < len(text):
        header = MSG_HEADER_RE.search(text, pos)
        if header is None:
            return

        body_start = header.end()
        end = MSG_END_RE.search(text, body_start)
        boundary = BODY_BOUNDARY_RE.search(text, body_start)
        body_end = end.start() if end is not None else len(text)
        closed = end is not None

        if boundary is not None and boundary.start() < body_end:
            body_end = boundary.start()
            closed = False
            next_pos = boundary.start()
        else:
            next_pos = end.end() if end is not None else len(text)

        body = text[body_start:body_end]
        # A complete body cannot contain <|start|>. If it appears here, the next
        # framed header is incomplete and must be held back until <|message|>.
        start_token = body.find("<|start|>")
        if start_token != -1:
            body = body[:start_token]
            closed = False

        yield header.group("recipient"), body, closed
        pos = next_pos


def _trailing_partial_marker_len(text: str) -> int:
    """Return the longest suffix that prefixes a structural marker."""
    max_overlap = min(len(text), _MAX_MARKER_LEN - 1)
    for overlap in range(max_overlap, 0, -1):
        suffix = text[-overlap:]
        if any(marker.startswith(suffix) for marker in _STRUCTURAL_MARKERS):
            return overlap
    return 0


def safe_open_body(body: str) -> str:
    """Trim a growing body's suffix until it is safe to emit."""
    while True:
        trimmed = body

        partial_marker = _trailing_partial_marker_len(trimmed)
        if partial_marker:
            trimmed = trimmed[:-partial_marker]

        header_tail = _OPEN_TAIL_HEADER_RE.search(trimmed)
        if header_tail is not None:
            trimmed = trimmed[: header_tail.start()]

        boundary = BODY_BOUNDARY_RE.search(trimmed, partial=True)
        if boundary is not None and boundary.partial and boundary.end() == len(trimmed):
            candidate = trimmed[boundary.start() :]
            if candidate.startswith(("to=", "<|start|>")):
                trimmed = trimmed[: boundary.start()]

        if trimmed == body:
            return body
        body = trimmed


def visible_channels(text: str) -> tuple[str, str, bool, bool]:
    """Return content, reasoning, and whether each last body is still open."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    content_open = False
    reasoning_open = False

    for recipient, body, closed in iter_messages(text):
        if recipient == REASONING_RECIPIENT:
            reasoning_parts.append(body)
            reasoning_open = not closed
        elif recipient is None or recipient == USER_RECIPIENT:
            # An UNTAGGED body carrying ATEM markup is a tool channel whose
            # ``to=`` never arrived, so surfacing it would leak markup that
            # tool-call parsing (scoped to recipient-tagged bodies) never
            # claims. A ``to=user`` body is addressed to the client and may
            # legitimately quote ATEM -- e.g. answering a question about
            # tool-call syntax -- so it is surfaced as written.
            if recipient is None and (
                FUNCTION_CALLS_OPEN in body or "<atem:invoke" in body
            ):
                continue
            content_parts.append(body)
            content_open = not closed

    return (
        "".join(content_parts),
        "\n".join(reasoning_parts),
        content_open,
        reasoning_open,
    )


def advance_emitted(emitted: str, current: str) -> tuple[str, str]:
    """Return ``(delta, new_emitted)`` for a body that must only ever grow.

    A reclassified body legitimately SHRINKS between deltas: a partial header
    becomes recognisable and is trimmed, or a body stops qualifying as content.
    Storing a shrunken value would move the cursor backwards and re-emit text
    that already went out, so a non-extending value yields no delta and leaves
    the cursor untouched.
    """
    if not current.startswith(emitted) or len(current) <= len(emitted):
        return "", emitted
    return current[len(emitted) :], current


def open_recipient(text: str) -> str | None:
    """Return the recipient of the last open message, if one exists."""
    recipient: str | None = None
    is_open = False
    for recipient, _body, closed in iter_messages(text):
        is_open = not closed
    return recipient if is_open else None
