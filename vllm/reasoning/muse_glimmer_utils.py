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
# A recipient name longer than this can never be a real channel header (the
# Rust port caps candidates identically): the whole candidate stays text.
_MAX_RECIPIENT_LEN = 1024
_RECIPIENT_CHAR = r"A-Za-z0-9_.\-"
_RECIPIENT = rf"[{_RECIPIENT_CHAR}]{{1,{_MAX_RECIPIENT_LEN}}}"
# A still-growing partial name; empty is allowed so a lone ` to=` holds.
_RECIPIENT_PARTIAL = rf"[{_RECIPIENT_CHAR}]{{0,{_MAX_RECIPIENT_LEN}}}"
MSG_HEADER_RE = re.compile(
    r"(?:<\|start\|>\s*assistant)?[^\S\n]*"
    rf"(?:to=(?P<recipient>{_RECIPIENT}))?<\|message\|>"
)
MSG_END_RE = re.compile(r"<\|eom\|>|<\|eot\|>")
# A run of channel terminators at the very end of the text (framing, not
# content). One pass strips the whole run: stripping a single marker per
# substitution is quadratic on a repetition-derailed <|eom|> flood.
TRAILING_MSG_END_RE = re.compile(
    r"(?:<\|eom\|>|<\|eot\|>)(?:\s*(?:<\|eom\|>|<\|eot\|>))*\s*$"
)

# Whitespace handling must match MSG_HEADER_RE exactly. If this pattern were
# stricter, a header it rejected but MSG_HEADER_RE accepted would be recognised
# as a header yet not as a body boundary: iter_messages would then cut the
# preceding body at the stray <|start|> and skip past the header, silently
# dropping that message's body.
FRAMED_HEADER_PATTERN = (
    rf"<\|start\|>\s*assistant[^\S\n]*(?:to={_RECIPIENT})?<\|message\|>"
)
# The bare-header defect switch: the header must start at a word boundary
# (whitespace or body start -- a match at body_start is preceded by the
# previous header's ``>`` and is rejected), and the ATEM markup must follow
# immediately, with no gap. Note the Rust port additionally fires a bare
# switch right after a channel header (empty-body defect); this pattern
# deliberately does not, preferring fewer phantom switches.
BARE_HEADER_WITH_ATEM_PATTERN = (
    rf"(?<!\S)to=(?!(?:self|user)<\|message\|>){_RECIPIENT}<\|message\|>"
    r"(?=<atem:(?:function_calls>|invoke(?:\s|>)))"
)
BODY_BOUNDARY_PATTERN = rf"(?:{FRAMED_HEADER_PATTERN}|{BARE_HEADER_WITH_ATEM_PATTERN})"
BODY_BOUNDARY_RE = re.compile(BODY_BOUNDARY_PATTERN)

_STRUCTURAL_MARKERS = (EOM, EOT, "<|start|>", "<|message|>")
_MAX_MARKER_LEN = max(len(marker) for marker in _STRUCTURAL_MARKERS)
# A ` to`-ish fragment that may grow into a bare header: preceded by any
# whitespace (including newlines, matching the boundary pattern's anchor).
_OPEN_TAIL_HEADER_RE = re.compile(rf"\s+(?:t|to|to={_RECIPIENT_PARTIAL})$")
# A complete bare TOOL header at the very end of a finished body: its body
# never arrived, so it is framing, not text. Anchored like the boundary
# pattern's Rust semantics: the header must follow whitespace -- glued to a
# word or at body start it is text (a body-start match would be preceded by
# the previous header's ``>`` in the full text, which the boundary pattern
# rejects). Self/user are excluded for the same reason: bare self/user
# headers never switch channels, so they were streamed as text. Only the
# header itself is stripped -- the preceding whitespace was legitimately
# emitted while streaming.
_TRAILING_BARE_HEADER_RE = re.compile(
    rf"(?<=\s)to=(?!(?:self|user)<\|message\|>){_RECIPIENT}<\|message\|>\Z"
)
# The start of an invoke block, word-boundaried like the boundary pattern.
_ATEM_INVOKE_OPEN_RE = re.compile(r"<atem:invoke(?=[\s>])")


def current_assistant_turn(text: str) -> str:
    """Return the text following the latest framed assistant marker.

    Only an occurrence that opens a real header counts -- or one that ends the
    text, as in the generation prompt's trailing turn opener. A quoted or
    partial ``<|start|>assistant`` inside a body is text, not a turn boundary.

    Forward scan, tracking the latest qualifying occurrence: re-anchoring with
    ``rfind`` in a loop is quadratic on a flood of quoted markers.
    """
    last: int | None = None
    # A header spans at most the turn opener, whitespace, a `to=` recipient
    # and <|message|>; bound the match window so a flood of quoted turn
    # openers costs O(1) per occurrence instead of a whole-tail scan for the
    # <|message|> literal. A header with a longer interior whitespace run is
    # derailment: missing it just widens the scan, which lands on the same
    # last open channel anyway.
    max_header_span = (
        len(ASSISTANT_TURN_OPEN) + 64 + 3 + _MAX_RECIPIENT_LEN + len("<|message|>")
    )
    for match in re.finditer(re.escape(ASSISTANT_TURN_OPEN), text):
        index = match.start()
        if MSG_HEADER_RE.match(text, index, index + max_header_span) is not None or (
            index + len(ASSISTANT_TURN_OPEN) == len(text)
        ):
            last = index
    if last is None:
        return text
    return text[last + len(ASSISTANT_TURN_OPEN) :]


def iter_messages(text: str) -> Iterator[tuple[str | None, str, bool]]:
    """Yield ``(recipient, body, closed)`` for each MuseGlimmer message.

    A body ends at an explicit end marker, a fully framed assistant header, or
    a bare recipient header immediately followed by ATEM tool-call markup.
    Bare ``to=self``/``to=user`` headers never bound a body (they are streamed
    as text), and a bare header must start at a word boundary -- glued to a
    preceding word it is ordinary text.
    """
    pos = 0
    end_marker_missing = False
    while pos < len(text):
        header = MSG_HEADER_RE.search(text, pos)
        if header is None:
            return

        body_start = header.end()
        # Once an end-marker search comes back empty no later body can have
        # one either; latching it keeps an open-channel flood (no <|eom|>
        # anywhere) linear instead of paying a whole-tail scan per message.
        end = None if end_marker_missing else MSG_END_RE.search(text, body_start)
        if end is None:
            end_marker_missing = True
        body_end = end.start() if end is not None else len(text)
        closed = end is not None

        # A boundary only matters when it starts before the end marker, so
        # stop the search there: scanning the rest of the text is wasted work
        # (quadratic on boundary-less message floods). The bare header's ATEM
        # lookahead never spans an end marker, so nothing is clipped.
        boundary = BODY_BOUNDARY_RE.search(text, body_start, body_end)

        if boundary is not None:
            body_end = boundary.start()
            closed = False
            next_pos = boundary.start()
        else:
            next_pos = end.end() if end is not None else len(text)

        body = text[body_start:body_end]
        # Hold back only a trailing prefix of a framed header, and only while
        # the body can still grow: a literal <|start|> inside a closed body is
        # user text and must be preserved. Only the LAST <|start|> can begin a
        # trailing partial header (a partial match must consume to end of
        # string, and nothing in the header pattern can absorb a second
        # <|start|>), so checking it alone is equivalent to scanning them all.
        if not closed:
            start_token = body.rfind("<|start|>")
            if start_token != -1:
                candidate = body[start_token:]
                partial_header = re.fullmatch(
                    FRAMED_HEADER_PATTERN, candidate, partial=True
                )
                if partial_header is not None and partial_header.partial:
                    body = body[:start_token]

        yield header.group("recipient"), body, closed
        pos = next_pos


def _trailing_live_marker_len(text: str) -> int:
    """Return the trailing suffix that could still grow into a marker.

    Anchored at the LAST ``<``: in a run of marker prefixes only that start
    is still live (a marker's third character is a letter), so this replaces
    re-scanning a run of prefixes to a fixpoint.
    """
    index = text.rfind("<")
    if index == -1:
        return 0
    suffix = text[index:]
    for marker in _STRUCTURAL_MARKERS:
        if len(suffix) < len(marker) and marker.startswith(suffix):
            return len(suffix)
    return 0


def safe_open_body(body: str) -> str:
    """Trim a growing body's suffix until it is safe to emit.

    Three suffixes are unsafe to emit from a body that can still grow: a live
    partial marker, a ` to=…` header fragment, and a partial channel boundary
    (e.g. a half-written framed header). A strip of one kind can expose
    another (``… to=skill<``), so a marker strip follows the header/boundary
    strips once more. Each strip anchors at its last possible start: in a run
    of marker prefixes only the last ``<`` is still live, and a fragment with
    more text after it can never complete. Longer chains are derailment
    debris and self-heal at the next delta.
    """
    partial_marker = _trailing_live_marker_len(body)
    if partial_marker:
        body = body[:-partial_marker]
    trimmed = body

    header_tail = _OPEN_TAIL_HEADER_RE.search(trimmed)
    if header_tail is not None:
        trimmed = trimmed[: header_tail.start()]

    boundary = BODY_BOUNDARY_RE.search(trimmed, partial=True)
    if boundary is not None and boundary.partial and boundary.end() == len(trimmed):
        candidate = trimmed[boundary.start() :]
        if candidate.startswith(("to=", "<|start|>")):
            trimmed = trimmed[: boundary.start()]

    if trimmed != body:
        # The header/boundary strip may have exposed a live marker tail.
        partial_marker = _trailing_live_marker_len(trimmed)
        if partial_marker:
            trimmed = trimmed[:-partial_marker]
    return trimmed


def safe_unframed_tail(text: str) -> str:
    """Trailing holdback for text with no channel framing yet.

    Never emit a tail that could still grow into a bare header: a trailing
    whitespace run may precede `to=…`, and the first header of a turn needs
    no preceding whitespace, so a whole-buffer `to=…` fragment is held too.
    """
    # Strip a trailing run of end markers first: exposing one after the
    # whitespace trim would leak it (the marker is framing, not content).
    # Fixpoint with safe_open_body: its partial-marker/header strip can expose
    # a complete end marker at the tail ("ok<|eom|><" -> "ok<|eom|>").
    # Bounded: this runs per delta over the full text, so a derailment flood
    # of strippable fragments must not multiply the scan cost. Holding back
    # too much is always safe -- finish flushes it as text.
    body = text
    converged = False
    for _ in range(4):
        stripped = TRAILING_MSG_END_RE.sub("", body)
        if stripped != body:
            body = stripped
            continue
        trimmed = safe_open_body(body)
        if trimmed == body:
            converged = True
            break
        body = trimmed
    if not converged:
        return ""
    tail = re.search(r"\s+$", body)
    if tail is not None:
        body = body[: tail.start()]
    if re.fullmatch(rf"(?:t|to|to={_RECIPIENT_PARTIAL})", body) is not None:
        return ""
    return body


def flush_open_body(body: str) -> str:
    """Trim trailing framing from a finished body.

    At end-of-stream nothing more arrives: a held-back ` to=…` fragment is
    real text and must flush, while a trailing partial marker (cut by the
    token limit) is framing and stays dropped. A trailing COMPLETE bare tool
    header (``to=…<|message|>`` with nothing after it) is framing too -- its
    body never arrived -- so it is stripped as well. Trailing end markers are
    likewise framing. The strips alternate to a fixpoint: stripping one layer
    can expose another ("ans<|eom|> to=x<|message|>" -> "ans").

    The partial-marker strip fires at most once: after it fires, a further
    live-looking suffix was followed by the stripped fragment in the true
    text, so it is dead text, not a cut marker ("a<|<|" keeps "a<|").
    """
    marker_stripped = False
    while True:
        # A trailing run of end markers is framing, not content (the
        # streaming path never surfaces them either).
        stripped = TRAILING_MSG_END_RE.sub("", body)
        if stripped != body:
            body = stripped
            continue

        # A lone "<" is ordinary text; only a marker actually in progress
        # ("<|…") is framing cut by the token limit.
        if not marker_stripped:
            partial = _trailing_live_marker_len(body)
            if partial >= 2:
                body = body[: len(body) - partial]
                marker_stripped = True
                continue

        header = _TRAILING_BARE_HEADER_RE.search(body)
        if header is not None:
            body = body[: header.start()]
            continue

        return body


def framing_start(text: str) -> int:
    """Start of the first channel framing, or ``len(text)`` when there is none.

    The earliest of the first message header (whose match may include the
    preceding whitespace run) and the first bare ``<|start|>`` (a framed
    header still in progress). Everything before it is pre-header text the
    segmenter will never admit into a channel.
    """
    header = MSG_HEADER_RE.search(text)
    start = header.start() if header is not None else len(text)
    bare = text.find("<|start|>")
    if bare != -1:
        start = min(start, bare)
    return start


def has_channel_framing(text: str) -> bool:
    """Whether the text contains channel framing or a possible start of it.

    ATEM markup alone does not count: a bare ``<atem:…`` block with no header
    is quoted text (no streaming path scans headerless markup; only the
    finish-time salvage and the non-streaming fallback do).
    """
    return MSG_HEADER_RE.search(text) is not None or "<|start|>" in text


def has_complete_channel(text: str) -> bool:
    """Whether the text contains at least one complete channel header."""
    return next(iter_messages(text), None) is not None


def visible_channels(
    text: str, *, withhold_open_untagged: bool = False, flush_growing: bool = False
) -> tuple[str, str, bool, bool]:
    """Return content, reasoning, and whether each last body is still growing.

    A body can still grow only when it is unterminated AND the last message
    in the text: a body cut by a later channel header is frozen, and its tail
    is emittable text -- holding it back (or stripping it at finish) would
    retract text the stream already validated.

    Open untagged bodies may later become ATEM tool channels, so streaming
    callers withhold them until their classification can no longer shrink --
    i.e. only while they can still grow.

    With ``flush_growing`` (the end-of-stream paths), the growing body's tail
    is flushed per-body BEFORE joining: on the joined string a frozen body's
    tail is indistinguishable from the growing body's framing.
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    content_open = False
    reasoning_open = False

    messages = list(iter_messages(text))
    last = len(messages) - 1
    for index, (recipient, body, closed) in enumerate(messages):
        # An empty growing tail contributes nothing to the joined text, so the
        # tail callers would trim belongs to a frozen body. Only a non-empty
        # growing body makes its channel's tail unsafe to emit.
        growing = not closed and index == last and bool(body)
        if recipient == REASONING_RECIPIENT:
            if flush_growing and growing:
                body = flush_open_body(body)
            reasoning_parts.append(body)
            reasoning_open = growing
        elif recipient is None or recipient == USER_RECIPIENT:
            if recipient is None and growing and withhold_open_untagged:
                # An open untagged body may still grow ATEM markup; hold it
                # back until its classification can no longer change.
                continue
            if recipient is None:
                # An untagged body carrying ATEM markup is a tool channel
                # whose ``to=`` never arrived: keep only the prefix before
                # the markup; the markup itself is never surfaced. The word
                # boundary matches the boundary pattern: "<atem:invokeful"
                # is ordinary text.
                invoke_open = _ATEM_INVOKE_OPEN_RE.search(body)
                atem_starts = [
                    i
                    for i in (
                        body.find(FUNCTION_CALLS_OPEN),
                        invoke_open.start() if invoke_open is not None else -1,
                    )
                    if i != -1
                ]
                if atem_starts:
                    body = body[: min(atem_starts)]
                    if not body:
                        continue
            if flush_growing and growing:
                body = flush_open_body(body)
            content_parts.append(body)
            content_open = growing

    return (
        "".join(content_parts),
        "\n".join(reasoning_parts),
        content_open,
        reasoning_open,
    )


def channel_seed(recipient: str | None) -> str | None:
    """Seed text for a generation continuing the prompt's open channel."""
    if recipient is None:
        return None
    if recipient == "":
        return "<|message|>"
    return f"to={recipient}<|message|>"


def strip_frozen_tail(text: str) -> str:
    """Trim a frozen unframed region's tail for the flip flush.

    Mirrors what the unframed streaming path never surfaces: a trailing
    end-marker run (framing, not content) and trailing whitespace. Anything
    else -- partial markers, ` to=…` fragments -- is dead text now that the
    region can no longer grow, and stays. The whitespace class must be the
    ``regex`` module's ``\\s`` (as in safe_unframed_tail): ``str.rstrip()``
    also eats U+001C-U+001F, which the streaming path keeps.
    """
    return re.sub(r"\s+$", "", TRAILING_MSG_END_RE.sub("", text))


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
    """Return the recipient of the last open message, if one exists.

    An open UNTAGGED channel (a bare ``<|message|>`` header) yields ``""``
    rather than None: the channel exists and is answer-side, it just carries
    no recipient name. None still means no open channel. A prompt that ends
    mid-header (a recipient with no ``<|message|>`` yet) is deliberately not
    reported -- the channel has not opened (documented limitation, exotic).
    """
    recipient: str | None = None
    is_open = False
    for recipient, _body, closed in iter_messages(text):
        is_open = not closed
    if not is_open:
        return None
    return recipient if recipient is not None else ""
