# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind structural-tag marker strings to their dedicated token IDs.

xgrammar's builtin structural tags spell tool-call markers such as
``<tool_call>`` as plain strings, so the grammar admits *any* tokenization of
that text. The streaming parser engine, once it has seen token IDs, recognises
the markers listed in ``token_id_terminals`` only by their dedicated token ID
and treats the same text spelled from ordinary tokens as content. Under
``tool_choice="required"`` a generation can therefore satisfy the grammar
while the parser reports no tool call at all.

:func:`bind_marker_tokens` rewrites a structural tag so that every marker the
parser keys by token ID is matched by xgrammar as that single token. The
grammar then admits exactly the encodings the parser recognises.
"""

from collections.abc import Collection, Mapping

from xgrammar import StructuralTag
from xgrammar.structural_tag import (
    ConstStringFormat,
    Format,
    OptionalFormat,
    OrFormat,
    PlusFormat,
    RepeatFormat,
    SequenceFormat,
    StarFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TokenFormat,
    TokenTriggeredTagsFormat,
    TriggeredTagsFormat,
)


def bind_marker_tokens(
    tag: StructuralTag,
    markers: Collection[str],
    vocab: Mapping[str, int],
) -> StructuralTag:
    """Match ``markers`` by dedicated token ID instead of by string.

    Args:
        tag: A structural tag, typically from xgrammar's builtin templates.
        markers: Marker strings the parser recognises by token ID.
        vocab: The tokenizer vocabulary. Markers that are not a single token
            are left as strings, which matches how the parser handles them.

    Returns:
        The rewritten tag, or ``tag`` itself when nothing needs rewriting.

    """
    resolved = {m for m in markers if m and m in vocab}
    if not resolved:
        return tag
    bound = _MarkerBinder(resolved, vocab).bind(tag.format)
    if bound is tag.format:
        return tag
    return tag.model_copy(update={"format": bound})


class _MarkerBinder:
    def __init__(self, markers: set[str], vocab: Mapping[str, int]) -> None:
        # Longest first, so a marker that extends another marker wins.
        self._markers = sorted(markers, key=len, reverse=True)
        self._vocab = vocab

    def bind(self, fmt: Format) -> Format:
        if isinstance(fmt, ConstStringFormat):
            return self._bind_const_string(fmt)
        if isinstance(fmt, TagFormat):
            return self._bind_tag(fmt)
        if isinstance(fmt, TriggeredTagsFormat):
            return self._bind_triggered_tags(fmt)
        if isinstance(fmt, TagsWithSeparatorFormat):
            return _replace(fmt, "tags", [self._bind_tag(tag) for tag in fmt.tags])
        if isinstance(fmt, SequenceFormat | OrFormat):
            return _replace(fmt, "elements", [self.bind(el) for el in fmt.elements])
        if isinstance(fmt, OptionalFormat | PlusFormat | StarFormat | RepeatFormat):
            return _replace(fmt, "content", self.bind(fmt.content))
        return fmt

    def _leading_marker(self, text: str) -> str | None:
        return next((m for m in self._markers if text.startswith(m)), None)

    def _trailing_marker(self, text: str) -> str | None:
        return next((m for m in self._markers if text.endswith(m)), None)

    def _split_literal(self, text: str) -> list[Format]:
        """Split a literal into const strings and marker tokens."""
        parts: list[Format] = []
        while text:
            best: tuple[int, str] | None = None
            for marker in self._markers:
                idx = text.find(marker)
                if idx != -1 and (best is None or idx < best[0]):
                    best = (idx, marker)
            if best is None:
                parts.append(ConstStringFormat(value=text))
                break
            idx, marker = best
            if idx:
                parts.append(ConstStringFormat(value=text[:idx]))
            parts.append(TokenFormat(token=marker))
            text = text[idx + len(marker) :]
        return parts

    def _bind_const_string(self, fmt: ConstStringFormat) -> Format:
        parts = self._split_literal(fmt.value)
        if not parts or (len(parts) == 1 and isinstance(parts[0], ConstStringFormat)):
            return fmt
        if len(parts) == 1:
            return parts[0]
        return SequenceFormat(elements=parts)

    def _bind_tag(
        self,
        tag: TagFormat,
        *,
        begin_marker: str | None = None,
        keep_begin: bool = False,
    ) -> TagFormat:
        """Move a leading begin marker and a trailing end marker onto tokens.

        The rest of ``begin`` and ``end`` is folded into the tag content so
        the accepted text stays byte-identical.
        """
        begin = tag.begin
        end = tag.end
        lead: list[Format] = []
        tail: list[Format] = []
        if isinstance(begin, str) and not keep_begin:
            marker = begin_marker or self._leading_marker(begin)
            if marker is not None and begin.startswith(marker):
                lead = self._split_literal(begin[len(marker) :])
                begin = TokenFormat(token=marker)
        if isinstance(end, str):
            marker = self._trailing_marker(end)
            if marker is not None:
                tail = self._split_literal(end[: -len(marker)])
                end = TokenFormat(token=marker)
        content = self.bind(tag.content)
        if lead or tail:
            content = SequenceFormat(elements=[*lead, content, *tail])
        if begin is tag.begin and end is tag.end and content is tag.content:
            return tag
        return tag.model_copy(update={"begin": begin, "content": content, "end": end})

    def _bind_triggered_tags(self, fmt: TriggeredTagsFormat) -> Format:
        """Dispatch on the marker token when every trigger starts with it."""
        marker = next(
            (
                m
                for m in self._markers
                if all(trigger.startswith(m) for trigger in fmt.triggers)
                and all(
                    isinstance(tag.begin, str) and tag.begin.startswith(m)
                    for tag in fmt.tags
                )
            ),
            None,
        )
        if marker is None:
            # The trigger is ordinary text (e.g. Llama's JSON prefix); keep
            # string dispatch but still bind markers inside the tags.
            return _replace(
                fmt, "tags", [self._bind_tag(tag, keep_begin=True) for tag in fmt.tags]
            )
        return TokenTriggeredTagsFormat(
            trigger_tokens=[marker],
            tags=[self._bind_tag(tag, begin_marker=marker) for tag in fmt.tags],
            # Only whole tokens can be excluded at the token level. A string
            # that is not a single token cannot be emitted as a marker anyway.
            exclude_tokens=[text for text in fmt.excludes if text in self._vocab],
            at_least_one=fmt.at_least_one,
            stop_after_first=fmt.stop_after_first,
        )


def _replace(fmt: Format, field: str, value: object) -> Format:
    """Copy ``fmt`` with ``field`` replaced, unless nothing actually changed."""
    current = getattr(fmt, field)
    if isinstance(value, list):
        unchanged = len(value) == len(current) and all(
            new is old for new, old in zip(value, current)
        )
    else:
        unchanged = value is current
    return fmt if unchanged else fmt.model_copy(update={field: value})
