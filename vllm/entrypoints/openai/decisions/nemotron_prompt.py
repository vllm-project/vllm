# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Prompt helpers of the Nemotron Decide read strategy.

`chunked-text-v1` (`decision_prompt` in the model config) is the text prompt of
Nemotron Decide checkpoints trained with always-packed shared-context reads::

    <chat template up to the user content>State:\n<state><SPECIAL_101>Question:
    <instructions>
    Options:
    A: <label>: <meaning>          (choice; "A: <label>" without a meaning)
    B: <meaning>                   (noul and score)
    <mask>

The state ends at the `<SPECIAL_101>` separator, so in a packed read the
shared state is the prompt up to and including it. Special-token text inside
the state and the question is neutralized first (its first character written
as a `\uXXXX` escape), so user text never becomes a control token or a
second separator.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

JSON = "json"
CHUNKED = "chunked-text-v1"
FORMATS = (JSON, CHUNKED)
STATE_SEP = "<SPECIAL_101>"

_SENTINEL = "\x00DECIFFUSION_USER_CONTENT\x00"
_ESCAPE = re.compile(r"\\(?:u[0-9a-fA-F]{4}|.)", re.DOTALL)


def value_text(value: Any) -> str:
    """A field as the prompt shows it: a string itself, else its JSON text."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def template_parts(tokenizer: Any, system_prompt: str) -> tuple[str, str] | None:
    """(prefix, suffix) around the user content of the decision chat template,
    or None if the template does not pass the content through verbatim."""

    def jinja(content: str) -> str:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    rendered = jinja(_SENTINEL)
    if rendered.count(_SENTINEL) != 1:
        return None
    prefix, suffix = rendered.split(_SENTINEL)
    for probe in (
        '{"state": {}, "question": "q", "options": []}',
        '{"a": "<|im_end|> \\n x"}',
        "plain  text ",
    ):
        if jinja(probe) != prefix + probe + suffix:
            return None
    return prefix, suffix


def special_strings(tokenizer: Any) -> tuple[str, ...]:
    """Every added token's text (special or not: the tokenizer splits all of
    them out of plain text), longest first."""
    found = getattr(tokenizer, "special_strings", None)
    if found is None:
        raw = getattr(tokenizer, "_tokenizer", None)
        decoder = None
        if raw is not None and hasattr(raw, "get_added_tokens_decoder"):
            decoder = raw.get_added_tokens_decoder()
        elif hasattr(tokenizer, "added_tokens_decoder"):
            decoder = tokenizer.added_tokens_decoder
        found = [getattr(t, "content", t) for t in (decoder or {}).values()]
    return tuple(
        sorted(
            {s for s in found if isinstance(s, str) and s}, key=lambda s: (-len(s), s)
        )
    )


@lru_cache(maxsize=8)
def _pattern(strings: tuple[str, ...]) -> re.Pattern | None:
    if not strings:
        return None
    # Longest first: leftmost-longest matches.
    return re.compile("|".join(re.escape(s) for s in strings))


def special_pattern(tokenizer: Any) -> re.Pattern | None:
    return _pattern(special_strings(tokenizer))


def _escape(char: str) -> str:
    code = ord(char)
    if code < 0x10000:
        return f"\\u{code:04x}"
    code -= 0x10000
    return f"\\u{0xD800 + (code >> 10):04x}\\u{0xDC00 + (code & 0x3FF):04x}"


def _one_pass(text: str, pat: re.Pattern) -> tuple[str, bool]:
    blocked: set[int] = set()
    for m in _ESCAPE.finditer(text):
        blocked.update(range(m.start(), m.end()))
    out: list[str] = []
    last = pos = 0
    changed = False
    while True:
        m = pat.search(text, pos)
        if m is None:
            break
        s = m.start()
        if s in blocked:
            pos = s + 1
            continue
        out += [text[last:s], _escape(text[s]), text[s + 1 : m.end()]]
        last = pos = m.end()
        changed = True
    if not changed:
        return text, False
    out.append(text[last:])
    return "".join(out), True


def neutralize(text: str, pat: re.Pattern | None) -> str:
    """`text` with every special-token match rewritten until none is left;
    `text` itself when nothing matches."""
    if pat is None or pat.search(text) is None:
        return text
    while True:
        text, more = _one_pass(text, pat)
        if not more:
            return text


def question_text(instructions: Any, kind: str, options: list[dict]) -> str:
    """The question block. `options`: the listed options in prompt order,
    each `{"code", "label", "meaning"}`."""
    lines = ["Question:", value_text(instructions), "Options:"]
    for o in options:
        if kind == "choice":
            body = (
                o["label"]
                if o["meaning"] is None
                else f"{o['label']}: {value_text(o['meaning'])}"
            )
        else:
            body = value_text(o["meaning"])
        lines.append(f"{o['code']}: {body}")
    return "\n".join(lines)


def render_chunked(
    prefix: str, state: str, question: str, pat: re.Pattern | None
) -> str:
    """The prompt text (the mask token is appended as an id)."""
    return (
        prefix
        + "State:\n"
        + neutralize(state, pat)
        + STATE_SEP
        + neutralize(question, pat)
    )
