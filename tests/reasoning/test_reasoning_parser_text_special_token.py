# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``_reasoning_parser_text``.

When the reasoning end marker (``</think>``) is a *special* token, the
per-request ``skip_special_tokens=True`` strips it from ``output.text``, so the
non-streaming string-based ``extract_reasoning`` can't find the
reasoning/content split. ``_reasoning_parser_text`` reinserts the marker from
the token ids so the split still works.
"""

from vllm.entrypoints.openai.chat_completion.serving import _reasoning_parser_text


class _FakeParser:
    """Minimal stand-in exposing the attributes the helper reads."""

    end_token = "</think>"
    end_token_id = 999


class _FakeTokenizer:
    """Maps sentinel ids back to text; id 999 is the special </think>.

    Decoding with ``skip_special_tokens=True`` drops id 999 (mirrors how vLLM
    strips a special end marker); ``False`` keeps its literal string form.
    """

    _ID_TEXT = {
        1: "reasoning body",
        2: "\n\n",
        3: "answer",
        4: "<|im_end|>",  # EOS special token
    }

    def decode(self, ids, skip_special_tokens=False):
        parts = []
        for i in ids:
            if i == 999:
                if not skip_special_tokens:
                    parts.append("</think>")
            elif i == 4:
                if not skip_special_tokens:
                    parts.append("<|im_end|>")
            else:
                parts.append(self._ID_TEXT.get(i, ""))
        return "".join(parts)


def test_reinserts_stripped_special_end_marker():
    tok = _FakeTokenizer()
    parser = _FakeParser()
    ids = [1, 999, 2, 3, 4]
    # output.text as vLLM produced it with skip_special_tokens=True:
    # </think> and <|im_end|> are gone.
    output_text = "reasoning body\n\nanswer"

    text = _reasoning_parser_text(
        tok, parser, output_text, ids, skip_special_tokens=True
    )
    # Marker reinserted at the split; EOS stays stripped.
    assert text == "reasoning body</think>\n\nanswer"


def test_noop_when_marker_already_present():
    tok = _FakeTokenizer()
    parser = _FakeParser()
    output_text = "reasoning body</think>answer"
    text = _reasoning_parser_text(
        tok, parser, output_text, [1, 999, 3], skip_special_tokens=False
    )
    assert text == output_text


def test_noop_when_marker_id_absent():
    tok = _FakeTokenizer()
    parser = _FakeParser()
    output_text = "no reasoning here"
    text = _reasoning_parser_text(
        tok, parser, output_text, [1, 3], skip_special_tokens=True
    )
    assert text == output_text


def test_noop_when_parser_lacks_end_token():
    tok = _FakeTokenizer()

    class _Bare:
        pass

    output_text = "whatever"
    text = _reasoning_parser_text(
        tok, _Bare(), output_text, [1, 999], skip_special_tokens=True
    )
    assert text == output_text
