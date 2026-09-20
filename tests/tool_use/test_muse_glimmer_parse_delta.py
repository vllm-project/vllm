# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""parse_delta-level regression tests for the MuseGlimmer parsers.

These drive the UNIFIED parser-engine streaming API the serving layer actually
uses (``DelegatingParser.parse_delta``), NOT the ``extract_*_streaming`` methods
directly. This is the coverage that was missing when a live no-tools streaming
request leaked raw harmony framing into ``content``: the phase machine marked
``reasoning_ended=True`` at the prompt boundary (because ``is_reasoning_end`` on
a prompt with no ``to=self`` wrongly returned True), skipping the reasoning
phase for the whole generation.

Requires a real MuseGlimmer tokenizer; skipped if the checkpoint is unavailable.
"""

import json
import os

import pytest

CKPT = os.environ.get("MUSE_GLIMMER_CKPT", "")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(CKPT), reason=f"MuseGlimmer checkpoint not found at {CKPT}"
)

_FRAMING = [
    "<|start|>",
    "<|message|>",
    "<|eom|>",
    "<|eot|>",
    "to=self",
    "to=user",
    "<atem:",
]

PROMPT = "<|start|>user<|message|>hi<|eom|><|start|>assistant"


class _Req:
    tools = None
    tool_choice = None
    include_reasoning = True


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)


@pytest.fixture(scope="module")
def parser_cls():
    from vllm.parser import ParserManager

    return ParserManager.get_parser(
        tool_parser_name="muse_glimmer",
        reasoning_parser_name="muse_glimmer",
        enable_auto_tools=True,
    )


def _drive(parser_cls, tok, gen_text, req=None):
    """Feed gen_text token-by-token through parse_delta (as serving does)."""
    from vllm.parser.abstract_parser import StreamState

    if req is None:
        req = _Req()
    parser = parser_cls(tok)
    parser._stream_state = StreamState()
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    gen_ids = tok.encode(gen_text, add_special_tokens=False)
    reasoning, content, tools = [], [], []
    for i, tid in enumerate(gen_ids):
        dm = parser.parse_delta(
            tok.decode([tid]),
            [tid],
            req,
            prompt_token_ids=prompt_ids if i == 0 else None,
            finished=(i == len(gen_ids) - 1),
        )
        if dm is None:
            continue
        if getattr(dm, "reasoning", None):
            reasoning.append(dm.reasoning)
        if getattr(dm, "content", None):
            content.append(dm.content)
        for tc in getattr(dm, "tool_calls", None) or []:
            fn = tc.function
            name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
            args = (
                fn.get("arguments")
                if isinstance(fn, dict)
                else getattr(fn, "arguments", None)
            )
            tools.append((tc.index, name, args))
    return "".join(reasoning), "".join(content), tools


def _assert_no_framing(text):
    for f in _FRAMING:
        assert f not in text, f"framing {f!r} leaked: {text!r}"


def test_parse_delta_no_tools_reasoning_then_answer(parser_cls, tok):
    # The core regression: reasoning phase must stay active for a no-tools turn.
    reasoning, content, tools = _drive(
        parser_cls,
        tok,
        " to=self<|message|>Let me think step by step about the sum.<|eom|>"
        "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>",
    )
    _assert_no_framing(content)
    assert reasoning == "Let me think step by step about the sum.", repr(reasoning)
    assert content == "The answer is 42.", repr(content)
    assert tools == []


def test_parse_delta_content_only(parser_cls, tok):
    reasoning, content, tools = _drive(
        parser_cls,
        tok,
        " to=user<|message|>Just a direct answer.<|eot|>",
    )
    _assert_no_framing(content)
    assert content == "Just a direct answer.", repr(content)
    assert tools == []


def test_parse_delta_tool_call(parser_cls, tok):
    reasoning, content, tools = _drive(
        parser_cls,
        tok,
        " to=self<|message|>I should read the hostname.<|eom|>"
        "<|start|>assistant to=read.read<|message|>"
        '<atem:function_calls>\n<atem:invoke name="read.read">\n'
        '<atem:parameter name="path">/etc/hostname</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>",
    )
    _assert_no_framing(content)
    assert reasoning == "I should read the hostname.", repr(reasoning)
    assert len(tools) == 1, tools
    idx, name, args = tools[0]
    assert idx == 0 and name == "read.read"
    assert json.loads(args) == {"path": "/etc/hostname"}


def test_parse_delta_truncated_cot_no_toolcall(parser_cls, tok):
    # Contemplated invoke inside an unterminated to=self block: no tool call,
    # no framing leak into content, partial reasoning recovered.
    reasoning, content, tools = _drive(
        parser_cls,
        tok,
        " to=self<|message|>Maybe I should call "
        '<atem:function_calls>\n<atem:invoke name="read.read">\n'
        '<atem:parameter name="path">/etc/hostname</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls> but wait",
    )
    _assert_no_framing(content)
    assert tools == [], f"contemplated invoke leaked as tool call: {tools}"
    assert "Maybe I should call" in reasoning, repr(reasoning)


def test_parse_delta_reasoning_suppressed_when_not_requested(parser_cls, tok):
    class _NoReasonReq:
        tools = None
        tool_choice = None
        include_reasoning = False

    reasoning, content, tools = _drive(
        parser_cls,
        tok,
        " to=self<|message|>secret thoughts<|eom|>"
        "<|start|>assistant to=user<|message|>Public answer.<|eot|>",
        req=_NoReasonReq(),
    )
    assert reasoning == "", repr(reasoning)  # suppressed
    assert content == "Public answer.", repr(content)
    _assert_no_framing(content)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
