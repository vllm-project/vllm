# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dependency-free conformance checks that pin each parser's documented wire format.

Complements ``common_tests.py``: that suite drives each parser through a broad
behavioural
matrix (parallel calls, data types, malformed input, streaming) using a **real
tokenizer**,
so it needs the model's tokenizer files to be fetchable. This module instead pins the
exact
documented raw bytes for a family and asserts the registered parser still extracts them,
using a mock tokenizer — so it runs anywhere, needs no download, no GPU and no model,
and
stays fast enough to be unconditional.

That makes it useful in two places the behavioural suite cannot reach:

* environments where tokenizer downloads are unavailable or undesirable in CI;
* parsers with no dedicated test module — ``qwen3_xml`` is registered but has no
  ``test_*_tool_parser.py`` of its own, so its format is otherwise unpinned.

Fixtures are authored from each parser's own source, not guessed, and ``raw_output`` is
byte-faithful to the documented format. A family whose parser is not registered in the
running build is skipped (version skew), never failed.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import regex as re

from vllm.tool_parsers import ToolParserManager

Call = tuple[str, dict]


@dataclass(frozen=True)
class FormatFixture:
    family: str
    parser: str  # registered --tool-call-parser name
    raw_output: str  # the exact documented wire format
    expected: tuple[Call, ...]  # calls a correct parser must extract
    markers: tuple[str, ...] = ()  # substrings the documented format must contain
    pattern: str | None = None  # or a regex the whole format must match (pythonic)
    doc_ref: str = ""
    # Tokens that must be PRESENT in the mock vocab. Some parsers branch on vocab
    # membership (mistral picks its pre-/post-v11 format by whether "[ARGS]" exists),
    # so this selects which code path the fixture exercises.
    vocab: tuple[str, ...] = ()


# Pinned from each parser's own tokens/regex in
# vllm/tool_parsers/<family>_tool_parser.py --
# authored from the source, not guessed; raw_output is byte-faithful to the documented
# format.
FIXTURES: tuple[FormatFixture, ...] = (
    FormatFixture(
        "hermes",
        "hermes",
        (
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": '
            '"SF"}}\n</tool_call>'
        ),
        (("get_weather", {"city": "SF"}),),
        markers=("<tool_call>", "</tool_call>"),
        doc_ref="hermes_tool_parser.py: <tool_call>{json name+arguments}</tool_call>",
    ),
    FormatFixture(
        "pythonic",
        "pythonic",
        '[get_weather(city="SF", units="metric")]',
        (("get_weather", {"city": "SF", "units": "metric"}),),
        pattern=r"^\s*\[\s*[A-Za-z_]\w*\(.*\)\s*\]\s*$",
        doc_ref="pythonic_tool_parser.py: [func(k=v, ...)] python-call syntax",
    ),
    FormatFixture(
        "mistral",
        "mistral",
        '[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "SF"}}]',
        (("get_weather", {"city": "SF"}),),
        markers=("[TOOL_CALLS]",),
        doc_ref="mistral_tool_parser.py: [TOOL_CALLS] + json array (pre-v11 tokenizer)",
    ),
    # Post-v11 mistral is a DIFFERENT wire format selected by the tokenizer, not by the
    # parser name: _is_pre_v11_tokeniser() returns "[ARGS]" not in vocab, so seeding the
    # vocab with "[ARGS]" forces the >= v11 path. Without this fixture the empty mock
    # vocab always selects pre-v11 and the post-v11 branch is never exercised.
    FormatFixture(
        "mistral_v11",
        "mistral",
        '[TOOL_CALLS]get_weather{"city": "SF"}',
        (("get_weather", {"city": "SF"}),),
        markers=("[TOOL_CALLS]",),
        doc_ref=(
            "mistral_tool_parser.py: [TOOL_CALLS]name{args} per call (v11+ tokenizer)"
        ),
        vocab=("[ARGS]",),
    ),
    FormatFixture(
        "qwen3_xml",
        "qwen3_xml",
        (
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nSF\n</parameter>\n"
            "</function>\n</tool_call>"
        ),
        (("get_weather", {"city": "SF"}),),
        markers=(
            "<tool_call>",
            "<function=",
            "<parameter=",
            "</function>",
            "</tool_call>",
        ),
        doc_ref="qwen3_engine_tool_parser.py: <tool_call><function=..><parameter=..>",
    ),
    FormatFixture(
        "kimi_k2",
        "kimi_k2",
        (
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0\n"
            '<|tool_call_argument_begin|>{"city": '
            '"SF"}<|tool_call_end|><|tool_calls_section_end|>'
        ),
        (("get_weather", {"city": "SF"}),),
        markers=(
            "<|tool_call_begin|>",
            "<|tool_call_argument_begin|>",
            "<|tool_call_end|>",
        ),
        doc_ref=(
            "kimi_k2_tool_parser.py: <|tool_call_begin|>functions.NAME:IDX "
            "<|arg|>{json}<|end|>"
        ),
    ),
    FormatFixture(
        "deepseek_v3",
        "deepseek_v3",
        (
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function"
            "<｜tool▁sep｜>get_weather\n"
            '```json\n{"city": "SF"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        ),
        (("get_weather", {"city": "SF"}),),
        markers=("<｜tool▁calls▁begin｜>", "<｜tool▁sep｜>", "<｜tool▁call▁end｜>"),
        doc_ref=(
            "deepseekv3_tool_parser.py: "
            "<|tool_call_begin|>type<|tool_sep|>name\\n```json\\n{args}\\n```"
        ),
    ),
    FormatFixture(
        "minimax_m2",
        "minimax_m2",
        (
            '<minimax:tool_call>\n<invoke name="get_weather">\n'
            '<parameter name="city">SF</parameter>\n</invoke>\n</minimax:tool_call>'
        ),
        (("get_weather", {"city": "SF"}),),
        markers=("<minimax:tool_call>", "<invoke name=", "<parameter name="),
        doc_ref=(
            "minimax_m2_tool_parser.py: <minimax:tool_call><invoke "
            "name=..><parameter name=..>"
        ),
    ),
    FormatFixture(
        "glm4_moe",
        "glm45",
        "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>SF</arg_value>\n</tool_call>",
        (("get_weather", {"city": "SF"}),),
        markers=("<tool_call>", "<arg_key>", "<arg_value>", "</tool_call>"),
        doc_ref=(
            "glm47_moe_tool_parser.py (registered 'glm45'): "
            "<tool_call>name<arg_key>k<arg_value>v"
        ),
    ),
    # Deliberately the same wire format as qwen3_xml: both parsers implement the same
    # XML
    # grammar. The fixtures do not distinguish the two parsers -- parser IDENTITY is
    # what
    # ToolParserManager.get_tool_parser(name) resolves; this pins the FORMAT for each
    # name.
    FormatFixture(
        "qwen3_coder",
        "qwen3_coder",
        (
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nSF\n</parameter>\n"
            "</function>\n</tool_call>"
        ),
        (("get_weather", {"city": "SF"}),),
        markers=(
            "<tool_call>",
            "<function=",
            "<parameter=",
            "</function>",
            "</tool_call>",
        ),
        doc_ref=(
            "qwen3_engine_tool_parser.py: <tool_call><function=..><parameter=..> "
        ),
    ),
    FormatFixture(
        "deepseek_v4",
        "deepseek_v4",
        (
            '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">SF</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        ),
        (("get_weather", {"city": "SF"}),),
        markers=(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=",
            "<｜DSML｜parameter name=",
        ),
        doc_ref=(
            "deepseekv4_engine_tool_parser.py: DSML invoke/parameter wrapped in "
            "<|DSML|tool_calls>"
        ),
    ),
)


class _FakeVocab(dict):
    """A ``get_vocab()`` result that yields an id for ANY token looked up with
    ``.get()``.

    Lets parsers that resolve special-token *ids* at construction (mistral, deepseek)
    build without a real tokenizer. The ids are arbitrary and only consistent *within a
    single process* (``hash`` is randomised per run by ``PYTHONHASHSEED``), which is all
    these tests need -- nothing here asserts on an id value.

    Membership (``in``) is deliberately NOT faked: it reports only the tokens seeded via
    ``FormatFixture.vocab``, so a parser that branches on vocab membership takes a
    predictable path.
    """

    def get(self, key, default=None):
        v = dict.get(self, key, None)
        return v if v is not None else abs(hash(key)) % 100000


class _MockTokenizer:
    """Truthy, non-Mistral tokenizer; no model needed.

    Non-streaming ``extract_tool_calls`` does not tokenize, so the fake vocab only has
    to
    satisfy construction-time special-token-id lookups. ``present`` seeds real
    membership
    for parsers that branch on it.
    """

    def __init__(self, present: tuple[str, ...] = ()):
        self._present = {tok: i for i, tok in enumerate(present)}

    def get_vocab(self):
        return _FakeVocab(self._present)

    def convert_tokens_to_ids(self, t):
        if isinstance(t, str):
            return abs(hash(t)) % 100000
        return [abs(hash(x)) % 100000 for x in t]

    def __bool__(self):
        return True


def _load(args) -> dict:
    return json.loads(args) if isinstance(args, str) else dict(args)


@pytest.mark.parametrize("fx", FIXTURES, ids=lambda fx: fx.family)
def test_documented_format_is_extracted(fx: FormatFixture):
    """The registered parser extracts the pinned documented format into the expected
    calls."""
    try:
        parser_cls = ToolParserManager.get_tool_parser(fx.parser)
    except Exception:
        pytest.skip(f"parser {fx.parser!r} not registered in this vLLM build")
    parser = parser_cls(_MockTokenizer(fx.vocab))
    request = SimpleNamespace(tools=None, tool_choice=None)
    info = parser.extract_tool_calls(fx.raw_output, request=request)
    assert getattr(info, "tools_called", False), (
        f"{fx.family}: parser reported no tool call"
    )
    got = [(tc.function.name, _load(tc.function.arguments)) for tc in info.tool_calls]
    want = [(name, dict(args)) for name, args in fx.expected]
    assert got == want


@pytest.mark.parametrize("fx", FIXTURES, ids=lambda fx: fx.family)
def test_doc_ref_names_the_real_implementation_module(fx: FormatFixture):
    """`doc_ref` points a reader at where the format is defined, so it has to stay true.

    Filenames drift (parsers get renamed and consolidated), and a stale pointer is worse
    than none in a suite whose purpose is to be an authoritative format reference — so
    the
    reference is checked rather than trusted.
    """
    try:
        parser_cls = ToolParserManager.get_tool_parser(fx.parser)
    except Exception:
        pytest.skip(f"parser {fx.parser!r} not registered in this vLLM build")
    module = Path(inspect.getfile(parser_cls)).name
    assert module in fx.doc_ref, (
        f"{fx.family}: doc_ref should name {module} (the module implementing "
        f"{fx.parser!r}), got: {fx.doc_ref!r}"
    )


@pytest.mark.parametrize("fx", FIXTURES, ids=lambda fx: fx.family)
def test_fixture_matches_documented_markers(fx: FormatFixture):
    """Offline guard on the fixtures themselves (no parser): the pinned raw output
    still carries
    its family's documented format signature, so a fixture typo can't mask a real
    regression."""
    if fx.pattern is not None:
        assert re.match(fx.pattern, fx.raw_output, re.DOTALL), (
            f"{fx.family}: fixture does not match documented pattern {fx.pattern!r}"
        )
    for marker in fx.markers:
        assert marker in fx.raw_output, (
            f"{fx.family}: fixture missing documented marker {marker!r}"
        )


def test_harmony_recipient_to_name():
    """gpt-oss/Harmony is not a raw-text format (its parser needs token_ids + the
    encoding); what
    is offline-checkable is the recipient->name logic (``functions.NAME``) in the pure
    helpers."""
    try:
        from vllm.entrypoints.openai.parser.harmony_utils import (
            extract_function_from_recipient,
            is_function_recipient,
        )
    except Exception:
        pytest.skip("harmony helpers not importable in this vLLM build")
    cases: tuple[tuple[str, str | None], ...] = (
        ("functions.get_weather", "get_weather"),
        ("functions.list_files", "list_files"),
        ("", None),
        ("<|constrain|>", None),
    )
    for recipient, expected_name in cases:
        is_call = bool(is_function_recipient(recipient))
        assert is_call == (expected_name is not None), (
            f"recipient {recipient!r} classification"
        )
        if expected_name is not None:
            assert extract_function_from_recipient(recipient) == expected_name
