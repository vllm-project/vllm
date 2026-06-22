# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CI-collected regression for the tool-call grammar whitespace ambiguity (#849).

Both EBNF builders must emit exactly one nullable ``ws`` between adjacent tokens
in the top-level ``tools`` array rule. The old form
``tool ws ("," ws tool)* ws "]"`` places two nullable ``ws*`` adjacent for the
single-tool case, which makes xgrammar's Earley parser accumulate ambiguous
states and turns ``fill_next_token_bitmask`` into an O(n^2) blowup that wedges
the serving replica.

This test guards **both** builders: ``collect_tool_schema_v2`` (guided-decoding
path) and ``collect_tool_schema`` (the live chat structural-tag path used in
prod). A regression in either one turns CI red.
"""

from __future__ import annotations

from vllm.cohere.guided_decoding.tool_grammar import collect_tool_schema_v2
from vllm.reasoning.cohere_command_reasoning_parser import collect_tool_schema

_EXPECTED = 'tools ::= ws "[" ws tool (ws "," ws tool)* ws "]" ws'
# The two adjacent-``ws*`` forms that re-introduce the #849 ambiguity.
_AMBIGUOUS_FORMS = ('tool ws ("," ws tool)*', '("," ws tool)*  ws "]"')

_TOOL_SCHEMA = [
    {
        "name": "add_numbers",
        "description": "adds two integers",
        "parameters": {
            "type": "object",
            "properties": {
                "num1": {"type": "number"},
                "num2": {"type": "number"},
            },
            "required": ["num1", "num2"],
        },
    }
]


def _assert_unambiguous(grammar: str, label: str) -> None:
    assert _EXPECTED in grammar, (label, grammar)
    for form in _AMBIGUOUS_FORMS:
        assert form not in grammar, (label, form)


def test_guided_decoding_builder_has_no_ambiguous_adjacent_whitespace() -> None:
    for arch in ("CohereForCausalLM", "Cohere2ForCausalLM"):
        grammar = collect_tool_schema_v2(arch, _TOOL_SCHEMA)
        _assert_unambiguous(grammar, f"collect_tool_schema_v2[{arch}]")


def test_structural_tag_builder_has_no_ambiguous_adjacent_whitespace() -> None:
    # The live chat structural-tag / prod path. Architecture-independent.
    grammar = collect_tool_schema(_TOOL_SCHEMA)
    _assert_unambiguous(grammar, "collect_tool_schema")
