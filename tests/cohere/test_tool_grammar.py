# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

from test_const import C2_grammar, C3_grammar, tool_schema_1

from vllm.cohere.guided_decoding.tool_grammar import collect_tool_schema_v2
from vllm.reasoning.cohere_command_reasoning_parser import collect_tool_schema

st = time.time()


def test_tools_rule_has_no_ambiguous_adjacent_whitespace() -> None:
    """Regression for #849.

    The top-level tool-call array rule must not place two nullable ``ws`` stars
    adjacent. The old form ``tool ws ("," ws tool)* ws "]"`` collapses to
    ``tool ws ws "]"`` for a single tool call, which lets a whitespace run be
    partitioned arbitrarily between the two ``ws*``. That made xgrammar's Earley
    parser accumulate ambiguous states and turned ``fill_next_token_bitmask``
    into an O(n^2) blowup on long whitespace runs, wedging the serving replica.

    Both EBNF builders must stay fixed: ``collect_tool_schema_v2`` (the
    guided-decoding path) and ``collect_tool_schema`` (the live chat
    structural-tag path used in prod).
    """
    expected = 'tools ::= ws "[" ws tool (ws "," ws tool)* ws "]" ws'

    def _check(grammar: str, label: str) -> None:
        assert expected in grammar, (label, grammar)
        # The ambiguous adjacent-``ws`` forms must be gone.
        assert 'tool ws ("," ws tool)*' not in grammar, label
        assert '("," ws tool)*  ws "]"' not in grammar, label

    for arch in ("CohereForCausalLM", "Cohere2ForCausalLM"):
        _check(collect_tool_schema_v2(arch, tool_schema_1), f"v2[{arch}]")
    # Architecture-independent prod builder used by the structural-tag path.
    _check(collect_tool_schema(tool_schema_1), "collect_tool_schema")


def validate_tool_grammar() -> None:
    response_schema_c3 = collect_tool_schema_v2("Cohere2ForCausalLM", tool_schema_1)
    response_schema_c2 = collect_tool_schema_v2("CohereForCausalLM", tool_schema_1)
    assert response_schema_c3 == C3_grammar
    assert response_schema_c2 == C2_grammar

    print("Tool grammar validation passed.")


if __name__ == "__main__":
    validate_tool_grammar()
