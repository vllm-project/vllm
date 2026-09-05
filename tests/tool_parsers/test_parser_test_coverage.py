# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coverage ratchet for the per-parser tool-calling tests.

Tool parsers are added per model, and nothing currently requires a new one to
arrive with tests. Coverage therefore drifts down silently: a parser registers,
works for the model it was written against, and no test fails when a later
refactor changes its behaviour.

This does not demand tests for every parser today, which would be unlandable. It
makes the gap explicit and non-regressing: every registered parser must be either
mapped to a test module or listed in PENDING_COVERAGE with a reason, and one that
is neither fails. A parser added without doing either is the case this catches.

The mapping is deliberately hand-maintained. No single automated signal is
reliable here: matching on the test filename, on the implementation class, and on
the registered name each disagree, because some modules resolve a parser through
``ToolParserManager.get_tool_parser("name")`` while others import the class
directly, and one module covers several parsers. A decision per parser is the
point, not overhead.
"""

import os

import pytest

from vllm.tool_parsers import ToolParserManager

TESTS_DIR = "tests/tool_parsers"

# Registered parser name -> the test module exercising it.
PARSER_TEST_MODULES: dict[str, str] = {
    "apertus": "test_apertus_tool_parser.py",
    "deepseek_v3": "test_deepseekv3_tool_parser.py",
    "deepseek_v31": "test_deepseekv31_tool_parser.py",
    "deepseek_v32": "test_deepseekv32_tool_parser.py",
    "deepseek_v4": "test_deepseekv4_tool_parser.py",
    "ernie45": "test_ernie45_moe_tool_parser.py",
    "functiongemma": "test_functiongemma_tool_parser.py",
    "gemma4": "test_gemma4_tool_parser.py",
    "gigachat3": "test_gigachat3_tool_parser.py",
    "glm45": "test_glm47_moe_tool_parser.py",
    "glm47": "test_glm47_moe_tool_parser.py",
    "granite": "test_granite_tool_parser.py",
    "granite-20b-fc": "test_granite_20b_fc_tool_parser.py",
    "granite4": "test_granite4_tool_parser.py",
    "hermes": "test_hermes_tool_parser.py",
    "hunyuan_a13b": "test_hunyuan_a13b_tool_parser.py",
    "hy_v3": "test_hy_v3_tool_parser.py",
    "internlm": "test_internlm2_tool_parser.py",
    "jamba": "test_jamba_tool_parser.py",
    "kimi_k2": "test_kimi_k2_tool_parser.py",
    "kimi_k3": "test_kimi_k3_named_tool_choice.py",
    "lfm2": "test_lfm2_tool_parser.py",
    "llama3_json": "test_llama3_json_tool_parser.py",
    "llama4_json": "test_llama3_json_tool_parser.py",
    "llama4_pythonic": "test_llama4_pythonic_tool_parser.py",
    "longcat": "test_longcat_tool_parser.py",
    "mimo": "test_qwen3coder_tool_parser.py",
    "minicpm5": "test_minicpm5xml_tool_parser.py",
    "minimax_m2": "test_minimax_m2_tool_parser.py",
    "minimax_m3": "test_minimax_m3_tool_parser.py",
    "olmo3": "test_olmo3_tool_parser.py",
    "phi4_mini_json": "test_phi4mini_tool_parser.py",
    "poolside_v1": "test_poolside_v1_tool_parser.py",
    "pythonic": "test_pythonic_tool_parser.py",
    "qwen3_coder": "test_qwen3coder_tool_parser.py",
    "qwen3_xml": "test_qwen3coder_tool_parser.py",
    "step3": "test_step3_tool_parser.py",
    "step3p5": "test_step3p5_tool_parser.py",
    "xlam": "test_xlam_tool_parser.py",
}

# Registered parsers with no test module, recorded so the gap is a decision and
# not an oversight. Entries may only be REMOVED, by adding tests.
PENDING_COVERAGE: dict[str, str] = {
    "cohere_command3": "shares cohere_command_tool_parser.py; no test module",
    "cohere_command4": "shares cohere_command_tool_parser.py; no test module",
    "inkling": "newly registered engine parser; no tests yet",
    "ling3": "no test module for the Ling3 parser",
    "mistral": "no dedicated test module for MistralToolParser",
    "openai": "GptOssToolParser (Harmony path) has no dedicated test module",
    "seed_oss": "no test module for the seed-oss engine parser",
}

# Floor on mapped parsers. Raise it when coverage improves; lowering it must be a
# deliberate edit with a reason in the commit message.
COVERAGE_FLOOR = 39


def _registered() -> set[str]:
    return set(ToolParserManager.list_registered())


def test_every_registered_parser_is_classified():
    """A new parser must have tests, or be explicitly recorded as pending."""
    unclassified = _registered() - set(PARSER_TEST_MODULES) - set(PENDING_COVERAGE)
    assert not unclassified, (
        f"tool parsers registered but not classified: {sorted(unclassified)}. Add "
        "tests and map the parser in PARSER_TEST_MODULES, or record it in "
        "PENDING_COVERAGE with a reason so the gap is explicit."
    )


@pytest.mark.parametrize(("parser", "module"), sorted(PARSER_TEST_MODULES.items()))
def test_mapped_test_module_exists(parser: str, module: str):
    """Deleting a parser's tests must fail here, not quietly reduce coverage."""
    assert os.path.isfile(os.path.join(TESTS_DIR, module)), (
        f"{parser} is mapped to {module}, which does not exist in {TESTS_DIR}/. "
        "Update PARSER_TEST_MODULES or restore the tests."
    )


def test_classification_is_not_stale():
    """Classified names must still be registered, and sit in exactly one bucket."""
    registered = _registered()
    for bucket, names in (
        ("PARSER_TEST_MODULES", set(PARSER_TEST_MODULES)),
        ("PENDING_COVERAGE", set(PENDING_COVERAGE)),
    ):
        stale = names - registered
        assert not stale, f"{bucket} names unregistered parsers: {sorted(stale)}"
    overlap = set(PARSER_TEST_MODULES) & set(PENDING_COVERAGE)
    assert not overlap, f"{sorted(overlap)} appear in both buckets; keep one"


def test_coverage_does_not_regress():
    """A floor, not a target: coverage drops only by a deliberate edit here."""
    assert len(PARSER_TEST_MODULES) >= COVERAGE_FLOOR, (
        f"mapped parsers dropped to {len(PARSER_TEST_MODULES)}, below the floor "
        f"of {COVERAGE_FLOOR}. If intended, lower the floor in the same commit "
        "and say why."
    )


def test_report_coverage(capsys):
    """Emit the coverage number so it can be tracked release over release."""
    registered = _registered()
    covered = len(PARSER_TEST_MODULES)
    with capsys.disabled():
        print(
            f"\ntool-parser test coverage: {covered}/{len(registered)} "
            f"({round(100 * covered / len(registered))}%), "
            f"{len(PENDING_COVERAGE)} pending"
        )
