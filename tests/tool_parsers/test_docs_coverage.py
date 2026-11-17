# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard test ensuring registered tool and reasoning parsers stay documented."""

import ast
from pathlib import Path

import pytest
import regex as re

ROOT = Path(__file__).resolve().parents[2]

TOOL_REGISTRY_PATH = ROOT / "vllm/tool_parsers/__init__.py"
REASONING_REGISTRY_PATH = ROOT / "vllm/reasoning/__init__.py"
TOOL_DOCS_PATH = ROOT / "docs/features/tool_calling.md"
REASONING_DOCS_PATH = ROOT / "docs/features/reasoning_outputs.md"

# Aliases whose primary parser is already documented in tool_calling.md.
TOOL_PARSER_ALIASES: frozenset[str] = frozenset(
    {
        "llama4_json",  # alias of llama3_json
        "qwen3_json",  # temporary alias of hermes
    }
)

# Registered tool parsers awaiting Batch 2 documentation sections.
# Shrink-only: remove a name from this set once documented in tool_calling.md.
UNDOCUMENTED_TOOL_PARSERS: frozenset[str] = frozenset(
    {
        "dots",
        "ernie45",
        "hy_v3",
        "hy_v4",
        "inkling",
        "k2_horizon",
        "ling3",
        "minicpm5",
        "plamo3",
        "poolside_v1",
        "step3",
        "step5",
    }
)

# Aliases whose primary parser is already documented in reasoning_outputs.md.
REASONING_PARSER_ALIASES: frozenset[str] = frozenset(
    {
        "musespark",  # alias of muse
        "step5",  # alias of deepseek_r1
    }
)

# Registered reasoning parsers awaiting Batch 2 table rows.
# Shrink-only: remove a name from this set once documented in reasoning_outputs.md.
UNDOCUMENTED_REASONING_PARSERS: frozenset[str] = frozenset(
    {
        "hy_v3",
        "hy_v4",
        "inkling",
        "k2_horizon",
        "ling3",
        "minicpmv",
        "plamo3",
        "poolside_v1",
        "step3",
    }
)

pytestmark = pytest.mark.cpu_test


def _load_registry_keys(path: Path, dict_name: str) -> set[str]:
    """Extract registered parser keys from a registry dict via AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == dict_name:
                    mapping = ast.literal_eval(node.value)
                    return set(mapping.keys())
    raise AssertionError(f"Could not find {dict_name} in {path}")


def _documented_tool_parsers(docs_text: str, registered: set[str]) -> set[str]:
    """Return registered tool parser names referenced in `tool_calling.md`."""
    return {
        name
        for name in registered
        if re.search(
            # `-` is not a word boundary: `granite-20b-fc` must not document `granite`.
            rf"(`{re.escape(name)}`|--tool-call-parser {re.escape(name)}(?![\w-]))",
            docs_text,
        )
    }


def _documented_reasoning_parsers(docs_text: str) -> set[str]:
    """Return parser names listed in `reasoning_outputs.md`'s Supported Models table."""
    section_match = re.search(
        r"^## Supported Models\b.*?(?=^## |\Z)",
        docs_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert section_match is not None, "Missing '## Supported Models' section"

    documented: set[str] = set()
    for line in section_match.group(0).splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if (
            len(cells) < 2
            or cells[1] in ("Parser Name", "---")
            or set(cells[1]) <= {"-"}
        ):
            continue
        documented.update(re.findall(r"`([^`]+)`", cells[1]))
    return documented


def test_tool_parsers_documented() -> None:
    """Fail when newly registered tool parsers are missing from `tool_calling.md`."""
    registered = _load_registry_keys(TOOL_REGISTRY_PATH, "_TOOL_PARSERS_TO_REGISTER")
    documented = _documented_tool_parsers(
        TOOL_DOCS_PATH.read_text(encoding="utf-8"), registered
    )
    allowlisted = TOOL_PARSER_ALIASES | UNDOCUMENTED_TOOL_PARSERS

    missing = sorted(registered - documented - allowlisted)
    assert not missing, (
        "Registered tool parsers missing from docs/features/tool_calling.md: "
        f"{missing}. Document them or add aliases to TOOL_PARSER_ALIASES."
    )

    stale = sorted(allowlisted - registered)
    assert not stale, (
        "Allowlisted tool parsers are no longer registered in "
        f"vllm/tool_parsers/__init__.py: {stale}"
    )

    newly_documented = sorted(allowlisted & documented)
    assert not newly_documented, (
        "Tool parsers are now documented in docs/features/tool_calling.md and must "
        f"be removed from the shrink-only allowlist: {newly_documented}"
    )


def test_reasoning_parsers_documented() -> None:
    """Fail when reasoning parsers are missing from `reasoning_outputs.md`."""
    registered = _load_registry_keys(
        REASONING_REGISTRY_PATH, "_REASONING_PARSERS_TO_REGISTER"
    )
    documented = _documented_reasoning_parsers(
        REASONING_DOCS_PATH.read_text(encoding="utf-8")
    )
    allowlisted = REASONING_PARSER_ALIASES | UNDOCUMENTED_REASONING_PARSERS

    missing = sorted(registered - documented - allowlisted)
    assert not missing, (
        "Registered reasoning parsers missing from Supported Models table in "
        f"docs/features/reasoning_outputs.md: {missing}. "
        "Add a table row or add aliases to REASONING_PARSER_ALIASES."
    )

    stale = sorted(allowlisted - registered)
    assert not stale, (
        "Allowlisted reasoning parsers are no longer registered in "
        f"vllm/reasoning/__init__.py: {stale}"
    )

    newly_documented = sorted(allowlisted & documented)
    assert not newly_documented, (
        "Reasoning parsers are now documented in docs/features/reasoning_outputs.md "
        f"and must be removed from the shrink-only allowlist: {newly_documented}"
    )


def _heading_section(text: str, heading: str) -> str:
    lines = text.splitlines(keepends=True)
    start = next(
        (i for i, line in enumerate(lines) if line.startswith(f"### {heading}")),
        None,
    )
    assert start is not None, f"Missing section '### {heading}'"
    end = next(
        (j for j in range(start + 1, len(lines)) if lines[j].startswith("### ")),
        len(lines),
    )
    return "".join(lines[start:end])


def test_tool_parser_flag_match_stops_at_hyphen() -> None:
    """A longer `--tool-call-parser` flag must not document a shorter name."""
    docs = "Flags: `--tool-call-parser granite-20b-fc`\n"
    found = _documented_tool_parsers(docs, {"granite", "granite-20b-fc"})
    assert found == {"granite-20b-fc"}

    exact = "--tool-call-parser granite\n`granite`\n"
    found_exact = _documented_tool_parsers(exact, {"granite", "granite-20b-fc"})
    assert found_exact == {"granite"}


def test_kimi_k3_required_tool_choice_docs() -> None:
    section = _heading_section(
        TOOL_DOCS_PATH.read_text(encoding="utf-8"), "Kimi-K3 Models"
    )
    assert "are rejected when it is disabled" not in section
    assert "Named `tool_choice`" in section
    assert "is not rejected" in section
    assert '"required"' in section


def test_deepseek_v4_default_does_not_contradict_later_section() -> None:
    text = REASONING_DOCS_PATH.read_text(encoding="utf-8")
    note = text.split("## Supported Models", 1)[1].split("## Quickstart", 1)[0]
    assert "implicit `reasoning_effort` of `high`" in note
    assert "omitting the field is not equivalent" in note
    assert "DeepSeek-V4 / V4.1" not in note
    assert "does not use DeepSeek-V4" in note
    auto = text.split("## Automatic `enable_thinking` Activation", 1)[1]
    paragraphs = [part.strip() for part in auto.split("\n\n") if part.strip()]
    first_para = paragraphs[0]
    assert first_para.startswith("Some models")
    assert "DeepSeek-V4" not in first_para
    assert "Gemma 4" in first_para
    assert "IBM Granite 3.2" in first_para


def test_minimax_m2_append_think_is_distinguished() -> None:
    text = REASONING_DOCS_PATH.read_text(encoding="utf-8")
    note = text.split("## Supported Models", 1)[1].split("## Quickstart", 1)[0]
    assert "`minimax_m2_append_think` leaves `reasoning` empty" in note
    assert "prepends `<think>` to `content`" in note
    assert "`minimax_m2` puts text before `</think>` in `reasoning`" in note


def test_minimax_m3_documents_thinking_mode_enabled() -> None:
    tool = _heading_section(
        TOOL_DOCS_PATH.read_text(encoding="utf-8"), "MiniMax-M3 Models"
    )
    assert 'thinking_mode="enabled"' in tool
    assert "prefilled" in tool
    reasoning = REASONING_DOCS_PATH.read_text(encoding="utf-8")
    note = reasoning.split("## Supported Models", 1)[1].split("## Quickstart", 1)[0]
    assert 'thinking_mode="enabled"' in note


def test_lfm2_lists_current_model_ids() -> None:
    section = _heading_section(
        TOOL_DOCS_PATH.read_text(encoding="utf-8"),
        "LiquidAI LFM2 and LFM2.5 Models",
    )
    assert "LFM2-8B-A1B-preview" not in section
    assert re.search(r"`LiquidAI/LFM2-8B-A1B`(?!-)", section)
    assert "`LiquidAI/LFM2.5-8B-A1B`" in section
    assert "`LiquidAI/LFM2-1.2B-Tool`" in section
    assert "`LiquidAI/LFM2-1.2B`" in section


def test_gemma4_skip_special_tokens_does_not_emit_syntax() -> None:
    section = _heading_section(
        TOOL_DOCS_PATH.read_text(encoding="utf-8"), "Gemma 4 Models"
    )
    assert "so the model emits native syntax" not in section
    assert "skips JSON guided decoding" in section
    assert "detokenized text the parser reads" in section


def test_nemotron_row_names_qwen3_coder_tool_parser() -> None:
    text = REASONING_DOCS_PATH.read_text(encoding="utf-8")
    row = next(
        line
        for line in text.splitlines()
        if line.startswith("|") and "`nemotron_v3`" in line
    )
    assert "`qwen3_coder`" in row
    note = text.split("## Supported Models", 1)[1].split("## Quickstart", 1)[0]
    assert "--tool-call-parser qwen3_coder" in note
    assert "no `nemotron_v3` tool parser" in note
