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
            rf"(`{re.escape(name)}`|--tool-call-parser {re.escape(name)}\b)",
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
