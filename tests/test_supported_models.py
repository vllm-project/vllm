# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Docs-vs-registry consistency checks.

Parses the model registry and the supported-models doc tables as plain text,
so this test needs neither torch nor a vLLM import. It guards against the
drift fixed by hand in the past: architectures listed in the docs must exist
in the registry, either as a registry key or as an implementation class
name (the docs mix both, e.g. ``NVLM_D_Model`` is the class registered
under the ``NVLM_D`` key).
"""

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
REGISTRY = REPO_ROOT / "vllm/model_executor/models/registry.py"
DOCS = REPO_ROOT / "docs/models/supported_models.md"

DOCS_TABLES_START = "## List of Text-only Language Models"
DOCS_TABLES_END = "## Model Support Policy"

# Listed in the docs but no longer supported: MiniMax-VL-01 was supported
# until v0.23.0 (see _PREVIOUSLY_SUPPORTED_MODELS) while its docs row
# remains. Kept as an explicit exception so the stale row stays visible;
# removing it is a docs follow-up, not something to hide inside this test.
KNOWN_STALE_DOCS_ARCHS = {"MiniMaxVL01ForConditionalGeneration"}

_EXCLUDED_REGISTRY_DICTS = {
    "_PREVIOUSLY_SUPPORTED_MODELS",
    "_OOT_SUPPORTED_MODELS",
}


def _registry_archs() -> set[str]:
    tree = ast.parse(REGISTRY.read_text())
    archs: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (
                isinstance(target, ast.Name)
                and target.id.endswith("_MODELS")
                and target.id not in _EXCLUDED_REGISTRY_DICTS
            ):
                continue
            if not isinstance(node.value, ast.Dict):
                continue
            for key, value in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    archs.add(key.value)
                if isinstance(value, ast.Tuple) and value.elts:
                    last = value.elts[-1]
                    if isinstance(last, ast.Constant) and isinstance(last.value, str):
                        archs.add(last.value)
    return archs


def _docs_archs() -> set[str]:
    lines = DOCS.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == DOCS_TABLES_START)
    end = next(i for i, line in enumerate(lines) if line.strip() == DOCS_TABLES_END)
    archs: set[str] = set()
    for line in lines[start:end]:
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells:
            archs.update(re.findall(r"`([^`]+)`", cells[0]))
    return archs


def test_registry_and_docs_parse_succeeds():
    # Guard against silent parser breakage: if either side parses to
    # nearly nothing (e.g. a format change), fail loudly instead of
    # passing vacuously.
    assert len(_registry_archs()) > 300
    assert len(_docs_archs()) > 200


def test_documented_archs_are_registered():
    unregistered = _docs_archs() - _registry_archs() - KNOWN_STALE_DOCS_ARCHS
    assert not unregistered, (
        "Architectures listed in docs/models/supported_models.md but "
        f"missing from the model registry: {sorted(unregistered)}"
    )
