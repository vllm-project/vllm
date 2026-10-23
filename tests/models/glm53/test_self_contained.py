# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The stock DSA package (vllm/models/deepseek_v32) does not import GLM-5.3's."""

import ast
from pathlib import Path

import vllm.models

PACKAGE = Path(vllm.models.__file__).resolve().parent / "glm53"


def _imports(path: Path) -> list[str]:
    """Return every absolute module a file imports (relative ones stay in the
    package).
    """
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
            names.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_deepseek_v32_does_not_import_glm53():
    deepseek = PACKAGE.parent / "deepseek_v32"
    offending = {
        str(path.relative_to(deepseek)): name
        for path in sorted(deepseek.rglob("*.py"))
        for name in _imports(path)
        if name.startswith("vllm.models.glm53")
    }
    assert offending == {}
