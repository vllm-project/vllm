# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
from pathlib import Path

import vllm.v1.core

WORKER_GPU = "vllm.v1.worker.gpu"


def _runtime_imports(tree: ast.AST):
    """Module-level imports, skipping ``if TYPE_CHECKING:`` blocks."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            continue
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, (ast.If, ast.Try)):
            yield from _runtime_imports(node)


def test_core_does_not_import_the_gpu_worker():
    """The scheduler runs in the engine core, which must not load the GPU worker."""
    root = Path(vllm.v1.core.__file__).parent
    bad = [
        f"{path.relative_to(root)}: {module}"
        for path in sorted(root.rglob("*.py"))
        for module in _runtime_imports(ast.parse(path.read_text()))
        if module.startswith(WORKER_GPU)
    ]
    assert not bad, bad
