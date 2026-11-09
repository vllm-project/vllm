# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
from collections.abc import Iterable
from pathlib import Path

import vllm.v1.core
import vllm.v1.spec_decode.resources

WORKER_GPU = "vllm.v1.worker.gpu"


def _runtime_imports(nodes: Iterable[ast.AST]):
    """``module.name`` of every import that runs, in functions too, skipping
    ``if TYPE_CHECKING:`` blocks."""
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and node.module:
            yield from (f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            yield from _runtime_imports(node.orelse)
        else:
            yield from _runtime_imports(ast.iter_child_nodes(node))


def _gpu_worker_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    return [m for m in _runtime_imports([tree]) if m.startswith(WORKER_GPU)]


def test_core_does_not_import_the_gpu_worker():
    """The scheduler runs in the engine core, which must not load the GPU worker;
    the engine core also lowers spec resources with ``resources.py``."""
    root = Path(vllm.v1.core.__file__).parent
    paths = [*sorted(root.rglob("*.py")), Path(vllm.v1.spec_decode.resources.__file__)]
    bad = [f"{path.name}: {m}" for path in paths for m in _gpu_worker_imports(path)]
    assert not bad, bad


def test_function_level_imports_are_found(tmp_path):
    path = tmp_path / "m.py"
    path.write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from vllm.v1.worker.gpu.a import A\n"
        "def f():\n"
        "    from vllm.v1.worker.gpu import b\n"
        "    import vllm.v1.worker.gpu.c\n"
    )
    assert _gpu_worker_imports(path) == [f"{WORKER_GPU}.b", f"{WORKER_GPU}.c"]
