# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path

import regex as re

import vllm

FORBIDDEN = re.compile(r"(^|\.)flash\w*_mla_sparse$")

BACKENDS = Path(vllm.__file__).parent / "v1/attention/backends/mla"
SOURCE = BACKENDS / "rocm_aiter_mla_sparse.py"


def _imported_modules(tree: ast.Module) -> set[str]:
    """Module names imported at module scope (function-local imports excluded)."""
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{a.name}" for a in node.names)
    return modules


def test_rocm_sparse_backend_avoids_cuda_sparse_modules():
    tree = ast.parse(SOURCE.read_text())
    offenders = sorted(m for m in _imported_modules(tree) if FORBIDDEN.search(m))
    assert not offenders, (
        f"{SOURCE.name} imports CUDA-only sparse backends {offenders}. Declare "
        "what this backend needs locally, or take it from a platform-neutral "
        "module (mla_attention, sparse_utils, index_group, v1.hisparse)."
    )
