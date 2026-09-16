# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The ROCm sparse-MLA backend must not import CUDA-only sparse modules.

``sparse_mla_attention.py`` and the ``flash*_mla_sparse`` backends pull in
flashinfer and flash-attn; they are imported by CUDA backends only. Every
non-CUDA MLA backend instead depends on the platform-neutral
``mla_attention.py`` / ``sparse_utils.py`` / ``index_group.py``. This test
reads source text, so it needs neither a GPU nor AITER and therefore runs in
CUDA CI too -- which is exactly where someone would unknowingly add the bad
import.
"""

import ast
import re
from pathlib import Path

import vllm

FORBIDDEN = re.compile(r"(^|\.)(sparse_mla_attention|flash\w*_mla_sparse)$")

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
        f"{SOURCE.name} imports CUDA-only sparse modules {offenders}. Declare "
        "what this backend needs locally, or take it from a platform-neutral "
        "module (mla_attention, sparse_utils, index_group, v1.hisparse)."
    )
