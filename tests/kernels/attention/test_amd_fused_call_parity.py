# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm deepseek_v32 module must not silently drop fused-kernel kwargs.

`fused_norm_rope` defaults several arguments to None and then substitutes
another value -- notably `indexer_slot_mapping`, which falls back to
`slot_mapping`. That fallback is invisible without HiSparse, where the indexer
and MLA groups share one slot mapping, and wrong with it, where they do not.
Reading the call site does not catch an omission; diffing it does.
"""

import ast
from pathlib import Path

import vllm

MODEL_DIR = Path(vllm.__file__).parent / "models" / "deepseek_v32"

# CUDA-only branches ROCm deliberately does not take.
NOT_APPLICABLE_TO_ROCM = {"index_k_out"}  # prefill context parallel


def _call_kwargs(path: Path, callee: str) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == callee:
            names |= {kw.arg for kw in node.keywords if kw.arg}
    return names


def test_amd_fused_norm_rope_passes_every_cuda_kwarg():
    cuda = _call_kwargs(MODEL_DIR / "attention.py", "fused_norm_rope")
    rocm = _call_kwargs(MODEL_DIR / "amd" / "rocm.py", "fused_norm_rope")
    assert cuda, "no fused_norm_rope call found on the CUDA path"
    assert not (cuda - rocm - NOT_APPLICABLE_TO_ROCM)
