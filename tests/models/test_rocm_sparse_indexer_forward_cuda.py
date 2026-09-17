"""Unit test: AMD SparseAttnIndexerKpool must be dispatchable via forward_hip.

CustomOp.dispatch_forward routes enabled ops to forward_hip on ROCm, which
falls back to forward_cuda. The AMD SparseAttnIndexerKpool implementation is
native-only; without a forward_cuda alias every enabled dispatch raises
NotImplementedError at the first forward (GLM-5.3-Flash boot crash on ROCm).
"""
import ast
import inspect
from pathlib import Path

import pytest

pytest.importorskip("vllm")

FILE = (
    Path(__file__).parent.parent.parent
    / "vllm"
    / "models"
    / "glm5next"
    / "amd"
    / "sparse_indexer.py"
)


@pytest.mark.parametrize(
    "impl",
    ["native"],
)
def test_sparse_attn_indexer_kpool_has_forward_cuda(impl):
    source = FILE.read_text()
    tree = ast.parse(source)
    cls = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "SparseAttnIndexerKpool"
    )
    # forward_native must exist (the actual implementation)
    assert any(
        isinstance(n, ast.FunctionDef) and n.name == "forward_native"
        for n in cls.body
    )
    # forward_cuda must be assigned (the class attr placeholder or a method)
    has_attr = any(
        isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "forward_cuda" for t in n.targets)
        for n in cls.body
    )
    has_method = any(
        isinstance(n, ast.FunctionDef) and n.name == "forward_cuda"
        for n in cls.body
    )
    has_alias = "SparseAttnIndexerKpool.forward_cuda = " in source
    assert has_attr or has_method or has_alias, (
        "SparseAttnIndexerKpool has no forward_cuda: CustomOp.forward_hip "
        "will raise NotImplementedError at the first forward on ROCm"
    )
