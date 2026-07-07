# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import ast
from pathlib import Path

MANAGER_PATH = Path(__file__).resolve().parents[2] / "vllm/knorm/manager.py"


def test_knorm_free_accepts_scheduler_kwarg_without_forwarding_to_base():
    tree = ast.parse(MANAGER_PATH.read_text(encoding="utf-8"))
    knorm_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KnormFullAttentionManager"
    )
    free_func = next(
        node
        for node in knorm_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "free"
    )

    assert any(
        arg.arg == "prioritize_uncached_for_reuse"
        for arg in free_func.args.args
    )

    super_free_calls = [
        node
        for node in ast.walk(free_func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "free"
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "super"
    ]
    assert len(super_free_calls) == 1
    assert not super_free_calls[0].keywords
