# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _projection_prefixes(path: Path) -> dict[str, ast.AST]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    layer_cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "DeepSeekV4MultiTokenPredictorLayer"
    )
    init = next(
        node
        for node in layer_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )

    prefixes: dict[str, ast.AST] = {}
    for node in ast.walk(init):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "self"
            and node.targets[0].attr in {"e_proj", "h_proj"}
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "ReplicatedLinear"
        ):
            prefix_kw = next(
                keyword.value
                for keyword in node.value.keywords
                if keyword.arg == "prefix"
            )
            prefixes[node.targets[0].attr] = prefix_kw

    return prefixes


def _is_prefix_attr(value: ast.AST, attr: str) -> bool:
    return (
        isinstance(value, ast.JoinedStr)
        and len(value.values) == 2
        and isinstance(value.values[0], ast.FormattedValue)
        and isinstance(value.values[0].value, ast.Name)
        and value.values[0].value.id == "prefix"
        and isinstance(value.values[1], ast.Constant)
        and value.values[1].value == f".{attr}"
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "vllm/models/deepseek_v4/nvidia/mtp.py",
        "vllm/models/deepseek_v4/amd/mtp.py",
    ],
)
def test_deepseek_v4_mtp_projection_prefixes(relative_path: str):
    prefixes = _projection_prefixes(ROOT / relative_path)

    assert _is_prefix_attr(prefixes["e_proj"], "e_proj")
    assert _is_prefix_attr(prefixes["h_proj"], "h_proj")
