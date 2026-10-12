# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host dispatch regressions, with no GPU allocation required.

Only the pure policy function is loaded: importing the kernel module requires
Triton and a configured accelerator even though launch selection is host logic.
"""

import ast
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def decode_launch():
    source = (
        Path(__file__).resolve().parents[3]
        / "vllm/v1/attention/ops/turboquant_soa"
        / "triton_turboquant_unified_attention.py"
    )
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_use_decode_launch"
    )
    namespace: dict[str, object] = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[function.name]


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("sequences", [1, 29, 64, 128])
def test_verify_uses_decode_launch(decode_launch, rows, sequences):
    # C128 must not switch back to serial history traversal as the grid grows.
    assert decode_launch(rows, rows * sequences, sequences)


@pytest.mark.parametrize("rows", [33, 64, 128, 1024, 32768])
def test_prefill_keeps_wide_launch(decode_launch, rows):
    assert not decode_launch(rows, rows * 64, 64)


def test_ragged_maximum_is_not_replaced_by_average(decode_launch):
    # One long query and many single-token decodes have a small average.
    assert not decode_launch(128, 191, 64)


@pytest.mark.parametrize(
    "tokens,sequences,expected", [(64, 64, True), (65, 64, False), (256, 64, False)]
)
def test_missing_maximum_preserves_conservative_fallback(
    decode_launch, tokens, sequences, expected
):
    assert decode_launch(None, tokens, sequences) is expected
