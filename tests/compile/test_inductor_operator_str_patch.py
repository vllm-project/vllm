# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Inductor implicit-fallback message patch in env_override.py."""

import pytest
import torch
from torch._inductor.exc import OperatorIssue

from vllm.env_override import _apply_inductor_operator_str_patch
from vllm.utils.torch_utils import direct_register_custom_op


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


direct_register_custom_op(
    op_name="test_operator_str_identity",
    op_func=_identity,
    fake_impl=_identity,
)


class _DeepIRNode:
    """Stands in for an IR node whose __str__ would recurse for minutes."""

    def __str__(self):
        raise AssertionError("IR node must not be stringified")

    __repr__ = __str__


@pytest.fixture
def patched_operator_str(monkeypatch):
    monkeypatch.setattr(OperatorIssue, "operator_str", OperatorIssue.operator_str)
    _apply_inductor_operator_str_patch()
    return OperatorIssue.operator_str


def test_vllm_op_message_is_target_only(patched_operator_str):
    op = torch.ops.vllm.test_operator_str_identity.default
    msg = patched_operator_str(op, [_DeepIRNode()], {"x": _DeepIRNode()})
    assert msg == f"  target: {op}"
