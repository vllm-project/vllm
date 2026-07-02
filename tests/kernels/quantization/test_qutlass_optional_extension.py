# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops


def test_missing_qutlass_op_error_message():
    with pytest.raises(
        RuntimeError,
        match="Missing op: __vllm_missing_qutlass_test_op__",
    ):
        ops._check_qutlass_op("__vllm_missing_qutlass_test_op__")


def test_matmul_mxf4_bf16_tn_requires_qutlass_op():
    if hasattr(torch.ops._qutlass_C, "matmul_mxf4_bf16_tn"):
        pytest.skip("QUTLASS matmul op is available in this build.")

    tensor = torch.empty(1)
    with pytest.raises(RuntimeError, match="Missing op: matmul_mxf4_bf16_tn"):
        ops.matmul_mxf4_bf16_tn(tensor, tensor, tensor, tensor, tensor)


@pytest.mark.parametrize(
    ("method", "op_name"),
    [
        ("quest", "fusedQuantizeMxQuest"),
        ("abs_max", "fusedQuantizeMxAbsMax"),
    ],
)
def test_fused_quantize_mx_requires_qutlass_op(method, op_name):
    if hasattr(torch.ops._qutlass_C, op_name):
        pytest.skip(f"QUTLASS {op_name} op is available in this build.")

    a = torch.empty(1, 32)
    b = torch.empty(32, 32)
    with pytest.raises(RuntimeError, match=f"Missing op: {op_name}"):
        ops.fusedQuantizeMx(a, b, method=method)


def test_fused_quantize_nv_requires_qutlass_op():
    if hasattr(torch.ops._qutlass_C, "fusedQuantizeNv"):
        pytest.skip("QUTLASS fusedQuantizeNv op is available in this build.")

    a = torch.empty(1, 16)
    b = torch.empty(16, 16)
    global_scale = torch.empty(1)
    with pytest.raises(RuntimeError, match="Missing op: fusedQuantizeNv"):
        ops.fusedQuantizeNv(a, b, global_scale)
