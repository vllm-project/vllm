# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.quantization.test_awq_triton import awq_dequantize_torch
from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops  # noqa: F401
from vllm.utils.torch_utils import set_random_seed


@pytest.mark.skipif(
    not hasattr(torch.ops._C, "awq_dequantize"),
    reason="AWQ is not supported on this GPU type.",
)
def test_awq_dequantize_opcheck(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_TRITON_AWQ", "0")
        qweight = torch.randint(
            -2000000000, 2000000000, (8192, 256), device="cuda", dtype=torch.int32
        )
        scales = torch.rand((64, 2048), device="cuda", dtype=torch.float16)
        zeros = torch.empty((64, 256), device="cuda", dtype=torch.int32)
        split_k_iters = 0
        thx = 0
        thy = 0
        opcheck(
            torch.ops._C.awq_dequantize,
            (qweight, scales, zeros, split_k_iters, thx, thy),
        )


@pytest.mark.skipif(
    not hasattr(torch.ops._C, "awq_dequantize"),
    reason="AWQ is not supported on this GPU type.",
)
@pytest.mark.parametrize("qweight_rows", [128, 3584])
@pytest.mark.parametrize("qweight_cols", [16, 64, 448, 576])
@pytest.mark.parametrize("group_size", [64, 128])
def test_awq_dequantize(qweight_rows, qweight_cols, group_size):
    """The CUDA dequant kernel must match the PyTorch reference element-wise."""
    set_random_seed(0)
    qweight = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (qweight_rows, qweight_cols),
        dtype=torch.int32,
        device="cuda",
    )
    scales = torch.rand(
        qweight_rows // group_size,
        qweight_cols * 8,
        dtype=torch.float16,
        device="cuda",
    )
    zeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (qweight_rows // group_size, qweight_cols),
        dtype=torch.int32,
        device="cuda",
    )

    out = torch.ops._C.awq_dequantize(qweight, scales, zeros, 0, 0, 0)

    torch.testing.assert_close(
        out, awq_dequantize_torch(qweight, scales, zeros, group_size), atol=0, rtol=0
    )
