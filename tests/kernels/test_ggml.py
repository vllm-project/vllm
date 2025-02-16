# SPDX-License-Identifier: Apache-2.0

import gguf
import pytest
import torch

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops  # noqa: F401


@pytest.mark.parametrize("quant_type", [12])
def test_ggml_opcheck(quant_type):
    block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    shape = [256, 1152]
    qweight = torch.randint(0, 100, shape, device='cuda', dtype=torch.uint8)
    m = qweight.shape[0]
    n = qweight.shape[1] // type_size * block_size
    opcheck(torch.ops._C.ggml_dequantize, (qweight, quant_type, m, n))

    x = torch.rand((m, 512), device='cuda', dtype=torch.float16)
    opcheck(torch.ops._C.ggml_mul_mat_a8,
            (qweight, x, quant_type, qweight.shape[0]))
    opcheck(torch.ops._C.ggml_mul_mat_vec_a8,
            (qweight, x, quant_type, qweight.shape[0]))
