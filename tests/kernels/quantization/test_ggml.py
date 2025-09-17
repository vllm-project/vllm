# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    opcheck(torch.ops._C.ggml_dequantize,
            (qweight, quant_type, m, n, torch.float16))

    x = torch.rand((m, 512), device='cuda', dtype=torch.float16)
    opcheck(torch.ops._C.ggml_mul_mat_a8,
            (qweight, x, quant_type, qweight.shape[0]))
    opcheck(torch.ops._C.ggml_mul_mat_vec_a8,
            (qweight, x, quant_type, qweight.shape[0]))

    shape = [256, 1024, 336]
    qweight = torch.randint(0, 100, shape, device='cuda', dtype=torch.uint8)
    x = torch.rand((1, 1024), device='cuda', dtype=torch.float16)
    sorted_token_ids = torch.arange(776, device='cuda')
    expert_ids = torch.randint(0, 256, (194, ), device='cuda')
    num_tokens_post_padded = torch.tensor([1],
                                          dtype=torch.int64,
                                          device='cuda')

    opcheck(torch.ops._C.ggml_moe_a8,
            (x, qweight, sorted_token_ids, expert_ids, num_tokens_post_padded,
             quant_type, qweight.shape[0], 1, x.shape[0]))

    topk_ids = torch.zeros((1, 1), device='cuda', dtype=torch.int32)

    opcheck(
        torch.ops._C.ggml_moe_a8_vec,
        (x, qweight, topk_ids, 1, quant_type, qweight.shape[0], x.shape[0]))
