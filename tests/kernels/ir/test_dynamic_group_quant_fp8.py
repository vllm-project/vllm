# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.kernels  # noqa: F401
from vllm import ir
from vllm.model_executor.layers.quantization.utils import fp8_utils
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="per_token_group_fp8_quant kernel path is CUDA/ROCm",
)
def test_dynamic_group_quant_fp8_matches_execute():
    torch.set_default_device(current_platform.device_type)
    group_size = 128
    x = torch.randn(32, 4096, dtype=torch.bfloat16)

    q_ref, s_ref = fp8_utils._execute_per_token_group_quant_fp8(
        x, group_size, column_major_scales=True
    )
    q_ir, s_ir = ir.ops.dynamic_group_quant_fp8(
        x, group_size, 1e-10, None, True, False, None, None
    )

    torch.testing.assert_close(q_ir, q_ref)
    torch.testing.assert_close(s_ir, s_ref)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="per_token_group_fp8_quant kernel path is CUDA/ROCm",
)
def test_dynamic_group_quant_fp8_optional_out_buffer():
    torch.set_default_device(current_platform.device_type)
    group_size = 128
    x = torch.randn(16, 2048, dtype=torch.bfloat16)
    dtype = current_platform.fp8_dtype()
    out_q = torch.empty(x.shape, device=x.device, dtype=dtype)

    q_ir, s_ir = ir.ops.dynamic_group_quant_fp8(
        x, group_size, 1e-10, None, True, False, None, out_q
    )
    q_ref, s_ref = fp8_utils._execute_per_token_group_quant_fp8(
        x, group_size, column_major_scales=True, out_q=out_q.clone()
    )

    assert q_ir is out_q
    torch.testing.assert_close(q_ir, q_ref)
    torch.testing.assert_close(s_ir, s_ref)
