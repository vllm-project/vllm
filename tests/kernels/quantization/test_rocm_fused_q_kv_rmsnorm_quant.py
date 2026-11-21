# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-major MXFP8 query quant fused with the Q/KV RMSNorm, on CDNA4.

This stays out of ``tests/kernels/core``: that directory is collected by the
MI300 Core Operation Kernels job, which skips every test that needs
``supports_mx``. The MI355 quantization jobs collect this directory.
"""

from __future__ import annotations

import pytest
import torch

from vllm.models.common.ops.fused_qk_rmsnorm import fused_q_kv_rmsnorm
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_mx()),
    reason="row-major MXFP8 query quant needs CDNA4",
)

DEVICE = current_platform.device_type


@pytest.mark.parametrize("num_tokens", [0, 1, 17, 128, 129, 224, 1024])
@pytest.mark.parametrize("q_size", [384, 1280, 1344])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_q_kv_rmsnorm_quant_matches_rocm_separate(num_tokens, q_size, dtype):
    """Row-major scales and fp8 values match RMSNorm then mxfp8_e4m3_quantize."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.models.deepseek_v41.common.ops.query_quant import (
        fused_q_kv_rmsnorm_quant,
    )

    torch.manual_seed(0)
    kv_size = 512
    x = torch.randn(num_tokens, q_size + kv_size, device=DEVICE, dtype=dtype)
    if num_tokens:
        x[:1] = 0
        x[1:2] *= 1e-30
    qr, kv = x.split([q_size, kv_size], -1)
    qw = torch.randn(q_size, device=DEVICE, dtype=x.dtype)
    kvw = torch.randn(kv_size, device=DEVICE, dtype=x.dtype)
    eps = 1e-20
    result = fused_q_kv_rmsnorm_quant(qr, kv, qw, kvw, eps)
    assert result[0].orig_dtype == dtype
    assert result[0].orig_shape == qr.shape
    if not num_tokens:
        assert result[0].data.shape == qr.shape
        assert result[0].scale.numel() == 0
        assert result[1].shape == kv.shape
        return
    assert result[0].scale.shape == (num_tokens, q_size // 32)
    qr_ref, kv_ref = fused_q_kv_rmsnorm(qr, kv, qw, kvw, eps)
    q_ref, scale_ref = mxfp8_e4m3_quantize(qr_ref, is_sf_swizzled_layout=False)
    for replay in (False, True):
        if replay:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = fused_q_kv_rmsnorm_quant(qr, kv, qw, kvw, eps)
            graph.replay()
        torch.testing.assert_close(
            result[0].data.view(torch.uint8), q_ref.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(result[0].scale, scale_ref, rtol=0, atol=0)
        torch.testing.assert_close(result[1], kv_ref, rtol=0, atol=0)
