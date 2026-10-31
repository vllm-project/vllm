# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Symmetric GPTQ layers on the ROCm W4A16 kernels, via `AutoGPTQLinearMethod`.

Run `pytest tests/kernels/quantization/test_rdna_hybrid_w4a16_gptq.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.model_executor.kernels.linear.mixed_precision import (  # noqa: E402
    RDNAHybridW4A16LinearKernel,
)
from vllm.model_executor.layers.quantization.auto_gptq import (  # noqa: E402
    AutoGPTQConfig,
    AutoGPTQLinearMethod,
)
from vllm.platforms.rocm import on_gfx1x, on_gfx1100  # noqa: E402

device = "cuda"
PACK = 8
# Symmetric GPTQ checkpoints store every zero point as 7 (8 - 1).
SYMMETRIC_QZEROS_WORD = 0x77777777


def _make_layer(K: int, N: int, G: int, dtype: torch.dtype):
    config = AutoGPTQConfig(
        weight_bits=4,
        group_size=G,
        desc_act=False,
        is_sym=True,
        lm_head_quantized=False,
        dynamic={},
        full_config={},
    )
    method = AutoGPTQLinearMethod(config)
    layer = torch.nn.Module()
    with torch.device(device):
        method.create_weights(
            layer,
            input_size_per_partition=K,
            output_partition_sizes=[N],
            input_size=K,
            output_size=N,
            params_dtype=dtype,
            weight_loader=lambda *args, **kwargs: None,
        )

    gen = torch.Generator().manual_seed(0)
    q = torch.randint(0, 16, (K, N), generator=gen, dtype=torch.int32)
    shifts = (torch.arange(PACK, dtype=torch.int32) * 4).view(1, PACK, 1)
    packed = torch.sum(q.view(K // PACK, PACK, N) << shifts, dim=1, dtype=torch.int32)
    scales = (0.01 + 0.04 * torch.rand((K // G, N), generator=gen)).to(dtype)

    with torch.no_grad():
        layer.qweight.data.copy_(packed)
        layer.scales.data.copy_(scales)
        layer.qzeros.data.fill_(SYMMETRIC_QZEROS_WORD)
    return method, layer, q, scales


@pytest.mark.skipif(
    not on_gfx1x() or on_gfx1100(), reason="Needs the RDNA hybrid kernel (gfx11/gfx12)"
)
@pytest.mark.parametrize("M", [1, 64], ids=["M=1_skinny", "M=64_triton"])
@pytest.mark.parametrize(
    "K, N, G",
    [
        (256, 128, 128),  # K/G != N/8: stale qzeros fail the zp shape check
        (2048, 128, 128),  # K/G == N/8: stale qzeros pass checks, output wrong
    ],
)
def test_symmetric_gptq_matches_reference(K, N, G, M, dist_init):
    dtype = torch.bfloat16
    method, layer, q, scales = _make_layer(K, N, G, dtype)
    assert isinstance(method.kernel, RDNAHybridW4A16LinearKernel)

    gen = torch.Generator().manual_seed(1)
    x = (0.25 * torch.randn((M, K), generator=gen)).to(device, dtype)
    w = (q.to(device, torch.float32) - 8.0) * scales.to(
        device, torch.float32
    ).repeat_interleave(G, dim=0)
    ref = (x.to(torch.float32) @ w).to(dtype)

    method.process_weights_after_loading(layer)
    out = method.apply(layer, x)

    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
