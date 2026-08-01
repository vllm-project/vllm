# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

from vllm.model_executor.layers.mamba.ops.lfm25_fused_short_conv import (
    fused_lfm25_short_conv_decode,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="LFM2.5 ShortConv fusion requires CUDA-alike",
)


def _short_conv_ref(b, c, x, conv_state, weight, indices):
    out = []
    for i in range(b.shape[0]):
        sidx = indices[i].item()
        bx = (b[i] * x[i]).to(b.dtype)
        cv = (
            conv_state[sidx, :, 0] * weight[:, 0].float()
            + conv_state[sidx, :, 1] * weight[:, 1].float()
            + bx.float() * weight[:, 2].float()
        ).to(b.dtype)
        out.append((c[i] * cv).to(b.dtype))
        conv_state[sidx, :, 0] = conv_state[sidx, :, 1].clone()
        conv_state[sidx, :, 1] = bx.to(torch.float32)
    return torch.stack(out)


@pytest.mark.parametrize("dim", [1536, 2048])
@pytest.mark.parametrize("num_tokens", [1, 4, 16])
def test_short_conv_fusion(dim: int, num_tokens: int):
    set_random_seed(0)
    nb = max(num_tokens * 4 + 16, 64)
    device = DEVICE

    b = torch.randn(num_tokens, dim, device=device, dtype=torch.bfloat16)
    c = torch.randn(num_tokens, dim, device=device, dtype=torch.bfloat16)
    x = torch.randn(num_tokens, dim, device=device, dtype=torch.bfloat16)
    w = torch.randn(dim, 3, device=device, dtype=torch.bfloat16)
    sr = torch.randn(nb, dim, 2, device=device, dtype=torch.float32)
    sf = sr.clone()
    # vLLM reserves block 0 as sentinel
    indices = torch.arange(1, num_tokens + 1, device=device, dtype=torch.int32)

    ref = _short_conv_ref(b.clone(), c.clone(), x.clone(), sr, w, indices.clone())
    fused = fused_lfm25_short_conv_decode(b, c, x, sf, w, indices)

    assert torch.allclose(ref, fused, rtol=1e-2, atol=1e-2)
    assert (sr - sf).abs().max().item() < 1e-4
