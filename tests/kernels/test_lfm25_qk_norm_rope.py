# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

from vllm.model_executor.layers.lfm25_fused_qk_norm_rope import (
    fused_lfm25_qk_rmsnorm_rope,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="LFM2.5 QK norm+RoPE fusion requires CUDA-alike",
)


def _qk_ref(q, k, qw, kw, cs, pos, eps, nqh, nkvh, hd, rd):
    dtype = q.dtype
    n = q.shape[0]
    hrd = rd // 2

    qr = q.float().reshape(n, nqh, hd)
    kr = k.float().reshape(n, nkvh, hd)
    rq = torch.rsqrt((qr**2).mean(-1, keepdim=True) + eps)
    rk = torch.rsqrt((kr**2).mean(-1, keepdim=True) + eps)
    qr = (qr * rq * qw.float()).to(dtype)
    kr = (kr * rk * kw.float()).to(dtype)

    for t in range(n):
        p = pos[t].item()
        cc = cs[p, :hrd].to(dtype)
        ss = cs[p, hrd:].to(dtype)
        for h in range(nqh):
            x1 = qr[t, h, :hrd].clone()
            x2 = qr[t, h, hrd:rd].clone()
            qr[t, h, :hrd] = (x1 * cc - x2 * ss).to(dtype)
            qr[t, h, hrd:rd] = (x2 * cc + x1 * ss).to(dtype)
        for h in range(nkvh):
            x1 = kr[t, h, :hrd].clone()
            x2 = kr[t, h, hrd:rd].clone()
            kr[t, h, :hrd] = (x1 * cc - x2 * ss).to(dtype)
            kr[t, h, hrd:rd] = (x2 * cc + x1 * ss).to(dtype)
    return qr.reshape(n, nqh * hd), kr.reshape(n, nkvh * hd)


@pytest.mark.parametrize("n", [4, 16, 64])
@pytest.mark.parametrize("nqh,nkvh,hd", [(8, 4, 128), (8, 4, 256), (16, 8, 128)])
def test_qk_norm_rope_fusion(n: int, nqh: int, nkvh: int, hd: int):
    set_random_seed(0)
    rd = hd
    device = DEVICE
    dtype = torch.bfloat16
    eps = 1e-6

    q = torch.randn(n, nqh * hd, device=device, dtype=dtype)
    k = torch.randn(n, nkvh * hd, device=device, dtype=dtype)
    qw = torch.randn(hd, device=device, dtype=dtype)
    kw = torch.randn(hd, device=device, dtype=dtype)
    cs = torch.randn(8192, rd, device=device, dtype=dtype)
    pos = torch.randint(0, 8192, (n,), device=device, dtype=torch.int32)

    ref_q, ref_k = _qk_ref(
        q.clone(), k.clone(), qw, kw, cs, pos, eps, nqh, nkvh, hd, rd
    )
    fused_q, fused_k = fused_lfm25_qk_rmsnorm_rope(
        q, k, qw, kw, cs, pos, eps, nqh, nkvh, hd, rd
    )

    assert torch.allclose(ref_q, fused_q, rtol=5e-2, atol=5e-2)
    assert torch.allclose(ref_k, fused_k, rtol=5e-2, atol=5e-2)
