# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.models.kimi_k3.nvidia.ops.vision_fa4_warmup import (
    KimiK3VisionFA4WarmupConfig,
    _iter_compile_units,
)
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda(), reason="FA4 requires CUDA")
@pytest.mark.parametrize("batch_size,seqlen", [(1, 257), (4, 1024)])
@torch.inference_mode()
def test_long_images_reuse_warmed_fa4_kernels(monkeypatch, batch_size, seqlen):
    """ViT requests past one Q tile must not compile after startup warmup."""
    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 10:
        pytest.skip("This warmup selects Blackwell FA4 kernels")

    from vllm.vllm_flash_attn.cute import interface as fa4

    from vllm.v1.attention.ops.vit_attn_wrappers import vit_flash_attn_wrapper

    config = KimiK3VisionFA4WarmupConfig(
        num_heads=12,
        head_dim=128,
        dtype=torch.bfloat16,
        max_batch_size=4,
        max_seqlen=1024,
    )
    for unit in _iter_compile_units(config):
        unit.compile()
    torch.accelerator.synchronize()

    def unexpected_compile(*args, **kwargs):
        pytest.fail("Image attention requested a kernel absent from startup warmup")

    monkeypatch.setattr(fa4.cute, "compile", unexpected_compile)
    device = torch.device("cuda")
    total = batch_size * seqlen
    qkv = torch.randn(total, 3, 12, 128, device=device, dtype=config.dtype) * 0.125
    q, k, v = qkv.unbind(1)
    # RoPE materializes Q/K; V remains a view into packed QKV in MoonViT.
    q, k = q.contiguous(), k.contiguous()
    cu_seqlens = torch.arange(0, total + 1, seqlen, device=device, dtype=torch.int32)
    output = vit_flash_attn_wrapper(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        batch_size=1,
        is_rocm_aiter=False,
        fa_version=4,
        scale=config.head_dim**-0.5,
        cu_seqlens=cu_seqlens,
        max_seqlen=torch.tensor(seqlen, device=device),
    )
    reference = F.scaled_dot_product_attention(
        *(
            t.reshape(batch_size, seqlen, 12, 128).transpose(1, 2).float()
            for t in (q, k, v)
        )
    ).transpose(1, 2)
    torch.testing.assert_close(
        output.reshape_as(reference).float(), reference, atol=1e-3, rtol=1e-2
    )
