# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashSM120Impl's sparse MLA decode on the GPU against a softmax-with-sink reference.

Contributed by hclsys in #68387; it fails when the learnable sink is dropped.
"""

import pytest
import torch

from vllm.config import CacheConfig
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.models.hy_v4_flash.nvidia.sm120 import (
    FlashSM120Impl,
    FlashSM120SparseBackend,
)
from vllm.platforms import current_platform


def _pack_fp8_ds_mla(kv: torch.Tensor) -> torch.Tensor:
    """bf16 [n, 576] -> 656-byte rows.

    512 fp8 values (4 tiles of 128), 4 pow2 fp32 scales, then 64 bf16 rope values.
    """
    n = kv.shape[0]
    rows = torch.zeros(n, 656, dtype=torch.uint8, device=kv.device)
    for t in range(4):
        tile = kv[:, t * 128 : (t + 1) * 128].float()
        amax = tile.abs().amax(-1).clamp(min=1e-4)
        scale = torch.pow(2, (amax / 448.0).log2().ceil())
        fp8 = (tile / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        rows[:, t * 128 : (t + 1) * 128] = fp8.view(torch.uint8)
        rows[:, 512 + t * 4 : 516 + t * 4] = scale.view(torch.uint8).view(n, 4)
    rows[:, 528:] = kv[:, 512:].contiguous().view(torch.uint8).view(n, 128)
    return rows


def _unpack_fp8_ds_mla(rows: torch.Tensor) -> torch.Tensor:
    n = rows.shape[0]
    out = torch.empty(n, 576, dtype=torch.float32, device=rows.device)
    for t in range(4):
        fp8 = rows[:, t * 128 : (t + 1) * 128].contiguous().view(torch.float8_e4m3fn)
        scale = rows[:, 512 + t * 4 : 516 + t * 4].contiguous().view(torch.float32)
        out[:, t * 128 : (t + 1) * 128] = fp8.float() * scale
    out[:, 512:] = rows[:, 528:].contiguous().view(torch.bfloat16).float()
    return out


def _sm120_impl(heads: int, topk: int, sinks: torch.Tensor) -> FlashSM120Impl:
    """The impl a Hy4-flash sparse layer builds, with Hy4-flash's MLA dimensions."""
    nope, rope, v_dim, kv_lora_rank = 192, 64, 256, 512
    layer = MLAAttention(
        num_heads=heads,
        scale=(kv_lora_rank + rope) ** -0.5,
        qk_nope_head_dim=nope,
        qk_rope_head_dim=rope,
        v_head_dim=v_dim,
        q_lora_rank=None,
        kv_lora_rank=kv_lora_rank,
        kv_b_proj=ColumnParallelLinear(
            kv_lora_rank, heads * (nope + v_dim), bias=False
        ),
        cache_config=CacheConfig(cache_dtype="fp8_ds_mla"),
        attn_backend=FlashSM120SparseBackend,
        use_sparse=True,
        topk_indices_buffer=torch.empty(64, topk, dtype=torch.int32, device="cuda"),
        sinks=sinks,
    )
    assert isinstance(layer.impl, FlashSM120Impl)
    return layer.impl


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120),
    reason="FlashInfer SM120 sparse MLA kernel",
)
@pytest.mark.parametrize("heads", [4, 16, 64, 128])
@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("num_tokens", [1, 32, 128])
def test_sm120_flash_impl_applies_the_sink_on_gpu(
    dist_init, default_vllm_config, heads: int, topk: int, num_tokens: int
) -> None:
    torch.manual_seed(0)
    dev = torch.device("cuda")
    num_blocks, block_size = 64, 64
    kv = (torch.randn(num_blocks * block_size, 576, device=dev) / 10).clamp(-1, 1)
    rows = _pack_fp8_ds_mla(kv.to(torch.bfloat16))
    kv_ref = _unpack_fp8_ds_mla(rows)
    q = (torch.randn(num_tokens, heads, 576, device=dev) / 10).clamp(-1, 1)
    q = q.to(torch.bfloat16)
    indices = torch.randint(
        0, num_blocks * block_size, (num_tokens, topk), device=dev, dtype=torch.int32
    )
    indices[:, topk // 2 :] = -1
    # Large enough to move the output well past the tolerance.
    sinks = torch.randn(heads, device=dev) * 2.0 + 6.0

    impl = _sm120_impl(heads, topk, sinks)
    out = impl._run_mqa_kernel(q, rows.view(num_blocks, block_size, 656), indices)

    # Reference: softmax over the selected keys plus one sink logit per head.
    keys = kv_ref[indices.clamp(min=0).long()]  # [t, k, 576]
    logits = torch.einsum("thd,tkd->thk", q.float(), keys) * impl.scale
    logits = logits.masked_fill((indices < 0)[:, None, :], float("-inf"))
    sink_col = sinks[None, :, None].expand(num_tokens, heads, 1)
    probs = torch.softmax(torch.cat([logits, sink_col], dim=-1), dim=-1)[..., :-1]
    ref = torch.einsum("thk,tkd->thd", probs, keys[..., :512])
    no_sink = torch.einsum("thk,tkd->thd", logits.softmax(-1), keys[..., :512])

    err = (out.float() - ref).abs().max().item()
    assert err < 5e-3
    assert (out.float() - no_sink).abs().max().item() > 5 * err
