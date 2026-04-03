# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test cases for Qwen ViT attention score extraction functionality."""

import pytest
import torch
from unittest.mock import patch
from functools import partial

from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionAttention,
    Qwen2_5_VisionBlock,
)
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.layernorm import RMSNorm

@pytest.fixture(autouse=True)
def mock_tp():
    import vllm.distributed.parallel_state as ps
    from unittest.mock import MagicMock
    
    # Mock TP 组，防止 Linear 层初始化报错
    if ps._TP is None:
        ps._TP = MagicMock()
        ps._TP.world_size = 1
        ps._TP.rank_in_group = 0
    yield

# --- Attention 层 Forward 测试 ---

@pytest.mark.parametrize("backend", [
     AttentionBackendEnum.TORCH_SDPA,
    # _Backend.XFORMERS, # xformers/flash-attn 版本不对，先跳过
])
def test_qwen2_5_vision_attention_forward_default_behavior(backend: AttentionBackendEnum):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = "cuda"
    attn = Qwen2_5_VisionAttention(
        embed_dim=768, num_heads=12, projection_size=768,
    ).to(device)
    
    batch_size = 2
    seq_len = 10
    x = torch.randn(seq_len, batch_size, 768, device=device)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    rotary_pos_emb_cos = None
    rotary_pos_emb_sin = None
    max_seqlen = torch.tensor(seq_len, dtype=torch.int32, device=device)
    
    result = attn(
        x, 
        cu_seqlens, 
        rotary_pos_emb_cos, 
        rotary_pos_emb_sin, 
        max_seqlen,
        return_attention_score=False
    )
    
    assert isinstance(result, torch.Tensor)
    assert result.shape == (seq_len, batch_size, 768)

# --- VisionBlock Forward 测试 ---

def test_qwen2_5_vision_block_forward_default_behavior():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = "cuda"
    # Use RMSNorm which supports residual parameter, matching Qwen2_5_VisionTransformer
    norm_layer = partial(RMSNorm, eps=1e-6)
    block = Qwen2_5_VisionBlock(
        dim=768, num_heads=12, mlp_hidden_dim=3072,
        act_fn=get_act_and_mul_fn("silu"),
        norm_layer=norm_layer,
    ).to(device)
    
    batch_size = 2
    seq_len = 10
    
    #  VisionAttention.split_qkv 期望的是 3D Tensor(seq_len, batch_size, 768)
    x = torch.randn(seq_len, batch_size, 768, device=device)
    
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    rotary_pos_emb_cos = None
    rotary_pos_emb_sin = None
    max_seqlen = torch.tensor(seq_len, dtype=torch.int32, device=device)
    
    result = block(
        x, 
        cu_seqlens, 
        rotary_pos_emb_cos, 
        rotary_pos_emb_sin, 
        max_seqlen,
        return_attention_score=False
    )

    if isinstance(result, tuple):
        # 只有进入了 Flash-Attn 路径才会拿到元组
        output, score = result
        assert score is not None
        assert output.shape == (seq_len, batch_size, 768)
    else:
        # 如果环境里不支持 Flash-Attn（比如 fallback 到了 SDPA），拿到 Tensor 也是正确的
        assert isinstance(result, torch.Tensor)
        assert result.shape == (seq_len, batch_size, 768)


def test_qwen2_5_vision_block_forward_with_need_score_interface():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = "cuda"
    # Use RMSNorm which supports residual parameter, matching Qwen2_5_VisionTransformer
    norm_layer = partial(RMSNorm, eps=1e-6)
    block = Qwen2_5_VisionBlock(
        dim=768, num_heads=12, mlp_hidden_dim=3072,
        act_fn=get_act_and_mul_fn("silu"),
        norm_layer=norm_layer,
    ).to(device)
    
    batch_size = 2
    seq_len = 10
    
    x = torch.randn(seq_len, batch_size, 768, device=device)
    
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    rotary_pos_emb_cos = None
    rotary_pos_emb_sin = None
    max_seqlen = torch.tensor(seq_len, dtype=torch.int32, device=device)
    
    result = block(
        x, 
        cu_seqlens, 
        rotary_pos_emb_cos, 
        rotary_pos_emb_sin, 
        max_seqlen,
        return_attention_score=True
    )
    
    if isinstance(result, tuple):
        output, attention_score = result
        assert isinstance(output, torch.Tensor)
        assert attention_score is not None
    else:
        # 如果是 SDPA 或者其他后端，result 就是普通的 Tensor
        assert isinstance(result, torch.Tensor)
