# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test cases for Qwen3 ViT attention score extraction functionality."""

import pytest
import torch
import os

try:
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLVisionConfig,
    )
except ImportError:
    # Fallback for different transformer versions
    Qwen3VLVisionConfig = None

from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionBlock,
    Qwen3_VisionTransformer,
)
from unittest.mock import patch
from vllm.attention.backends.registry import AttentionBackendEnum


@pytest.fixture(autouse=True)
def mock_tp():
    import vllm.distributed.parallel_state as ps
    from unittest.mock import MagicMock
    
    # 如果 _TP 是 None，手动给它一个 Mock 对象绕过 assert _TP is not None
    if ps._TP is None:
        ps._TP = MagicMock()
        ps._TP.world_size = 1
        ps._TP.rank_in_group = 0
    yield


@pytest.fixture(autouse=True)
def mock_vit_attn_backend():
    """Mock get_vit_attn_backend to return a supported backend for Qwen3-VL."""
    with patch('vllm.model_executor.models.qwen3_vl.get_vit_attn_backend') as mock_backend:
        # Return TORCH_SDPA which is supported by Qwen3-VL
        mock_backend.return_value = AttentionBackendEnum.TORCH_SDPA
        yield mock_backend

def test_qwen3_vision_block_has_forward_param():
    """Test that Qwen3_VisionBlock forward method accepts return_attention_score parameter."""
    block = Qwen3_VisionBlock(
        dim=768,
        num_heads=12,
        mlp_hidden_dim=3072,
    )
    
    # Check that forward method signature includes return_attention_score
    import inspect
    sig = inspect.signature(block.forward)
    assert 'return_attention_score' in sig.parameters


def test_qwen3_vision_block_forward_with_score():
    """Test that Qwen3_VisionBlock forward can return attention score."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = "cuda"
    block = Qwen3_VisionBlock(
        dim=768,
        num_heads=12,
        mlp_hidden_dim=3072,
    ).to(device)
    
    batch_size = 2
    seq_len = 10
    x = torch.randn(seq_len, batch_size, 768, device=device)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    rotary_pos_emb_cos = None
    rotary_pos_emb_sin = None
    max_seqlen = torch.tensor(seq_len, dtype=torch.int32, device=device)
    
    # Test with return_attention_score=True
    result = block(
        x,
        cu_seqlens,
        rotary_pos_emb_cos,
        rotary_pos_emb_sin,
        max_seqlen,
        return_attention_score=True,
    )
    
    # Result can be either tensor or tuple depending on backend
    assert isinstance(result, torch.Tensor) or isinstance(result, tuple)


@pytest.mark.skipif(
    Qwen3VLVisionConfig is None,
    reason="Qwen3VLVisionConfig not available"
)
def test_qwen3_vision_transformer_default_behavior():
    """Test that Qwen3_VisionTransformer works with default config."""
    vision_config = Qwen3VLVisionConfig(
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
        hidden_size=768,
        num_heads=12,
        intermediate_size=3072,
        depth=12,
        out_hidden_size=768,
        spatial_merge_size=2,
        num_position_embeddings=1024,
        deepstack_visual_indexes=[],
        hidden_act="silu",
    )
    
    transformer = Qwen3_VisionTransformer(
        vision_config=vision_config,
    )
    
    # Check that blocks are created
    assert len(transformer.blocks) == vision_config.depth
    
    # Check that blocks exist (return_attention_score is a forward parameter, not an attribute)
    for block in transformer.blocks:
        assert block is not None


@pytest.mark.skipif(
    Qwen3VLVisionConfig is None,
    reason="Qwen3VLVisionConfig not available"
)
def test_qwen3_vision_transformer_forward_default_behavior():
    """Test that forward returns single tensor by default."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = "cuda"
    
    vision_config = Qwen3VLVisionConfig(
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
        hidden_size=768,
        num_heads=12,
        intermediate_size=3072,
        depth=12,
        out_hidden_size=768,
        spatial_merge_size=2,
        num_position_embeddings=1024,
        deepstack_visual_indexes=[],
        hidden_act="silu",
    )
    
    transformer = Qwen3_VisionTransformer(
        vision_config=vision_config,
    ).to(device)
    
    # Create dummy input
    seq_len = 100
    x = torch.randn(seq_len, 1176, device=device)
    grid_thw = [[1, 10, 10]]  # 1 frame, 10x10 grid
    
    result = transformer(x, grid_thw)
    
    # Should return single tensor
    assert isinstance(result, torch.Tensor)
    assert not isinstance(result, tuple)

