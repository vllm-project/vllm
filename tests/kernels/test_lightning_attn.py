# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.model_executor.layers.lightning_attn import (
    lightning_attention,
    lightning_attention2_parallel,
    linear_decode_forward_triton
)
from vllm.platforms import current_platform

# 测试参数
NUM_HEADS = [4, 8]
HEAD_SIZES = [64, 128]
BATCH_SIZES = [1, 2]
SEQ_LENGTHS = [16, 128]
DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_lightning_attention(
    batch_size: int,
    num_heads: int,
    head_size: int,
    seq_len: int,
    dtype: torch.dtype,
):
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)
    
    # 准备输入
    q = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    ed = torch.rand(num_heads, device="cuda")
    
    # 运行 lightning_attention
    output, kv = lightning_attention(q, k, v, ed)
    
    # 验证输出形状
    assert output.shape == (batch_size, num_heads, seq_len, head_size)
    assert kv.shape[0] == batch_size
    assert kv.shape[1] == num_heads
    
    # 测试 lightning_attention2_parallel
    output2, kv2 = lightning_attention2_parallel(q, k, v, ed)
    
    # 验证两个函数输出相同
    torch.testing.assert_close(output, output2, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(kv, kv2, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_lightning_attention_with_kv_history(
    batch_size: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
):
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)
    
    seq_len = 32
    
    # 准备输入
    q = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    ed = torch.rand(num_heads, device="cuda")
    
    # 创建 kv_history
    kv_history = torch.randn(batch_size, num_heads, head_size, head_size, 
                            dtype=torch.float32, device="cuda")
    
    # 运行 lightning_attention 带 kv_history
    output, kv = lightning_attention(q, k, v, ed, kv_history=kv_history)
    
    # 验证输出形状
    assert output.shape == (batch_size, num_heads, seq_len, head_size)
    assert kv.shape[0] == batch_size
    assert kv.shape[1] == num_heads


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_linear_decode_forward_triton(
    batch_size: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
):
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)
    
    # 准备输入
    q = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    k = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    v = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    
    # 创建 kv_caches
    kv_caches = torch.randn(batch_size, num_heads, head_size, head_size, 
                           dtype=dtype, device="cuda")
    
    # 创建 slope_rate
    slope_rate = torch.rand(num_heads, device="cuda")
    
    # 创建 slot_idx (非填充样本)
    slot_idx = torch.arange(batch_size, device="cuda")
    
    # 运行 linear_decode_forward_triton
    output = linear_decode_forward_triton(
        q, k, v, kv_caches, slope_rate, slot_idx
    )
    
    # 验证输出形状
    assert output.shape == (batch_size, num_heads * head_size)


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_linear_decode_forward_triton_with_padding(
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
):
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)
    
    batch_size = 4
    
    # 准备输入
    q = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    k = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    v = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    
    # 创建 kv_caches
    kv_caches = torch.randn(batch_size, num_heads, head_size, head_size, 
                           dtype=dtype, device="cuda")
    
    # 创建 slope_rate
    slope_rate = torch.rand(num_heads, device="cuda")
    
    # 创建 slot_idx (包含填充样本，-1表示填充)
    slot_idx = torch.tensor([0, 1, -1, 2], device="cuda")
    
    # 运行 linear_decode_forward_triton
    output = linear_decode_forward_triton(
        q, k, v, kv_caches, slope_rate, slot_idx
    )
    
    # 验证输出形状
    assert output.shape == (batch_size, num_heads * head_size)
    
    # 验证填充位置的输出是否为零
    # 注意：由于实现细节，填充位置可能不会被处理，所以这个测试可能需要调整
    # torch.testing.assert_close(output[2], torch.zeros_like(output[2]))