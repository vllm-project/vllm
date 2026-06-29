import pytest
import torch
import math

from vllm.v1.attention.ops.oscar_ops import (
    build_rotation_matrix,
    quantize_int2,
    dequantize_int2,
    store_kv_to_cache,
    load_kv_from_cache,
)

@pytest.mark.parametrize("head_dim", [64, 80, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_oscar_pack_unpack_roundtrip(head_dim, dtype):
    device = torch.device("cuda:0")
    # Generate random test data
    k = torch.randn(2, 4, head_dim, dtype=dtype, device=device)
    
    # 1. Quantize
    k_packed, scale, zero = quantize_int2(k)
    
    # 2. Dequantize
    k_deq = dequantize_int2(k_packed, scale, zero, head_dim).to(dtype)
    
    # Values should be roughly equal (quantization error is expected)
    # The key is that the shape and general magnitude matches.
    assert k_deq.shape == k.shape
    
    # Max error should be bounded by the quantization step (range / 3)
    k_min = k.amin(dim=-1, keepdim=True)
    k_max = k.amax(dim=-1, keepdim=True)
    step = (k_max - k_min) / 3.0
    
    diff = (k_deq - k).abs()
    # The max error for nearest neighbor rounding is half the step size,
    # plus some small fp16 rounding error.
    assert (diff <= (step / 2.0 + 1e-3)).all()


@pytest.mark.parametrize("head_dim", [64, 80, 128])
def test_oscar_rotation_matrix(head_dim):
    device = torch.device("cuda:0")
    R = build_rotation_matrix(head_dim, device)
    
    assert R.shape == (head_dim, head_dim)
    assert R.device == device
    
    # Check orthogonality: R @ R^T == I
    I = torch.eye(head_dim, device=device)
    RR_T = R @ R.T
    
    assert torch.allclose(RR_T, I, atol=1e-5)


@pytest.mark.parametrize("head_dim", [80, 128])
def test_oscar_cache_store_load(head_dim):
    device = torch.device("cuda:0")
    
    # 1 block, block_size=16, 2 KV heads, slot_size
    from vllm.v1.attention.ops.oscar_ops import get_cache_layout
    _, _, _, _, slot_size = get_cache_layout(head_dim)
    
    cache = torch.zeros(1, 16, 2, slot_size, dtype=torch.uint8, device=device)
    
    N = 3
    Hk = 2
    
    # Random K/V in rotated space
    k_rot = torch.randn(N, Hk, head_dim, dtype=torch.float32, device=device)
    v_rot = torch.randn(N, Hk, head_dim, dtype=torch.float32, device=device)
    
    slot_mapping = torch.tensor([0, 5, 15], dtype=torch.long, device=device)
    
    # Store
    store_kv_to_cache(
        k_rot, v_rot, cache, slot_mapping, 
        clip_ratio_k=0.0, clip_ratio_v=0.0
    )
    
    # Load
    k_loaded, v_loaded = load_kv_from_cache(cache, slot_mapping, head_dim)
    
    assert k_loaded.shape == k_rot.shape
    assert v_loaded.shape == v_rot.shape
    
    # Loaded values should be quantized versions of the original (similar to the first test)
    # They should not be identically equal due to quantization, but they shouldn't be zero.
    assert not torch.allclose(k_loaded, torch.zeros_like(k_loaded))
    assert not torch.allclose(v_loaded, torch.zeros_like(v_loaded))
