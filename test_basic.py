import torch
from vllm import _custom_ops as ops

print("="*60)
print("TEST 1: Basic Functionality")
print("="*60)

# Small test
batch_size = 4
hidden_size = 1024
group_size = 128

input_tensor = torch.randn(
    batch_size, hidden_size * 2,
    dtype=torch.float16,
    device='cuda'
)

print(f"Input shape: {input_tensor.shape}")

try:
    output, scales = ops.silu_and_mul_per_block_quant(
        input_tensor,
        group_size=[1, group_size],
        quant_dtype=torch.float8_e4m3fn,
        is_scale_transposed=False,
    )
    
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Scales shape: {scales.shape}")
    print(f"✅ Output dtype: {output.dtype}")
    print(f"✅ Scales dtype: {scales.dtype}")
    
    # Check values are reasonable
    print(f"\nOutput stats:")
    print(f"  Min: {output.float().min().item():.4f}")
    print(f"  Max: {output.float().max().item():.4f}")
    print(f"  Mean: {output.float().mean().item():.4f}")
    
    print(f"\nScale stats:")
    print(f"  Min: {scales.min().item():.6f}")
    print(f"  Max: {scales.max().item():.6f}")
    print(f"  Mean: {scales.mean().item():.6f}")
    
    print("\n🎉 BASIC TEST PASSED!\n")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
