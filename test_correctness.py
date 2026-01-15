import torch
import torch.nn.functional as F
from vllm import _custom_ops as ops

print("="*60)
print("TEST 2: Correctness (vs Unfused Implementation)")
print("="*60)

def unfused_silu_mul_group_quant(input_tensor, group_size):
    """Reference implementation: separate SiLU+Mul and quantization"""
    batch, width = input_tensor.shape
    hidden = width // 2
    
    # Split into gate and up
    gate = input_tensor[:, :hidden]
    up = input_tensor[:, hidden:]
    
    # SiLU + Mul
    silu_out = F.silu(gate) * up
    
    # Group quantization (simple version)
    num_groups = hidden // group_size
    output = torch.empty(batch, hidden, dtype=torch.float8_e4m3fn, device='cuda')
    scales = torch.empty(batch, num_groups, dtype=torch.float32, device='cuda')
    
    for b in range(batch):
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            group = silu_out[b, start:end]
            
            # Compute scale
            group_max = group.abs().max().item()
            scale = group_max / 448.0 if group_max > 0 else 1e-10
            scales[b, g] = scale
            
            # Quantize
            scaled = group / scale
            output[b, start:end] = scaled.to(torch.float8_e4m3fn)
    
    return output, scales

# Test parameters
batch_size = 2
hidden_size = 256
group_size = 128

torch.manual_seed(42)
input_tensor = torch.randn(
    batch_size, hidden_size * 2,
    dtype=torch.float16,
    device='cuda'
) * 10  # Scale up for better FP8 range usage

print(f"Testing with batch={batch_size}, hidden={hidden_size}, group_size={group_size}")

# Unfused reference
ref_output, ref_scales = unfused_silu_mul_group_quant(input_tensor.float(), group_size)

# Fused implementation
fused_output, fused_scales = ops.silu_and_mul_per_block_quant(
    input_tensor,
    group_size=[1, group_size],
    quant_dtype=torch.float8_e4m3fn,
    is_scale_transposed=False,
)

# Compare outputs
output_diff = (fused_output.float() - ref_output.float()).abs()
scale_diff = (fused_scales - ref_scales).abs()

print(f"\nOutput comparison:")
print(f"  Max difference: {output_diff.max().item():.6f}")
print(f"  Mean difference: {output_diff.mean().item():.6f}")
print(f"  Relative error: {(output_diff / (ref_output.float().abs() + 1e-8)).mean().item():.6f}")

print(f"\nScale comparison:")
print(f"  Max difference: {scale_diff.max().item():.6f}")
print(f"  Mean difference: {scale_diff.mean().item():.6f}")
print(f"  Relative error: {(scale_diff / (ref_scales.abs() + 1e-8)).mean().item():.6f}")

# Check if close enough (FP8 has limited precision)
output_close = output_diff.max() < 1.0  # FP8 tolerance
scale_close = scale_diff.max() < 0.01   # Scale should be very close

if output_close and scale_close:
    print("\n✅ CORRECTNESS TEST PASSED!")
    print("   Fused and unfused implementations produce similar results.")
else:
    print("\n⚠️  CORRECTNESS TEST FAILED!")
    print("   Differences exceed tolerance.")
    if not output_close:
        print(f"   Output max diff: {output_diff.max().item():.6f} (threshold: 1.0)")
    if not scale_close:
        print(f"   Scale max diff: {scale_diff.max().item():.6f} (threshold: 0.01)")
