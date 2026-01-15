import torch
import torch.nn.functional as F
from vllm import _custom_ops as ops
import time

print("="*60)
print("TEST 3: Performance Benchmark")
print("="*60)

def unfused_implementation(input_tensor):
    """Unfused: SiLU+Mul then quantize separately"""
    hidden = input_tensor.shape[-1] // 2
    gate, up = input_tensor.split(hidden, dim=-1)
    silu_out = F.silu(gate) * up
    
    # Simplified group quant (not optimized)
    output = silu_out.to(torch.float8_e4m3fn)
    scales = torch.ones(silu_out.shape[0], silu_out.shape[1] // 128, 
                       dtype=torch.float32, device='cuda')
    return output, scales

def fused_implementation(input_tensor):
    """Fused: Single kernel"""
    return ops.silu_and_mul_per_block_quant(
        input_tensor,
        group_size=[1, 128],
        quant_dtype=torch.float8_e4m3fn,
        is_scale_transposed=False,
    )

# Test configurations
configs = [
    (16, 4096, "Small (16 tokens, 4K hidden)"),
    (128, 4096, "Medium (128 tokens, 4K hidden)"),
    (512, 4096, "Large (512 tokens, 4K hidden)"),
    (16, 14336, "LLaMA-7B FFN (16 tokens, 14K hidden)"),
]

num_warmup = 10
num_iterations = 100

for batch_size, hidden_size, desc in configs:
    print(f"\n{desc}")
    print("-" * 60)
    
    input_tensor = torch.randn(
        batch_size, hidden_size * 2,
        dtype=torch.float16,
        device='cuda'
    )
    
    # Warmup
    for _ in range(num_warmup):
        _ = unfused_implementation(input_tensor)
        _ = fused_implementation(input_tensor)
    torch.cuda.synchronize()
    
    # Benchmark unfused
    start = time.time()
    for _ in range(num_iterations):
        _ = unfused_implementation(input_tensor)
    torch.cuda.synchronize()
    unfused_time = (time.time() - start) / num_iterations * 1000  # ms
    
    # Benchmark fused
    start = time.time()
    for _ in range(num_iterations):
        _ = fused_implementation(input_tensor)
    torch.cuda.synchronize()
    fused_time = (time.time() - start) / num_iterations * 1000  # ms
    
    speedup = unfused_time / fused_time
    
    print(f"  Unfused: {unfused_time:.3f} ms")
    print(f"  Fused:   {fused_time:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    if speedup > 1.0:
        print(f"  ✅ Fused is faster!")
    else:
        print(f"  ⚠️  Fused is slower (might need optimization)")

print("\n" + "="*60)
print("PERFORMANCE TEST COMPLETE")
print("="*60)
