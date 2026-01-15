import torch
from vllm import _custom_ops as ops

print("="*60)
print("TEST 4: Integration Test (Different Configurations)")
print("="*60)

test_cases = [
    # (batch, hidden, group_size, transposed, description)
    (4, 1024, 128, False, "Standard config"),
    (4, 1024, 64, False, "Smaller groups"),
    (4, 1024, 128, True, "Transposed scales"),
    (1, 4096, 128, False, "Single token, large hidden"),
    (256, 512, 128, False, "Many tokens, small hidden"),
]

for batch, hidden, group_size, transposed, desc in test_cases:
    print(f"\nTesting: {desc}")
    print(f"  Config: batch={batch}, hidden={hidden}, group={group_size}, transposed={transposed}")
    
    try:
        input_tensor = torch.randn(batch, hidden * 2, dtype=torch.float16, device='cuda')
        output, scales = ops.silu_and_mul_per_block_quant(
            input_tensor,
            group_size=[1, group_size],
            quant_dtype=torch.float8_e4m3fn,
            is_scale_transposed=transposed,
        )
        
        expected_scale_shape = [batch, hidden // group_size]
        if transposed:
            expected_scale_shape = [hidden // group_size, batch]
        
        assert output.shape == (batch, hidden), f"Wrong output shape: {output.shape}"
        assert list(scales.shape) == expected_scale_shape, f"Wrong scale shape: {scales.shape}"
        
        print(f"  ✅ Passed")
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")

print("\n" + "="*60)
print("INTEGRATION TEST COMPLETE")
print("="*60)
