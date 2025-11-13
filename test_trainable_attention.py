#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that TrainableFlashAttention can be imported from vLLM and works correctly.
"""

import torch

print("Testing vLLM TrainableFlashAttention import and usage...")
print("=" * 70)

# Test 1: Import
print("\n1. Testing import from vllm.model_executor.layers...")
try:
    from vllm.model_executor.layers import TrainableFlashAttention

    print("   ✓ Import successful")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

# Test 2: Instantiation
print("\n2. Testing instantiation...")
try:
    attn = TrainableFlashAttention(hidden_size=256, num_heads=4, dropout=0.1)
    print("   ✓ Created TrainableFlashAttention")
    print(f"   - hidden_size: {attn.hidden_size}")
    print(f"   - num_heads: {attn.num_heads}")
    print(f"   - head_dim: {attn.head_dim}")
except Exception as e:
    print(f"   ✗ Instantiation failed: {e}")
    exit(1)

# Test 3: Forward pass
print("\n3. Testing forward pass...")
try:
    batch_size = 2
    seq_len = 16
    hidden_size = 256

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    output = attn(hidden_states)

    print("   ✓ Forward pass successful")
    print(f"   - Input shape: {hidden_states.shape}")
    print(f"   - Output shape: {output.shape}")

    assert output.shape == (batch_size, seq_len, hidden_size)
    print("   ✓ Output shape correct")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Test 4: Backward pass
print("\n4. Testing backward pass (training mode)...")
try:
    attn.train()
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)

    output = attn(hidden_states)
    loss = output.sum()
    loss.backward()

    print("   ✓ Backward pass successful")
    print(f"   - Input gradients computed: {hidden_states.grad is not None}")
    print(f"   - QKV weight gradients computed: {attn.qkv.weight.grad is not None}")
    print(
        f"   - Output projection gradients computed: "
        f"{attn.o_proj.weight.grad is not None}"
    )

    assert hidden_states.grad is not None
    assert attn.qkv.weight.grad is not None
    assert attn.o_proj.weight.grad is not None
    print("   ✓ All gradients computed correctly")
except Exception as e:
    print(f"   ✗ Backward pass failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Test 5: Eval mode
print("\n5. Testing eval mode (inference)...")
try:
    attn.eval()
    with torch.no_grad():
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        output = attn(hidden_states)

    print("   ✓ Eval mode successful")
    print(f"   - Output shape: {output.shape}")
except Exception as e:
    print(f"   ✗ Eval mode failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print("\n" + "=" * 70)
print("✓ All TrainableFlashAttention tests passed!")
print("=" * 70)
print("\nPhase 1 Complete: vLLM TrainableFlashAttention module working!")
