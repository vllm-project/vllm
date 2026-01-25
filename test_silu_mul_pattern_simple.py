# test_silu_mul_pattern_simple.py
import torch
import torch.nn.functional as F

print("="*80)
print("SIMPLE PATTERN TEST")
print("="*80)

# Step 1: Check if pattern is registered
print("\n[1/3] Checking pattern registration...")

from vllm.config import VllmConfig, ModelConfig, CompilationConfig
from vllm.compilation.pass_manager import PostGradPassManager
from vllm.config import set_current_vllm_config

config = VllmConfig(
    model_config=ModelConfig(model="facebook/opt-125m", dtype=torch.float16, seed=0),
    compilation_config=CompilationConfig(
        custom_ops=["+quant_fp8"],
        pass_config={"fuse_act_quant": True}
    )
)

with set_current_vllm_config(config):
    pm = PostGradPassManager()
    pm.configure(config)
    
    found = False
    for p in pm.passes:
        if "ActivationQuant" in type(p).__name__:
            print(f"   ✓ Found: {type(p).__name__}")
            if hasattr(p, 'patterns'):
                print(f"   ✓ Patterns: {len(p.patterns.patterns)}")
            found = True
    
    if not found:
        print("   ✗ Pattern NOT found!")
        exit(1)

# Step 2: Check if kernel is callable
print("\n[2/3] Checking if kernel is callable...")

import vllm._custom_ops as ops

x = torch.randn(16, 4096*2, dtype=torch.float16, device="cuda")

try:
    out, scales = ops.silu_and_mul_per_block_quant(
        x,
        group_size=128,
        quant_dtype=torch.float8_e4m3fn,
        is_scale_transposed=False,
    )
    print(f"   ✓ Kernel callable: out={out.shape}, scales={scales.shape}")
except Exception as e:
    print(f"   ✗ Kernel failed: {e}")
    exit(1)

# Step 3: Test pattern matching with torch.compile
print("\n[3/3] Testing pattern matching with torch.compile...")

from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

def test_function(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Function that should trigger pattern matching."""
    hidden = x.shape[-1] // 2
    gate, up = x.split(hidden, dim=-1)
    silu_out = F.silu(gate) * up
    result, scales = per_token_group_quant_fp8(silu_out, group_size=128, use_ue8m0=False)
    return result, scales

# Baseline
print("   Running baseline (no compilation)...")
x_test = torch.randn(16, 4096*2, dtype=torch.float16, device="cuda")
with torch.no_grad():
    baseline_out, baseline_scales = test_function(x_test)
print(f"   Baseline: out={baseline_out.shape}")

# Compiled
print("   Compiling function...")
print("   (If pattern fires, you'll see: 🔥 FUSED KERNEL TRIGGERED!)\n")

import torch._inductor.config as inductor_config

with set_current_vllm_config(config):
    pm_compile = PostGradPassManager()
    pm_compile.configure(config)
    inductor_config.post_grad_custom_post_pass = pm_compile
    
    compiled_fn = torch.compile(test_function, backend="inductor")
    
    with torch.no_grad():
        compiled_out, compiled_scales = compiled_fn(x_test)
    
    print(f"\n   Compiled: out={compiled_out.shape}")
    
    match = torch.allclose(baseline_out.float(), compiled_out.float(), atol=1.0)
    print(f"   Results match: {match}")
    
    if match:
        print("\n   Note: Results matching doesn't guarantee pattern fired")
        print("   Look for '🔥 FUSED KERNEL TRIGGERED!' message above")

print("\n" + "="*80)
print("DONE")
print("="*80)
```

**What this tests:**

1. ✅ **Pattern is registered** in `ActivationQuantFusionPass`
2. ✅ **Kernel is callable** directly via `ops.silu_and_mul_per_block_quant`
3. ⚠️ **Pattern matching with torch.compile** (look for debug print)

**Expected output if working:**
```
[1/3] Checking pattern registration...
   ✓ Found: ActivationQuantFusionPass
   ✓ Patterns: 8

[2/3] Checking if kernel is callable...
   ✓ Kernel callable: out=torch.Size([16, 4096]), scales=torch.Size([16, 32])

[3/3] Testing pattern matching...
   Running baseline...
   Baseline: out=torch.Size([16, 4096])
   Compiling...
   
🔥 FUSED KERNEL TRIGGERED! input.shape=torch.Size([16, 8192]), group_size=128

   Compiled: out=torch.Size([16, 4096])
   Results match: True