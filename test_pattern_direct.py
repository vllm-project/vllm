# test_pattern_direct.py
import torch
import torch.nn.functional as F

print("="*80)
print("DIRECT PATTERN MATCHING TEST")
print("="*80)

# Set up minimal config
from vllm.config import VllmConfig, ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig, CompilationConfig

compilation_config = CompilationConfig(
    custom_ops=["+quant_fp8"],
    pass_config={
        "fuse_act_quant": True,  # Explicitly enable
    }
)

model_config = ModelConfig(
    model="facebook/opt-125m",
    tokenizer="facebook/opt-125m",
    tokenizer_mode="auto",
    trust_remote_code=False,
    dtype="float16",
    seed=0,
)

scheduler_config = SchedulerConfig(
    max_model_len=256,
    is_encoder_decoder=False,
)

config = VllmConfig(
    model_config=model_config,
    cache_config=CacheConfig(block_size=16),
    parallel_config=ParallelConfig(),
    scheduler_config=scheduler_config,
    compilation_config=compilation_config,
)

print("✓ Config created with fuse_act_quant enabled")

# ============================================================================
# PART 1: Check if pattern is registered (your existing code)
# ============================================================================
from vllm.compilation.pass_manager import PostGradPassManager
from vllm.config import set_current_vllm_config

with set_current_vllm_config(config):
    pass_manager = PostGradPassManager()
    pass_manager.configure(config)
    
    print(f"\n✓ Pass manager configured")
    print(f"  Number of passes: {len(pass_manager.passes)}")
    
    # Find your fusion pass
    found_pass = False
    for i, pass_obj in enumerate(pass_manager.passes):
        pass_name = type(pass_obj).__name__
        if "Activation" in pass_name and "Quant" in pass_name:
            print(f"\n🎯 FOUND: {pass_name} at index {i}")
            if hasattr(pass_obj, 'patterns'):
                num_patterns = len(pass_obj.patterns.patterns)
                print(f"  Number of patterns registered: {num_patterns}")
                
                # Look for your specific pattern
                for j, pattern in enumerate(pass_obj.patterns.patterns):
                    if "SiluMul" in str(type(pattern)):
                        print(f"    Pattern {j}: {type(pattern).__name__}")
            found_pass = True
    
    if not found_pass:
        print("\n⚠️  ActivationQuantFusionPass NOT in pass manager")

print("\n" + "="*80)

# ============================================================================
# PART 2: Actually test if pattern matching works (NEW!)
# ============================================================================
print("\nTESTING ACTUAL PATTERN MATCHING")
print("="*80)

from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

# Create a function with the exact pattern
def silu_mul_then_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """This should match your pattern!"""
    hidden = x.shape[-1] // 2
    gate, up = x.split(hidden, dim=-1)
    
    # SiLU + Mul (this is what MatcherSiluAndMul looks for)
    silu_out = F.silu(gate) * up
    
    # Block quantization (this is what MatcherQuantFP8 looks for)
    result, scales = per_token_group_quant_fp8(
        silu_out, 
        group_size=128,  # Must match one of your registered patterns
        use_ue8m0=False
    )
    
    return result, scales

# Test without compilation (baseline)
print("\n1. Running WITHOUT compilation (baseline)...")
x = torch.randn(16, 4096 * 2, dtype=torch.float16, device="cuda")

with torch.no_grad():
    baseline_out, baseline_scales = silu_mul_then_quant(x)

print(f"   Output: {baseline_out.shape}, dtype: {baseline_out.dtype}")
print(f"   Scales: {baseline_scales.shape}")

# Test WITH compilation (should trigger pattern matching)
print("\n2. Compiling with torch.compile...")

with set_current_vllm_config(config):
    # This should use the pass_manager with your patterns
    compiled_fn = torch.compile(
        silu_mul_then_quant,
        backend="inductor",
        fullgraph=True,
    )
    
    print("\n3. Running compiled version...")
    print("   (Watch for 'FUSED KERNEL TRIGGERED!' message)\n")
    
    with torch.no_grad():
        compiled_out, compiled_scales = compiled_fn(x)
    
    print(f"\n   Compiled output: {compiled_out.shape}")
    print(f"   Compiled scales: {compiled_scales.shape}")
    
    # Check if results match
    outputs_match = torch.allclose(
        baseline_out.float(), 
        compiled_out.float(), 
        atol=1.0  # Allow ±1 for quantization precision
    )
    scales_match = torch.allclose(baseline_scales, compiled_scales, rtol=1e-5)
    
    print(f"\n   Outputs match: {outputs_match}")
    print(f"   Scales match: {scales_match}")
    
    if outputs_match and scales_match:
        print("\n   ✓ Pattern matching appears to work correctly!")
    else:
        print("\n   ⚠️  Results don't match - pattern may not have fired")

print("\n" + "="*80)
print("If you saw '🔥 FUSED KERNEL TRIGGERED!', your pattern matched!")
print("="*80)
