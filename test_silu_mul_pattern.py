# test_silu_mul_pattern.py
import torch
import torch.nn.functional as F

from vllm.compilation.pass_manager import PostGradPassManager
from vllm.compilation.inductor_pass import pass_context
from vllm.config import VllmConfig, ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig, CompilationConfig
from vllm.config.utils import Range
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

print("="*80)
print("TESTING SILU+MUL+BLOCK QUANT PATTERN")
print("="*80)

# Create config
compilation_config = CompilationConfig(
    custom_ops=["+quant_fp8"],
    pass_config={
        "fuse_act_quant": True,
    }
)

model_config = ModelConfig(
    model="facebook/opt-125m",
    tokenizer="facebook/opt-125m",
    tokenizer_mode="auto",
    trust_remote_code=False,
    dtype=torch.float16,
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

print("✓ Config created")

# Set up pass manager (like the test does)
with pass_context(Range(start=1, end=8)):
    pass_manager = PostGradPassManager()
    pass_manager.configure(config)

    print(f"✓ Pass manager configured with {len(pass_manager.passes)} passes")

    # Check for ActivationQuantFusionPass
    for pass_obj in pass_manager.passes:
        if "ActivationQuant" in type(pass_obj).__name__:
            print(f"  ✓ Found: {type(pass_obj).__name__}")
            if hasattr(pass_obj, 'patterns'):
                print(f"    Patterns: {len(pass_obj.patterns.patterns)}")
                # NEW: Print what patterns are registered
                for i, pattern in enumerate(pass_obj.patterns.patterns):
                    print(f"      Pattern {i}: {pattern}")

# Create test function
def silu_mul_then_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = x.shape[-1] // 2
    gate, up = x.split(hidden, dim=-1)
    silu_out = F.silu(gate) * up
    result, scales = per_token_group_quant_fp8(silu_out, group_size=128, use_ue8m0=False)
    return result, scales

print("\n" + "="*80)
print("RUNNING TEST")
print("="*80)

x = torch.randn(16, 4096 * 2, dtype=torch.float16, device="cuda")

# Baseline
print("\n1. Baseline (no compilation)...")
with torch.no_grad():
    baseline_out, baseline_scales = silu_mul_then_quant(x)
print(f"   Output: {baseline_out.shape}, Scales: {baseline_scales.shape}")

# Compiled - set pass manager in inductor config
print("\n2. Setting up compilation with custom passes...")
import torch._inductor.config as inductor_config

with pass_context(Range(start=1, end=8)):
    # Re-create to ensure fresh state
    pass_manager_for_compile = PostGradPassManager()
    pass_manager_for_compile.configure(config)
    
    # Register with inductor
    inductor_config.post_grad_custom_post_pass = pass_manager_for_compile
    print("   ✓ Pass manager registered with inductor")
    
    # Compile AND run inside the context
    compiled_fn = torch.compile(silu_mul_then_quant, backend="inductor", fullgraph=True)
    
    with torch.no_grad():
        compiled_out, compiled_scales = compiled_fn(x)

print(f"\n   Output: {compiled_out.shape}, Scales: {compiled_scales.shape}")

# Check results
match = torch.allclose(baseline_out.float(), compiled_out.float(), atol=1.0)
print(f"   Results match: {match}")

if not match:
    print("\n   ⚠️  Pattern did not fire - checking why...")
    print("   - Did you see '🔥 FUSED KERNEL TRIGGERED!' above?")
    print("   - If not, the pattern didn't match")

print("\n" + "="*80)
