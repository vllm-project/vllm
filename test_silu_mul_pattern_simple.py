# test_with_custom_ops_debug.py
import torch
import os
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

# Enable pattern matching debug
os.environ['TORCHINDUCTOR_PATTERN_MATCH_DEBUG'] = '1'

print("="*80)
print("TESTING WITH CUSTOM OPS + DEBUG")
print("="*80)

def test_function_with_custom_ops(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Uses vLLM's custom silu_and_mul op (should match pattern!)"""
    # Pre-allocate output for silu_and_mul (mutation-based op)
    hidden = x.shape[-1] // 2
    silu_out = torch.empty(
        x.shape[:-1] + (hidden,),
        dtype=x.dtype,
        device=x.device
    )
    
    # Call the custom op with pre-allocated output
    torch.ops._C.silu_and_mul(silu_out, x)
    
    # This calls the custom quant op that MatcherQuantFP8 expects
    result, scales = per_token_group_quant_fp8(silu_out, group_size=128, use_ue8m0=False)
    return result, scales

x = torch.randn(4, 256*2, dtype=torch.float16, device="cuda")

# Step 1: Verify custom ops are present
print("\n[1] Verifying custom ops in graph...")

captured_graphs = []

def capturing_backend(gm, example_inputs):
    captured_graphs.append(gm)
    return gm.forward

compiled_fn = torch.compile(test_function_with_custom_ops, backend=capturing_backend)

with torch.no_grad():
    _ = compiled_fn(x)

has_silu_and_mul = False
has_quant = False

if captured_graphs:
    for node in captured_graphs[0].graph.nodes:
        if node.op == 'call_function':
            op_name = str(node.target)
            if 'silu_and_mul' in op_name:
                has_silu_and_mul = True
                print(f"  ✓ Found: {op_name}")
            if 'per_token_group' in op_name and 'fp8' in op_name:
                has_quant = True
                print(f"  ✓ Found: {op_name}")

if not (has_silu_and_mul and has_quant):
    print("\n❌ Missing custom ops - stopping here")
    exit(1)

print("\n✅ Both custom ops present")

# Step 2: Test with pass manager and detailed debugging
print("\n[2] Testing with vLLM pass manager...")

from vllm.config import VllmConfig, ModelConfig, CompilationConfig
from vllm.compilation.pass_manager import PostGradPassManager
from vllm.config import set_current_vllm_config
from vllm.compilation.inductor_pass import pass_context
from vllm.config.utils import Range
import torch._inductor.config as inductor_config

config = VllmConfig(
    model_config=ModelConfig(model="facebook/opt-125m", dtype=torch.float16, seed=0),
    compilation_config=CompilationConfig(
        custom_ops=["+quant_fp8"],
        pass_config={"fuse_act_quant": True}
    )
)

with pass_context(Range(start=1, end=8)):
    with set_current_vllm_config(config):
        pm = PostGradPassManager()
        pm.configure(config)
        
        # Check what patterns are actually registered
        print("\n[3] Checking registered patterns in ActivationQuantFusionPass...")
        for pass_obj in pm.passes:
            if "ActivationQuant" in type(pass_obj).__name__:
                print(f"  Found pass: {type(pass_obj).__name__}")
                if hasattr(pass_obj, 'patterns'):
                    print(f"  Number of patterns: {len(pass_obj.patterns.patterns)}")
                    
                    # Try to inspect pattern details
                    for i, pattern in enumerate(pass_obj.patterns.patterns):
                        print(f"    Pattern {i}: {pattern}")
        
        # Register with inductor
        inductor_config.post_grad_custom_post_pass = pm
        
        # Enable more debugging
        inductor_config.trace.enabled = True
        inductor_config.trace.debug_log = True
        
        print("\n[4] Compiling with pattern matching...")
        print("   Environment: TORCHINDUCTOR_PATTERN_MATCH_DEBUG=1")
        print("   (Look for pattern match attempts below)\n")
        print("-" * 80)
        
        compiled_fn_fused = torch.compile(
            test_function_with_custom_ops,
            backend="inductor",
            fullgraph=True,
        )
        
        with torch.no_grad():
            result, scales = compiled_fn_fused(x)
        
        print("-" * 80)
        print(f"\n[5] Compilation complete")
        print(f"   Output: {result.shape}")
        print(f"   Scales: {scales.shape}")
        
        # Check if pattern fired by looking for the debug message
        print("\n[6] Did pattern fire?")
        print("   Look above for: '🔥 FUSED KERNEL TRIGGERED!'")
        print("   OR look for pattern match debug output")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("If you see pattern match debug output above but NO '🔥' message,")
print("it means the pattern was ATTEMPTED but didn't match.")
print("\nCheck the pattern match debug output for WHY it didn't match.")
print("="*80)