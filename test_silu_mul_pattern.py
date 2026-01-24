# test_silu_mul_pattern.py
import torch
import torch.nn.functional as F
from vllm.compilation.pass_manager import PostGradPassManager
from vllm.compilation.inductor_pass import pass_context
from vllm.config import VllmConfig, ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig, CompilationConfig, set_current_vllm_config
from vllm.config.utils import Range
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
import torch._inductor.config as inductor_config

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

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

with set_current_vllm_config(config):
    quant_fp8 = QuantFP8(
        static=False,
        group_shape=GroupShape(1, 128),
        column_major_scales=False,
        use_ue8m0=False,
    )

# Create test function
# Create test function
def silu_mul_then_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    d = x.shape[-1] // 2
    silu_out = F.silu(x[..., :d]) * x[..., d:]
    result, scales = quant_fp8(silu_out)
    return result, scales

# Create a debug backend to see the graph
def debug_backend(gm: torch.fx.GraphModule, example_inputs):
    print("\n" + "="*80)
    print("CAPTURED GRAPH BEFORE FUSION:")
    print("="*80)
    print(gm.graph)
    print("\nGRAPH OPS:")
    for node in gm.graph.nodes:
        print(f"  {node.op:15} {node.target if hasattr(node, 'target') else ''}")
    print("="*80 + "\n")
    
    # Now apply the pass manager and continue compilation
    from torch._inductor.compile_fx import compile_fx
    return compile_fx(gm, example_inputs)

print("\n" + "="*80)
print("RUNNING TEST")
print("="*80)

x = torch.randn(16, 4096 * 2, dtype=torch.float16, device="cuda")

# Baseline
print("\n1. Baseline (no compilation)...")
with torch.no_grad():
    baseline_out, baseline_scales = silu_mul_then_quant(x)
print(f"   Output: {baseline_out.shape}, Scales: {baseline_scales.shape}")

# Compiled
print("\n2. Setting up compilation with custom passes...")

print("\n3. Compiling and running inside pass_context...")

with pass_context(Range(start=1, end=8)):
    pass_manager = PostGradPassManager()
    pass_manager.configure(config)
    
    # Debug: Check patterns
    for pass_obj in pass_manager.passes:
        if "ActivationQuant" in type(pass_obj).__name__:
            print(f"  ✓ Found: {type(pass_obj).__name__}")
            print(f"    Patterns: {len(pass_obj.patterns.patterns)}")
    
    # Register with inductor
    inductor_config.post_grad_custom_post_pass = pass_manager
    print("   ✓ Pass manager registered with inductor")
    
    # Use debug backend to see the graph
    compiled_fn = torch.compile(silu_mul_then_quant, backend=debug_backend, fullgraph=True)
    
    print("\n4. Running compiled function...")
    
    with torch.no_grad():
        compiled_out, compiled_scales = compiled_fn(x)

print(f"\n   Output: {compiled_out.shape}, Scales: {compiled_scales.shape}")

# Check results
match = torch.allclose(baseline_out.float(), compiled_out.float(), atol=1.0)
print(f"   Results match: {match}")

if not match:
    print("\n   Pattern did not fire - checking why...")
    print("   - Did you see 'FUSED KERNEL TRIGGERED!' above?")
    print("   - If not, check the captured graph ops above")

print("\n" + "="*80)