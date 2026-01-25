# test_with_custom_ops.py
import torch
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

print("="*80)
print("TESTING WITH CUSTOM OPS")
print("="*80)

def test_function_with_custom_ops(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Uses vLLM's custom silu_and_mul op (should match pattern!)"""
    # This calls the custom op that MatcherSiluAndMul expects
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

# Step 1: Capture graph
print("\n[1] Capturing graph to verify custom ops are present...")

from torch._dynamo import optimize

captured_graphs = []

def capturing_backend(gm, example_inputs):
    print("\n[CAPTURED GRAPH]")
    for node in gm.graph.nodes:
        if node.op == 'call_function':
            print(f"  {node.target}")
    captured_graphs.append(gm)
    return gm.forward

compiled_fn = torch.compile(test_function_with_custom_ops, backend=capturing_backend)

with torch.no_grad():
    _ = compiled_fn(x)

# Step 2: Check if both custom ops are present
print("\n[2] Checking for required custom ops...")

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

if has_silu_and_mul and has_quant:
    print("\n✅ Both custom ops present - pattern CAN match!")
else:
    print("\n❌ Missing custom ops - pattern CANNOT match")
    if not has_silu_and_mul:
        print("   Missing: silu_and_mul")
    if not has_quant:
        print("   Missing: per_token_group_fp8_quant")

# Step 3: Test with vLLM's pass manager
if has_silu_and_mul and has_quant:
    print("\n[3] Testing with vLLM pass manager...")
    
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
            inductor_config.post_grad_custom_post_pass = pm
            
            print("   Compiling with pattern matching enabled...")
            print("   (Watch for 🔥 FUSED KERNEL TRIGGERED!)\n")
            
            compiled_fn_fused = torch.compile(
                test_function_with_custom_ops,
                backend="inductor",
                fullgraph=True,
            )
            
            with torch.no_grad():
                result, scales = compiled_fn_fused(x)
            
            print(f"\n   Output: {result.shape}")
            print(f"   Scales: {scales.shape}")

print("\n" + "="*80)