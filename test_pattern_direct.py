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

# Now test pattern matching
from vllm.compilation.pass_manager import PostGradPassManager
from vllm.config import set_current_vllm_config

with set_current_vllm_config(config):
    pass_manager = PostGradPassManager()
    pass_manager.configure(config)
    
    print(f"\n✓ Pass manager configured")
    print(f"  Number of passes: {len(pass_manager.passes)}")
    
    # Find your fusion pass
    found = False
    for i, pass_obj in enumerate(pass_manager.passes):
        pass_name = type(pass_obj).__name__
        if "SiluAndMul" in pass_name:
            print(f"\n🎯 FOUND: {pass_name} at index {i}")
            if hasattr(pass_obj, 'patterns'):
                print(f"  Number of patterns registered: {len(pass_obj.patterns.patterns)}")
            found = True
    
    if not found:
        print("\n⚠️  SiluAndMulQuantFusionPass NOT in pass manager")
        print("  Registered passes:")
        for i, p in enumerate(pass_manager.passes):
            print(f"    [{i}] {type(p).__name__}")

print("\n" + "="*80)
