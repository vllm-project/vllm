#!/usr/bin/env python3
"""
Check the expert configuration for DeepSeek models
"""
import sys
from transformers import AutoConfig

def check_model_experts(model_name):
    print(f"\n{'='*60}")
    print(f"Checking model: {model_name}")
    print('='*60)
    
    # Load the model config
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # Check various expert-related attributes
    expert_attrs = [
        'n_routed_experts',
        'num_experts',
        'num_experts_per_tok',
        'n_shared_experts',
        'n_group',
        'moe_layer_freq',
        'first_k_dense_replace',
        'num_hidden_layers',
    ]
    
    print("\nModel Configuration:")
    for attr in expert_attrs:
        value = getattr(config, attr, 'Not found')
        print(f"  {attr}: {value}")
    
    # Calculate what would be EP_SIZE with different configurations
    print("\n" + "="*60)
    print("EP_SIZE calculations (EP_SIZE = TP_SIZE × DP_SIZE):")
    print("-"*60)
    
    n_routed = getattr(config, 'n_routed_experts', None)
    n_shared = getattr(config, 'n_shared_experts', 0) or 0
    
    if n_routed:
        print(f"  Routed experts: {n_routed}")
        print(f"  Shared experts: {n_shared}")
        print(f"  Total experts: {n_routed + n_shared}")
        print()
        
        # Check divisibility for different EP sizes
        for tp in [1, 2]:
            for dp in [1, 2, 4]:
                ep_size = tp * dp
                total = n_routed  # Note: usually only routed experts are distributed
                divisible = (total % ep_size == 0)
                result = "✓ OK" if divisible else f"✗ FAIL ({total} % {ep_size} = {total % ep_size})"
                print(f"  TP={tp}, DP={dp} → EP_SIZE={ep_size}: {result}")

# Test both models
models = [
    "deepseek-ai/DeepSeek-V2-Lite",
    "deepseek-ai/deepseek-moe-16b-base",
]

for model in models:
    try:
        check_model_experts(model)
    except Exception as e:
        print(f"Error checking {model}: {e}")

print("\n" + "="*60)
print("Summary:")
print("-"*60)
print("EPLB requires: global_num_experts % ep_size == 0")
print("For DP=2 with TP=1, EP_SIZE=2, so experts must be divisible by 2")
print("="*60)
