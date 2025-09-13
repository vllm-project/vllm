#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GTS Prefill Worker - following v1 pattern
"""

import os
import sys
import json

# Add mock GTS to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../tests/v1/kv_connector'))
from test_gts_connector_mock import MockGTS

# Mock the gts module
sys.modules['gts'] = MockGTS()

# Set platform environment variables to fix device detection
os.environ['VLLM_PLATFORM'] = 'cuda'

# Workaround for platform detection issue when C extensions aren't built
try:
    from vllm.platforms import current_platform
    if not hasattr(current_platform, 'device_type') or not current_platform.device_type:
        current_platform.device_type = 'cuda'
except:
    pass

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def main():
    print("🚀 GTS Prefill Worker")
    print("=" * 40)
    
    prompts = [
        "Hello, my name is",
        "The capital of France is", 
        "Tell me about artificial intelligence",
    ]
    
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=5)

    # GTS config for prefill (kv_producer)
    ktc = KVTransferConfig(
        kv_connector="GTSConnector",
        kv_role="kv_producer",
        kv_rank=0,
        engine_id="prefill_engine_gts",
        gts_server_address="localhost:6174"
    )

    print(f"📊 Configuration: Prefill worker (kv_producer)")
    print(f"   - Model: meta-llama/Meta-Llama-3-8B-Instruct")
    print(f"   - GTS Server: localhost:6174")
    print(f"   - Engine ID: prefill_engine_gts")
    
    try:
        llm = LLM(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            kv_transfer_config=ktc,
            max_model_len=512,
            gpu_memory_utilization=0.6,
            enforce_eager=True,
            trust_remote_code=True
        )

        print(f"📝 Processing {len(prompts)} prompts...")
        outputs = llm.generate(prompts, sampling_params)
        
        print("✅ Prefill completed!")
        
        # Save results for decode worker
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append({
                'prompt': output.prompt,
                'prefill_output': generated_text,
                'full_prompt': output.prompt + generated_text
            })
            print(f"  '{output.prompt}' → '{generated_text}'")
        
        # Save to JSON for decode worker
        with open('prefill_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("💾 Results saved to prefill_results.json")
        print("🔄 KV cache should be available in GTS for decode workers")
        
    except Exception as e:
        print(f"❌ Prefill failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()