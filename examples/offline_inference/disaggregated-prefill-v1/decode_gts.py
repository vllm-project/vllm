#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GTS Decode Worker - following v1 pattern
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


def read_prefill_results():
    """Read prefill results"""
    try:
        with open("prefill_results.json") as f:
            results = json.load(f)
        print(f"📖 Loaded {len(results)} prefill results from prefill_results.json")
        return results
    except FileNotFoundError:
        print("❌ Error: prefill_results.json file not found")
        exit(1)


def main():
    print("🎯 GTS Decode Worker")
    print("=" * 40)
    
    # Load prefill results
    prefill_results = read_prefill_results()
    
    # Use the original prompts (decode worker would receive partial prompts + KV cache)
    prompts = [r['prompt'] for r in prefill_results]
    
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=15)

    # GTS config for decode (kv_consumer)
    ktc = KVTransferConfig(
        kv_connector="GTSConnector",
        kv_role="kv_consumer",
        kv_rank=1,
        engine_id="decode_engine_gts",
        gts_server_address="localhost:6174"
    )

    print(f"📊 Configuration: Decode worker (kv_consumer)")
    print(f"   - Model: meta-llama/Meta-Llama-3-8B-Instruct")
    print(f"   - GTS Server: localhost:6174")
    print(f"   - Engine ID: decode_engine_gts")
    
    try:
        llm = LLM(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            kv_transfer_config=ktc,
            max_model_len=512,
            gpu_memory_utilization=0.6,
            enforce_eager=True,
            trust_remote_code=True
        )

        print("🔄 Loading KV cache from GTS and continuing generation...")
        outputs = llm.generate(prompts, sampling_params)
        
        print("🎉 Decode completed! Final results:")
        print("=" * 60)
        
        for i, (output, prefill_result) in enumerate(zip(outputs, prefill_results)):
            prompt = output.prompt
            decode_text = output.outputs[0].text
            prefill_text = prefill_result['prefill_output']
            
            print(f"Request {i+1}:")
            print(f"  Original: '{prompt}'")
            print(f"  Prefill:  '{prefill_text}'")  
            print(f"  Decode:   '{decode_text}'")
            print(f"  Full:     '{prompt}{decode_text}'")
            print("-" * 40)
        
        print("✅ GTS disaggregated inference completed!")
        
    except Exception as e:
        print(f"❌ Decode failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()