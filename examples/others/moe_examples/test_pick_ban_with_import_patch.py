#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pick & Ban Routing Test with Import-based Patching

This script imports the patch module before vLLM to ensure patches
are applied in all processes.
"""

import logging

import torch

# Set up logging to see our debug messages
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Now import vLLM (patches should be applied)
from vllm import LLM, SamplingParams  # noqa: E402


def clear_gpu_memory():
    """Clear GPU memory and garbage collect."""
    import gc
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def test_pick_ban_routing():
    """Test with Pick & Ban routing."""
    print("=" * 50)
    print("TESTING PICK & BAN ROUTING WITH IMPORT PATCHING")
    print("=" * 50)
    
    try:
        # Load model (patches should already be applied via import)
        model_path = (
            "/home/lifd/.cache/modelscope/hub/models/Qwen/Qwen1.5-MoE-A2.7B-Chat"
        )
        
        print("🔄 Loading model...")
        llm = LLM(
            model=model_path,
            gpu_memory_utilization=0.8,  # Increase back to 60% since we have 2 GPUs
            max_model_len=64,  # Very short sequence length
            dtype="half",
            trust_remote_code=True,
            tensor_parallel_size=2,  # Use 2 GPUs
            pipeline_parallel_size=1,
            max_num_batched_tokens=128,  # Very small batch size
            max_num_seqs=1,
            enforce_eager=True,  # Disable CUDA graphs
            disable_custom_all_reduce=True,  # Disable custom kernels
            kv_cache_dtype="fp8",  # Use FP8 for KV cache to save memory
            cpu_offload_gb=4,  # Offload some weights to CPU
            swap_space=4,  # Use swap space for additional memory
        )
        
        print("✅ Model loaded successfully!")
        
        # Simple test prompt
        test_prompt = "你好，请介绍一下你自己。"
        
        sampling_params = SamplingParams(temperature=0.7, max_tokens=50, top_p=0.8)
        
        print(f"\n🧪 Testing with prompt: {test_prompt}")
        print("🔍 Watch for Pick & Ban routing debug messages...")
        print(
            "🔍 Look for '🚀🚀🚀 PATCHED Qwen2MoeSparseMoeBlock.__init__ called!' "
            "messages above!"
        )
        print(
            "🔍 Look for '🚀🚀🚀 PICK & BAN ROUTING FUNCTION CALLED!' "
            "messages during generation!"
        )
        print("-" * 50)
        
        # Generate response - this should trigger our routing function
        outputs = llm.generate([test_prompt], sampling_params)
        
        response = outputs[0].outputs[0].text
        print(f"📝 Response: {response}")
        
        del llm
        clear_gpu_memory()
        
        print("\n✅ Test completed successfully!")
        print("🔍 Check the logs above for Pick & Ban routing debug messages.")
        return True
        
    except Exception as e:
        print(f"❌ Pick & Ban routing test failed: {e}")
        import traceback
        
        traceback.print_exc()
        return False


def main():
    """Main function."""
    
    print("Qwen1.5-MoE-A2.7B Pick & Ban Routing Test with Import Patching")
    print("=" * 50)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    print(f"🔍 Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(
            f"  💾 GPU {i} Memory: "
            f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB"
        )
    
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    # Run test
    success = test_pick_ban_routing()
    
    if success:
        print("\n🎉 Pick & Ban routing test completed!")
        print(
            "💡 If you see '🚀🚀🚀 PATCHED Qwen2MoeSparseMoeBlock.__init__ called!' "
            "messages above,"
        )
        print("   it means our patch is working in worker processes!")
        print("💡 If you see '🚀🚀🚀 PICK & BAN ROUTING FUNCTION CALLED!' messages,")
        print("   it means the Pick & Ban algorithm is being executed!")
    else:
        print("\n❌ Test failed. Check the error messages above.")


if __name__ == "__main__":
    main()