# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Comprehensive test for the Hybrid Attention Architecture (Attention + SSM).

This script performs two levels of testing:
1.  **Synthetic Verification**: Validates the "plumbing" of the Hybrid backend and
    SSM adapter using the deterministic 'prefix-sum' mode described in the
    research paper. This runs without a real model checkpoint.
2.  **Integration Test**: Runs the full vLLM engine with a Hybrid model (if provided)
    to ensure end-to-end stability.

Usage:
    python examples/offline_inference/test_hybrid_architecture.py
    
    # To run the integration test with a specific model:
    export HYBRID_MODEL="your-org/step3-text-hybrid"
    python examples/offline_inference/test_hybrid_architecture.py
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Optional

# Ensure we can import from vllm source
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from vllm import LLM, SamplingParams
from vllm.model_executor.layers.hybrid_attn_layer import HybridAttentionImpl
from vllm.model_executor.layers.hybrid_ssm_adapter import HybridSSMAdapter
from vllm.config import ModelConfig, CacheConfig, VllmConfig
from vllm.utils import is_hip

# Mocking utilities to avoid needing a full model config for the synthetic test
from unittest.mock import MagicMock

def create_mock_configs():
    """Create minimal mock configs to satisfy HybridSSMAdapter initialization."""
    model_config = MagicMock(spec=ModelConfig)
    model_config.dtype = torch.float32
    model_config.get_num_attention_heads.return_value = 4
    model_config.get_head_size.return_value = 16
    
    cache_config = MagicMock(spec=CacheConfig)
    cache_config.mamba_cache_dtype = "auto"
    cache_config.mamba_ssm_cache_dtype = "auto"
    cache_config.sliding_window = None
    
    return model_config, cache_config

def run_synthetic_verification():
    """
    Executes the 'Prefix-Sum Verification' described in the research paper (Section 4.1).
    
    This test:
    1.  Instantiates the real `HybridSSMAdapter` and `HybridAttentionImpl`.
    2.  Sets `VLLM_HYBRID_SSM_MODE='prefix_sum'`.
    3.  Feeds a sequence of tokens [x1, x2, x3...].
    4.  Mocks the attention branch to return 0.
    5.  Verifies that the output is [x1, x1+x2, x1+x2+x3...].
    
    This proves that the SSM history branch is correctly wired and can carry 
    state across tokens, independent of the standard attention mechanism.
    """
    print("\n" + "=" * 80)
    print("Running Synthetic Prefix-Sum Verification (Paper Section 4.1)")
    print("=" * 80)
    
    # Force the environment variable for the adapter
    os.environ["VLLM_HYBRID_SSM_MODE"] = "prefix_sum"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Setup parameters
    num_heads = 2
    head_size = 8
    num_kv_heads = 2
    intermediate_size = 32
    ssm_state_size = 16
    num_tokens = 5
    
    # Create mock configs
    model_config, cache_config = create_mock_configs()
    
    # Instantiate the REAL adapter
    print("Initializing HybridSSMAdapter...")
    adapter = HybridSSMAdapter(
        hidden_size=num_heads * head_size,
        ssm_state_size=ssm_state_size,
        conv_kernel_size=3,
        intermediate_size=intermediate_size,
        model_config=model_config,
        cache_config=cache_config,
        prefix="test_layer.ssm"
    ).to(device)
    
    # Instantiate the backend
    print("Initializing HybridAttentionImpl...")
    impl = HybridAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=1.0,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    
    # Mock the Triton attention implementation to return zeros
    # This isolates the contribution of the SSM branch.
    class _ZeroTritonImpl:
        def forward(self, *args, output, **kwargs):
            output.zero_()
            
    impl._triton_impl = _ZeroTritonImpl()
    
    # Create inputs: Simple 1, 2, 3, 4, 5 sequence broadcast across heads
    base = torch.arange(1, num_tokens + 1, dtype=torch.float32, device=device)
    # Shape: (num_tokens, num_heads, head_size)
    query = base.view(num_tokens, 1, 1).expand(num_tokens, num_heads, head_size)
    key = torch.zeros_like(query)
    value = torch.zeros_like(query)
    kv_cache = torch.empty(0, device=device)
    output = torch.empty_like(query)
    
    # Create minimal metadata
    class _Meta:
        def __init__(self, n):
            self.num_prefill_tokens = n
            self.num_actual_tokens = n
            # For the adapter's logic to trigger, it checks attn_metadata.mamba_metadata
            # or direct attributes if not wrapped. The adapter unwraps HybridAttentionMetadata.
            # Here we pass a simple object that has the attributes the adapter looks for
            # in the 'prefix_sum' path (mainly num_prefill_tokens).
            self.mamba_metadata = self
            self.triton_metadata = None

    attn_metadata = _Meta(num_tokens)
    
    # Create a dummy layer object as expected by HybridAttentionImpl
    class _ToyLayer:
        def __init__(self, ssm_adapter):
            self.ssm_adapter = ssm_adapter
    
    layer = _ToyLayer(adapter)
    
    print("Executing forward pass...")
    # The backend calls: triton_impl.forward (zeros) + ssm_adapter.forward (prefix sum)
    out = impl.forward(
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
    )
    
    # Verify results
    expected_prefix = torch.cumsum(base, dim=0).view(num_tokens, 1, 1).expand_as(out)
    
    try:
        assert torch.allclose(out, expected_prefix)
        print("\nSUCCESS: Output matches expected prefix sums!")
        print(f"Input (first head, first dim): {base.tolist()}")
        print(f"Output (first head, first dim): {out[:, 0, 0].tolist()}")
        print("State persistence verification passed.")
    except AssertionError:
        print("\nFAILURE: Output does not match expected prefix sums.")
        print(f"Expected: {expected_prefix[:, 0, 0].tolist()}")
        print(f"Got:      {out[:, 0, 0].tolist()}")
        raise

def run_integration_test(model_name: str):
    """
    Runs the full vLLM engine with the specified hybrid model.
    """
    print("\n" + "=" * 80)
    print(f"Running Integration Test with Model: {model_name}")
    print("=" * 80)
    
    prompts = [
        "The future of AI is",
        "Hybrid architectures allow",
    ]
    sampling_params = SamplingParams(temperature=0.0)
    
    try:
        llm = LLM(model=model_name)
        outputs = llm.generate(prompts, sampling_params)
        
        print("\nGenerated Outputs:\n" + "-" * 60)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt:    {prompt!r}")
            print(f"Output:    {generated_text!r}")
            print("-" * 60)
        print("\nIntegration test completed successfully (no crashes).")
        
    except Exception as e:
        print(f"\nIntegration test failed with error: {e}")
        print("Note: Ensure the model config has 'model_type': 'step3_text' and 'use_hybrid_step3_attn': true")

if __name__ == "__main__":
    # 1. Run the synthetic verification (no model required)
    try:
        run_synthetic_verification()
    except Exception as e:
        print(f"Synthetic verification failed: {e}")
        sys.exit(1)
        
    # 2. Run integration test if a model is provided
    model_name = os.getenv("HYBRID_MODEL")
    if model_name:
        run_integration_test(model_name)
    else:
        print("\n[INFO] Skipping integration test. Set HYBRID_MODEL env var to run end-to-end.")
        print("Example: export HYBRID_MODEL='your-org/step3-text-hybrid'")

