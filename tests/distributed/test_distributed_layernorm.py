import os
import random

import pytest
import ray
import torch
import torch.distributed as dist

from ..utils import (ensure_model_parallel_initialized,
                     init_test_distributed_environment, multi_process_parallel)

from vllm.distributed import tensor_model_parallel_all_gather

def parallel_rms_norm_test(tp_size, pp_size, rank, distributed_init_port):
    """Test function that runs on each GPU to compare RMSNorm implementations."""
    # Setup distributed environment
    del os.environ["CUDA_VISIBLE_DEVICES"]
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    ensure_model_parallel_initialized(tp_size, pp_size)

    hidden_size = 1024
    num_tokens = 16
    dtype = torch.float16

    # Create both norm layers
    standard_norm = RMSNorm(hidden_size, eps=1e-6).to(device)
    parallel_norm = ParallelRMSNorm(hidden_size, eps=1e-6).to(device)

    # Create input tensor
    torch.manual_seed(42 + rank)  # Ensure deterministic but different inputs per rank
    parallel_input = torch.randn(num_tokens, hidden_size // tp_size, dtype=dtype, device=device)
    reference_input = tensor_model_parallel_all_gather(parallel_input)
    
    # Create a residual tensor for testing residual path
    residual = torch.randn_like(input_tensor)
    reference_residual = tensor_model_parallel_all_gather(residual)

    # Run forward passes
    with torch.no_grad():
        # Test without residual
        standard_out = standard_norm.forward(input_tensor)
        parallel_out = parallel_norm.forward(input_tensor)
        
        # Test with residual
        standard_out_residual, residual1 = standard_norm.forward_native(reference_input_tensor, reference_residual.clone())
        parallel_out_residual, residual2 = parallel_norm.forward_native(input_tensor, residual.clone())

    # Synchronize before comparing
    torch.cuda.synchronize()

    gathered_parallel_output = tensor_model_parallel_all_gather(parallel_out)
    gathered_parallel_residual = tensor_model_parallel_all_gather(residual2)
    
    # Convert to float32 for comparison
    standard_out = standard_out.float()
    parallel_out = parallel_out.float()
    standard_out_residual = standard_out_residual.float()
    parallel_out_residual = parallel_out_residual.float()

    # Compare outputs
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    
    assert torch.allclose(standard_out, parallel_out, rtol=rtol, atol=atol), \
        f"Outputs without residual don't match. Max diff: {(standard_out - parallel_out).abs().max()}"
    
    assert torch.allclose(standard_out_residual, parallel_out_residual, rtol=rtol, atol=atol), \
        f"Outputs with residual don't match. Max diff: {(standard_out_residual - parallel_out_residual).abs().max()}"
    
    assert torch.allclose(residual1, residual2, rtol=rtol, atol=atol), \
        f"Residual outputs don't match. Max diff: {(residual1 - residual2).abs().max()}"

@pytest.mark.parametrize("tp_size", [2])
def test_parallel_rms_norm(tp_size):
    """Main test function that spawns multiple processes."""
    world_size = tp_size
    if world_size > torch.cuda.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    
    multi_process_parallel(
        tp_size=tp_size,
        pp_size=1,  # We only need TP for this test
        test_target=parallel_rms_norm_test
    )

