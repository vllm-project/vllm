import os

import pytest
import ray
import torch
import torch.distributed

from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.layers.layernorm import ParallelRMSNorm, RMSNorm

from ..utils import (ensure_model_parallel_initialized,
                     init_test_distributed_environment, multi_process_parallel)


@ray.remote(num_gpus=1, max_calls=1)
def parallel_rms_norm_test(tp_size, pp_size, rank, distributed_init_port):
    """Test function that runs on each GPU comparing RMSNorm implementations."""
    # Setup distributed environment
    del os.environ["CUDA_VISIBLE_DEVICES"]
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(tp_size, pp_size, rank,
                                      distributed_init_port)
    ensure_model_parallel_initialized(tp_size, pp_size)

    hidden_size = 1024
    num_tokens = 16
    dtype = torch.float16

    # Create both norm layers
    reference_norm = RMSNorm(hidden_size, eps=1e-6).to(device)
    parallel_norm = ParallelRMSNorm(hidden_size, eps=1e-6).to(device)

    # Create input tensor
    torch.manual_seed(
        42 + rank)  # Ensure deterministic but different inputs per rank
    parallel_input = torch.randn(num_tokens,
                                 hidden_size // tp_size,
                                 dtype=dtype,
                                 device=device)
    reference_input = tensor_model_parallel_all_gather(parallel_input)

    # Create a residual tensor for testing residual path
    residual = torch.randn_like(parallel_input)
    reference_residual = tensor_model_parallel_all_gather(residual)

    # Run forward passes
    with torch.no_grad():
        # Test without residual
        reference_out = reference_norm.forward(reference_input)
        parallel_out = parallel_norm.forward(parallel_input)

        # Test with residual
        reference_out_residual, residual1 = reference_norm.forward_native(
            reference_input, reference_residual.clone())
        parallel_out_residual, residual2 = parallel_norm.forward_native(
            parallel_input, residual.clone())

    # Synchronize before comparing
    torch.cuda.synchronize()

    gathered_parallel_output = tensor_model_parallel_all_gather(parallel_out)
    gathered_parallel_residual = tensor_model_parallel_all_gather(residual2)

    # Convert to float32 for comparison
    reference_out = reference_out.float()
    parallel_out = gathered_parallel_output.float()
    reference_out_residual = reference_out_residual.float()
    parallel_out_residual = gathered_parallel_residual.float()

    # Compare outputs
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-5

    assert torch.allclose(reference_out, parallel_out, rtol=rtol, atol=atol)
    assert torch.allclose(reference_out_residual,
                          parallel_out_residual,
                          rtol=rtol,
                          atol=atol)
    assert torch.allclose(residual1, residual2, rtol=rtol, atol=atol)


@pytest.mark.parametrize("tp_size", [2])
def test_parallel_rms_norm(tp_size):
    """Main test function that spawns multiple processes."""
    world_size = tp_size
    if world_size > torch.cuda.device_count():
        pytest.skip("Not enough GPUs to run the test.")

    multi_process_parallel(
        tp_size=tp_size,
        pp_size=1,  # We only need TP for this test
        test_target=parallel_rms_norm_test)
