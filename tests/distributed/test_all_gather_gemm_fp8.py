# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import torch
import torch.distributed as dist
import pytest
from vllm.platforms import current_platform
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.distributed.all_gather_gemm_fp8 import (
    helion_all_gather_fp8_gemm  # This triggers the direct_register_custom_op call
)

FP8_DTYPE = current_platform.fp8_dtype()

# TODO: test the helion_picker! and add more shapes to test.
def skip_if_platform_unsupported():
    try:
        from vllm.kernels.helion.utils import get_canonical_gpu_name

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        platform = get_canonical_gpu_name()

        try:
            config_manager = ConfigManager.get_instance()
        except RuntimeError:
            config_manager = ConfigManager()

        configs = config_manager.get_platform_configs("helion_matmul_w_progress_fp8", platform)
        if len(configs) == 0:
            pytest.skip("Current GPU platform not supported for helion_matmul_w_progress_fp8 kernel")

    except (ImportError, RuntimeError, KeyError):
        pytest.skip("Error detecting platform support for helion_matmul_w_progress_fp8 kernel")

@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


TEST_SHAPES = [
    #(512, 8192, 2048),
    (128, 32, 64)
]
@pytest.mark.parametrize("M,N,K", TEST_SHAPES)
def test_helion_fp8_all_gather_matmul(M, K, N):
    """Test Helion FP8 all-gather followed by matmul operation.
    
    Run with:
        VLLM_USE_HELION_BACKEND=1 torchrun --nproc_per_node=2 -m pytest tests/distributed/test_all_gather_gemm_fp8.py -v -s
        or
        VLLM_USE_HELION_BACKEND=1  python -m torch.distributed.run --standalone     --nproc-per-node 4     --rdzv-backend c10d --rdzv-endpoint localhost:0     -m pytest tests/distributed/test_all_gather_gemm_fp8.py 
    """
    # torchrun sets these environment variables automatically
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Setup device - CRITICAL: each process gets its own GPU
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # Initialize distributed with torchrun's env vars
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    
    # Register dist.group.WORLD in vLLM's _groups registry
    from vllm.distributed.parallel_state import _groups, GroupCoordinator
    import weakref
    
    # Create a minimal GroupCoordinator wrapping WORLD
    world_group = GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=local_rank,
        torch_distributed_backend="nccl",
        use_device_communicator=False,
        group_name="world",
    )
    
    # Register it in _groups using dist.group.WORLD's name as key
    dist_group = dist.group.WORLD
    assert dist_group is not None
    # Store a weak reference to the GroupCoordinator in _groups so the kernel can access it without preventing garbage collection.
    _groups[dist_group.group_name] = weakref.ref(world_group)
    
    # Test parameters
    M_per_rank = M // world_size

    # Create inputs
    a_shared = torch.rand(M_per_rank, K, device=device, dtype=torch.float32) * 0.05
    a_shared = a_shared.to(FP8_DTYPE)

    b = (torch.rand(K, N, device=device, dtype=torch.float32)  * 0.05).T.contiguous().T
    b= b.to(FP8_DTYPE)
    scale_a = torch.rand((M_per_rank, 1), device=device, dtype=torch.float32) * 0.1 + 0.05
    scale_b = torch.rand((1, N), device=device, dtype=torch.float32) * 0.1 + 0.05
    # call the HelionOp
    a_out, c = torch.ops.vllm.helion_all_gather_fp8_gemm(
        a_shared,
        b,
        scale_a,
        scale_b,
        world_size,
        dist_group.group_name,
    )
    # Compute golden reference
    ag_golden, mm_golden = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
        a_shared,
        [b],
        scale_a,
        [scale_b],
        gather_dim=0,
        biases=[None],
        result_scales=[None],
        out_dtypes=[torch.bfloat16],
        use_fast_accum=[False],
        group_name=dist_group.group_name,
    )
    # Compare results
    torch.testing.assert_close(a_out, ag_golden)
    torch.testing.assert_close(
        c, mm_golden[0].to(torch.bfloat16), 
        rtol=1e-1, 
        atol=1e-1
    )
    
