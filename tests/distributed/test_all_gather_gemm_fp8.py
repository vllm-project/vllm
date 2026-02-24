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

@pytest.fixture(scope="function",autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


def init_distributed():
    """Initialize distributed environment and GroupCoordinator."""
    import weakref
    from vllm.distributed.parallel_state import _groups, GroupCoordinator

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if not dist.is_initialized():
        print(f"Initializing distributed: rank {rank}, local_rank {local_rank}, world_size {world_size}")
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

    world_group = GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=local_rank,
        torch_distributed_backend="nccl",
        use_device_communicator=False,
        group_name="world",
    )
    _groups[dist.group.WORLD.group_name] = weakref.ref(world_group)

    return rank, local_rank, world_size, device, dist.group.WORLD, world_group


def run_shape_test(M, K, N, rank, world_size, device, dist_group, world_group):
    """Run a single shape through Helion FP8 all-gather + matmul."""
    torch.manual_seed(41)  # deterministic for all ranks

    M_per_rank = M // world_size

    # Inputs
    a_shared = torch.rand(M_per_rank, K, device=device, dtype=torch.bfloat16) * 0.05
    a_shared = a_shared.to(FP8_DTYPE)
    b = (torch.rand(K, N, device=device, dtype=torch.bfloat16) *0.1+ 0.05).T.contiguous().T
    b = b.to(FP8_DTYPE)

    scale_a = torch.rand((M_per_rank, 1), device=device, dtype=torch.float32) * 0.05 + 0.01
    scale_b = torch.rand((1, N), device=device, dtype=torch.float32) * 0.05 + 0.01

    #adding clamping to avoid nan, inf (overflow)
    min_val=1e-3 
    max_val = 0.02 * (1024 / max(K, N))

    scale_a = scale_a.clamp(min=min_val, max=max_val)
    scale_b = scale_b.clamp(min=min_val, max=max_val)
    # call the HelionOp
    candidate_splits = [1, 2, 4]
    for sp in candidate_splits:
        if M_per_rank % sp != 0:
            continue  # skip invalid splits
        print(f"Testing shape ({M}, {K}, {N}) with split {sp} (tokens per rank: {M_per_rank})")
        a_out, c = torch.ops.vllm.helion_all_gather_fp8_gemm(
            a_shared,
            b,
            scale_a,
            scale_b,
            world_size,
            dist_group.group_name,
            SPLITS_PER_RANK=sp,
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

#TODO: if we run this test with one shape it will pass, if. we run multiple it will fail to intalize with nccl.
@pytest.mark.parametrize("M,K,N", [
    #small shapes
    #(128, 32, 64),
    #(256, 1024, 1024),
    #medium shapes
    #(2048, 1024, 2048),
    #(2048, 4096, 4096),
    #(4096, 2048, 4096),
    #large shapes
    (4096, 5120, 5120),
    #(8192, 8192, 8192),
])
def test_helion_fp8_all_gather_matmul(M, K, N):
    rank, local_rank, world_size, device, dist_group, world_group = init_distributed()
    
    dist.barrier()
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    ConfigManager.reset_instance()
    _ = ConfigManager()

    try:
        run_shape_test(M, K, N, rank, world_size, device, dist_group, world_group)
        torch.cuda.synchronize()
        if rank == 0:
            print(f"Shape ({M}, {K}, {N}) PASSED")

    except Exception as e:
        print(f"Shape ({M}, {K}, {N}) FAILED: {e}")
        raise
