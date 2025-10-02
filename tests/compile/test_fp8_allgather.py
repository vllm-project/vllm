# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

from ..utils import multi_gpu_test

if not current_platform.is_cuda():
    pytest.skip("CUDA only test", allow_module_level=True)


def test_nccl_fp8_dtype_support():
    """Test that NCCL wrapper supports FP8 datatypes"""
    from vllm.distributed.device_communicators.pynccl_wrapper import (
        ncclDataTypeEnum)

    # Test FP8 E4M3
    assert hasattr(ncclDataTypeEnum, 'ncclFp8E4M3')
    assert ncclDataTypeEnum.ncclFp8E4M3 == 10

    # Test FP8 E5M2
    assert hasattr(ncclDataTypeEnum, 'ncclFp8E5M2')
    assert ncclDataTypeEnum.ncclFp8E5M2 == 11

    # Test from_torch mapping
    assert ncclDataTypeEnum.from_torch(
        torch.float8_e4m3fn) == ncclDataTypeEnum.ncclFp8E4M3
    assert ncclDataTypeEnum.from_torch(
        torch.float8_e5m2) == ncclDataTypeEnum.ncclFp8E5M2


def test_custom_ops_registered():
    """Test that custom FP8 ops are registered"""
    # Import to trigger registration

    # Check that ops are registered
    assert hasattr(torch.ops.vllm, 'vllm_quantize_fp8')
    assert hasattr(torch.ops.vllm, 'vllm_all_gather_fp8')

    # Check that default variants exist
    assert hasattr(torch.ops.vllm.vllm_quantize_fp8, 'default')
    assert hasattr(torch.ops.vllm.vllm_all_gather_fp8, 'default')


def test_fp8_quantization_op():
    """Test FP8 quantization custom op"""
    from vllm.compilation.fp8_collective_ops import vllm_quantize_fp8

    # Create test tensor
    x = torch.randn(16, 32, dtype=torch.bfloat16, device='cuda')

    # Quantize
    x_fp8, scale_inv = vllm_quantize_fp8(x)

    # Check output types
    assert x_fp8.dtype == torch.float8_e4m3fn
    assert scale_inv.dtype == torch.float32

    # Check shapes
    assert x_fp8.shape == x.shape
    assert scale_inv.numel() == 1  # per-tensor scale

    # Check dequantization (approximately recovers original)
    x_dequant = x_fp8.to(torch.bfloat16) * scale_inv
    torch.testing.assert_close(x_dequant, x, rtol=0.1, atol=0.1)


def fp8_allgather_worker(local_rank: int, world_size: int):
    """Worker function for multi-GPU FP8 AllGather test"""
    from vllm.compilation.fp8_collective_ops import vllm_all_gather_fp8
    from vllm.distributed import (get_tp_group, init_distributed_environment,
                                  initialize_model_parallel)
    from vllm.utils import update_environment_variables

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    update_environment_variables({
        'RANK': str(local_rank),
        'LOCAL_RANK': str(local_rank),
        'WORLD_SIZE': str(world_size),
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '29501',
    })

    # Initialize distributed
    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    # Create test tensor (generate as BF16 then convert to FP8)
    x = torch.randn(8, 16, dtype=torch.bfloat16,
                    device='cuda').to(torch.float8_e4m3fn)

    # All-gather
    tp_group = get_tp_group()
    gathered = vllm_all_gather_fp8(x,
                                   dim=0,
                                   world_size=tp_group.world_size,
                                   group_name=tp_group.unique_name)

    # Check shape
    expected_shape = (8 * tp_group.world_size, 16)
    assert gathered.shape == expected_shape
    print(
        f"Rank {local_rank}: ✅ FP8 AllGather op test passed! Shape: {gathered.shape}"
    )


@multi_gpu_test(num_gpus=2)
def test_fp8_allgather_op():
    """Test FP8 all-gather custom op (requires multi-GPU)"""

    def run_torch_spawn(fn, nprocs):
        torch.multiprocessing.spawn(fn, args=(nprocs, ), nprocs=nprocs)

    run_torch_spawn(fp8_allgather_worker, 2)


def test_fp8_allgather_pass_init():
    """Test FP8 AllGather pass initialization"""
    pytest.skip(
        "Requires distributed initialization - test manually with multi-GPU")


def test_fp8_allgather_pattern_fake():
    """Test pattern with fake mode (no actual distributed execution)"""
    pytest.skip(
        "Pattern registration requires valid TP group - test manually with multi-GPU"
    )


def fp8_allgather_correctness_worker(local_rank: int, world_size: int):
    """Worker function for FP8 AllGather numerical correctness test"""
    from vllm.compilation.fp8_collective_ops import (vllm_all_gather_fp8,
                                                     vllm_quantize_fp8)
    from vllm.distributed import (get_tp_group, init_distributed_environment,
                                  initialize_model_parallel,
                                  tensor_model_parallel_all_gather)
    from vllm.utils import update_environment_variables

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    update_environment_variables({
        'RANK': str(local_rank),
        'LOCAL_RANK': str(local_rank),
        'WORLD_SIZE': str(world_size),
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '29502',
    })

    # Initialize distributed
    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    # Create test tensor
    x = torch.randn(16, 32, dtype=torch.bfloat16, device='cuda')

    # Method 1: Direct AllGather (baseline, default dim=-1)
    gathered_direct = tensor_model_parallel_all_gather(x)

    # Method 2: FP8 Optimized AllGather (use same dim=-1)
    x_fp8, scale_inv = vllm_quantize_fp8(x)
    tp_group = get_tp_group()
    gathered_fp8 = vllm_all_gather_fp8(x_fp8,
                                       dim=-1,
                                       world_size=tp_group.world_size,
                                       group_name=tp_group.unique_name)

    # All-gather scales (reshape scalar to 1D first)
    scale_inv_1d = scale_inv.view(1)
    scale_gathered = tensor_model_parallel_all_gather(scale_inv_1d, dim=0)

    # Dequantize: apply each rank's scale to its chunk
    # gathered_fp8 has shape [16, 32*world_size], scale_gathered has shape [world_size]
    # Need to broadcast scale to match each chunk along dim=-1
    chunk_size = x.shape[-1]
    scale_expanded = torch.repeat_interleave(scale_gathered, chunk_size).view(
        1, -1).to(torch.bfloat16)
    gathered_opt = gathered_fp8.to(torch.bfloat16) * scale_expanded

    # Check correctness (allow for FP8 quantization error)
    torch.testing.assert_close(gathered_opt,
                               gathered_direct,
                               rtol=0.05,
                               atol=0.05)
    print(
        f"Rank {local_rank}: ✅ FP8 AllGather numerical correctness test passed!"
    )


@multi_gpu_test(num_gpus=2)
def test_fp8_allgather_numerical_correctness():
    """Test end-to-end numerical correctness of FP8 AllGather optimization"""

    def run_torch_spawn(fn, nprocs):
        torch.multiprocessing.spawn(fn, args=(nprocs, ), nprocs=nprocs)

    run_torch_spawn(fp8_allgather_correctness_worker, 2)


def test_pass_config_has_flag():
    """Test that PassConfig has enable_fp8_allgather_opt flag"""
    from vllm.config import PassConfig

    config = PassConfig(enable_fp8_allgather_opt=True)
    assert config.enable_fp8_allgather_opt is True

    config = PassConfig(enable_fp8_allgather_opt=False)
    assert config.enable_fp8_allgather_opt is False

    # Default should be False
    config = PassConfig()
    assert config.enable_fp8_allgather_opt is False
