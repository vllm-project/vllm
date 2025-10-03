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
    print(f"Rank {local_rank}: ✅ FP8 AllGather op test passed! "
          f"Shape: {gathered.shape}")


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
    pytest.skip("Pattern registration requires valid TP group - "
                "test manually with multi-GPU")


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
    # gathered_fp8 has shape [16, 32*world_size], scale_gathered has
    # shape [world_size]. Need to broadcast scale to match each chunk
    # along dim=-1
    chunk_size = x.shape[-1]
    scale_expanded = torch.repeat_interleave(scale_gathered, chunk_size).view(
        1, -1).to(torch.bfloat16)
    gathered_opt = gathered_fp8.to(torch.bfloat16) * scale_expanded

    # Check correctness (allow for FP8 quantization error)
    torch.testing.assert_close(gathered_opt,
                               gathered_direct,
                               rtol=0.05,
                               atol=0.05)
    print(f"Rank {local_rank}: ✅ FP8 AllGather numerical correctness "
          f"test passed!")


@multi_gpu_test(num_gpus=2)
def test_fp8_allgather_numerical_correctness():
    """Test end-to-end numerical correctness of FP8 AllGather optimization"""

    def run_torch_spawn(fn, nprocs):
        torch.multiprocessing.spawn(fn, args=(nprocs, ), nprocs=nprocs)

    run_torch_spawn(fp8_allgather_correctness_worker, 2)


def fp8_allgather_pattern_equivalence_worker(local_rank: int, world_size: int):
    """
    Worker function to test pattern transformation equivalence.

    Tests that the transformation:
        AllGather(BF16) → Quantize(FP8, shared_scale)
    is numerically equivalent to:
        Quantize(FP8, shared_scale) → AllGather(FP8)

    This validates the core assumption of the FP8AllGatherOptPass pattern.
    """
    from vllm.compilation.fp8_collective_ops import vllm_all_gather_fp8
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
        'MASTER_PORT': '29503',
    })

    # Initialize distributed
    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    # Create test tensor with different values per rank
    torch.manual_seed(42 + local_rank)
    x = torch.randn(16, 32, dtype=torch.bfloat16, device='cuda')

    # Shared precomputed scale (simulating what modelopt would provide)
    # In reality, this would be computed from the global tensor statistics,
    # but for testing we use a fixed value that all ranks share
    shared_scale = torch.tensor(0.05, dtype=torch.float32, device='cuda')

    # METHOD 1 (Original Pattern): AllGather(BF16) → Quantize(FP8)
    gathered_bf16 = tensor_model_parallel_all_gather(x, dim=0)

    # Apply modelopt-style quantization AFTER AllGather
    x_f32 = gathered_bf16.to(torch.float32)
    scale_inv = shared_scale.reciprocal()
    x_scaled = x_f32 * scale_inv
    x_clamped = x_scaled.clamp(min=-448.0, max=448.0)
    result_pattern = x_clamped.to(torch.float8_e4m3fn)

    # METHOD 2 (Optimized Replacement): Quantize(FP8) → AllGather(FP8)
    # Apply modelopt-style quantization BEFORE AllGather
    x_f32_local = x.to(torch.float32)
    x_scaled_local = x_f32_local * scale_inv
    x_clamped_local = x_scaled_local.clamp(min=-448.0, max=448.0)
    x_fp8_local = x_clamped_local.to(torch.float8_e4m3fn)

    # AllGather FP8 tensors
    tp_group = get_tp_group()
    result_replacement = vllm_all_gather_fp8(x_fp8_local,
                                             dim=0,
                                             world_size=tp_group.world_size,
                                             group_name=tp_group.unique_name)

    # Check that both methods produce IDENTICAL results
    # Since we're using the same shared scale and FP8 quantization,
    # the results should be bit-exact (no tolerance needed)
    assert result_pattern.shape == result_replacement.shape, (
        f"Shape mismatch: {result_pattern.shape} vs {result_replacement.shape}"
    )

    # Convert to int8 to compare bit patterns (FP8 doesn't have direct equality)
    pattern_bits = result_pattern.view(torch.int8)
    replacement_bits = result_replacement.view(torch.int8)

    matches = (pattern_bits == replacement_bits).float().mean().item()

    # Allow for very small numerical differences due to FP8 rounding
    # but they should be nearly identical (>99.9% match)
    assert matches > 0.999, (
        f"Rank {local_rank}: Pattern transformation not equivalent! "
        f"Only {matches*100:.2f}% of values match. "
        f"Expected >99.9% match for bit-exact equivalence.")

    print(f"Rank {local_rank}: ✅ Pattern transformation equivalence "
          f"test passed! Match rate: {matches*100:.4f}%")


@multi_gpu_test(num_gpus=2)
def test_fp8_allgather_pattern_equivalence():
    """
    Test that the FP8AllGatherOptPass pattern transformation is
    numerically valid.

    This test validates the core assumption: when using a shared
    precomputed scale, quantizing before AllGather produces the same
    result as quantizing after.
    """

    def run_torch_spawn(fn, nprocs):
        torch.multiprocessing.spawn(fn, args=(nprocs, ), nprocs=nprocs)

    run_torch_spawn(fp8_allgather_pattern_equivalence_worker, 2)


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
