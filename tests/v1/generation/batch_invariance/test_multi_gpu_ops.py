# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
from functools import reduce

import pytest
import torch
import torch.multiprocessing as mp

from tests.utils import multi_gpu_test
from vllm.distributed.parallel_state import (init_distributed_environment,
                                             initialize_model_parallel)
from vllm.model_executor.layers.batch_invariant import init_batch_invariance
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.platforms import current_platform
from vllm.utils import update_environment_variables


def get_open_port():
    """Get an available port for distributed testing."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def run_parallel_op_test_worker(local_rank: int, world_size: int,
                                master_port: int, test_config: dict, fn):
    """Worker function that runs on each GPU process."""
    # Set up distributed environment
    device = f"cuda:{local_rank}"
    current_platform.set_device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    update_environment_variables({
        'RANK': str(local_rank),
        'LOCAL_RANK': str(local_rank),
        'WORLD_SIZE': str(world_size),
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': str(master_port),
    })

    # Initialize distributed
    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    # Set seed for reproducibility
    current_platform.seed_everything(42)
    init_batch_invariance()

    # Run the specific test based on op_name
    fn(local_rank, world_size, test_config)


class ULPChecker:
    FP_SPECS = {
        torch.float8_e4m3fn: {
            'mantissa_bits': 3,
            'exponent_bits': 4,
            'total_bits': 8,
            'int_dtype': torch.uint8
        },
        torch.float8_e5m2: {
            'mantissa_bits': 2,
            'exponent_bits': 5,
            'total_bits': 8,
            'int_dtype': torch.uint8
        },
        torch.bfloat16: {
            'mantissa_bits': 7,
            'exponent_bits': 8,
            'total_bits': 16,
            'int_dtype': torch.int16
        },
        torch.float16: {
            'mantissa_bits': 10,
            'exponent_bits': 5,
            'total_bits': 16,
            'int_dtype': torch.int16
        },
        torch.float32: {
            'mantissa_bits': 23,
            'exponent_bits': 8,
            'total_bits': 32,
            'int_dtype': torch.int32
        },
        torch.float64: {
            'mantissa_bits': 52,
            'exponent_bits': 11,
            'total_bits': 64,
            'int_dtype': torch.int64
        },
    }

    @staticmethod
    def to_int_bits(tensor: torch.Tensor) -> torch.Tensor:
        dtype = tensor.dtype
        if dtype not in ULPChecker.FP_SPECS:
            raise ValueError(f"Unsupported dtype: {dtype}")

        spec = ULPChecker.FP_SPECS[dtype]
        int_dtype = spec['int_dtype']

        return tensor.view(int_dtype)

    @staticmethod
    def ulp_distance_int(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.dtype != b.dtype:
            raise ValueError(f"Dtype mismatch: {a.dtype} vs {b.dtype}")

        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

        spec = ULPChecker.FP_SPECS[a.dtype]
        total_bits = spec['total_bits']

        a_int = ULPChecker.to_int_bits(a)
        b_int = ULPChecker.to_int_bits(b)

        sign_bit = 1 << (total_bits - 1)

        a_ordered = torch.where(
            (a_int & sign_bit) != 0,
            sign_bit - (a_int & ~sign_bit),  # Negative: flip magnitude bits
            a_int + sign_bit  # Positive: offset by sign bit
        )
        b_ordered = torch.where((b_int & sign_bit) != 0,
                                sign_bit - (b_int & ~sign_bit),
                                b_int + sign_bit)

        ulp_dist = torch.abs(a_ordered - b_ordered)
        return ulp_dist


def create_needle_tensor(
        batch_size: int,
        shape: list[int],
        device: torch.device,
        dtype: torch.dtype,
        needle_idx: int = 0) -> torch.Tensor:
    input_tensor = torch.randn(batch_size, *shape, device=device, dtype=dtype)

    numel = reduce(lambda x, y: x * y, shape)
    needle_pattern = torch.sin(
        torch.arange(numel, device=device).float().view(*shape) *
        0.1).to(dtype)

    assert needle_idx < input_tensor.shape[0]
    input_tensor[needle_idx] = needle_pattern

    return input_tensor


def verify(outputs: list[torch.Tensor],
                              needle_idxs: list[int]) -> bool:
    if len(outputs) < 2:
        return True

    needle_outputs = []
    for output, needle_idx in zip(outputs, needle_idxs):
        needle_outputs.append(output[needle_idx])

    reference = needle_outputs[0]
    for i, needle_output in enumerate(needle_outputs[1:], 1):
        dist_t = ULPChecker.ulp_distance_int(reference, needle_output)
        if torch.max(dist_t) != 0:
            print(f"Needle consistency failed at batch size comparison {i}")
            print(f"Max difference (ULP): {torch.max(dist_t)}")
            print(f"Max difference: {torch.max(reference - needle_output)}")
            return False

    return True


def validate(func, batch_sizes, shape, device, dtype):
    random.seed(123)
    outputs = []
    needle_idxs = []

    for batch_size in batch_sizes:
        needle_idx = random.randint(0, batch_size - 1)
        input_tensor = create_needle_tensor(batch_size, shape, device, dtype,
                                            needle_idx)

        with torch.no_grad():
            output = func(input_tensor)
            assert isinstance(output, torch.Tensor)
            outputs.append(output)
            needle_idxs.append(needle_idx)

    assert verify(outputs, needle_idxs), \
        "Needle consistency failed"


def _test_column_parallel_linear(local_rank: int, world_size: int,
                                 config: dict):
    device = torch.device(f"cuda:{local_rank}")
    batch_sizes = [1, 8, 32]
    dtype = config['dtype']
    hidden_size = config['reduction_size']
    seq_len = 4096
    input_size = hidden_size
    output_size = hidden_size * 2
    layer = ColumnParallelLinear(
        input_size=input_size,
        output_size=output_size,
        bias=True,
        gather_output=False,
        params_dtype=dtype,
    )
    layer = layer.to(device)
    validate(lambda x: layer(x)[0], batch_sizes, (seq_len, hidden_size),
             device, dtype)


def _test_row_parallel_linear(local_rank: int, world_size: int, config: dict):
    device = torch.device(f"cuda:{local_rank}")
    batch_sizes = [1, 8, 32]
    dtype = config['dtype']
    hidden_size = config['reduction_size']
    seq_len = 4096
    input_size = hidden_size * 2
    output_size = hidden_size
    layer = RowParallelLinear(
        input_size=input_size,
        output_size=output_size,
        bias=True,
        reduce_results=True,
        params_dtype=dtype,
    )
    layer = layer.to(device)
    validate(lambda x: layer(x)[0], batch_sizes,
             (seq_len, input_size // world_size), device, dtype)


def _test_rms_norm(local_rank: int, world_size: int,
                                      config: dict):
    """Test RMSNorm with needle consistency."""
    device = torch.device(f"cuda:{local_rank}")
    dtype = config['dtype']
    hidden_size = config['reduction_size']
    batch_sizes = [1, 32, 1024]

    layer = RMSNorm(hidden_size, eps=1e-6)
    layer = layer.to(device).to(dtype)
    validate(layer, batch_sizes, (hidden_size, ), device, dtype)


def _test_fused_rms_norm(local_rank: int, world_size: int,
                                            config: dict):
    device = torch.device(f"cuda:{local_rank}")
    dtype = config['dtype']
    hidden_size = config['reduction_size']
    batch_sizes = [1, 32, 1024]

    layer = RMSNorm(hidden_size, eps=1e-6)
    layer = layer.to(device).to(dtype)
    validate(lambda x: layer(x, x)[0], batch_sizes, (hidden_size, ), device,
             dtype)


def _test_fused_moe(local_rank: int, world_size: int,
                                       config: dict):
    """Test FusedMoE with needle consistency."""
    device = torch.device(f"cuda:{local_rank}")
    dtype = config['dtype']
    hidden_size = config['reduction_size']
    batch_sizes = [1, 8, 32]

    # MoE configuration parameters
    num_experts = 8
    top_k = 2
    intermediate_size = hidden_size * 4

    from vllm.config import VllmConfig
    from vllm.forward_context import get_forward_context, set_forward_context
    from vllm.model_executor.layers.fused_moe import FusedMoE

    vllm_config = VllmConfig()

    # Create FusedMoE layer similar to how it's used in models
    layer = FusedMoE(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        params_dtype=dtype,
        reduce_results=True,
        renormalize=True,
        use_grouped_topk=False,
    )
    layer = layer.to(device)

    # Test function that takes hidden states and generates router logits
    def test_func(hidden_states):
        # Generate router logits (this would normally come from a router layer)
        router_logits = torch.randn(hidden_states.shape[0],
                                    hidden_states.shape[1],
                                    num_experts,
                                    device=device,
                                    dtype=dtype)

        # Set forward context with minimal required parameters
        # attn_metadata can be None for testing purposes
        with set_forward_context(attn_metadata=None,
                                 vllm_config=vllm_config,
                                 num_tokens=hidden_states.shape[0] *
                                 hidden_states.shape[1]):
            fwdctx = get_forward_context()
            fwdctx.no_compile_layers[''] = layer
            return layer(hidden_states, router_logits)

    validate(test_func, batch_sizes, (hidden_size, ), device, dtype)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("world_size", [2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("reduction_size", [1, 5, 1024, 1024 + 1])
@pytest.mark.parametrize("func", [
    _test_column_parallel_linear,
    _test_row_parallel_linear,
    _test_rms_norm,
    _test_fused_rms_norm,
    _test_fused_moe,
])
def test_parallel_reduction_batch_invariance(world_size: int,
                                             dtype: torch.dtype,
                                             reduction_size: int, func):
    """Test parallel operators on 2 GPUs."""
    test_config = {
        "dtype": dtype,
        "reduction_size": reduction_size,
    }

    mp.spawn(run_parallel_op_test_worker,
             args=(world_size, get_open_port(), test_config, func),
             nprocs=world_size)
