# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm projection/collective contract corresponding to the SM100 GEMM-RS/AR.

Exercise the production RowParallelLinear and sequence-parallel reduction
paths: native BF16/MXFP8 GEMM, AITER all-reduce, and RCCL reduce-scatter. This
checks the same observable contract, not NVIDIA's fused GEMM/communication
implementation or its performance.
"""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.kernels.test_gemm_rs_ar import _N, _SHAPES
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.config.quantization import QuantizationConfigArgs
from vllm.distributed import cleanup_dist_env_and_memory, get_tp_group
from vllm.distributed.parallel_state import (
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_torch,
    dequant_mxfp8_to_bf16,
)
from vllm.models.common.ops.sequence_parallel import sp_reduce_scatter
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

# Retain the expanded parent's shape transitions, plus fully padded RS ranks.
_ROCM_SHAPES = ((1, 512), (7, 768), *_SHAPES)


def _reference(
    x: torch.Tensor,
    layer: RowParallelLinear,
    all_reduce: bool,
) -> torch.Tensor:
    weight = layer.weight
    if hasattr(layer, "weight_scale"):
        # The production ROCm path uses the Triton quantizer and dot_scaled;
        # use the independent torch quantizer and FP32 matmul as the oracle.
        x_q, x_scale = _mxfp8_e4m3_quantize_torch(x)
        x = dequant_mxfp8_to_bf16(x_q, x_scale)
        weight = dequant_mxfp8_to_bf16(weight, layer.weight_scale)
    partial = (x.float() @ weight.float().T).to(torch.bfloat16).float()
    group = get_tp_group()
    # Always use FP32 RCCL AR for the oracle, including the RS candidate.
    dist.all_reduce(partial, group=group.device_group)
    expected = partial.to(torch.bfloat16)
    if all_reduce:
        return expected
    padding = -x.shape[0] % group.world_size
    expected = torch.nn.functional.pad(expected, (0, 0, 0, padding))
    return expected.chunk(group.world_size)[group.rank_in_group]


def _apply(x: torch.Tensor, layer: RowParallelLinear) -> torch.Tensor:
    output, _ = layer(x)
    return output if layer.reduce_results else sp_reduce_scatter(output)


def _assert_output(
    actual: torch.Tensor, expected: torch.Tensor, M: int, all_reduce: bool
) -> None:
    assert actual.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    # Match the parent's tolerance at 0.1 weight scale, then bound total error.
    # BF16 partial/reduction rounding can dominate near a cancelled output.
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=0.4)
    error = (actual.float() - expected.float()).norm()
    assert error <= 5e-3 * expected.float().norm() + 1e-6
    if not all_reduce:
        rank_start = get_tp_group().rank_in_group * actual.shape[0]
        valid_rows = min(max(M - rank_start, 0), actual.shape[0])
        assert not actual[valid_rows:].any(), "RS padding must remain zero"


def _run_mode(
    device: torch.device, backend: str, all_reduce: bool, world_size: int
) -> None:
    group = get_tp_group()
    generator = torch.Generator(device=device)
    quant_config = (
        OnlineQuantizationConfig(QuantizationConfigArgs(linear="mxfp8"))
        if backend == "mxfp8"
        else None
    )
    layers = {}
    for K in sorted({K for _, K in _ROCM_SHAPES}):
        with torch.device(device):
            layer = RowParallelLinear(
                K * world_size,
                _N,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=quant_config,
                reduce_results=all_reduce,
            )
        generator.manual_seed(1000 + group.rank_in_group * 100 + K)
        weight = (
            torch.randn(_N, K, dtype=torch.bfloat16, device=device, generator=generator)
            * 0.1
        )
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.quant_method.process_weights_after_loading(layer)
        if backend == "mxfp8":
            from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
                RocmDotScaledMxfp8LinearKernel,
            )

            assert isinstance(layer.quant_method.kernel, RocmDotScaledMxfp8LinearKernel)
            assert layer.weight.dtype == torch.float8_e4m3fn
        layers[K] = layer

    def make_input(M: int, K: int, seed: int) -> torch.Tensor:
        generator.manual_seed(seed + group.rank_in_group * 100)
        return torch.randn(
            M, K, dtype=torch.bfloat16, device=device, generator=generator
        )

    comm = group.device_communicator
    assert comm is not None
    assert comm.use_aiter_allreduce
    assert comm.aiter_ar_comm is not None and not comm.aiter_ar_comm.disabled
    assert comm.qr_comm is None or comm.qr_comm.disabled
    assert comm.pynccl_comm is not None and not comm.pynccl_comm.disabled
    for M, K in (*_ROCM_SHAPES, *_ROCM_SHAPES[::-1]):
        x = make_input(M, K, 2000 + M + K)
        if all_reduce:
            probe = torch.empty(M, _N, dtype=x.dtype, device=device)
            assert comm.aiter_ar_comm.should_custom_ar(probe)
        expected = _reference(x, layers[K], all_reduce)
        actual = _apply(x, layers[K])
        _assert_output(actual, expected, M, all_reduce)

    # Retained outputs must survive the next call's collective buffer reuse.
    x = make_input(257, 768, 2500)
    first = _apply(x, layers[768])
    snapshot = first.clone()
    second = _apply(-x, layers[768])
    torch.accelerator.synchronize(device)
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, snapshot, rtol=0, atol=0)

    M, K = 1025, 4224
    graph_x = make_input(M, K, 3000)
    for _ in range(3):
        _apply(graph_x, layers[K])
    torch.accelerator.synchronize(device)
    dist.barrier(group=group.device_group)
    graph = torch.cuda.CUDAGraph()
    # vLLM's context registers the AITER IPC addresses used by graph replay.
    with (
        graph_capture(device=device) as context,
        torch.cuda.graph(graph, stream=context.stream),
    ):
        graph_output = _apply(graph_x, layers[K])
    torch.cuda.current_stream().wait_stream(context.stream)
    dist.barrier(group=group.device_group)
    for replay in range(3):
        graph_x.copy_(make_input(M, K, 3001 + replay))
        expected = _reference(graph_x, layers[K], all_reduce)
        graph.replay()
        torch.accelerator.synchronize(device)
        _assert_output(graph_output, expected, M, all_reduce)
    dist.barrier(group=group.device_group)


def _worker(rank: int, world_size: int, port: int, backend: str) -> None:
    device = torch.device("cuda", rank)
    torch.accelerator.set_device_index(device)
    update_environment_variables(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
        }
    )
    config = VllmConfig()
    config.model_config = ModelConfig(dtype="bfloat16")
    try:
        with set_current_vllm_config(config), torch.inference_mode():
            init_distributed_environment()
            initialize_model_parallel(tensor_model_parallel_size=world_size)
            for all_reduce in (False, True):
                _run_mode(device, backend, all_reduce, world_size)
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm GEMM-RS/AR contract")
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize(
    "backend",
    [
        "bf16",
        pytest.param(
            "mxfp8",
            marks=pytest.mark.skipif(
                not current_platform.supports_mx(), reason="native MXFP8 requires CDNA4"
            ),
        ),
    ],
)
@pytest.mark.distributed(num_gpus=4)
def test_rocm_gemm_rs_ar(
    monkeypatch: pytest.MonkeyPatch, world_size: int, backend: str
) -> None:
    """Preserve rank reduction, tails, shape reuse, lifetime, and graph replay."""
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"GEMM-RS/AR requires {world_size} GPUs")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_CUSTOM_AR", "1")
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", "NONE")
    try:
        mp.spawn(
            _worker, args=(world_size, get_open_port(), backend), nprocs=world_size
        )
    finally:
        cleanup_dist_env_and_memory()
