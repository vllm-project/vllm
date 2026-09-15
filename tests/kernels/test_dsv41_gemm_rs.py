# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


def _worker(rank: int, port: int) -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        swizzle_mxfp8_scale,
    )
    from vllm.models.deepseek_v41.nvidia.ops.gemm_rs import Mxfp8GemmRS

    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="4",
        MASTER_ADDR="localhost",
        MASTER_PORT=str(port),
    )
    torch.accelerator.set_device_index(rank)
    torch.manual_seed(1234 + rank)
    init_distributed_environment()
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(tensor_model_parallel_size=4)
    n, k = 512, 768
    fused = Mxfp8GemmRS(4097, n, k)
    b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    bs = torch.randint(119, 128, (n, k // 32), device="cuda", dtype=torch.uint8)
    b_scale = swizzle_mxfp8_scale(bs, M=n, K=k)
    b_float = b.float() * torch.exp2(bs.float() - 127).repeat_interleave(32, dim=1)

    def check(m, runner=fused):
        a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
        sf = torch.randint(119, 128, (m, k // 32), device="cuda", dtype=torch.uint8)
        a_scale = swizzle_mxfp8_scale(sf, M=m, K=k)
        a_float = a.float() * torch.exp2(sf.float() - 127).repeat_interleave(32, dim=1)
        expected = (a_float @ b_float.T).bfloat16().float()
        # Different MMA accumulation orders can round a partial to adjacent
        # BF16 values. Bound the error before cancellation across TP ranks.
        error_bound = expected.abs() * torch.finfo(torch.bfloat16).eps
        dist.all_reduce(error_bound)
        dist.all_reduce(expected)
        local_m = (m + 3) // 4
        expected = torch.nn.functional.pad(expected, (0, 0, 0, local_m * 4 - m))
        expected = expected[rank * local_m : (rank + 1) * local_m].bfloat16()
        error_bound = torch.nn.functional.pad(error_bound, (0, 0, 0, local_m * 4 - m))[
            rank * local_m : (rank + 1) * local_m
        ]
        actual = runner.quantized(a, b, a_scale, b_scale)
        delta = (actual.float() - expected.float()).abs()
        assert torch.all(delta <= error_bound + expected.float().abs() * 0.008)
        assert (
            delta.square().mean().sqrt()
            < expected.float().square().mean().sqrt() * 0.005
        )
        return a, a_scale, actual, expected

    # Grid changes and workspace reuse must preserve rank ownership and padding.
    for m in (128, 129, 1024, 4097, 257, 128):
        check(m)
    a, a_scale, first, expected = check(129)
    snapshot = first.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused.quantized(a, b, a_scale, b_scale)
    stream.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fused.quantized(a, b, a_scale, b_scale)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(5):
        # Changing values detects stale producer flags during graph replay.
        a.copy_((-a.float()).to(a.dtype))
        expected.neg_()
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)
        torch.testing.assert_close(first, snapshot, rtol=0, atol=0)
    del graph, fused
    cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100)
    or torch.accelerator.device_count() < 4,
    reason="Requires four SM100-family GPUs in one NVLink domain",
)
def test_mxfp8_gemm_rs_tp4_reuse_and_graph() -> None:
    """Compare native MXFP8 GEMM-RS to dequantized matmul and FP32 reduction."""
    mp.spawn(_worker, args=(get_open_port(),), nprocs=4, join=True)
