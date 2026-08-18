# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.utils import ensure_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutedsl_gemm_allreduce
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

_SUPPORTED_DTYPES = (torch.bfloat16,)
_TOLERANCES = {
    torch.bfloat16: (5e-2, 1.0),
}


def _random_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    return (
        torch.randn(
            shape,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        .mul_(0.25)
        .to(dtype)
    )


def test_flashinfer_gemm_allreduce_bucket_sizes() -> None:
    from vllm.model_executor.kernels.linear.cute_dsl.gemm_allreduce import (
        gemm_ar_bucket_sizes,
    )

    assert gemm_ar_bucket_sizes(1) == (128,)
    assert gemm_ar_bucket_sizes(128) == (128,)
    assert gemm_ar_bucket_sizes(129) == (128, 256)
    buckets = gemm_ar_bucket_sizes(16384)
    assert len(buckets) == 128
    assert buckets == tuple(range(128, 16385, 128))
    with pytest.raises(ValueError, match="max_M >= 1"):
        gemm_ar_bucket_sizes(0)


def _worker(local_rank: int, world_size: int, master_port: int) -> None:
    from vllm.model_executor.kernels.linear.cute_dsl.gemm_allreduce import (
        get_or_create_cutedsl_fused_gemm_ar_workspace,
        make_cutedsl_fused_gemm_ar,
    )
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    device = torch.device("cuda", local_rank)
    torch.accelerator.set_device_index(device)
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
        }
    )

    init_distributed_environment()
    with ensure_current_vllm_config():
        initialize_model_parallel(tensor_model_parallel_size=world_size)

    group = dist.group.WORLD
    rank = dist.get_rank(group)
    max_M, N, K = 384, 7168, 1536
    generator = torch.Generator(device=device)
    generator.manual_seed(4000 + rank)
    for dtype in _SUPPORTED_DTYPES:
        weight = _random_tensor((N, K), dtype, device, generator)
        linear = cast(
            LinearBase,
            SimpleNamespace(
                quant_method=UnquantizedLinearMethod(),
                bias=None,
                reduce_results=True,
                weight=weight,
            ),
        )
        projection = make_cutedsl_fused_gemm_ar(linear, max_M=max_M)
        assert projection is not None
        op = projection.workspace
        assert (
            get_or_create_cutedsl_fused_gemm_ar_workspace(
                max_M=max_M, N=N, K=K, dtype=dtype
            )
            is op
        )
        assert op.compile() == len(op.bucket_sizes)
        assert op.compile() == 0

        mismatched_dtype = torch.bfloat16 if dtype == torch.float16 else torch.float16
        mismatched_x = torch.empty(256, K, dtype=mismatched_dtype, device=device)
        assert not projection.should_run(mismatched_x)

        for M in (1, 127, 128, 129, 255, 256, 257):
            x = _random_tensor((M, K), dtype, device, generator)
            expected = torch.mm(x.float(), weight.float().T).to(dtype).float()
            dist.all_reduce(expected, group=group)
            actual = projection(x)
            torch.accelerator.synchronize(device)
            assert actual.dtype == dtype
            rtol, atol = _TOLERANCES[dtype]
            torch.testing.assert_close(actual.float(), expected, rtol=rtol, atol=atol)

        first_output = projection(x)
        first_output_snapshot = first_output.clone()
        second_output = projection(-x)
        torch.accelerator.synchronize(device)
        assert first_output.data_ptr() != second_output.data_ptr()
        torch.testing.assert_close(
            first_output,
            first_output_snapshot,
            rtol=0,
            atol=0,
        )

        if dtype == torch.bfloat16:
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            graph_output = torch.empty((x.shape[0], N), dtype=dtype, device=device)
            with torch.cuda.stream(capture_stream):
                projection(x)
                graph_output.copy_(projection(x))
            capture_stream.synchronize()
            dist.barrier(group=group)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                projection(x)
                graph_output.copy_(projection(x))
            torch.cuda.current_stream().wait_stream(capture_stream)
            dist.barrier(group=group)
            graph.replay()
            torch.accelerator.synchronize(device)
            torch.testing.assert_close(
                graph_output.float(), expected, rtol=rtol, atol=atol
            )
            del capture_stream, graph, graph_output

        dist.barrier(group=group)
        del linear, mismatched_x, projection, op, weight, x
        gc.collect()

    gc.collect()
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()


@pytest.mark.distributed(num_gpus=8)
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100)
    or not has_flashinfer_cutedsl_gemm_allreduce(),
    reason="FlashInfer GEMM-allreduce requires SM100-family GPUs and its kernel",
)
def test_flashinfer_gemm_allreduce(monkeypatch: pytest.MonkeyPatch) -> None:
    world_size = 8
    if torch.accelerator.device_count() < world_size:
        pytest.skip("FlashInfer GEMM-allreduce requires eight GPUs")

    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "1")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "1")
    try:
        mp.spawn(
            _worker,
            args=(world_size, get_open_port()),
            nprocs=world_size,
        )
    finally:
        cleanup_dist_env_and_memory()
