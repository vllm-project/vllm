# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Column-sharded MuseSpark1x pooler/unpooler on ROCm: every TP rank projects
its block of output columns and the blocks are all-gathered, which must give
the replicated projection's output."""

from types import SimpleNamespace

import pytest
import ray
import torch
import torch.nn.functional as F

from vllm._aiter_ops import is_aiter_found, rocm_aiter_ops
from vllm.distributed.parallel_state import get_tp_group, graph_capture
from vllm.envs import disable_envs_cache
from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)

pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_rocm(), reason="ROCm-only MuseSpark1x model path"
    ),
    pytest.mark.skipif(not is_aiter_found(), reason="AITER is not installed"),
]

HIDDEN, POOLED = 8192, 4096
# At TP8, up to 128 tokens take the split-K GEMM, and 8192 tokens gather the
# unpooler output in chunks that fit AITER's buffer.
NUM_TOKENS = [1, 5, 64, 128, 129, 1000, 8192]


def _check_sharded_matches_replicated(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
    use_aiter: bool,
) -> None:
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.setenv("VLLM_ROCM_USE_AITER", "1" if use_aiter else "0")
        m.setenv("VLLM_ROCM_USE_AITER_CUSTOM_AR", "1" if use_aiter else "0")
        disable_envs_cache()
        rocm_aiter_ops.refresh_env_variables()

        from vllm.models.muse_spark.amd.model import (
            MuseSparkShardedProjMethod,
            _gathered_column_proj,
        )

        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        assert (rocm_aiter_ops.get_aiter_allreduce() is not None) == use_aiter

        method = MuseSparkShardedProjMethod()

        def project(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return method.apply(SimpleNamespace(weight=weight), x)

        gen = torch.Generator(device=device).manual_seed(0)
        for n, k in ((POOLED, HIDDEN), (HIDDEN, POOLED)):
            weight = torch.randn(n, k, device=device, generator=gen) * k**-0.5
            blocks = [b.contiguous() for b in weight.bfloat16().chunk(tp_size)]

            def layer(x: torch.Tensor, w: torch.Tensor = blocks[rank]):
                return project(w, x), None

            for num_tokens in NUM_TOKENS:
                x = torch.randn(num_tokens, k, device=device, generator=gen)
                x = x.bfloat16()
                out = _gathered_column_proj(layer, x)
                # The gather only moves bits: each block is what its rank's
                # GEMM produced, recomputed here from the same weights.
                local = torch.cat([project(b, x) for b in blocks], dim=1)
                assert torch.equal(out, local), (n, k, num_tokens)
                # Both GEMMs accumulate in fp32 and differ only in summation
                # order, so outputs differ by at most an ulp, and rarely.
                replicated = F.linear(x, torch.cat(blocks))
                mismatch = (out != replicated).float().mean().item()
                assert mismatch < 5e-2, (n, k, num_tokens, mismatch)
                torch.testing.assert_close(out, replicated, atol=2e-2, rtol=1e-2)

            # Decode steps replay CUDA graphs.
            x = torch.randn(4, k, device=device, generator=gen).bfloat16()
            expected = _gathered_column_proj(layer, x)
            with graph_capture(device=device) as ctx:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=ctx.stream):
                    out = _gathered_column_proj(layer, x)
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(out, expected)
        get_tp_group().barrier()


@ray.remote(num_gpus=1, max_calls=1)
def gather_with_aiter(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _check_sharded_matches_replicated(
        monkeypatch, tp_size, pp_size, rank, distributed_init_port, use_aiter=True
    )


@ray.remote(num_gpus=1, max_calls=1)
def gather_with_rccl(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _check_sharded_matches_replicated(
        monkeypatch, tp_size, pp_size, rank, distributed_init_port, use_aiter=False
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("test_target", [gather_with_aiter, gather_with_rccl])
def test_column_sharded_projection_tp2(monkeypatch: pytest.MonkeyPatch, test_target):
    multi_process_parallel(monkeypatch, 2, 1, test_target)


@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize("test_target", [gather_with_aiter, gather_with_rccl])
def test_column_sharded_projection_tp8(monkeypatch: pytest.MonkeyPatch, test_target):
    multi_process_parallel(monkeypatch, 8, 1, test_target)
