# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.hier_all_reduce import (
    _TWO_SHOT_MIN_ELEMS,
    HierarchicalAllReduce,
)
from vllm.distributed.parallel_state import get_tp_group

from ..utils import init_test_distributed_environment, multi_process_parallel

# Straddle the one-shot/two-shot dispatch threshold in both directions, and
# include a size that is not a multiple of the CTA tiling to exercise the
# masked tails.
TEST_SIZES = [
    1024,
    _TWO_SHOT_MIN_ELEMS // 2,
    _TWO_SHOT_MIN_ELEMS,
    _TWO_SHOT_MIN_ELEMS * 2 + 512,
]


@ray.remote(num_gpus=1, max_calls=1)
def hier_allreduce_matches_nccl(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

        tp_group = get_tp_group()
        group = tp_group.cpu_group
        # Islands are group-local rank indices; split the TP group in half so
        # the two halves stand in for the two PCIe islands.
        half = tp_size // 2
        islands = [list(range(half)), list(range(half, tp_size))]
        comm = HierarchicalAllReduce(group, device, islands)

        for numel in TEST_SIZES:
            inp = torch.randn(numel, dtype=torch.bfloat16, device=device)
            ref = inp.clone()
            dist.all_reduce(ref, group=tp_group.device_group)
            assert comm.should_use(inp)
            # Run twice: the flag protocol alternates buffer halves by
            # sequence-token parity, so the second call takes the other half.
            for _ in range(2):
                out = comm.all_reduce(inp)
                torch.cuda.synchronize()
                # Reduction order differs from NCCL's, so compare within the
                # dtype's tolerance rather than bit-exactly.
                torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("pipeline_parallel_size", [1])
def test_hier_all_reduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pipeline_parallel_size,
):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_parallel(
        monkeypatch, tp_size, pipeline_parallel_size, hier_allreduce_matches_nccl
    )
