# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.utils import init_test_distributed_environment
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory, get_tp_group
from vllm.distributed.device_communicators.low_sm_all_reduce import LowSMAllReduce
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


def _worker(rank: int, world_size: int, port: int) -> None:
    device = torch.device("cuda", rank)
    torch.accelerator.set_device_index(device)
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4))
    with set_current_vllm_config(config):
        init_test_distributed_environment(
            world_size, 1, rank, str(port), local_rank=rank
        )
        tp = get_tp_group()
        hidden_size, capacity_rows = 136, 641
        max_num_bytes = capacity_rows * hidden_size * torch.bfloat16.itemsize
        kwargs = dict(
            group=tp.cpu_group,
            device=device,
            max_num_bytes=max_num_bytes,
            num_slots=2,
        )
        mismatched = kwargs | {"max_num_bytes": max_num_bytes + rank * 16}
        assert LowSMAllReduce.initialize(**mismatched) is None
        op = LowSMAllReduce.initialize(**kwargs)
        assert op is not None
        assert LowSMAllReduce.initialize(**kwargs) is op

        for rows in (1, 513, capacity_rows):
            torch.manual_seed(100 + rank + rows)
            value = torch.randn(rows, hidden_size, dtype=torch.bfloat16, device=device)
            original = value.clone()
            expected = value.clone()
            dist.all_reduce(expected, group=tp.device_group)
            result = op(value)
            torch.testing.assert_close(value, original, atol=0, rtol=0)
            torch.testing.assert_close(result, expected, atol=4e-2, rtol=2e-2)

        first = op(value, slot=0)
        normal_ar = tp.device_communicator.symm_mem_comm
        assert normal_ar is not None
        normal = normal_ar.all_reduce(
            torch.full((8,), rank + 1, dtype=torch.bfloat16, device=device)
        )
        assert normal is not None
        assert torch.all(normal == world_size * (world_size + 1) // 2)
        second = op(value + 1, slot=1)
        assert first.data_ptr() != second.data_ptr()
        torch.testing.assert_close(first, expected, atol=4e-2, rtol=2e-2)
        torch.testing.assert_close(second, expected + world_size, atol=4e-2, rtol=2e-2)
        assert not op.supports(
            torch.empty(
                capacity_rows * hidden_size + 8,
                dtype=torch.bfloat16,
                device=device,
            )
        )
        with pytest.raises(ValueError, match="alias"):
            op(first)
        with pytest.raises(ValueError, match="out of range"):
            op(value, slot=2)

        static = torch.full(
            (513, hidden_size), rank + 1, dtype=torch.bfloat16, device=device
        )
        graph = torch.cuda.CUDAGraph()
        torch.accelerator.synchronize()
        with torch.cuda.graph(graph):
            reduced = op(static)
            consumed = reduced + 5
            reused = op(static + 7)
            graph_output = consumed + reused
        assert reduced.data_ptr() == reused.data_ptr()
        for replay in range(2):
            static.fill_(rank + replay + 1)
            graph.replay()
            torch.accelerator.synchronize()
            rank_sum = world_size * (world_size + 1) // 2 + replay * world_size
            assert torch.all(graph_output == 2 * rank_sum + 7 * world_size + 5)

    LowSMAllReduce._instances.clear()
    cleanup_dist_env_and_memory()


@pytest.mark.distributed(num_gpus=4)
@pytest.mark.skipif(
    not (
        current_platform.is_device_capability((10, 0))
        or current_platform.is_device_capability((10, 3))
    ),
    reason="low-SM all-reduce requires SM100 or SM103",
)
def test_low_sm_all_reduce(monkeypatch: pytest.MonkeyPatch) -> None:
    if torch.accelerator.device_count() < 4:
        pytest.skip("low-SM all-reduce requires four GPUs")
    monkeypatch.setenv("VLLM_ALLREDUCE_USE_SYMM_MEM", "1")
    try:
        mp.spawn(_worker, args=(4, get_open_port()), nprocs=4)
    finally:
        cleanup_dist_env_and_memory()
