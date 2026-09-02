# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing as mp
import tempfile
from pathlib import Path

import pytest
import torch.distributed as dist

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncCoordinator

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _GraphManager:
    def dispatch(
        self,
        num_reqs,
        num_tokens,
        uniform_token_count,
        num_active_loras,
        max_query_len=None,
    ):
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            uniform_token_count=uniform_token_count,
            max_query_len=max_query_len,
            num_active_loras=num_active_loras,
        )


def _run_concurrent_lanes(rank, init_path, result_queue):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
    )
    target_group = dist.new_group(ranks=[0, 1], backend="gloo")
    speculator_group = dist.new_group(ranks=[0, 1], backend="gloo")
    try:
        manager = _GraphManager()
        target = DPSyncCoordinator(
            2,
            rank,
            group=target_group,
            lane="target",
            execution_contract=True,
        )
        speculator = DPSyncCoordinator(
            2,
            rank,
            group=speculator_group,
            lane="speculator",
            execution_contract=True,
        )

        live_reqs = rank * 2
        live_tokens = rank * 4

        def start_target():
            return target.start(
                manager,
                live_reqs,
                live_tokens,
                uniform_token_count=2 if rank else None,
                max_query_len=2 if rank else None,
            )

        def start_speculator():
            return speculator.start(
                manager,
                live_reqs,
                live_reqs,
                uniform_token_count=1 if rank else None,
                parent_generation=0,
            )

        # Issue the logical stages in opposite orders. They can coexist only
        # because each stage has its own process-group sequence.
        if rank == 0:
            target_future = start_target()
            speculator_future = start_speculator()
        else:
            speculator_future = start_speculator()
            target_future = start_target()

        target_desc, target_sync = target_future.result(manager)
        speculator_desc, speculator_sync = speculator_future.result(manager)
        assert target_sync is not None
        assert speculator_sync is not None
        result_queue.put(
            (
                rank,
                target_desc.num_tokens,
                target_desc.num_reqs,
                target_sync.live_num_tokens_across_dp,
                speculator_desc.num_tokens,
                speculator_sync.parent_generation,
            )
        )
        target_future.release()
        speculator_future.release()
    finally:
        dist.destroy_process_group(speculator_group)
        dist.destroy_process_group(target_group)
        dist.destroy_process_group()


def test_target_and_speculator_collectives_can_overlap_on_gloo():
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    with tempfile.TemporaryDirectory() as temp_dir:
        init_path = str(Path(temp_dir) / "gloo_init")
        processes = [
            context.Process(
                target=_run_concurrent_lanes,
                args=(rank, init_path, result_queue),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0

    results = sorted(result_queue.get(timeout=1) for _ in range(2))
    assert results == [
        (0, 4, 2, (0, 4), 2, 0),
        (1, 4, 2, (0, 4), 2, 0),
    ]
