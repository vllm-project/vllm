# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise eager cache snapshots through multiple actual process boundaries."""

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.models.deepseek_v4_1.common.pipeline_transfer import (
    restore_cache_blocks,
    snapshot_cache_blocks,
)


def _relay_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=30),
    )
    try:
        expected = torch.arange(6 * 3 * 4, dtype=torch.uint8).reshape(6, 3, 4)
        cache = expected.clone() if rank == 0 else torch.zeros_like(expected)
        table = torch.tensor([[1, 4, -1]])
        if rank:
            ids = torch.empty(2, dtype=torch.int64)
            blocks = torch.empty(2, 3, 4, dtype=torch.uint8)
            dist.recv(ids, src=rank - 1)
            dist.recv(blocks, src=rank - 1)
            restore_cache_blocks(cache, ids, blocks)
        torch.testing.assert_close(cache[[1, 4]], expected[[1, 4]], rtol=0, atol=0)
        if rank < 2:
            ids, blocks = snapshot_cache_blocks(cache, [table], max_bytes=1024)
            dist.send(ids, dst=rank + 1)
            dist.send(blocks, dst=rank + 1)
    finally:
        dist.destroy_process_group()


def test_cache_blocks_survive_two_pipeline_hops(tmp_path):
    mp.spawn(_relay_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=3)


def _mixed_relay_worker(rank, port, asynchronous):
    from tests.utils import init_test_distributed_environment
    from vllm.distributed import get_pp_group
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    torch.accelerator.set_device_index(rank)
    init_test_distributed_environment(1, 4, rank, port, local_rank=rank)
    group = get_pp_group()
    try:
        # Distinct rounds refresh a prefix block and then reuse a physical block.
        for round_id, selected in enumerate(([1, 4], [1, 4], [4], [])):
            expected = (
                torch.arange(6 * 2 * 3 * 4, device="cuda")
                .reshape(6, 2, 3, 4)
                .add(round_id * 19)
                .to(torch.uint8)[:, 0]
            )
            cache = torch.full((6, 2, 3, 4), 17, device="cuda", dtype=torch.uint8)[:, 0]
            if rank == 0:
                cache.copy_(expected)
            else:
                if asynchronous:
                    payload, handles, postprocess = group.irecv_tensor_dict(
                        src=rank - 1
                    )
                    for handle in handles:
                        handle.wait()
                    for callback in postprocess:
                        callback()
                else:
                    payload = group.recv_tensor_dict(src=rank - 1)
                assert payload["ids"].device.type == "cpu"
                assert payload["blocks"].device.type == "cuda"
                assert payload["round"] == round_id
                restore_cache_blocks(cache, payload["ids"], payload["blocks"])
                untouched = [i for i in range(6) if i not in selected]
                assert torch.all(cache[untouched] == 17)
            torch.testing.assert_close(
                cache[selected], expected[selected], rtol=0, atol=0
            )
            if rank < 3:
                table = torch.tensor([selected + [-1]], dtype=torch.int32)
                ids, blocks = snapshot_cache_blocks(cache, [table], max_bytes=1024)
                payload = {"ids": ids, "blocks": blocks, "round": round_id}
                if asynchronous:
                    handles = group.isend_tensor_dict(payload, dst=rank + 1)
                    for handle in handles:
                        handle.wait()
                    group._reap_completed_isends()
                    # The CPU Gloo handle must also tolerate a caller's second wait.
                    for handle in handles:
                        handle.wait()
                else:
                    group.send_tensor_dict(payload, dst=rank + 1)
            group.barrier()
        torch.accelerator.synchronize()
    finally:
        cleanup_dist_env_and_memory()


def test_host_ids_and_cuda_cache_survive_three_pipeline_hops():
    import pytest

    from vllm.utils.network_utils import get_open_port

    if not torch.cuda.is_available() or torch.accelerator.device_count() < 4:
        pytest.skip("Four CUDA devices required")
    for asynchronous in (False, True):
        mp.spawn(
            _mixed_relay_worker, args=(str(get_open_port()), asynchronous), nprocs=4
        )
