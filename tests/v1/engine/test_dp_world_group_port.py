# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DP world group picks its port when it binds, not ahead of time."""

import multiprocessing as mp
import socket
import traceback
from multiprocessing.queues import Queue

import pytest
import torch

from vllm.distributed.utils import create_tcp_store
from vllm.utils.network_utils import get_open_port

RESULT_TIMEOUT_S = 240


def _join_dp_world_group(
    dp_rank: int, coord_store_port: int, dp_master_port: int, result: Queue
) -> None:
    """Bring up one DP rank of the world group and report what it saw."""
    try:
        import torch.distributed as dist

        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.config.parallel import ParallelConfig
        from vllm.distributed.parallel_state import init_distributed_environment

        parallel_config = ParallelConfig(
            data_parallel_size=2, data_parallel_rank=dp_rank
        )
        parallel_config.data_parallel_master_ip = "127.0.0.1"
        parallel_config._data_parallel_master_port_list = [dp_master_port]
        parallel_config._coord_store_port = coord_store_port

        with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
            init_distributed_environment(
                world_size=1, rank=0, local_rank=0, backend="gloo"
            )

        reduced = torch.tensor([dp_rank + 1])
        dist.all_reduce(reduced)
        result.put((dp_rank, int(reduced.item()), None))
    except Exception:
        result.put((dp_rank, None, traceback.format_exc()))


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="gloo backend required"
)
@pytest.mark.parametrize("with_coord_store", [True, False])
def test_dp_world_group_survives_a_taken_preallocated_port(with_coord_store: bool):
    """With a coordination store, the two ranks agree on a port the group binds
    itself, so a pre-allocated port taken in the meantime cannot stop startup.
    Without one, the pre-allocated port is still used as before.
    """
    squatter = None
    coord_store = None
    try:
        if with_coord_store:
            # Something else grabbed the pre-allocated port after it was picked.
            squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            squatter.bind(("127.0.0.1", 0))
            squatter.listen(1)
            dp_master_port = squatter.getsockname()[1]
            coord_store = create_tcp_store(
                "127.0.0.1", 0, is_master=True, world_size=-1, wait_for_workers=False
            )
            coord_store_port = coord_store.port
        else:
            dp_master_port = get_open_port()
            coord_store_port = 0

        ctx = mp.get_context("spawn")
        result: Queue = ctx.Queue()
        procs = [
            ctx.Process(
                target=_join_dp_world_group,
                args=(dp_rank, coord_store_port, dp_master_port, result),
            )
            for dp_rank in range(2)
        ]
        for proc in procs:
            proc.start()
        reports: list[tuple[int, int | None, str | None]] = []
        try:
            while len(reports) < len(procs):
                reports.append(result.get(timeout=RESULT_TIMEOUT_S))
                if reports[-1][2] is not None:
                    break  # the peer would wait on this rank; do not wait for it
        finally:
            for proc in procs:
                proc.join(timeout=30)
                if proc.is_alive():
                    proc.kill()

        for dp_rank, reduced, error in sorted(reports):
            assert error is None, f"DP rank {dp_rank} failed:\n{error}"
            # Both ranks joined the same group: 1 + 2 over the two ranks.
            assert reduced == 3, f"DP rank {dp_rank} all-reduced to {reduced}"

        if with_coord_store:
            assert coord_store is not None
            bound_port = int(coord_store.get("world_pg_port").decode())
            assert bound_port != dp_master_port
    finally:
        if squatter is not None:
            squatter.close()
