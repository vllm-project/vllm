# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import pytest
import ray
import torch

from tests.utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)
from vllm.platforms import current_platform


def _on_cdna4() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import get_cdna_version

    return get_cdna_version() == 4


@ray.remote(num_gpus=1, max_calls=1)
def _sync_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm.distributed import get_tp_group
    from vllm.models.minimax_m3.amd.mono.sync import (
        ELEMENTS,
        BoundedAllReducePrototype,
    )

    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    ensure_model_parallel_initialized(tp_size, pp_size)

    prototype = BoundedAllReducePrototype(rank, get_tp_group().cpu_group, device)
    try:
        partial = torch.full((ELEMENTS,), rank + 1, dtype=torch.float32, device=device)
        output = prototype.run(partial, epoch=1)
        torch.accelerator.synchronize()
        torch.testing.assert_close(output, torch.full_like(output, 10.0))

        start = time.monotonic()
        prototype.run(partial, epoch=2, fault_rank=2)
        torch.accelerator.synchronize()
        elapsed = time.monotonic() - start
        health = prototype.health()
        assert elapsed < 10.0
        assert health.failed
        assert health.rank == 2
        assert health.epoch == 2
        assert 0 <= health.waiter_rank < tp_size
    finally:
        prototype.close()


@pytest.mark.skipif(not _on_cdna4(), reason="the sync prototype targets CDNA4")
def test_bounded_peer_allreduce(monkeypatch: pytest.MonkeyPatch):
    if torch.accelerator.device_count() < 4:
        pytest.skip("needs four GPUs")
    pytest.importorskip("flydsl")
    multi_process_parallel(monkeypatch, 4, 1, _sync_worker)
