# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two TP ranks whose local autotune timings disagree must pick the same Triton config
under VLLM_TRITON_SYNC_AUTOTUNE_TP, and different configs without it, also when only
one rank has the key in its on-disk autotune cache.

Run `pytest tests/distributed/test_tp_synced_autotune.py`.
"""

import tempfile
from types import SimpleNamespace

import pytest
import ray
import torch

from ..utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)


def _picked_block(rank: int, mode: str) -> list[tuple[int, int, int]]:
    """Autotune a copy kernel with rank-dependent fake timings; gather the picks."""
    import torch.distributed as dist

    from vllm.distributed import get_tp_group
    from vllm.forward_context import override_forward_context
    from vllm.triton_utils import tl, tp_synced_autotune, triton

    # A private cache dir per rank, so a rank sees only the autotune results it wrote.
    triton.knobs.cache.dir = tempfile.mkdtemp(prefix=f"triton_rank{rank}_")
    triton.knobs.autotuning.cache = mode == "warm_rank1"
    Autotuner = triton.runtime.autotuner.Autotuner
    configs = [triton.Config({"BLOCK": 64}), triton.Config({"BLOCK": 128})]

    fastest_here = 64 if rank == 0 else 128
    benched: list[int] = []

    def fake_bench(self, *args, config, **meta):
        # Each rank believes a different config is the fastest.
        benched.append(config.kwargs["BLOCK"])
        return [1.0] if config.kwargs["BLOCK"] == fastest_here else [2.0]

    Autotuner._bench = fake_bench
    if mode != "unsynced":
        tp_synced_autotune.install()
        tp_synced_autotune.configure(
            SimpleNamespace(
                enable_expert_parallel=False,
                use_sequence_parallel_moe=False,
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            )
        )

    @triton.autotune(configs=configs, key=["n"])
    @triton.jit
    def copy_kernel(src, dst, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)

    n = 1000
    src = torch.arange(n, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)

    def launch():
        copy_kernel[lambda m: (triton.cdiv(n, m["BLOCK"]),)](src, dst, n)

    group = get_tp_group().cpu_group
    if mode == "warm_rank1" and rank == 1:
        launch()  # outside a forward pass: tuned locally, written to rank 1's disk
        copy_kernel.cache.clear()  # as after a restart: only the disk entry is left
        benched.clear()
    dist.barrier(group=group)

    with override_forward_context(SimpleNamespace()):  # type: ignore[arg-type]
        launch()
    torch.accelerator.synchronize()
    torch.testing.assert_close(dst, src)

    picks: list[tuple[int, int, int]] = [(0, 0, 0)] * get_tp_group().world_size
    dist.all_gather_object(
        picks,
        (copy_kernel.best_config.kwargs["BLOCK"], rank, len(benched)),
        group=group,
    )
    return picks


def _worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port, mode):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    torch.accelerator.set_device_index(torch.device(f"cuda:{rank}"))
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    full = _picked_block(rank, mode)
    picks = [p[0] for p in full]
    benches = [p[2] for p in full]
    if mode == "unsynced":
        assert picks == [64, 128], full  # the test can tell the difference
        assert benches == [2, 2], full
    else:
        assert picks == [64, 64], full  # every rank follows rank 0
        assert benches == [2, 0], full  # only rank 0 tunes; rank 1 skips even its disk


@ray.remote(num_gpus=1, max_calls=1)
def synced_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port, "synced")


@ray.remote(num_gpus=1, max_calls=1)
def unsynced_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port, "unsynced")


@ray.remote(num_gpus=1, max_calls=1)
def warm_rank1_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port, "warm_rank1")


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "test_target", [synced_worker, unsynced_worker, warm_rank1_worker]
)
def test_tp_synced_autotune(monkeypatch: pytest.MonkeyPatch, test_target):
    multi_process_parallel(monkeypatch, 2, 1, test_target)
