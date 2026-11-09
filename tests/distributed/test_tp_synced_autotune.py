# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two TP ranks whose local autotune timings disagree must pick the same Triton config
under VLLM_TRITON_SYNC_AUTOTUNE_TP, and different configs without it.

Run `pytest tests/distributed/test_tp_synced_autotune.py`.
"""

from types import SimpleNamespace

import pytest
import ray
import torch

from ..utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)


def _picked_block(rank: int, sync: bool) -> list[int]:
    """Autotune a copy kernel with rank-dependent fake timings; gather the picks."""
    import torch.distributed as dist

    from vllm.distributed import get_tp_group
    from vllm.forward_context import override_forward_context
    from vllm.triton_utils import tl, tp_synced_autotune, triton

    Autotuner = triton.runtime.autotuner.Autotuner
    configs = [triton.Config({"BLOCK": 64}), triton.Config({"BLOCK": 128})]

    def fake_bench(self, *args, config, **meta):
        # Each rank believes "its own" config is the fastest.
        return [1.0] if configs.index(config) == rank else [2.0]

    Autotuner._bench = fake_bench
    if sync:
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

    # A per-case size gives each case its own autotune key: Triton may reuse cached
    # autotune results across processes, which would hide the per-rank choice.
    n = 1000 if sync else 1001
    src = torch.arange(n, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    with override_forward_context(SimpleNamespace()):  # type: ignore[arg-type]
        copy_kernel[lambda m: (triton.cdiv(n, m["BLOCK"]),)](src, dst, n)
    torch.accelerator.synchronize()
    torch.testing.assert_close(dst, src)

    picks: list[int] = [0] * get_tp_group().world_size
    dist.all_gather_object(
        picks, copy_kernel.best_config.kwargs["BLOCK"], group=get_tp_group().cpu_group
    )
    return picks


def _worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port, sync):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    torch.accelerator.set_device_index(torch.device(f"cuda:{rank}"))
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    picks = _picked_block(rank, sync)
    if sync:
        assert picks == [64, 64], picks  # every rank follows rank 0's timings
    else:
        assert picks == [64, 128], picks  # the test can tell the difference


@ray.remote(num_gpus=1, max_calls=1)
def synced_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port, sync=True)


@ray.remote(num_gpus=1, max_calls=1)
def unsynced_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port, sync=False)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("test_target", [synced_worker, unsynced_worker])
def test_tp_synced_autotune(monkeypatch: pytest.MonkeyPatch, test_target):
    multi_process_parallel(monkeypatch, 2, 1, test_target)
