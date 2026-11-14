# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The copy-engine PP transport carries every step's tensors intact through a
PP chain (one per TP rank), with varying sizes and stages that take turns
lagging behind, and every rank makes the same transport choice when one rank
cannot set it up.

Every scenario runs on one shared set of Ray workers (one per GPU): a worker
keeps its vLLM import and CUDA context across scenarios and only
re-initializes the distributed environment, so the startup cost is paid once
per file instead of once per scenario. Sharing workers across tests requires
one process for the whole file, hence the plain marks of `multi_gpu_marks`
instead of `multi_gpu_test` (which forks per test, and a forked child cannot
use the parent's Ray connection).
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray
import torch

from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_pp_group,
    get_tp_group,
)
from vllm.distributed.pp_transport import create_pp_transport
from vllm.distributed.pp_transport.copy_engine import (
    CopyEngineTransport,
    CopyEngineUnavailable,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from ..utils import (
    VLLM_PATH,
    _test_rendezvous_path,
    get_open_port,
    init_test_distributed_environment,
    multi_gpu_marks,
)

MAX_ROWS, WIDTH, STEPS = 4096, 1024, 40
KEYS = ("hidden_states", "residual")


def _rows(step: int) -> int:
    return [MAX_ROWS, 1, 1337, 512][step % 4]


def _config(tp_size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size, prefill_context_parallel_size=1
        ),
    )


@ray.remote(num_gpus=1)
class PPTransportWorker:
    """One rank's Ray worker, kept alive across this file's scenarios."""

    def __init__(self, rank: int):
        self.rank = rank
        torch.accelerator.set_device_index(rank)

    def ping(self) -> None:
        """Lets the driver wait for construction."""

    def _init(self, pp_size: int, port: int, tp_size: int):
        init_test_distributed_environment(tp_size, pp_size, self.rank, str(port))
        return get_pp_group()

    @staticmethod
    def _teardown() -> None:
        destroy_model_parallel()
        destroy_distributed_environment()

    def run_chain(
        self, pp_size: int, port: int, tp_size: int = 1, fabric: bool = False
    ) -> None:
        """Every step's tensors flow through the chain with varying row
        counts and stages that take turns lagging; the last stage checks
        every step's values. With TP > 1, each TP rank's chain carries its
        own values, so crossed chains would show. `fabric` makes the chain
        look multi-node, so it shares buffers through FABRIC handles."""
        if self.rank >= pp_size * tp_size:
            return
        pp_group = self._init(pp_size, port, tp_size)
        # Imported here: Ray cannot pickle the module as a global of this actor.
        import vllm.distributed.parallel_state as parallel_state

        in_the_same_node_as = parallel_state.in_the_same_node_as
        if fabric:
            parallel_state.in_the_same_node_as = lambda *args, **kwargs: [
                rank == 0 for rank in range(pp_size)
            ]
        try:
            # Small integers, exact in bf16.
            base = 120 * get_tp_group().rank_in_group
            first, last = pp_group.is_first_rank, pp_group.is_last_rank

            persistent = None
            if not first:
                persistent = {
                    k: torch.zeros(MAX_ROWS, WIDTH, dtype=torch.bfloat16, device="cuda")
                    for k in KEYS
                }
            transport = CopyEngineTransport(_config(tp_size), pp_group, persistent)

            lag = torch.randn(4096, 4096, device="cuda")
            for step in range(STEPS):
                # Senders often lag (a receiver reading too early sees stale
                # values), the last stage sometimes (a sender overwriting too
                # early corrupts values it has not read yet).
                if (step % 2 == 0) if not last else (step % 7 == 3):
                    for _ in range(20):
                        lag @ lag
                if first:
                    h = torch.full(
                        (_rows(step), WIDTH),
                        base + step % 100,
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    transport.send({"hidden_states": h, "residual": -h}, pp_group)
                    continue
                received = transport.recv(pp_group).tensors
                if not last:
                    transport.send({k: v + 1 for k, v in received.items()}, pp_group)
                    continue
                sent = base + step % 100
                want = sent + pp_size - 2
                assert received["hidden_states"].shape == (_rows(step), WIDTH)
                assert (received["hidden_states"] == want).all(), step
                assert (received["residual"] == want - 2 * sent).all(), step
            torch.accelerator.synchronize()
            # A stage's last metadata send completes only once the next stage
            # receives it, so no stage may exit before the chain is done.
            torch.distributed.barrier(group=pp_group.cpu_group)
        finally:
            parallel_state.in_the_same_node_as = in_the_same_node_as
            self._teardown()

    def run_one_rank_failing(
        self, pp_size: int, port: int, tp_size: int = 1, fail_rank: int = 1
    ) -> None:
        """Rank `fail_rank` cannot map its peers' buffers: under `copy_engine`
        every rank raises, under `auto` every rank falls back to NCCL, and
        nothing stays allocated."""
        pp_group = self._init(pp_size, port, tp_size)
        # Imported here: Ray cannot pickle the module as a global of this actor.
        from vllm import _custom_ops as ops

        open_mem_handle = ops.open_mem_handle
        transport_env = os.environ.get("VLLM_PP_TRANSPORT")
        try:
            if self.rank == fail_rank:

                def fail(handle):
                    raise RuntimeError("injected open failure")

                ops.open_mem_handle = fail
            persistent = None
            if not pp_group.is_first_rank:
                persistent = IntermediateTensors(
                    {
                        k: torch.zeros(
                            MAX_ROWS, WIDTH, dtype=torch.bfloat16, device="cuda"
                        )
                        for k in KEYS
                    }
                )
            torch.accelerator.synchronize()
            free_before = torch.accelerator.get_memory_info()[0]

            os.environ["VLLM_PP_TRANSPORT"] = "copy_engine"
            with pytest.raises(CopyEngineUnavailable, match="injected open failure"):
                create_pp_transport(_config(tp_size), persistent)
            os.environ["VLLM_PP_TRANSPORT"] = "auto"
            assert create_pp_transport(_config(tp_size), persistent) is None

            # Every rank's buffers and peer mappings were released.
            torch.distributed.barrier(group=pp_group.cpu_group)
            torch.accelerator.synchronize()
            assert torch.accelerator.get_memory_info()[0] >= free_before - (2 << 20)
            torch.distributed.barrier(group=pp_group.cpu_group)
        finally:
            ops.open_mem_handle = open_mem_handle
            if transport_env is None:
                os.environ.pop("VLLM_PP_TRANSPORT", None)
            else:
                os.environ["VLLM_PP_TRANSPORT"] = transport_env
            self._teardown()


class _WorkerPool:
    """This file's shared Ray cluster: one worker per GPU, started lazily so
    that a skipped file never pays for it."""

    def __init__(self) -> None:
        self.workers: list | None = None

    def get(self) -> list:
        if self.workers is None:
            if not ray.is_initialized():
                # As in multi_process_parallel: the repo on PYTHONPATH is
                # enough for local Ray workers, and Ray must not narrow
                # device visibility (each worker picks its device by rank).
                env_vars = {
                    "PYTHONPATH": os.pathsep.join(
                        filter(None, [str(VLLM_PATH), os.environ.get("PYTHONPATH")])
                    ),
                    **{
                        env_var: "1"
                        for env_var in current_platform.ray_noset_device_env_vars
                    },
                }
                ray.init(runtime_env={"env_vars": env_vars})
            # The PP=4 scenarios size the pool; with fewer GPUs only PP=2 runs.
            n = 4 if current_platform.device_count() >= 4 else 2
            self.workers = [
                PPTransportWorker.remote(rank)  # type: ignore[attr-defined]
                for rank in range(n)
            ]
            ray.get([w.ping.remote() for w in self.workers])
        return self.workers

    def kill(self) -> None:
        """Kill every worker and forget the pool. A rank whose scenario
        raised keeps its actor (and its gloo connections) alive, so its peers
        would hang on the collective it abandoned; the next test starts a
        fresh pool."""
        if self.workers is not None:
            for worker in self.workers:
                ray.kill(worker)
            self.workers = None


_POOL = _WorkerPool()


@pytest.fixture(scope="module", autouse=True)
def _shutdown_pool():
    yield
    if ray.is_initialized():
        ray.shutdown()


def _multi_gpu(*, num_gpus: int):
    """`multi_gpu_test` without the per-test fork, so the tests share one
    process and can share the Ray workers."""

    def wrap(f):
        for mark in reversed(multi_gpu_marks(num_gpus=num_gpus)):
            f = mark(f)
        return f

    return wrap


def _run_scenario(method: str, pp_size: int, tp_size: int = 1, **kwargs) -> None:
    # NVML, not CUDA: the test process must not initialize CUDA.
    if not (
        current_platform.is_cuda()
        and current_platform.is_fully_connected(list(range(pp_size * tp_size)))
    ):
        pytest.skip("copy engines need NVLink-connected CUDA GPUs")
    port = get_open_port()
    refs = [
        getattr(w, method).remote(pp_size, port, tp_size, **kwargs) for w in _POOL.get()
    ]
    try:
        ray.get(refs)
    except BaseException:
        _POOL.kill()
        raise
    finally:
        # All ranks are done with the store, so the file has no readers left.
        Path(_test_rendezvous_path(port)).unlink(missing_ok=True)


@_multi_gpu(num_gpus=2)
def test_copy_engine_pp2():
    _run_scenario("run_chain", 2)


@_multi_gpu(num_gpus=4)
def test_copy_engine_pp4():
    _run_scenario("run_chain", 4)


@_multi_gpu(num_gpus=4)
def test_one_rank_failing_setup_gives_the_same_choice_everywhere():
    """Rank 1 (a middle stage) cannot map its peers' buffers: under
    `copy_engine` every rank raises, under `auto` every rank falls back to
    NCCL, and nothing stays allocated."""
    _run_scenario("run_one_rank_failing", 4)


@_multi_gpu(num_gpus=4)
def test_copy_engine_tp2_pp2():
    _run_scenario("run_chain", 2, tp_size=2)


@_multi_gpu(num_gpus=4)
def test_one_tp_chain_failing_setup_gives_the_same_choice_everywhere():
    """Only TP rank 1's chain fails (at its last stage, global rank 3), so TP
    rank 0's chain must fall back with it."""
    _run_scenario("run_one_rank_failing", 2, tp_size=2, fail_rank=3)


@_multi_gpu(num_gpus=4)
def test_copy_engine_fabric_pp4():
    """The multi-node NVLink path (FABRIC handles), on one node."""
    from vllm.utils.vmm_driver import imex_channel_available

    if not imex_channel_available():
        pytest.skip("FABRIC handles need an IMEX channel")
    _run_scenario("run_chain", 4, fabric=True)
