# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The copy-engine PP transport (CUDA copy engines, or SDMA engines on ROCm
gfx950) and the MoRI-IO one (ROCm) carry every step's tensors intact through a
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

import gc
import os
import weakref
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import ray
import torch

from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_pp_group,
    get_tp_group,
)
from vllm.distributed.pp_transport import create_pp_transport, row_specs
from vllm.distributed.pp_transport.copy_engine import (
    CopyEngineTransport,
    CopyEngineUnavailable,
)
from vllm.distributed.pp_transport.mori_io import (
    MoriIOTransport,
    check_available,
    pack_offsets,
    row_bytes,
    slot_bytes,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.import_utils import has_mori

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


def _inject_mori_setup_failure(stack: ExitStack, phase: str) -> None:
    """Make one MoRI-IO setup phase raise on this rank, for as long as `stack`
    is open: `dependency` (MoRI availability check), `register` (receive slot
    registration), `connect` (session creation) or `worker` (a background
    thread selecting its GPU)."""
    import threading

    import mori.io

    def fail(*args, **kwargs):
        raise RuntimeError(f"injected MoRI {phase} failure")

    if phase == "dependency":
        stack.enter_context(patch("vllm.utils.import_utils.has_mori", new=fail))
    elif phase == "worker":
        set_device = torch.accelerator.set_device_index

        def fail_in_thread(device):
            if threading.current_thread() is not threading.main_thread():
                fail()
            return set_device(device)

        stack.enter_context(
            patch.object(torch.accelerator, "set_device_index", fail_in_thread)
        )
    else:
        methods = {"register": "register_torch_tensor", "connect": "create_session"}
        stack.enter_context(patch.object(mori.io.IOEngine, methods[phase], new=fail))


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
        self,
        pp_size: int,
        port: int,
        tp_size: int = 1,
        fabric: bool = False,
        transport_cls: type = CopyEngineTransport,
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
        transport = None
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
            transport = transport_cls(_config(tp_size), pp_group, persistent)

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
            if transport is not None:
                transport.close()
                if isinstance(transport, MoriIOTransport):
                    released = weakref.ref(transport)
                    del transport
                    gc.collect()
                    assert released() is None
            parallel_state.in_the_same_node_as = in_the_same_node_as
            self._teardown()

    def run_mori_one_rank_failing(
        self,
        pp_size: int,
        port: int,
        tp_size: int = 1,
        phase: str = "connect",
        # Not used here: `_run_scenario` reads it to pick the MoRI platform check.
        transport_cls: type = MoriIOTransport,
    ) -> None:
        """A failed dependency, allocation, connection or GPU thread on one
        TP chain must reject setup on all chains and release receive slots."""
        if self.rank >= pp_size * tp_size:
            return
        pp_group = self._init(pp_size, port, tp_size)
        try:
            persistent = (
                None
                if pp_group.is_first_rank
                else IntermediateTensors(
                    {
                        k: torch.empty(
                            MAX_ROWS, WIDTH, dtype=torch.bfloat16, device="cuda"
                        )
                        for k in KEYS
                    }
                )
            )
            before = torch.accelerator.memory_allocated()
            # Receive slots are registered only on non-first stages, so a
            # register failure is injected on PP1 / TP0 (rank tp_size); the
            # other phases fail on rank 0. The other TP chain must fail too.
            fail_rank = tp_size if phase == "register" else 0
            with ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, VLLM_PP_TRANSPORT="mori_io"))
                if self.rank == fail_rank:
                    _inject_mori_setup_failure(stack, phase)
                with pytest.raises(
                    RuntimeError, match=f"injected MoRI {phase} failure"
                ):
                    create_pp_transport(_config(tp_size), persistent)
            gc.collect()
            torch.accelerator.synchronize()
            assert torch.accelerator.memory_allocated() <= before
        finally:
            self._teardown()

    def run_mori_row_limit(
        self,
        pp_size: int,
        port: int,
        tp_size: int = 1,
        transport_cls: type = MoriIOTransport,
    ) -> None:
        """One oversized tensor must be rejected even when another smaller
        tensor leaves enough space in the combined slot."""
        if self.rank >= pp_size * tp_size:
            return
        pp_group = self._init(pp_size, port, tp_size)
        transport = None
        try:
            persistent = (
                None
                if pp_group.is_first_rank
                else {
                    k: torch.empty(MAX_ROWS, WIDTH, dtype=torch.bfloat16, device="cuda")
                    for k in KEYS
                }
            )
            transport = transport_cls(_config(tp_size), pp_group, persistent)
            if pp_group.is_first_rank:
                tensors = {
                    "hidden_states": torch.empty(
                        MAX_ROWS + 1, WIDTH, dtype=torch.bfloat16, device="cuda"
                    ),
                    "residual": torch.empty(
                        1, WIDTH, dtype=torch.bfloat16, device="cuda"
                    ),
                }
                with pytest.raises(RuntimeError, match="hidden_states.*at most"):
                    transport.send(tensors, pp_group)
            torch.distributed.barrier(group=pp_group.cpu_group)
        finally:
            if transport is not None:
                transport.close()
            self._teardown()

    def run_mori_close_pending(
        self,
        pp_size: int,
        port: int,
        tp_size: int = 1,
        transport_cls: type = MoriIOTransport,
    ) -> None:
        """A bounded close must retain a credit-blocked write's buffers; after
        the consumer drains every step, retrying close releases them."""
        if self.rank >= pp_size * tp_size:
            return
        pp_group = self._init(pp_size, port, tp_size)
        transport = None
        try:
            persistent = (
                None
                if pp_group.is_first_rank
                else {
                    k: torch.empty(MAX_ROWS, WIDTH, dtype=torch.bfloat16, device="cuda")
                    for k in KEYS
                }
            )
            transport = transport_cls(_config(tp_size), pp_group, persistent)
            if pp_group.is_first_rank:
                for step in range(4):
                    h = torch.full(
                        (_rows(step), WIDTH), step, dtype=torch.bfloat16, device="cuda"
                    )
                    transport.send({"hidden_states": h, "residual": -h}, pp_group)
                torch.accelerator.synchronize()
                allocated = torch.accelerator.memory_allocated()
                with pytest.raises(RuntimeError, match="still using its buffers"):
                    transport.close()
                assert torch.accelerator.memory_allocated() == allocated
                # The receiver has not read a slot yet, so the fourth write
                # cannot finish before this explicit handoff.
                pp_group.send_object("close kept the buffers", dst=1)
                assert pp_group.recv_object(src=1) == "all four steps arrived"
                transport.close()
                assert torch.accelerator.memory_allocated() < allocated
            else:
                assert pp_group.recv_object(src=0) == "close kept the buffers"
                for step in range(4):
                    received = transport.recv(pp_group).tensors
                    assert received["hidden_states"].shape == (_rows(step), WIDTH)
                    assert (received["hidden_states"] == step).all(), step
                    assert (received["residual"] == -step).all(), step
                transport.close()
                pp_group.send_object("all four steps arrived", dst=0)
            transport.close()
        finally:
            if transport is not None:
                transport.close()
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
            # Under "auto", NCCL: on ROCm always, elsewhere by the failure.
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
    # NVML / amdsmi, not CUDA: the test process must not initialize CUDA.
    if kwargs.get("transport_cls") is MoriIOTransport:
        supported = current_platform.is_rocm() and has_mori()
    elif current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx950

        supported = on_gfx950()
    else:
        supported = current_platform.is_cuda()
    gpus = list(range(pp_size * tp_size))
    if not (supported and current_platform.is_fully_connected(gpus)):
        pytest.skip(
            "PP transport needs NVLink-connected CUDA GPUs or xGMI-connected "
            "gfx950, or MoRI on xGMI-connected ROCm GPUs"
        )
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

    if not current_platform.is_cuda():
        pytest.skip("multi-node chains need CUDA")
    if not imex_channel_available():
        pytest.skip("FABRIC handles need an IMEX channel")
    _run_scenario("run_chain", 4, fabric=True)


@_multi_gpu(num_gpus=2)
def test_mori_io_pp2():
    _run_scenario("run_chain", 2, transport_cls=MoriIOTransport)


@_multi_gpu(num_gpus=4)
def test_mori_io_pp4():
    _run_scenario("run_chain", 4, transport_cls=MoriIOTransport)


@_multi_gpu(num_gpus=4)
def test_mori_io_tp2_pp2():
    _run_scenario("run_chain", 2, tp_size=2, transport_cls=MoriIOTransport)


@_multi_gpu(num_gpus=4)
@pytest.mark.parametrize("phase", ["dependency", "register", "connect", "worker"])
def test_mori_io_setup_failure_reaches_other_tp_chains(phase):
    _run_scenario(
        "run_mori_one_rank_failing",
        2,
        tp_size=2,
        phase=phase,
        transport_cls=MoriIOTransport,
    )


@_multi_gpu(num_gpus=2)
def test_mori_io_rejects_one_tensor_exceeding_its_row_capacity():
    _run_scenario("run_mori_row_limit", 2, transport_cls=MoriIOTransport)


@_multi_gpu(num_gpus=2)
def test_mori_io_close_retains_pending_transfer_buffers():
    _run_scenario("run_mori_close_pending", 2, transport_cls=MoriIOTransport)


def test_mori_io_slot_holds_every_relayed_tensor():
    """A MoRI-IO receive slot holds the largest step of every intermediate
    tensor the stage receives, relayed aux hidden states and positions
    included, and each tensor of a step starts 256-byte aligned without
    overlapping the next."""
    persistent = {
        key: torch.empty(MAX_ROWS, WIDTH, dtype=torch.bfloat16)
        for key in ("hidden_states", "residual", "aux_3", "aux_17", "aux_31")
    }
    persistent["positions"] = torch.empty(MAX_ROWS, 3, dtype=torch.int64)
    assert slot_bytes(persistent) == (5 * MAX_ROWS * WIDTH * 2 + MAX_ROWS * 3 * 8)

    sizes = row_bytes(row_specs(persistent))
    rows = dict.fromkeys(persistent, 1337)
    offsets, used = pack_offsets(sizes, rows)
    assert used <= slot_bytes(persistent)
    ends = [offsets[k] + rows[k] * sizes[k] for k in persistent]
    assert all(offset % 256 == 0 for offset in offsets.values())
    assert all(end <= offsets[k] for end, k in zip(ends, list(persistent)[1:]))
    assert ends[-1] <= used


def test_mori_io_unavailable_is_a_clear_error():
    if current_platform.is_rocm() and has_mori():
        pytest.skip("MoRI-IO is available here")
    with pytest.raises(RuntimeError, match="VLLM_PP_TRANSPORT=mori_io needs"):
        check_available()
