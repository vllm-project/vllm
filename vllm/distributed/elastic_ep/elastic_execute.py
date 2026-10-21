# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import faulthandler
import gc
import os
import threading
import time
import weakref
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributed import P2POp

from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.compilation.wrapper import reset_compile_wrapper
from vllm.config import ParallelConfig, set_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pcp_group,
    get_tp_group,
    get_world_group,
)
from vllm.distributed.elastic_ep import rocm_debug
from vllm.distributed.elastic_ep.standby_state import (
    create_standby_groups,
    get_standby_dp_group,
    get_standby_ep_group,
    get_standby_eplb_group,
    get_standby_world_group,
    pop_standby_groups,
)
from vllm.distributed.eplb.eplb_communicator import (
    EplbCommunicator,
)
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    _replace_active_groups,
)
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    get_ep_all2all_manager,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.eep_reconfigure import (
    make_eep_staged_quant_method,
)
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    sync_flashinfer_autotune_cache,
)
from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
from vllm.platforms import current_platform
from vllm.utils import is_moe_layer
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.worker.dp_utils import skip_dp_coordination
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.eplb_utils import preserve_serving_state
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.workspace import lock_workspace, unlock_workspace

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )


def can_reuse_fused_moe_kernel(parallel_config: ParallelConfig) -> bool:
    """Whether fused MoE kernels can be reused across EP rank changes.

    The EP backend must preserve its CUDA graph entries when ranks are connected,
    disconnected, masked, or unmasked.
    """
    return parallel_config.all2all_backend == "nixl_ep"


def batch_transfer_weights(
    model: nn.Module,
    is_sender: bool,
    peer_rank: int,
    dp_group: StatelessGroupCoordinator,
    expert_weights: Sequence[Iterable[torch.Tensor]],
) -> None:
    device_comm = dp_group.device_communicator
    if device_comm is None:
        raise ValueError("No device communicator found")

    expert_weights_set = set()
    for weight_group in expert_weights:
        for weight in weight_group:
            expert_weights_set.add(weight.data_ptr())

    state_dict = model.state_dict()
    all_params = []

    for name, param in state_dict.items():
        if name.endswith("expert_map") or name.find("._shared_experts") != -1:
            continue
        if param.data_ptr() not in expert_weights_set:
            all_params.append(param.data)

    assert len(all_params) > 0
    p2p_ops = []
    for param in all_params:
        # PyNccl transfers flat memory and does not honor tensor strides.
        transfer_param = param.contiguous()
        op = object.__new__(P2POp)
        op.op = torch.distributed.isend if is_sender else torch.distributed.irecv
        op.tensor = transfer_param
        op.group_peer = peer_rank
        p2p_ops.append(op)
        if transfer_param is not param:
            device_comm.batch_isend_irecv(p2p_ops)
            p2p_ops.clear()
            if not is_sender:
                param.copy_(transfer_param)
    if p2p_ops:
        device_comm.batch_isend_irecv(p2p_ops)


def broadcast_expert_mapping(
    physical_to_logical: torch.Tensor | None,
    dp_group: StatelessGroupCoordinator,
    device: torch.device,
    src_rank: int = 0,
) -> torch.Tensor:
    if dp_group.rank_in_group == src_rank:
        assert physical_to_logical is not None
        assert physical_to_logical.dtype == torch.int64
        shape_tensor = torch.tensor(
            list(physical_to_logical.shape), dtype=torch.int64, device="cpu"
        )
    else:
        shape_tensor = torch.empty(2, dtype=torch.int64, device="cpu")

    shape_tensor = dp_group.tcp_store_group.broadcast(shape_tensor, src_rank)

    if dp_group.rank_in_group != src_rank:
        assert device is not None
        physical_to_logical = torch.empty(
            tuple(shape_tensor.tolist()),
            dtype=torch.int64,
            device=device,
        )

    assert physical_to_logical is not None
    return dp_group.broadcast(physical_to_logical, src_rank)


class ElasticEPScalingExecutor:
    def __init__(self, worker):
        self.worker_ref = weakref.ref(worker)
        self.reconfig_request = None
        self._staged_moe_quant_methods: dict[nn.Module, FusedMoEMethodBase] = {}
        self._prepared_eplb_communicator: EplbCommunicator | None = None
        # Groups held back because a shrunk communicator still runs on them.
        self._deferred_parent_groups: list[GroupCoordinator] = []
        self._async_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ElasticEPAsync"
        )
        self._async_future: Future[None] | None = None
        self._group_cleanup_future: Future[None] | None = None

    @property
    def worker(self):
        worker = self.worker_ref()
        if worker is None:
            raise RuntimeError("Worker has been garbage collected")
        return worker

    def _can_reuse_fused_moe_kernel(self) -> bool:
        return can_reuse_fused_moe_kernel(
            self.worker.vllm_config.parallel_config,
        )

    @contextmanager
    def _suppress_eplb(self) -> Iterator[None]:
        # A rank warming up alone must not advance the EPLB step counter.
        runner = self.worker.model_runner
        was_suppressed = runner.eep_eplb_suppressed
        runner.eep_eplb_suppressed = True
        try:
            yield
        finally:
            runner.eep_eplb_suppressed = was_suppressed

    @contextmanager
    def _disable_flashinfer_autotune(self) -> Iterator[None]:
        kernel_config = self.worker.vllm_config.kernel_config
        enabled = kernel_config.enable_flashinfer_autotune
        kernel_config.enable_flashinfer_autotune = False
        try:
            yield
        finally:
            kernel_config.enable_flashinfer_autotune = enabled

    def execute(self, execute_method: str, *args, **kwargs):
        method = getattr(self, execute_method, None)
        if method is None:
            raise ValueError(f"Unknown execute method: {execute_method}")
        return method(*args, **kwargs)

    def start_async(self, execute_method: str, *args, **kwargs) -> str:
        if self._async_future is not None:
            raise RuntimeError("Another Elastic EP async method is active")
        if args and isinstance(args[0], ReconfigureDistributedRequest):
            self.reconfig_request = args[0]
        dp_rank = self.worker.vllm_config.parallel_config.data_parallel_rank
        done_key = f"eep_async/{execute_method}/{dp_rank}/{self.worker.rank}"
        self._async_future = self._async_executor.submit(
            self._run_async, execute_method, *args, **kwargs
        )
        self._async_future.add_done_callback(lambda _: self._mark_async_done(done_key))
        return done_key

    def _run_async(self, execute_method: str, *args, **kwargs) -> None:
        from vllm.platforms import current_platform

        self.worker.vllm_config.enable_trace_function_call_for_thread()
        assert hasattr(self.worker, "device")
        current_platform.set_device(self.worker.device)
        with set_current_vllm_config(self.worker.vllm_config):
            self.execute(execute_method, *args, **kwargs)

    def _mark_async_done(self, done_key: str) -> None:
        from vllm.distributed.utils import get_cached_tcp_store_client

        assert self.reconfig_request is not None
        get_cached_tcp_store_client(
            self.reconfig_request.new_data_parallel_master_ip,
            self.reconfig_request.coord_store_port,
        ).set(done_key, b"1")

    def clear_async(self) -> None:
        future = self._async_future
        if future is None:
            raise RuntimeError("No Elastic EP async method is active")
        if not future.done():
            raise RuntimeError("Elastic EP async method is not done")
        self._async_future = None
        future.result()

    def _destroy_retired_groups(
        self, groups: tuple[GroupCoordinator | None, ...]
    ) -> None:
        from vllm.platforms import current_platform

        current_platform.set_device(self.worker.device)
        self._destroy_released_parent_groups()
        for group in groups:
            if group is None:
                continue
            if self._defer_for_shrink_child(group):
                continue
            group.destroy()

    def _start_group_cleanup(self, groups: tuple[GroupCoordinator | None, ...]) -> None:
        # Only `_retire_groups` can hand this an empty tuple, by having already
        # done the work. Goes away with the flag, and then this is `main` again.
        if not groups:
            return
        assert self._group_cleanup_future is None
        self._group_cleanup_future = self._async_executor.submit(
            self._destroy_retired_groups, groups
        )

    def _wait_for_group_cleanup(self) -> None:
        if (future := self._group_cleanup_future) is not None:
            self._group_cleanup_future = None
            future.result()

    def _destroy_released_parent_groups(self) -> None:
        """Retire deferred groups once their last child has gone.

        A deferred group is one retirement behind, which is the earliest point
        at which its child can already be dead, and leaving it to a later one
        would hold the buffers for the rest of the process.
        """
        if not self._deferred_parent_groups:
            return
        still_held = []
        for group in self._deferred_parent_groups:
            communicator = getattr(group, "device_communicator", None)
            pynccl_comm = getattr(communicator, "pynccl_comm", None)
            if pynccl_comm is not None and pynccl_comm.has_live_shrink_children:
                still_held.append(group)
                continue
            group.destroy()
        logger.warning(
            "[EEP-DEBUG] retired %d deferred parent group(s), %d still held",
            len(self._deferred_parent_groups) - len(still_held),
            len(still_held),
        )
        self._deferred_parent_groups = still_held

    def _defer_for_shrink_child(self, group: GroupCoordinator) -> bool:
        """Hold back a group whose communicator a shrunk child is running on.

        `ncclCommShrink` with NCCL_COMM_SHRINK_SHARE_RESOURCES=1 gives the child
        the parent's buffers rather than its own, so retiring the parent here
        frees memory a live communicator is still using. Both surviving ranks
        faulted five seconds after a scale down that did exactly that.

        This exists only because the shrink does, so if the trio goes then so
        does this. While it stays, deferring is only ever right for a group
        that actually has a child, because holding anything else back would
        keep buffers alive for no reason.
        """
        if os.getenv("VLLM_EEP_SHRINK_KEEP_PARENT") != "1":
            return False
        communicator = getattr(group, "device_communicator", None)
        pynccl_comm = getattr(communicator, "pynccl_comm", None)
        if pynccl_comm is None or not pynccl_comm.has_live_shrink_children:
            return False
        self._deferred_parent_groups.append(group)
        logger.warning(
            "[EEP-DEBUG] keeping a retired group alive because a shrunk "
            "communicator still runs on its resources, %d deferred so far",
            len(self._deferred_parent_groups),
        )
        return True

    @contextmanager
    def _watchdog(self, label: str) -> Iterator[None]:
        """Dump every thread's stack while a reconfiguration phase is stuck.

        A scale down hung inside drain_async for two hours without printing a
        single line. A rank stuck there cannot report anything by itself: the
        loop spins on an all_reduce over the EP CPU group and never returns to
        any logging statement, so the stacks of all its threads are the only
        evidence of who is waiting and on what. Samples repeat, because one
        snapshot cannot tell a slow phase from a stopped one.

        Diagnostic only.
        """
        seconds = float(os.getenv("VLLM_EEP_DEBUG_WATCHDOG") or 0)
        if seconds <= 0:
            yield
            return
        done = threading.Event()

        def bark() -> None:
            samples = 0
            while not done.wait(seconds):
                samples += 1
                logger.warning(
                    "[EEP-DEBUG] watchdog: %s has been running for %.0f s, "
                    "dumping all thread stacks (sample %d)",
                    label,
                    samples * seconds,
                    samples,
                )
                faulthandler.dump_traceback(all_threads=True)

        barker = threading.Thread(target=bark, name="eep-watchdog", daemon=True)
        barker.start()
        started = time.perf_counter()
        try:
            yield
        finally:
            done.set()
            barker.join(timeout=5)
            logger.warning(
                "[EEP-DEBUG] %s finished in %.3f s",
                label,
                time.perf_counter() - started,
            )

    @contextmanager
    def _timed(self, label: str) -> Iterator[None]:
        """Record how long one step of a reconfiguration took.

        commit_scale_up was measured running past 120 s on the CI machine,
        against a few seconds locally, and a rank crashes somewhere inside that
        window. The watchdog says the phase is slow but not which step is slow,
        and the stacks it dumps are a single instant rather than a breakdown.

        Diagnostic only.
        """
        if not os.getenv("VLLM_EEP_DEBUG_WATCHDOG"):
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            logger.warning(
                "[EEP-DEBUG] step %s took %.3f s",
                label,
                time.perf_counter() - started,
            )
            self._probe_ipc_health(label)

    def _probe_ipc_health(self, label: str) -> None:
        """Ask after each step whether IPC still works in this process.

        The refusal belongs to the process and not to the call that trips over
        it: standby groups are built 22 times out of 22 for the first
        reconfiguration a process performs and refused 10 times out of 10 for
        the second, and a probe immediately before a failure exports fine while
        the same probe immediately after refuses everything, including a buffer
        on a different card. Something between those two moments switches IPC
        off, and every step in between already has a name, so asking after each
        of them turns an interval into a step. That is worth more than another
        measurement of the failure itself, because an operation we can name is
        an operation we can try not to perform.

        Diagnostic only, and separate from VLLM_EEP_DEBUG_OFF on purpose: the
        passes that need this are exactly the ones that silence the probes
        which export enough to be suspected of causing the failure.
        """
        if os.getenv("VLLM_EEP_DEBUG_POISON_POINT") != "1":
            return
        from vllm.distributed.elastic_ep import rocm_debug

        logger.warning("[EEP-DEBUG] ipc after %s: %s", label, rocm_debug.ipc_health())

    def _retire_groups(
        self, groups: tuple[GroupCoordinator | None, ...]
    ) -> tuple[GroupCoordinator | None, ...]:
        """Destroy the groups the reconfiguration replaced, right here.

        This has to happen before the first collective on the new groups, and
        it has to be synchronous. RCCL exports IPC handles when it sets up the
        P2P transport for a new communicator, and on MI355 the export is
        refused with hsa_status 4097 whenever the communicators of the previous
        configuration are still alive: three passes that only waited two
        seconds here were refused, five that destroyed the groups were not.
        Handing the teardown to a background thread is not enough either, since
        it then runs while the new transport is being set up, which faulted the
        GPU with a write to a read-only page.

        Every one of those measurements predates the expert arena registration
        fix, and the run of 21 October showed they were measuring that fix
        instead: six passes retired nothing and none was refused, against zero
        of three before the arena stopped being deregistered under a live
        reader. The read-only page fault quoted above is the same one the arena
        controls produced, so it was failure 2 all along.

        So this is now the control arm rather than the default. With the flag
        unset the replaced groups go back to the background cleanup `main`
        already has, which is what should ship. The flag survives for one more
        run because the eager cases cannot speak for captured graphs, where a
        released graph still holds pointers into the arena.

        Returns the groups still needing cleanup, which is nothing when this
        has already done it.
        """
        live = sum(1 for group in groups if group is not None)
        if not live:
            return ()
        if os.getenv("VLLM_EEP_GROUP_RETIRE") != "1":
            # So a pass that took the background path can be told from one that
            # did not. Drop with the rest of the diagnostics.
            logger.info(
                "[Elastic EP] Handed %d replaced group(s) to the background "
                "cleanup, synchronous retirement is off",
                live,
            )
            return groups
        started = time.perf_counter()
        self._destroy_retired_groups(groups)
        logger.info(
            "[Elastic EP] Retired %d stale group(s) in %.3f s",
            live,
            time.perf_counter() - started,
        )
        return ()

    def _dump_group_pressure(self, label: str) -> None:
        """Record what is alive at the moments where the refusal happens.

        RCCL refuses to create IPC memory inside ncclCommInitRank at scale down
        and during the first collective on the new group at scale up, and both
        times there is no teardown in flight, so the only remaining candidate is
        how much is registered and alive right then. Nothing exposes the IPC
        registrations themselves, so count the communicators that own them.
        Diagnostic only.
        """
        if os.getenv("VLLM_EEP_DEBUG_GROUP_PRESSURE") != "1":
            return
        free_bytes, total_bytes = torch.accelerator.get_memory_info()
        comms = sum(
            1 for obj in gc.get_objects() if type(obj).__name__ == "PyNcclCommunicator"
        )
        logger.warning(
            "[EEP-DEBUG] pressure %s: %d pynccl comm(s), reserved %.2f GiB, "
            "allocated %.2f GiB, free %.2f of %.2f GiB",
            label,
            comms,
            torch.accelerator.memory_reserved() / 2**30,
            torch.accelerator.memory_allocated() / 2**30,
            free_bytes / 2**30,
            total_bytes / 2**30,
        )

    def _probe_bootstrap_if_requested(
        self,
        new_dp_size: int,
        new_world_size_across_dp: int,
        master_ip: str,
        coord_store_port: int,
    ) -> None:
        """Spend one stateless-group bootstrap before the real one.

        Diagnostic only. See probe_stateless_bootstrap for what this answers.
        """
        if os.getenv("VLLM_EEP_DEBUG_PROBE_BOOTSTRAP") != "1":
            return
        from vllm.distributed.elastic_ep.standby_state import (
            probe_stateless_bootstrap,
        )

        self._dump_group_pressure("before throwaway bootstrap")
        started = time.perf_counter()
        probe_stateless_bootstrap(
            new_dp_size=new_dp_size,
            new_world_size_across_dp=new_world_size_across_dp,
            master_ip=master_ip,
            coord_store_port=coord_store_port,
        )
        logger.warning(
            "[EEP-DEBUG] throwaway stateless group built and destroyed in %.3f s",
            time.perf_counter() - started,
        )

    def _drain_reshuffle_if_requested(self) -> None:
        """Wait for the previous reshuffle before touching the groups again.

        prepare_reconfiguration waits for the group cleanup but not for the
        expert copies that commit_scale_up scheduled asynchronously, and the
        drain that does wait for them runs later, inside the commit. On MI355
        the scale down request arrives 34 to 36 s into a 39 s copy window, and
        create_standby_groups then either gets hipIpcGetMemHandle refused with
        hsa_status 4097 or faults the GPU inside ncclCommInitRank. Draining
        here means no reconfiguration ever overlaps copies in flight.

        Diagnostic only.
        """
        if os.getenv("VLLM_EEP_DEBUG_DRAIN_BEFORE_RECONFIG") != "1":
            return
        eplb_state = self.worker.model_runner.eplb_state
        if eplb_state is None:
            return
        started = time.perf_counter()
        eplb_state.drain_async()
        torch.accelerator.synchronize()
        logger.warning(
            "[EEP-DEBUG] drained in-flight expert resharding in %.3f s",
            time.perf_counter() - started,
        )

    def shutdown(self) -> None:
        try:
            self._wait_for_group_cleanup()
        finally:
            self._async_executor.shutdown()

    def load_model(self) -> None:
        self.worker.load_model(load_dummy_weights=True)

    def prepare_reconfiguration(
        self, reconfig_request: ReconfigureDistributedRequest, use_all2all: bool
    ) -> None:
        self._wait_for_group_cleanup()
        self._drain_reshuffle_if_requested()
        self.reconfig_request = reconfig_request
        new_dp_size = reconfig_request.new_data_parallel_size
        old_dp_size = get_dp_group().world_size
        parallel_config = self.worker.vllm_config.parallel_config
        world_size = parallel_config.world_size
        new_world_size_across_dp = world_size * new_dp_size
        self._dump_group_pressure(f"before standby groups for dp={new_dp_size}")
        with self._timed("prepare create_standby_groups"):
            create_standby_groups(
                new_dp_size=new_dp_size,
                new_world_size_across_dp=new_world_size_across_dp,
                master_ip=reconfig_request.new_data_parallel_master_ip,
                coord_store_port=reconfig_request.coord_store_port,
                use_all2all=use_all2all,
                enable_eplb=parallel_config.enable_eplb,
            )
        self._dump_group_pressure(f"after standby groups for dp={new_dp_size}")
        self._probe_bootstrap_if_requested(
            new_dp_size=new_dp_size,
            new_world_size_across_dp=new_world_size_across_dp,
            master_ip=reconfig_request.new_data_parallel_master_ip,
            coord_store_port=reconfig_request.coord_store_port,
        )
        standby_ep_group = get_standby_ep_group()
        assert standby_ep_group is not None
        if current_platform.is_rocm():
            rocm_debug.dump_static()
            rocm_debug.dump(
                "prepare",
                {"dp": get_standby_dp_group(), "ep": standby_ep_group},
                heavy=True,
            )
        all2all_manager = get_ep_all2all_manager(standby_ep_group)
        with self._timed("prepare stage_ep_size"):
            all2all_manager.stage_ep_size()
        if not self._can_reuse_fused_moe_kernel():
            with self._timed("prepare stage_quant_methods"):
                self.stage_standby_moe_quant_methods(all2all_manager)
        with self._timed("prepare eplb_communicator"):
            self._prepare_eplb_communicator(get_standby_eplb_group())
        if new_dp_size > old_dp_size:
            with self._timed("prepare transfer_weights"):
                self.transfer_weights(old_dp_size, new_dp_size)
        # On ROCm this deadlocks while serving; warm_and_capture warms these at commit.
        if not current_platform.is_rocm():
            self._warm_target_groups(get_standby_dp_group(), standby_ep_group)
        if new_dp_size > old_dp_size and self._can_reuse_fused_moe_kernel():
            target_world_group = get_standby_world_group()
            assert target_world_group is not None
            sync_flashinfer_autotune_cache(self.worker.model_runner, target_world_group)

    def _prepare_eplb_communicator(self, eplb_group) -> None:
        assert eplb_group is not None
        model_runner = self.worker.model_runner
        eplb_state = model_runner.eplb_state
        assert eplb_state is not None
        self._prepared_eplb_communicator = eplb_state.create_communicator(
            model_runner.model_config, eplb_group
        )

    def transfer_weights(self, old_dp_size: int, new_dp_size: int) -> None:
        standby_dp_group = get_standby_dp_group()
        assert standby_dp_group is not None
        # Broadcast old_dp_size to all workers in standby group
        if standby_dp_group.rank_in_group < old_dp_size:
            old_dp_size_tensor = torch.tensor(
                [old_dp_size], dtype=torch.int64, device="cpu"
            )
        else:
            old_dp_size_tensor = torch.empty(1, dtype=torch.int64, device="cpu")
        old_dp_size_tensor = standby_dp_group.tcp_store_group.broadcast(
            old_dp_size_tensor, 0
        )

        num_new_workers = new_dp_size - old_dp_size
        dp_rank = self.worker.vllm_config.parallel_config.data_parallel_rank

        # Sender-receiver pairing: the first new_workers % old_dp_size
        # senders get (k+1) contiguous receivers, the rest get k
        # receivers.
        num_dst_per_sender = num_new_workers // old_dp_size
        remainder = num_new_workers % old_dp_size

        if dp_rank < remainder:
            recv_begin = dp_rank * (num_dst_per_sender + 1)
            recv_end = recv_begin + num_dst_per_sender + 1
        else:
            recv_begin = (
                remainder * (num_dst_per_sender + 1)
                + (dp_rank - remainder) * num_dst_per_sender
            )
            recv_end = recv_begin + num_dst_per_sender

        ranks_to_send = list(range(old_dp_size + recv_begin, old_dp_size + recv_end))

        model = self.worker.model_runner.get_model()
        for new_worker_rank in sorted(ranks_to_send):
            batch_transfer_weights(
                model=model,
                is_sender=True,
                peer_rank=new_worker_rank,
                dp_group=standby_dp_group,
                expert_weights=model.expert_weights,
            )
        torch.accelerator.synchronize()

    def _warm_target_groups(self, dp_group, ep_group) -> None:
        assert dp_group is not None and ep_group is not None
        stream = torch.Stream(device=dp_group.device)
        with stream:
            tensor = torch.zeros(1, dtype=torch.int32, device=dp_group.device)
            for group in (dp_group, ep_group):
                torch.distributed.all_reduce(tensor, group=group.device_group)
                stream.synchronize()

    def broadcast_expert_mapping(self) -> None:
        standby_dp_group = get_standby_dp_group()
        assert standby_dp_group is not None
        model_config = self.worker.model_runner.model_config
        eplb_state = self.worker.model_runner.eplb_state
        assert eplb_state is not None
        eplb_state.drain_async()
        eplb_model_state = eplb_state.model_states[model_config.compute_hash()]
        physical_to_logical = eplb_model_state.physical_to_logical_map_buffer
        broadcast_expert_mapping(
            physical_to_logical=physical_to_logical,
            dp_group=standby_dp_group,
            src_rank=0,
            device=self.worker.device,
        )

    def _make_eep_moe_config(self, module, dp_group, ep_group):
        parallel_config = self.worker.vllm_config.parallel_config
        tp_size = get_tp_group().world_size
        sp_size = tp_size if parallel_config.use_sequence_parallel_moe else 1
        moe_parallel_config = FusedMoEParallelConfig.make(
            tp_size_=tp_size,
            pcp_size_=get_pcp_group().world_size,
            dp_size_=dp_group.world_size,
            sp_size_=sp_size,
            vllm_parallel_config=parallel_config,
        )
        return replace(
            module.moe_config,
            num_experts=module.moe_config.num_local_experts * ep_group.world_size,
            moe_parallel_config=moe_parallel_config,
        )

    def stage_standby_moe_quant_methods(self, all2all_manager) -> None:
        standby_dp_group = get_standby_dp_group()
        standby_ep_group = get_standby_ep_group()
        model = self.worker.model_runner.get_model()
        moe_modules = [module for module in model.modules() if is_moe_layer(module)]
        self._staged_moe_quant_methods.clear()
        with set_current_vllm_config(self.worker.vllm_config):
            for module in moe_modules:
                staged_quant_method = make_eep_staged_quant_method(
                    module,
                    self._make_eep_moe_config(
                        module,
                        standby_dp_group,
                        standby_ep_group,
                    ),
                    all2all_manager=all2all_manager,
                )
                if staged_quant_method is not None:
                    self._staged_moe_quant_methods[module] = staged_quant_method

    def _commit_staged_moe_quant_methods(self) -> None:
        model = self.worker.model_runner.get_model()
        moe_modules = [module for module in model.modules() if is_moe_layer(module)]
        for module in moe_modules:
            staged_quant_method = self._staged_moe_quant_methods.pop(module, None)
            if staged_quant_method is None:
                continue
            assert staged_quant_method.moe_kernel is not None
            module._replace_quant_method(staged_quant_method)
        self._staged_moe_quant_methods.clear()

    def _release_cuda_graphs(self) -> None:
        manager = getattr(self.worker.model_runner, "cudagraph_manager", None)
        speculator = getattr(self.worker.model_runner, "speculator", None)
        if manager is not None:
            # MRV2 captures through CudaGraphManager instead of wrapping the
            # model, so neither wrapper branch below ever fires.
            manager.release_graphs()
            for attr in vars(speculator).values() if speculator else ():
                if isinstance(attr, CudaGraphManager):
                    attr.release_graphs()

        elif isinstance(self.worker.model_runner.model, CUDAGraphWrapper):
            wrapper = self.worker.model_runner.model
            wrapper.concrete_cudagraph_entries = {}

        elif isinstance(self.worker.model_runner.model, UBatchWrapper):
            self.worker.model_runner.model.clear_graphs()

        torch.compiler.reset()
        with set_current_vllm_config(self.worker.vllm_config):
            reset_compile_wrapper(self.worker.model_runner.get_model())
            if draft_model := getattr(speculator, "model", None):
                reset_compile_wrapper(draft_model)

        gc.collect()
        if os.getenv("VLLM_EEP_DEBUG_LIVE_GRAPHS") == "1":
            self._report_live_graphs()
        torch.accelerator.synchronize()
        torch.accelerator.empty_cache()

    def _report_live_graphs(self) -> None:
        """Count graphs that outlived the release and name what holds them.

        rocm-systems#13018 reports that RCCL teardown waits without a bound on
        references that kernel plans captured into a graph hold, and that they
        drop only when the graph itself is destroyed. One graph missed here is
        enough for ncclCommAbort never to return, which is what two ranks of
        the last run reported five seconds before being refused an IPC export.
        The branch above only reaches the containers it knows about and the
        breakable registry only when a runner exists, so the referrer types
        are the half of the answer that says where to look.
        """
        graphs = [o for o in gc.get_objects() if isinstance(o, torch.cuda.CUDAGraph)]
        if not graphs:
            logger.warning("[EEP-DEBUG] no captured graph outlived the release")
            return
        holders: dict[str, int] = {}
        for graph in graphs:
            for referrer in gc.get_referrers(graph):
                name = type(referrer).__name__
                holders[name] = holders.get(name, 0) + 1
        ranked = sorted(holders.items(), key=lambda item: -item[1])[:8]
        logger.warning(
            "[EEP-DEBUG] %d captured graph(s) outlived the release, held by %s",
            len(graphs),
            ", ".join(f"{name} x{count}" for name, count in ranked),
        )

    def switch_and_remove(self) -> None:
        # Removing ranks skipped preparation, so wait for prior cleanup here.
        self._wait_for_group_cleanup()
        # The EPLB worker is a daemon thread that runs collectives on the EPLB
        # group, and the group is destroyed a few lines below. Every removed
        # worker was observed with that thread still alive here, and one of
        # them segfaulted with no Python frames, only __clone, which is what a
        # daemon thread killed inside a native call looks like.
        eplb_state = self.worker.model_runner.eplb_state
        if eplb_state is not None:
            started = time.perf_counter()
            stopped = eplb_state.stop_async_worker()
            # Diagnostic only, to tell a pass that ran this from one that did
            # not. Drop together with the rest of the EEP-DEBUG lines.
            logger.warning(
                "[EEP-DEBUG] eplb async worker %s in %.3f s",
                "stopped" if stopped else "STILL RUNNING",
                time.perf_counter() - started,
            )
        self._release_cuda_graphs()
        retired_groups = _replace_active_groups(
            world=None, dp=None, ep=None, eplb=None, node_count=None
        )
        self._start_group_cleanup(self._retire_groups(retired_groups))
        # Finish collective cleanup before this worker is shut down.
        self._wait_for_group_cleanup()

    def switch_and_prepare(self) -> tuple[GroupCoordinator | None, ...]:
        if not self._can_reuse_fused_moe_kernel():
            self._release_cuda_graphs()
        retired_groups = _replace_active_groups(**pop_standby_groups())

        parallel_config = self.worker.vllm_config.parallel_config
        reconfig_request = self.reconfig_request
        assert reconfig_request is not None
        new_dp_size = reconfig_request.new_data_parallel_size
        new_ep_size = get_ep_group().world_size

        parallel_config.data_parallel_size = new_dp_size
        if (
            reconfig_request.new_data_parallel_rank
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        if (
            reconfig_request.new_data_parallel_rank_local
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank_local = (
                reconfig_request.new_data_parallel_rank_local
            )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )

        # Reconfigure MoE modules with new EP size
        moe_modules = [
            module
            for module in self.worker.model_runner.model.modules()
            if is_moe_layer(module)
        ]
        num_local_experts = moe_modules[0].moe_config.num_local_experts
        assert all(
            module.moe_config.num_local_experts == num_local_experts
            for module in moe_modules
        ), "All MoE modules must have the same number of experts"
        dp_group = get_dp_group()
        ep_group = get_ep_group()
        for module in moe_modules:
            new_moe_config = self._make_eep_moe_config(module, dp_group, ep_group)
            module._set_moe_config(new_moe_config)

        # Update EPLB state
        eplb_state = self.worker.model_runner.eplb_state
        assert eplb_state is not None
        model_config = self.worker.model_runner.model_config
        eplb_model_state = eplb_state.model_states[model_config.compute_hash()]

        num_physical_experts = num_local_experts * new_ep_size
        num_logical_experts = eplb_model_state.logical_replica_count.shape[1]
        parallel_config.eplb_config.num_redundant_experts = (
            num_physical_experts - num_logical_experts
        )
        eplb_state.reconfigure_physical_expert_slots(model_config, num_physical_experts)

        model = self.worker.model_runner.get_model()
        model.expert_weights = []
        with set_current_vllm_config(self.worker.vllm_config):
            model.set_eplb_state(
                eplb_model_state.expert_load_pass_buffer,
                eplb_model_state.logical_to_physical_map,
                eplb_model_state.logical_replica_count,
            )
            eplb_state._propagate_shared_tensors(
                model, eplb_model_state.num_unpadded_tokens_tensors
            )
            model.update_physical_experts_metadata(
                num_physical_experts=num_physical_experts,
                num_local_physical_experts=num_local_experts,
            )
            self._commit_staged_moe_quant_methods()
            get_ep_all2all_manager().commit_ep_size()
            # Legacy modular methods need to be recreated for the new EP size.
            for module in moe_modules:
                if getattr(module._quant_method, "wraps_legacy_quant_method", False):
                    module._replace_quant_method(module._quant_method.old_quant_method)

        assert self._prepared_eplb_communicator is not None
        eplb_state.update_communicator(model_config, self._prepared_eplb_communicator)
        self._prepared_eplb_communicator = None

        return retired_groups

    def _perform_eplb_reshuffle(
        self,
        rank_mapping: dict[int, int] | None = None,
        async_op: bool = False,
    ) -> None:
        if async_op and os.getenv("VLLM_EEP_ASYNC_RESHUFFLE") != "1":
            # A reconfiguration reshuffle cannot be scheduled. The copies land
            # on a stream the caller does not wait for, so the engine resumes
            # serving against expert weights that are still being moved, and
            # answers come back wrong for as long as the copies run. Opting
            # back in exists only to measure that difference.
            async_op = False
            logger.info("[Elastic EP] Expert resharding runs synchronously")

        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Starting expert resharding...")

        eplb_state = self.worker.model_runner.eplb_state
        assert eplb_state is not None

        is_async_enabled = eplb_state.is_async
        run_async = async_op and is_async_enabled
        eplb_state.is_async = run_async
        eplb_state.rearrange(rank_mapping=rank_mapping, use_last_expert_load=True)
        if not run_async:
            # Wait for non-blocking expert resharding copies before continuing
            # the Elastic EP reconfiguration.
            torch.accelerator.synchronize()
        # reset expert_rearrangement_step to ensure all ranks are synchronized
        eplb_state.expert_rearrangement_step = 0
        eplb_state.is_async = is_async_enabled
        # Start the async worker thread if it doesn't exist yet (idempotent).
        # This is needed for new workers after scale-up: they create EplbState
        # in setup_eplb_from_mapping() but don't start the thread there because
        # groups aren't ready yet.
        eplb_state.start_async_loop()
        if get_ep_group().rank == 0:
            logger.info(
                "[Elastic EP] Expert resharding %s",
                "scheduled" if run_async else "completed",
            )

    def commit_scale_up(self, is_existing_worker: bool) -> None:
        retired_groups: tuple[GroupCoordinator | None, ...] = ()
        with self._watchdog("commit_scale_up"):
            with self._timed("scale_up switch"):
                if is_existing_worker:
                    self.broadcast_expert_mapping()
                    retired_groups = self._retire_groups(self.switch_and_prepare())
                else:
                    mapping = self.receive_expert_mapping()
                    self.worker.model_runner.setup_eplb_from_mapping(mapping)
            if current_platform.is_rocm():
                with self._timed("scale_up empty_cache"):
                    groups = {"dp": get_dp_group(), "ep": get_ep_group()}
                    rocm_debug.dump("scale_up before empty_cache", groups)
                    torch.accelerator.empty_cache()
                    rocm_debug.dump("scale_up after empty_cache", groups)
                    rocm_debug.warm_each(groups, "scale_up")
            if not self._can_reuse_fused_moe_kernel():
                with self._timed("scale_up warm_and_capture"):
                    self.warm_and_capture()
            with self._timed("scale_up reshuffle"):
                self._perform_eplb_reshuffle(async_op=True)
            self._start_group_cleanup(retired_groups)

    def commit_scale_down(self, new_dp_size: int, removing: bool) -> None:
        retired_groups: tuple[GroupCoordinator | None, ...] = ()
        with self._watchdog("commit_scale_down"):
            with self._timed("scale_down reshuffle"):
                self.perform_scale_down_eplb_reshuffle(new_dp_size)
            if removing:
                with self._timed("scale_down switch_and_remove"):
                    self.switch_and_remove()
            else:
                with self._timed("scale_down switch"):
                    retired_groups = self._retire_groups(self.switch_and_prepare())
                if current_platform.is_rocm():
                    with self._timed("scale_down empty_cache"):
                        groups = {"dp": get_dp_group(), "ep": get_ep_group()}
                        rocm_debug.dump("scale_down before empty_cache", groups)
                        torch.accelerator.empty_cache()
                        rocm_debug.dump("scale_down after empty_cache", groups)
                        rocm_debug.warm_each(groups, "scale_down")
                if not self._can_reuse_fused_moe_kernel():
                    with self._timed("scale_down warm_and_capture"):
                        self.warm_and_capture()
                self._start_group_cleanup(retired_groups)

    def perform_scale_down_eplb_reshuffle(self, new_dp_size: int) -> None:
        eplb_state = self.worker.model_runner.eplb_state
        if eplb_state is not None:
            eplb_state.drain_async()
        parallel_config = self.worker.vllm_config.parallel_config
        tp_size = parallel_config.tensor_parallel_size
        old_ep_size = parallel_config.data_parallel_size * tp_size
        new_ep_size = new_dp_size * tp_size
        rank_mapping = {
            old_ep_rank: old_ep_rank if old_ep_rank < new_ep_size else -1
            for old_ep_rank in range(old_ep_size)
        }
        self._perform_eplb_reshuffle(rank_mapping=rank_mapping)

    def prepare_new_worker(self) -> None:
        dp_group = get_dp_group()
        assert isinstance(dp_group, StatelessGroupCoordinator)
        new_dp_size = dp_group.world_size
        dp_rank = self.worker.vllm_config.parallel_config.data_parallel_rank

        # Receive old_dp_size broadcasted during transfer_weights
        old_dp_size_tensor = torch.empty(1, dtype=torch.int64, device="cpu")
        old_dp_size_tensor = dp_group.tcp_store_group.broadcast(old_dp_size_tensor, 0)
        old_dp_size = int(old_dp_size_tensor[0].item())

        # Calculate which existing worker will send to this new worker
        num_new_workers = new_dp_size - old_dp_size
        new_worker_idx = dp_rank - old_dp_size
        num_dst_per_sender = num_new_workers // old_dp_size
        remainder = num_new_workers % old_dp_size

        if new_worker_idx < remainder * (num_dst_per_sender + 1):
            sender_rank = new_worker_idx // (num_dst_per_sender + 1)
        else:
            sender_rank = (
                remainder
                + (new_worker_idx - remainder * (num_dst_per_sender + 1))
                // num_dst_per_sender
            )

        model = self.worker.model_runner.get_model()
        expert_weights = [
            module.get_expert_weights()
            for module in model.modules()
            if is_moe_layer(module)
        ]
        batch_transfer_weights(
            model=model,
            is_sender=False,
            peer_rank=sender_rank,
            dp_group=dp_group,
            expert_weights=expert_weights,
        )
        torch.accelerator.synchronize()
        if not current_platform.is_rocm():
            self._warm_target_groups(get_dp_group(), get_ep_group())
        if self._can_reuse_fused_moe_kernel():
            sync_flashinfer_autotune_cache(self.worker.model_runner, get_world_group())

    def receive_expert_mapping(self) -> torch.Tensor:
        dp_group = get_dp_group()
        assert isinstance(dp_group, StatelessGroupCoordinator)
        return broadcast_expert_mapping(
            physical_to_logical=None,
            dp_group=dp_group,
            src_rank=0,
            device=self.worker.device,
        )

    def warmup_new_worker(self) -> None:
        with set_current_vllm_config(self.worker.vllm_config):
            kernel_warmup(self.worker, process_local_only=True)
        if self._can_reuse_fused_moe_kernel():
            self.warm_and_capture()

    def warm_and_capture(self) -> None:
        # _ensure_workspace_size allocates a fresh tensor on grow, leaving
        # any captured CUDA graph with a stale data pointer; drop graphs
        # before re-warm so captures realign with the resized buffer.
        self._release_cuda_graphs()
        unlock_workspace()

        runner = self.worker.model_runner
        all2all_manager = get_ep_all2all_manager()
        reuse_kernel = self._can_reuse_fused_moe_kernel()
        serving_state = (
            preserve_serving_state(runner)
            if self.worker.use_v2_model_runner
            else runner.preserve_serving_state()
        )
        with (
            skip_dp_coordination() if reuse_kernel else nullcontext(),
            all2all_manager.mask_remote_ranks() if reuse_kernel else nullcontext(),
            self._disable_flashinfer_autotune() if reuse_kernel else nullcontext(),
            self._suppress_eplb(),
            serving_state,
        ):
            runner._dummy_run(
                runner.max_num_tokens,
                is_profile=True,
                skip_eplb=True,
                randomize_inputs=self.worker.randomize_dummy_inputs,
            )
            self.worker.compile_or_warm_up_model()

        lock_workspace()
