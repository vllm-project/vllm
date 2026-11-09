# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async EPLB: transfer experts in the background, one layer per step.

`AsyncEplbRunner` lives on the main thread. It submits one job at a time to a
daemon worker thread and acts on the result only after every EP rank finished
the same job. A job gets everything it touches as arguments and returns a
value or raises, so the two threads share no mutable state.
"""

import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed import ProcessGroup

from vllm.distributed.parallel_state import get_ep_group, get_eplb_group
from vllm.logger import init_logger

from .eplb_communicator import EplbCommunicator
from .eplb_utils import device_stream
from .migration_scheduler import MigrationFlow, schedule_migration_batches_for_layers
from .policy import AbstractEplbPolicy
from .rebalance_execute import (
    AsyncEplbLayerResult,
    TransferMetadata,
    transfer_layer,
)

if TYPE_CHECKING:
    from .eplb_state import EplbModelState, EplbStats

logger = init_logger(__name__)

AsyncEplbJob = Callable[[torch.Stream], Any]
ApplyLayerFn = Callable[["EplbModelState", int, torch.Tensor, TransferMetadata], None]


class AsyncEplbExecutor:
    """Runs async EPLB jobs one at a time on a daemon thread.

    The thread is bound to one device and owns one side stream, which every
    job receives as its first argument. It is a daemon so that a job stuck in
    a transfer with a dead peer cannot block interpreter exit;
    `ThreadPoolExecutor` threads are joined at exit.
    """

    def __init__(self, device_index: int, rank: int) -> None:
        self._rank = rank
        self._jobs: queue.SimpleQueue[
            tuple[Future, AsyncEplbJob, torch.Event | None]
        ] = queue.SimpleQueue()
        self.thread = threading.Thread(
            target=self._run, args=(device_index,), daemon=True
        )
        self.thread.start()

    def submit(self, job: AsyncEplbJob, after: torch.Event | None = None) -> Future:
        """Queue `job`; the worker waits for `after` before running it."""
        future: Future = Future()
        self._jobs.put((future, job, after))
        return future

    def _run(self, device_index: int) -> None:
        stream: torch.Stream | None = None
        while True:
            future, job, after = self._jobs.get()
            try:
                if stream is None:
                    torch.accelerator.set_device_index(device_index)
                    device = torch.accelerator.current_accelerator()
                    stream = torch.Stream(device=device)
                if after is not None:
                    after.synchronize()
                future.set_result(job(stream))
            except BaseException as exc:
                logger.exception("Async EPLB job failed (Rank %d)", self._rank)
                future.set_exception(exc)


@dataclass
class AsyncEplbPlan:
    """Old and new placement of one model, on CPU."""

    old_physical_to_logical_map: torch.Tensor
    new_physical_to_logical_map: torch.Tensor
    migration_batches_by_layer: list[list[list[MigrationFlow]]] | None


class AsyncEplbRunner:
    """Main-thread driver of async EPLB cycles.

    A cycle is a list of layers to rearrange. `poll()` applies a layer and
    submits the next one once every EP rank finished the current job, so all
    ranks apply the same layer in the same step.
    """

    def __init__(
        self,
        device_index: int,
        apply_layer: ApplyLayerFn,
        policy: type[AbstractEplbPolicy],
        migration_batching: bool,
    ) -> None:
        self._executor = AsyncEplbExecutor(device_index, get_ep_group().rank_in_group)
        self._apply_layer = apply_layer
        self._policy = policy
        self._migration_batching = migration_batching
        # Layers of the running cycle that are not applied yet. The job of
        # the first one is running or finished.
        self._layers_left: deque[tuple[EplbModelState, int]] = deque()
        # The last submitted job, kept until its cycle ends. After a reset or
        # a failure its result is never applied.
        self._job: Future | None = None
        self._error: BaseException | None = None
        self._start_time = 0.0

    @property
    def cycle_running(self) -> bool:
        return bool(self._layers_left)

    def start(self, model_states: list["EplbModelState"]) -> None:
        """Start a cycle. Every EP rank must call this in the same step."""
        if self._error is not None:
            raise self._error
        assert not self.cycle_running
        self._layers_left.extend(
            (model_state, layer_idx)
            for model_state in model_states
            for layer_idx in range(model_state.model.num_moe_layers)
        )
        self._submit_next(plan=None)

    def poll(self) -> None:
        """Advance the cycle if every rank finished its job. Called by every EP
        rank once per step; raises on every rank if a job failed on any rank.
        """
        if self._error is not None:
            raise self._error
        if not self._layers_left:
            return
        num_waiting_models = len({id(m) for m, _ in self._layers_left}) - 1
        if self._all_ranks_done():
            self._finish_layer(apply=True)
        # step() has always done one all-reduce per model that is still being
        # rebalanced. Keep that count, although a model whose first job is not
        # submitted yet is busy on every rank.
        for _ in range(num_waiting_models):
            self._count_failed_and_busy(failed=False, busy=True)

    def reset(self, timeout: float | None) -> None:
        """Clear the error and drop the cycle, for fault recovery.

        Not collective, so it is safe while the EP groups are broken. Every
        rank must call it between the same two steps. Waits for a running job,
        because the caller must not rebuild the groups under it; a job blocked
        on a peer ends within the cpu group timeout.
        """
        self._layers_left.clear()
        if self._job is not None:
            try:
                self._job.exception(timeout)
            except FutureTimeoutError:
                raise RuntimeError(
                    f"Async EPLB worker still in a transfer job after {timeout}s."
                ) from None
        self._job = None
        self._error = None

    def drain(self) -> None:
        """Run the cycle's remaining transfers in lockstep and apply none of
        them; a full rearrange is expected to follow.
        """
        if self._error is not None:
            raise self._error
        while self._layers_left:
            model_state = self._layers_left[0][0]
            logger.info(
                "Draining async EPLB worker for model %s", model_state.model_name
            )
            while self._layers_left and self._layers_left[0][0] is model_state:
                if self._all_ranks_done():
                    self._finish_layer(apply=False)
                else:
                    time.sleep(0.001)
            logger.info(
                "Async EPLB worker drained for model %s", model_state.model_name
            )

    def _finish_layer(self, apply: bool) -> None:
        assert self._job is not None
        result: AsyncEplbLayerResult = self._job.result()
        model_state, layer_idx = self._layers_left.popleft()
        if apply:
            self._apply_layer(
                model_state,
                layer_idx,
                result.new_physical_to_logical_map,
                result.transfer_metadata,
            )
        num_layers = model_state.model.num_moe_layers
        if layer_idx == num_layers - 1 and get_ep_group().rank_in_group == 0:
            logger.info(
                "Async EPLB rearranged %d/%d layers of %s in %.2f s.",
                num_layers,
                num_layers,
                model_state.model_name,
                time.perf_counter() - self._start_time,
            )
        if self._layers_left:
            self._submit_next(result.plan)
        else:
            # Do not keep the plan alive between cycles.
            self._job = None

    def _submit_next(self, plan: AsyncEplbPlan | None) -> None:
        model_state, layer_idx = self._layers_left[0]
        if layer_idx == 0:
            # The job of a model's first layer computes the plan.
            plan = None
            self._start_time = time.perf_counter()
        # expert_buffer is on the device and the previous layer's copy out of
        # it is only queued on the main stream. The job's writes are not
        # ordered against that stream (e.g. NIXL RDMA reads are host-issued),
        # so the worker waits for this event before it runs the job.
        main_stream_done = torch.Event()
        main_stream_done.record()
        self._job = self._executor.submit(
            partial(
                transfer_layer_to_buffer,
                plan=plan,
                policy=self._policy,
                eplb_stats=model_state.eplb_stats,
                physical_to_logical_map=model_state.physical_to_logical_map,
                migration_batching=self._migration_batching,
                layer_idx=layer_idx,
                expert_weights=model_state.model.expert_weights[layer_idx],
                expert_buffer=model_state.expert_buffer,
                communicator=model_state.communicator,
                ep_group=get_eplb_group().device_group,
                ep_cpu_group=get_eplb_group().cpu_group,
            ),
            after=main_stream_done,
        )

    def _all_ranks_done(self) -> bool:
        assert self._job is not None
        done = self._job.done()
        error = self._job.exception() if done else None
        num_failed, num_busy = self._count_failed_and_busy(
            failed=error is not None, busy=not done
        )
        if num_failed:
            self._error = RuntimeError(
                "Async EPLB worker failed on at least one rank; see the worker logs."
            )
            self._error.__cause__ = error
            self._layers_left.clear()
            raise self._error
        return num_busy == 0

    @staticmethod
    def _count_failed_and_busy(failed: bool, busy: bool) -> tuple[int, int]:
        flags = torch.tensor((int(failed), int(busy)), dtype=torch.int32, device="cpu")
        # cpu_all_reduce_ only sums, so count failed and busy ranks.
        get_ep_group().cpu_all_reduce_(flags)
        num_failed, num_busy = flags.tolist()
        return num_failed, num_busy


def run_rebalance_experts(
    policy: type[AbstractEplbPolicy],
    eplb_stats: "EplbStats",
    physical_to_logical_map_cpu: torch.Tensor,
    stream: torch.Stream,
) -> torch.Tensor:
    # Move the global expert load window to CPU for computation.
    with device_stream(stream):
        global_expert_load_window = eplb_stats.global_expert_load_window.cpu()
    # Compute new expert mappings for the model
    new_physical_to_logical_map = policy.rebalance_experts(
        global_expert_load_window,
        eplb_stats.num_replicas,
        eplb_stats.num_groups,
        eplb_stats.num_nodes,
        eplb_stats.num_gpus,
        physical_to_logical_map_cpu,
    )
    assert new_physical_to_logical_map.device == torch.device("cpu")

    return new_physical_to_logical_map


def transfer_layer_to_buffer(
    stream: torch.Stream,
    *,
    plan: AsyncEplbPlan | None,
    policy: type[AbstractEplbPolicy],
    eplb_stats: "EplbStats",
    physical_to_logical_map: torch.Tensor,
    migration_batching: bool,
    layer_idx: int,
    expert_weights: Sequence[torch.Tensor],
    expert_buffer: Sequence[torch.Tensor],
    communicator: EplbCommunicator,
    ep_group: ProcessGroup,
    ep_cpu_group: ProcessGroup,
) -> AsyncEplbLayerResult:
    """Fetch the new experts of one layer into `expert_buffer`.

    The job of a model's first layer gets no `plan` and computes it first.
    """
    # Set the async worker's CUDA stream on the communicator
    communicator.set_stream(stream)

    if plan is None:
        # Snapshot the physical_to_logical_map and copy it to CPU
        with device_stream(stream):
            physical_to_logical_map_cpu = physical_to_logical_map.cpu()

        new_physical_to_logical_map = run_rebalance_experts(
            policy, eplb_stats, physical_to_logical_map_cpu, stream
        )

        migration_batches_by_layer = None
        if migration_batching:
            profile_name = "eplb: schedule migration batches"
            num_local_experts = expert_weights[0].shape[0]
            # The placement snapshots stay fixed for this rebalance cycle.
            # Precompute all layer schedules once, but execute transfers one
            # layer at a time to preserve the migration lifecycle. This
            # profiling range therefore covers scheduling for every MoE layer.
            with torch.profiler.record_function(profile_name):
                migration_batches_by_layer = schedule_migration_batches_for_layers(
                    num_local_experts,
                    physical_to_logical_map_cpu.numpy(),
                    new_physical_to_logical_map.numpy(),
                )
        plan = AsyncEplbPlan(
            physical_to_logical_map_cpu,
            new_physical_to_logical_map,
            migration_batches_by_layer,
        )

    # No rank starts a transfer before every rank's worker is ready for it.
    torch.distributed.barrier(group=ep_cpu_group)

    migration_batches_by_layer = plan.migration_batches_by_layer
    transfer_metadata = transfer_layer(
        old_layer_indices=plan.old_physical_to_logical_map[layer_idx],
        new_layer_indices=plan.new_physical_to_logical_map[layer_idx],
        expert_weights=expert_weights,
        expert_weights_buffer=expert_buffer,
        communicator=communicator,
        ep_group=ep_group,
        stream=stream,
        layer_idx=layer_idx,
        enable_migration_batching=migration_batching,
        migration_batches=(
            # This is only the current layer's schedule; it does not
            # combine transfers from different layers.
            migration_batches_by_layer[layer_idx]
            if migration_batches_by_layer is not None
            else None
        ),
    )

    # Wait until all writes to expert_buffer have finished before making the
    # result visible to the main thread.
    stream.synchronize()
    return AsyncEplbLayerResult(
        layer_idx=layer_idx,
        new_physical_to_logical_map=plan.new_physical_to_logical_map[layer_idx],
        transfer_metadata=transfer_metadata,
        plan=plan,
    )
