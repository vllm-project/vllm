# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The async worker that transfers experts in the background."""

import threading
import time
from typing import TYPE_CHECKING

import torch

from vllm.distributed.parallel_state import get_eplb_group
from vllm.logger import init_logger

from .eplb_utils import CpuGpuEvent, device_stream
from .migration_scheduler import schedule_migration_batches_for_layers
from .policy_process import EplbPolicyProcess, EplbPolicyProcessError
from .rebalance_execute import AsyncEplbLayerResult, transfer_layer

if TYPE_CHECKING:
    from .eplb_state import EplbModelState, EplbState

logger = init_logger(__name__)


def start_async_worker(
    state: "EplbState",
    is_profile: bool = False,
) -> threading.Thread:
    rank = get_eplb_group().device_group.rank()
    device_index = state.device_index
    assert state.is_async

    policy_process = EplbPolicyProcess(state.policy) if not is_profile else None
    state.policy_process = policy_process

    def thread_target() -> None:
        try:
            assert device_index is not None
            torch.accelerator.set_device_index(device_index)
            stream = torch.Stream(device=torch.accelerator.current_accelerator())
            transfer_run_periodically(
                state=state,
                stream=stream,
                is_profile=is_profile,
            )
        except Exception as exc:
            state.async_worker_error = exc
            logger.exception("async loop error (Rank %d): %s", rank, str(exc))
        finally:
            if policy_process is not None:
                policy_process.close()

    thread = threading.Thread(target=thread_target, daemon=True)
    try:
        thread.start()
    except BaseException:
        if policy_process is not None:
            policy_process.close()
        state.policy_process = None
        raise
    return thread


def run_rebalance_experts(
    model_state: "EplbModelState",
    eplb_state: "EplbState",
    physical_to_logical_map_cpu: torch.Tensor,
    stream: torch.Stream,
) -> torch.Tensor:
    assert model_state.eplb_stats is not None
    eplb_stats = model_state.eplb_stats

    # Move the global expert load window to CPU for computation.
    with device_stream(stream):
        eplb_stats.global_expert_load_window = (
            eplb_stats.global_expert_load_window.cpu()
        )
    # Compute new expert mappings for the model
    policy = eplb_state.policy_process or eplb_state.policy
    new_physical_to_logical_map = policy.rebalance_experts(
        eplb_stats,
        physical_to_logical_map_cpu,
    )
    assert new_physical_to_logical_map.device == torch.device("cpu")

    return new_physical_to_logical_map


def transfer_run_periodically(
    state: "EplbState",
    stream: torch.Stream,
    is_profile: bool = False,
) -> None:
    rank = get_eplb_group().device_group.rank()
    while not state.async_worker_stop.is_set():
        signalled = True
        try:
            signalled = state.rearrange_event.wait(stream=stream, timeout=0.1)
            if not signalled:
                continue
            if state.async_worker_stop.is_set():
                return
            _transfer_cycle(state, stream, is_profile)
        except Exception as exc:
            if state.async_worker_stop.is_set():
                return
            if isinstance(exc, EplbPolicyProcessError) or not (
                state.parallel_config.enable_fault_tolerance
            ):
                raise
            # step() raises this on every rank; the thread stays up for retry.
            state.async_worker_error = exc
            logger.exception(
                "async loop error (Rank %d); continuing under fault tolerance",
                rank,
            )
        finally:
            if signalled:
                with state.async_cycle_done:
                    state.async_cycles_pending -= 1
                    state.async_cycle_done.notify_all()


def _transfer_cycle(
    state: "EplbState",
    stream: torch.Stream,
    is_profile: bool,
) -> None:
    """Run one rebalance pass over all model states and layers."""
    # Snapshot the FT reset epoch. A change in this value means a fault
    # recovery happened, and the in-flight rebalance must be abandoned.
    reset_epoch = state.ft_reset_epoch

    eplb_group = get_eplb_group().device_group
    eplb_cpu_group = get_eplb_group().cpu_group
    ep_rank = eplb_group.rank()

    assert state.is_async
    for model_state in state.model_states.values():
        layer_idx = 0
        # Set the async worker's CUDA stream on the communicator
        model_state.communicator.set_stream(stream)
        num_layers = model_state.model.num_moe_layers

        t_start = time.perf_counter()
        physical_to_logical_map_cpu = None
        policy_error = None
        new_physical_to_logical_map = None
        if state.ft_reset_epoch == reset_epoch and not state.async_worker_stop.is_set():
            # Snapshot the physical_to_logical_map (synchronized with
            # rearrange_event) and copy it to CPU
            with device_stream(stream):
                physical_to_logical_map_cpu = model_state.physical_to_logical_map.cpu()
            try:
                new_physical_to_logical_map = run_rebalance_experts(
                    model_state, state, physical_to_logical_map_cpu, stream
                )
            except EplbPolicyProcessError as exc:
                policy_error = exc

        migration_batches_by_layer = None
        if (
            policy_error is None
            and new_physical_to_logical_map is not None
            and state.ft_reset_epoch == reset_epoch
            and not state.async_worker_stop.is_set()
            and state.parallel_config.eplb_config.migration_batching_enabled
        ):
            assert physical_to_logical_map_cpu is not None
            profile_name = "eplb: schedule migration batches"
            num_local_experts = model_state.model.expert_weights[0][0].shape[0]
            # The placement snapshots stay fixed for this rebalance cycle.
            # Precompute all layer schedules once, but execute transfers one
            # layer at a time below to preserve the migration lifecycle. This
            # profiling range therefore covers scheduling for every MoE layer.
            with torch.profiler.record_function(profile_name):
                migration_batches_by_layer = schedule_migration_batches_for_layers(
                    num_local_experts,
                    physical_to_logical_map_cpu.numpy(),
                    new_physical_to_logical_map.numpy(),
                )

        # Execute one EPLB layer transfer per model forward pass. Each iteration
        # of this loop will copy the new set of expert weights into
        # model_state.expert_buffer, which will be consumed by the main thread in
        # move_to_workspace.
        # We sync the rebalanced flag across ranks before each iteration so
        # all ranks make a coordinated decision to continue or stop.
        while layer_idx < num_layers:
            if (
                state.ft_reset_epoch != reset_epoch
                and eplb_cpu_group is not get_eplb_group().cpu_group
            ):
                return
            if state.ft_reset_epoch == reset_epoch:
                model_state.worker_progress = 2 * layer_idx
            stopping = state.async_worker_stop.is_set()
            flag = torch.tensor(
                [
                    int(
                        model_state.rebalanced
                        and state.ft_reset_epoch == reset_epoch
                        and not stopping
                        and policy_error is None
                    ),
                    int(policy_error is not None and not stopping),
                ],
                dtype=torch.int32,
                device="cpu",
            )
            torch.distributed.all_reduce(flag, group=eplb_cpu_group)
            if int(flag[1].item()):
                raise EplbPolicyProcessError(
                    "EPLB policy process failed on at least one rank"
                ) from policy_error
            if int(flag[0].item()) != eplb_cpu_group.size():
                log = (
                    logger.debug if state.async_worker_stop.is_set() else logger.warning
                )
                log(
                    "async worker (rank=%d): layer %d coordinated stop "
                    "(flag_sum=%d, group_size=%d)",
                    ep_rank,
                    layer_idx,
                    int(flag[0].item()),
                    eplb_cpu_group.size(),
                )
                if state.ft_reset_epoch == reset_epoch:
                    for cancelled_model in state.model_states.values():
                        cancelled_model.rebalanced = False
                return

            assert physical_to_logical_map_cpu is not None
            assert new_physical_to_logical_map is not None
            transfer_metadata = transfer_layer(
                old_layer_indices=physical_to_logical_map_cpu[layer_idx],
                new_layer_indices=new_physical_to_logical_map[layer_idx],
                expert_weights=model_state.model.expert_weights[layer_idx],
                expert_weights_buffer=model_state.expert_buffer,
                communicator=model_state.communicator,
                ep_group=eplb_group,
                is_profile=is_profile,
                stream=stream,
                layer_idx=layer_idx,
                enable_migration_batching=(
                    state.parallel_config.eplb_config.migration_batching_enabled
                ),
                migration_batches=(
                    # This is only the current layer's schedule; it does not
                    # combine transfers from different layers.
                    migration_batches_by_layer[layer_idx]
                    if migration_batches_by_layer is not None
                    else None
                ),
            )

            # Wait until all writes to expert_buffer have finished before making the
            # AsyncEplbLayerResult visible to the main thread.
            if stream is not None:
                stream.synchronize()

            # Do not publish a stale transfer. The next layer's collective
            # coordinates cancellation with peers still finishing this layer.
            if state.async_worker_stop.is_set():
                return
            if state.ft_reset_epoch == reset_epoch:
                # This event guarantees that expert_buffer will not be overwritten by
                # subsequent iterations of this loop until the main thread has consumed
                # it. Record is called by the main thread after move_from_buffer().
                consumed_event = CpuGpuEvent()
                model_state.pending_result = AsyncEplbLayerResult(
                    layer_idx=layer_idx,
                    new_physical_to_logical_map=new_physical_to_logical_map[layer_idx],
                    transfer_metadata=transfer_metadata,
                    consumed_event=consumed_event,
                )
                model_state.worker_progress = 2 * layer_idx + 1
                if state.ft_reset_epoch != reset_epoch:
                    model_state.pending_result = None
                else:
                    # Wait for the main thread to consume or discard the buffer.
                    while not consumed_event.wait(stream=stream, timeout=0.1):
                        if state.async_worker_stop.is_set():
                            return
                    # Host-issued writes (e.g. RDMA reads) are not stream-ordered.
                    if stream is not None:
                        stream.synchronize()
                    assert model_state.pending_result is None
            layer_idx += 1
        if (
            ep_rank == 0
            and state.ft_reset_epoch == reset_epoch
            and not state.async_worker_stop.is_set()
        ):
            logger.info(
                "Async EPLB rearranged %d/%d layers of %s in %.2f s.",
                layer_idx,
                num_layers,
                model_state.model_name,
                time.perf_counter() - t_start,
            )
