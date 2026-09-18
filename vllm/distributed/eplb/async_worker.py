# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The async worker that transfers experts in the background."""

import threading
from typing import TYPE_CHECKING

import torch

from vllm.distributed.parallel_state import get_eplb_group
from vllm.logger import init_logger

from .eplb_utils import CpuGpuEvent, device_stream
from .migration_scheduler import schedule_migration_batches_for_layers
from .policy import EplbPlan, EplbRebalanceContext, EplbTopology
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

    def thread_target() -> None:
        assert device_index is not None
        torch.accelerator.set_device_index(device_index)
        stream = torch.Stream(device=torch.accelerator.current_accelerator())
        try:
            transfer_run_periodically(
                state=state,
                stream=stream,
                is_profile=is_profile,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            logger.exception("async loop error (Rank %d): %s", rank, str(exc))

    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()
    return thread


def run_rebalance_experts(
    model_state: "EplbModelState",
    eplb_state: "EplbState",
    physical_to_logical_map_cpu: torch.Tensor,
    stream: torch.Stream,
) -> EplbPlan:
    assert model_state.eplb_stats is not None
    stats = model_state.eplb_stats
    with device_stream(stream):
        load_window_cpu = stats.global_expert_load_window.cpu()
    context = EplbRebalanceContext(
        load_window_cpu=load_window_cpu,
        physical_to_logical_map_cpu=physical_to_logical_map_cpu,
        topology=EplbTopology(
            num_groups=stats.num_groups,
            num_nodes=stats.num_nodes,
            num_ranks=stats.num_gpus,
        ),
        num_replicas=stats.num_replicas,
        cpu_group=get_eplb_group().cpu_group,
    )
    plan = eplb_state.plan_rebalance(model_state, context)

    return plan


def transfer_run_periodically(
    state: "EplbState",
    stream: torch.Stream,
    is_profile: bool = False,
) -> None:
    while True:
        state.rearrange_event.wait(stream=stream)

        eplb_group = get_eplb_group().device_group
        eplb_cpu_group = get_eplb_group().cpu_group
        ep_rank = eplb_group.rank()

        assert state.is_async
        for model_state in state.model_states.values():
            layer_idx = 0
            # Set the async worker's CUDA stream on the communicator
            model_state.communicator.set_stream(stream)
            num_layers = model_state.model.num_moe_layers

            # Snapshot the physical_to_logical_map (synchronized with
            # rearrange_event) and copy it to CPU
            with device_stream(stream):
                physical_to_logical_map_cpu = model_state.physical_to_logical_map.cpu()

            plan = run_rebalance_experts(
                model_state, state, physical_to_logical_map_cpu, stream
            )
            new_physical_to_logical_map = plan.physical_to_logical_map

            migration_batches_by_layer = None
            if state.parallel_config.eplb_config.migration_batching_enabled:
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
                flag = torch.tensor(
                    [int(model_state.rebalanced)],
                    dtype=torch.int32,
                    device="cpu",
                )
                torch.distributed.all_reduce(flag, group=eplb_cpu_group)
                if int(flag.item()) != eplb_cpu_group.size():
                    logger.warning(
                        "async worker (rank=%d): layer %d coordinated stop "
                        "(flag_sum=%d, group_size=%d)",
                        ep_rank,
                        layer_idx,
                        int(flag.item()),
                        eplb_cpu_group.size(),
                    )
                    model_state.rebalanced = False
                    break

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
                stream.synchronize()

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

                # Block this thread until the main thread and main stream
                # finish copying model_state.expert_buffer into
                # model_state.model.expert_weights[layer_idx]
                consumed_event.wait(stream=stream)
                assert model_state.pending_result is None
                layer_idx += 1
