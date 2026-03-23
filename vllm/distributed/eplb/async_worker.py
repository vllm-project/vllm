# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
The async worker that transfers experts in the background.
"""

import asyncio
import threading
from typing import TYPE_CHECKING

import torch
from torch.distributed import ProcessGroup

from vllm.distributed.parallel_state import get_eplb_group
from vllm.logger import init_logger

from .eplb_utils import EPLBEvent
from .rebalance_execute import AsyncEPLBLayerResult, transfer_layer

if TYPE_CHECKING:
    from .eplb_state import EplbModelState, EplbState

logger = init_logger(__name__)


def start_async_worker(
    state: "EplbState",
    is_profile: bool = False,
) -> threading.Thread:
    eplb_group = get_eplb_group().device_group
    rank = eplb_group.rank()
    device_index = state.cuda_device_index
    assert state.is_async

    def thread_target() -> None:
        assert device_index is not None
        torch.accelerator.set_device_index(device_index)
        cuda_stream = torch.cuda.Stream(device=device_index)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                transfer_run_periodically(
                    state=state,
                    eplb_group=eplb_group,
                    cuda_stream=cuda_stream,
                    is_profile=is_profile,
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            logger.exception("async loop error (Rank %d): %s", rank, str(exc))
        finally:
            loop.close()

    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()
    return thread


def run_rebalance_experts(
    model_state: "EplbModelState",
    eplb_state: "EplbState",
    physical_to_logical_map_cpu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute new expert mappings for one rebalancing cycle.

    Returns: new_physical_to_logical_map
    """
    assert model_state.eplb_stats is not None
    eplb_stats = model_state.eplb_stats

    # Wait for global_expert_load_window tensor to be ready.
    assert model_state.window_ready_event is not None
    model_state.window_ready_event.wait()
    model_state.window_ready_event = None

    # Move the global expert load window to CPU for computation.
    global_expert_load_window = eplb_stats.global_expert_load_window.cpu()
    # Compute new expert mappings for the model
    (
        new_physical_to_logical_map,
        new_logical_to_physical_map,
        new_logical_replica_count,
    ) = eplb_state.policy.rebalance_experts(
        global_expert_load_window,
        eplb_stats.num_replicas,
        eplb_stats.num_groups,
        eplb_stats.num_nodes,
        eplb_stats.num_gpus,
        physical_to_logical_map_cpu,
    )
    assert new_physical_to_logical_map.device == torch.device("cpu")

    max_slots = model_state.logical_to_physical_map.shape[-1]
    new_logical_to_physical_map = torch.nn.functional.pad(
        new_logical_to_physical_map,
        (0, max(0, max_slots - new_logical_to_physical_map.shape[-1])),
        value=-1,
    ).to(model_state.logical_to_physical_map.device)
    new_logical_replica_count = new_logical_replica_count.to(
        model_state.logical_replica_count.device
    )
    return (
        new_physical_to_logical_map,
        new_logical_to_physical_map,
        new_logical_replica_count,
    )


async def transfer_run_periodically(
    state: "EplbState",
    eplb_group: ProcessGroup,
    cuda_stream: torch.cuda.Stream,
    is_profile: bool = False,
) -> None:
    while True:
        await asyncio.to_thread(state.rearrange_event.wait)
        logger.info("async worker woke up for EPLB transfer")

        assert state.is_async
        for model_state in state.model_states.values():
            layer_idx = 0
            current_num_layers = model_state.model.num_moe_layers

            # Snapshot the physical_to_logical_map and copy it to CPU
            with model_state.map_lock:
                physical_to_logical_map_cpu = model_state.physical_to_logical_map.cpu()

            (
                new_physical_to_logical_map,
                new_logical_to_physical_map,
                new_logical_replica_count,
            ) = run_rebalance_experts(model_state, state, physical_to_logical_map_cpu)
            logger.info(
                "Async worker computed new indices for model %s",
                model_state.model_name,
            )

            while model_state.rebalanced and layer_idx < current_num_layers:
                (
                    is_unchanged,
                    is_received_locally,
                    recv_metadata,
                ) = await transfer_layer(
                    old_layer_indices=physical_to_logical_map_cpu[layer_idx],
                    new_layer_indices=new_physical_to_logical_map[layer_idx],
                    expert_weights=model_state.model.expert_weights[layer_idx],
                    expert_weights_buffer=model_state.expert_buffer,
                    ep_group=eplb_group,
                    is_profile=is_profile,
                    cuda_stream=cuda_stream,
                )

                # Wait until all writes to expert_buffer have finished before making the
                # AsynEPLBLayerResult visible to the main thread.
                cuda_stream.synchronize()

                # This event guarantees that expert_buffer will not be overwritten by
                # subsequent iterations of this loop until the main thread has consumed
                # it. Record is called by the main thread after move_from_buffer().
                consumed_event = EPLBEvent()

                model_state.pending_result = AsyncEPLBLayerResult(
                    layer_idx=layer_idx,
                    new_physical_to_logical_map=new_physical_to_logical_map,
                    new_logical_to_physical_map=new_logical_to_physical_map,
                    new_logical_replica_count=new_logical_replica_count,
                    is_unchanged=is_unchanged,
                    is_received_locally=is_received_locally,
                    recv_metadata=recv_metadata,
                    consumed_event=consumed_event,
                )

                consumed_event.wait(stream=cuda_stream)
                layer_idx += 1

        state.rearrange_event.clear()
