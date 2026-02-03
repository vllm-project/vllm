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

from .rebalance_execute import transfer_layer

if TYPE_CHECKING:
    from .eplb_state import EplbModelState, EplbState

logger = init_logger(__name__)


def start_async_worker(
    state: "EplbState",
    rank_mapping: dict[int, int] | None = None,
    is_profile: bool = False,
) -> threading.Thread:
    eplb_group = get_eplb_group().device_group
    rank = eplb_group.rank()
    device_index = state.cuda_device_index
    assert state.is_async

    def thread_target() -> None:
        assert device_index is not None
        torch.cuda.set_device(device_index)
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
                    rank_mapping=rank_mapping,
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
) -> None:
    assert model_state.eplb_stats is not None
    eplb_stats = model_state.eplb_stats

    # Wait for the main thread's all-reduce and clone to complete before
    # accessing the global_expert_load_window tensor.
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

    model_state.new_physical_to_logical_map = new_physical_to_logical_map

    max_slots = model_state.logical_to_physical_map.shape[-1]
    padded_logical = torch.nn.functional.pad(
        new_logical_to_physical_map,
        (0, max(0, max_slots - new_logical_to_physical_map.shape[-1])),
        value=-1,
    ).to(model_state.logical_to_physical_map.device)
    new_replica = new_logical_replica_count.to(model_state.logical_replica_count.device)
    model_state.new_logical_to_physical_map = padded_logical
    model_state.new_logical_replica_count = new_replica


async def transfer_run_periodically(
    state: "EplbState",
    eplb_group: ProcessGroup,
    cuda_stream: torch.cuda.Stream,
    is_profile: bool = False,
    rank_mapping: dict[int, int] | None = None,
) -> None:
    while True:
        await asyncio.to_thread(state.rearrange_event.wait)
        logger.info("async worker woke up for EPLB transfer")

        assert state.is_async
        for model_state in state.model_states.values():
            rebalancing_algorithm_executed = False

            current_num_layers = model_state.model.num_moe_layers
            while (
                model_state.rebalanced
                and model_state.layer_to_transfer < current_num_layers
            ):
                if not model_state.ep_buffer_ready and model_state.rebalanced:
                    # Polling the lock directly in the async thread avoids
                    # the thread switch overhead of asyncio.to_thread.
                    # This is typically faster than offloading to a worker thread.
                    while not model_state.buffer_lock.acquire(blocking=False):
                        await asyncio.sleep(0)
                    try:
                        if model_state.layer_to_transfer >= current_num_layers:
                            break
                        if (
                            not rebalancing_algorithm_executed
                            or model_state.new_physical_to_logical_map is None
                        ):
                            # Move the physical_to_logical_map to CPU
                            # for rebalancing and transfer_layer.
                            physical_to_logical_map_cpu = (
                                model_state.physical_to_logical_map.cpu()
                            )
                            run_rebalance_experts(
                                model_state, state, physical_to_logical_map_cpu
                            )
                            rebalancing_algorithm_executed = True
                            logger.info(
                                "Async worker computed new indices for model %s",
                                model_state.model_name,
                            )

                        assert model_state.new_physical_to_logical_map is not None

                        # Wait for the main thread to finish consuming the buffer
                        # before initiating an EPLB transfer on another layer.
                        if model_state.buffer_consumed_event is not None:
                            cuda_stream.wait_event(model_state.buffer_consumed_event)
                            model_state.buffer_consumed_event = None

                        layer_idx = model_state.layer_to_transfer
                        old_layer_indices = physical_to_logical_map_cpu[layer_idx]
                        new_layer_indices = model_state.new_physical_to_logical_map[
                            layer_idx
                        ]

                        (
                            model_state.is_unchanged,
                            model_state.is_received_locally,
                            model_state.recv_metadata,
                        ) = await transfer_layer(
                            old_layer_indices=old_layer_indices,
                            new_layer_indices=new_layer_indices,
                            expert_weights=model_state.model.expert_weights[layer_idx],
                            expert_weights_buffer=model_state.expert_buffer,
                            ep_group=eplb_group,
                            is_profile=is_profile,
                            cuda_stream=cuda_stream,
                            rank_mapping=rank_mapping,
                        )
                        event = torch.cuda.Event(blocking=False)
                        cuda_stream.record_event(event)
                        model_state.buffer_ready_event = event
                        model_state.ep_buffer_ready = 1
                    finally:
                        model_state.buffer_lock.release()
                else:
                    if not model_state.rebalanced:
                        break
                    await asyncio.sleep(0.001)

        state.rearrange_event.clear()
