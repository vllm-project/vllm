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

from vllm.distributed.parallel_state import get_ep_group
from vllm.logger import init_logger

from .rebalance_execute import transfer_layer

if TYPE_CHECKING:
    from .eplb_state import EplbState

logger = init_logger(__name__)


def start_async_worker(
    state: "EplbState",
    rank_mapping: dict[int, int] | None = None,
    is_profile: bool = False,
) -> threading.Thread:
    ep_group = get_ep_group().device_group
    rank = ep_group.rank()
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
                    ep_group=ep_group,
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


async def transfer_run_periodically(
    state: "EplbState",
    ep_group: ProcessGroup,
    cuda_stream: torch.cuda.Stream,
    is_profile: bool = False,
    rank_mapping: dict[int, int] | None = None,
) -> None:
    while True:
        await asyncio.to_thread(state.rearrange_event.wait)
        logger.info("async worker woke up for EPLB transfer")

        assert state.is_async
        for model_state in state.model_states.values():
            current_num_layers = model_state.model.num_moe_layers
            while (
                model_state.rebalanced
                and model_state.layer_to_transfer < current_num_layers
            ):
                # Set the async worker's CUDA stream on the communicator
                model_state.communicator.set_stream(cuda_stream)
                if (
                    not model_state.ep_buffer_ready
                    and model_state.rebalanced
                    and model_state.new_physical_to_logical_map is not None
                ):
                    await asyncio.to_thread(model_state.buffer_lock.acquire)
                    try:
                        if model_state.layer_to_transfer >= current_num_layers:
                            break

                        (
                            model_state.is_unchanged,
                            model_state.is_received_locally,
                            model_state.recv_metadata,
                        ) = await transfer_layer(
                            old_global_expert_indices=model_state.physical_to_logical_map,
                            new_global_expert_indices=model_state.new_physical_to_logical_map,
                            expert_weights=model_state.model.expert_weights,
                            expert_weights_buffer=model_state.expert_buffer,
                            ep_group=ep_group,
                            communicator=model_state.communicator,
                            is_profile=is_profile,
                            layer=model_state.layer_to_transfer,
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
