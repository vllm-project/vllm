# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
The async worker that transfers experts in the background.
"""
import asyncio
import threading
from typing import TYPE_CHECKING, Optional

import torch
from torch.distributed import ProcessGroup

from vllm.distributed.parallel_state import get_ep_group
from vllm.logger import init_logger

from .rebalance_execute import transfer_layer

if TYPE_CHECKING:
    from .eplb_state import EplbState

logger = init_logger(__name__)


def start_async_worker(state: "EplbState",
                       model,
                       rank_mapping: Optional[dict[int, int]] = None,
                       is_profile: bool = False) -> threading.Thread:
    ep_group = get_ep_group().device_group
    rank = ep_group.rank()
    device_index = state.cuda_device_index

    def thread_target() -> None:
        if device_index is not None:
            torch.cuda.set_device(device_index)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                transfer_run_periodically(state=state,
                                          model=model,
                                          ep_group=ep_group,
                                          is_profile=is_profile,
                                          rank_mapping=rank_mapping,
                                          device_index=device_index))
        except Exception as exc:  # pragma: no cover - diagnostic path
            logger.exception("async loop error (Rank %d): %s", rank, str(exc))
        finally:
            loop.close()

    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()
    return thread


async def transfer_run_periodically(
        state: "EplbState",
        model,
        ep_group: ProcessGroup,
        is_profile: bool = False,
        rank_mapping: Optional[dict[int, int]] = None,
        device_index: Optional[int] = None) -> None:
    experts_stream = (torch.cuda.Stream(device=device_index)
                      if device_index is not None else torch.cuda.Stream())

    while not state.shutdown_event.is_set():
        await asyncio.to_thread(state.rearrange_event.wait)

        if state.shutdown_event.is_set():
            break

        current_num_layers = model.num_moe_layers
        while (state.layer_to_transfer < current_num_layers
               and not state.shutdown_event.is_set()):
            if not state.ep_buffer_ready and state.rebalanced:
                assert state.new_physical_to_logical_map is not None
                await asyncio.to_thread(state.buffer_lock.acquire)
                try:
                    if state.layer_to_transfer >= current_num_layers:
                        break

                    for i, weight in enumerate(model.expert_weights[0]):
                        state.expert_buffer[i] = torch.empty_like(weight)

                    (state.is_unchanged, state.is_received_locally,
                     state.experts_recv_loc) = await transfer_layer(
                         old_global_expert_indices=state.
                         physical_to_logical_map,
                         new_global_expert_indices=state.
                         new_physical_to_logical_map,
                         expert_weights=model.expert_weights,
                         expert_weights_buffer=state.expert_buffer,
                         ep_group=ep_group,
                         is_profile=is_profile,
                         layer=state.layer_to_transfer,
                         cuda_stream=experts_stream,
                         rank_mapping=rank_mapping,
                     )
                    state.ep_buffer_ready = 1
                finally:
                    state.buffer_lock.release()
            else:
                await asyncio.sleep(0.001)

        state.rearrange_event.clear()
