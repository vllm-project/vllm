# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# EWSJF MODIFICATION: This file implements the EWSJF policy by inheriting from
# the default vLLm v1 Scheduler and overriding its waiting queue management.

from __future__ import annotations

import threading
from collections import deque
from typing import TYPE_CHECKING, cast

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.output import SchedulerOutput

# EWSJF MODIFICATION: Import the parent Scheduler class to inherit from it.
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.structured_output import StructuredOutputManager

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


class SJFScheduler(Scheduler):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            mm_registry,
            include_finished_set,
            log_stats,
        )

        self.lock = threading.Lock()

    def schedule(self) -> SchedulerOutput:
        self.sort_waiting_queue()
        return super().schedule()

    def sort_waiting_queue(self):
        with self.lock:
            # Cast self.waiting to deque to access clear/extend and iteration
            # We assume self.waiting behaves like a deque/list in this implementation
            waiting_queue = cast(deque, self.waiting)

            # Fix: Handle potential None type for prompt_token_ids
            sorted_items = sorted(
                waiting_queue, key=lambda req: len(req.prompt_token_ids or [])
            )

            waiting_queue.clear()
            waiting_queue.extend(sorted_items)
