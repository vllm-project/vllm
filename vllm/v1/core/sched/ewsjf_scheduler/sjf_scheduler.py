# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# EWSJF MODIFICATION: This file implements the EWSJF policy by inheriting from
# the default vLLm v1 Scheduler and overriding its waiting queue management.

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple, Deque, Union
from sortedcontainers import SortedDict

from vllm.v1.core.sched.ewsjf_scheduler.waiting_queue import WaitingQueue, QueueInfo
# EWSJF MODIFICATION: Import the parent Scheduler class to inherit from it.
from vllm.v1.core.sched.scheduler import Scheduler

from vllm.v1.core.sched.ewsjf_scheduler.scoring import SimpleScoreCalculator
import time
from collections import defaultdict
from collections.abc import Iterable

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVEventBatch

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry

from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

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
        super().__init__(vllm_config, kv_cache_config, structured_output_manager,
                         block_size, mm_registry, include_finished_set, log_stats)

        self.lock = threading.Lock()

    def schedule(self) -> SchedulerOutput:
        self.sort_waiting_queue()

        return super().schedule()

    def sort_waiting_queue(self):
        with self.lock:
            sorted_items = sorted(self.waiting, key=lambda req: len(req.prompt_token_ids))

            self.waiting.clear()
            self.waiting.extend(sorted_items)
