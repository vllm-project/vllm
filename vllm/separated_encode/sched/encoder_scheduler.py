# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import itertools
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Optional, Union

from vllm.config import VllmConfig

from vllm.distributed.kv_transfer.kv_connector.v1 import (KVConnectorBase_V1,
                                                          KVConnectorRole)
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.separated_encode.ec_transfer.connector.redis import (
    RedisECConnector)
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                              create_request_queue)
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


class EncoderScheduler(SchedulerInterface):

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager
        self.epd_disagg_config = vllm_config.epd_disagg_config

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: Optional[dict[int, set[str]]] = (
            defaultdict(set) if include_finished_set else None)

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        self.policy = SchedulingPolicy.FCFS
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        self.finished_req_ids: set[str] = set()

        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        self.max_num_encoder_input_tokens = encoder_compute_budget

        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size*128)
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1

        self.separated_encode = True
        self.instance_type = self.epd_disagg_config.instance_type
        if self.instance_type != "encode":
            raise RuntimeError("Incorrect instance initialization")
        
        self.ec_connector = RedisECConnector(
            vllm_config = self.vllm_config,
            intra_instance_type = "scheduler",
            preallocate_callback = None,
            injection_callback = None
        )
        self._allocated: dict[tuple[str, int], int] = {}

    def schedule(self) -> SchedulerOutput:
        scheduled_new_reqs: list[Request] = []

        token_budget = self.max_num_scheduled_tokens

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens
        
        # For logging.
        scheduled_timestamp = time.monotonic()


        while self.waiting and token_budget > 0:
            if len(self.running) == self.max_num_running_reqs:
                break

            request = self.waiting.peek_request()
            encoder_inputs_to_schedule = None
            new_encoder_budget = encoder_budget
            if not request.has_encoder_inputs:
                raise RuntimeError("Request without encoder input")
   
            #Schedule all mm inputs at once:
            mm_positions = request.mm_positions
            total_size, required_budget = 0, 0
            for i, pos_info in enumerate(mm_positions):
                if self.encoder_cache_manager.has_cache(request, i):
                    continue
                total_size += request.get_num_encoder_tokens(i)
                required_budget += pos_info.length

            if ((not self.encoder_cache_manager.can_preallocate(total_size)) 
                or required_budget > new_encoder_budget):
                break
            encoder_inputs_to_schedule = []
            for i, pos_info in enumerate(mm_positions):
                num_encoder_tokens = pos_info.length
                new_encoder_budget -= num_encoder_tokens
                encoder_inputs_to_schedule.append(i)
            
            # Sanity check:
            assert (new_encoder_budget + required_budget == encoder_budget)

            # Request was already popped from self.waiting
            # unless it was re-added above due to new_blocks being None.
            request = self.waiting.pop_request()
            self.running.append(request)

            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED,
                                        scheduled_timestamp)
            if request.status == RequestStatus.WAITING:
                scheduled_new_reqs.append(request)
            else:
                raise RuntimeError(
                    f"Invalid request status: {request.status}")

            request.status = RequestStatus.RUNNING
            # Encoder-related.
            scheduled_encoder_inputs[request.request_id] = (
                encoder_inputs_to_schedule)
            # Allocate the encoder cache.
            for i in encoder_inputs_to_schedule:
                self.encoder_cache_manager.allocate(request, i)
                self.ec_connector.schedule_send_encoder_cache_metadata(
                    request.request_id,
                    i,
                    request.get_num_encoder_tokens(i)
                )
                self._allocated[(request.request_id, i)] = request.get_num_encoder_tokens(i)
            encoder_budget = new_encoder_budget



        assert len(self.running) <= self.max_num_running_reqs

        new_reqs_data = [
            NewRequestData.from_request(req, ([],))
            for req in scheduled_new_reqs
        ]

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=0,
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids={},
            grammar_bitmask=None,
        )

        self.finished_req_ids = set()
        return scheduler_output

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:            
        transfered_mm_data = model_runner_output.transfered_mm_data

        for (req_id, input_id) in transfered_mm_data:
            assert (req_id, input_id) in self._allocated
            cache_size = self._allocated[(req_id, input_id)]
            self._allocated.pop((req_id, input_id))
            self.encoder_cache_manager.deallocate(req_id, input_id, cache_size)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)

        model_finished = []
        for request in self.running:
            req_id = request.request_id
            model_finished.append(req_id)
            outputs[request.client_index].append(
                EngineCoreOutput(request_id=req_id,
                    new_token_ids=[],
                    finish_reason=RequestStatus.get_finished_reason(
                        RequestStatus.FINISHED_STOPPED
                    ),
                    stop_reason="stop",
                    kv_transfer_params={}
                )
            )
        self.finish_requests(model_finished, RequestStatus.FINISHED_STOPPED)
        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        if engine_core_outputs:
            # Return stats to only one of the front-ends.
            next(iter(engine_core_outputs.values())).scheduler_stats = (
                self.make_stats(None))

        return engine_core_outputs

    def add_request(self, request: Request) -> None:
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)

        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        else:
            request_ids = set(request_ids)

        running_requests_to_remove = []
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.append(request)
            else:
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        for request in running_requests_to_remove:
            self.running.remove(request)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> Optional[dict[str, Any]]:
        assert request.is_finished()
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)
        del self.requests[request.request_id]
        return None 

# no changes v

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting)

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def make_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats] = None,
    ) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
        )

# Placeholder functions v
    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats],
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> Optional[SpecDecodingStats]:
        return None

    def shutdown(self) -> None:
        pass

    def get_kv_connector(self) -> Optional[KVConnectorBase_V1]:
        return None

    def reset_prefix_cache(self) -> bool:
        pass