# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.tracing import instrument
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.cpu.model_runner import CPUModelRunner
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.simulated.sampler import SimulatedSampler

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class SimulatedCPUModelRunner(CPUModelRunner):
    """CPU runner that simulates model execution while preserving scheduler/KV logic."""

    def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
        super().load_model(load_dummy_weights, *args, **kwargs)
        self.simulated_sampler = SimulatedSampler(self.req_states)

    @instrument(span_name="Warmup (Simulated CPU)")
    def warming_up_model(self) -> None:
        logger.info("Skipping model warmup for simulated forward.")

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
    ) -> ModelRunnerOutput:
        if dummy_run:
            return EMPTY_MODEL_RUNNER_OUTPUT

        self.finish_requests(scheduler_output)
        self.free_states(scheduler_output)
        self.add_requests(scheduler_output)
        for request in scheduler_output.scheduled_new_reqs:
            sampling_params = request.sampling_params
            assert sampling_params is not None
            req_idx = self.req_states.req_id_to_index[request.req_id]
            self.simulated_sampler.add_request(req_idx, sampling_params)
        self.update_requests(scheduler_output)
        self.block_tables.apply_staged_writes()

        if not scheduler_output.total_num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT

        num_reqs = len(scheduler_output.num_scheduled_tokens)
        input_batch = self.prepare_inputs(
            scheduler_output,
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=scheduler_output.total_num_scheduled_tokens,
                num_reqs=num_reqs,
            ),
        )
        req_ids = input_batch.req_ids
        req_id_to_index = {req_id: i for i, req_id in enumerate(req_ids)}
        sampler_output = self.simulated_sampler.sample(input_batch)
        num_sampled = sampler_output.num_sampled
        num_rejected = sampler_output.num_rejected
        assert num_sampled is not None and num_rejected is not None

        self.postprocess_sampled(
            idx_mapping=input_batch.idx_mapping,
            sampled_tokens=sampler_output.sampled_token_ids,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            query_start_loc=input_batch.query_start_loc,
        )
        sampled_token_ids_list = [
            token_ids[:num_tokens]
            for token_ids, num_tokens in zip(
                sampler_output.sampled_token_ids.tolist(), num_sampled.tolist()
            )
        ]

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids_list,
        )

    def initialize_kv_cache_tensors(self) -> None:
        self.kv_caches = []
        logger.info(
            "Initialized virtual KV cache with %d groups and %d blocks; "
            "skipped KV tensor allocation.",
            len(self.kv_cache_config.kv_cache_groups),
            self.kv_cache_config.num_blocks,
        )

    def _init_kv_zero_meta(self) -> None:
        """Skip zeroing metadata because simulated forward has no KV tensors."""

    def _apply_kv_cache_memory_updates(
        self, scheduler_output: "SchedulerOutput"
    ) -> None:
        """Skip physical KV updates because simulated forward has no KV tensors."""
