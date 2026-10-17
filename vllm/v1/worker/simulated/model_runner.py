# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.tracing import instrument
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.cpu.model_runner import CPUModelRunner
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class SimulatedCPUModelRunner(CPUModelRunner):
    """CPU runner that simulates model execution while preserving scheduler/KV logic."""

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
        context_len: int = 0,
        valid_dummy_state_slots: bool = False,
    ) -> ModelRunnerOutput:
        if dummy_run:
            return EMPTY_MODEL_RUNNER_OUTPUT

        self.finish_requests(scheduler_output)
        self.free_states(scheduler_output)
        self.add_requests(scheduler_output)
        self.update_requests(scheduler_output)
        self.block_tables.apply_staged_writes()

        if not scheduler_output.total_num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT

        batch_req_state, _ = self.gather_batch_req_state(scheduler_output, dummy_run)
        assert batch_req_state is not None
        input_batch = self.prepare_inputs(
            scheduler_output,
            batch_req_state,
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=batch_req_state.num_tokens,
                num_reqs=len(batch_req_state.req_ids),
            ),
            num_active_loras=0,
        )
        req_ids = input_batch.req_ids
        req_id_to_index = {req_id: i for i, req_id in enumerate(req_ids)}

        assert self.sampler is not None
        trace_replay_state = self.sampler.trace_replay_state
        assert trace_replay_state is not None
        sampled = torch.zeros(
            input_batch.num_reqs, dtype=torch.int64, device=self.device
        )
        trace_replay_state.apply_trace(sampled, input_batch.idx_mapping)
        sampled_token_ids = sampled.view(-1, 1)

        seq_lens = input_batch.seq_lens[: input_batch.num_reqs]
        prefill_lens = torch.from_numpy(input_batch.prefill_len_np)
        num_sampled = (seq_lens >= prefill_lens).to(torch.int32)
        num_rejected = torch.zeros_like(num_sampled)

        self.postprocess_sampled(
            idx_mapping=input_batch.idx_mapping,
            sampled_tokens=sampled_token_ids,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            query_start_loc=input_batch.query_start_loc,
        )
        sampled_token_ids_list = [
            token_ids[:num_tokens]
            for token_ids, num_tokens in zip(
                sampled_token_ids.tolist(), num_sampled.tolist()
            )
        ]

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids_list,
        )

    def initialize_kv_cache_tensors(
        self,
        *,
        is_profiling: bool,
        kv_cache_allocation_context: AbstractContextManager | None,
    ) -> None:
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
