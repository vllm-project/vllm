# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import numpy as np
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.cpu.kernels import input_batch as kernels
from vllm.v1.worker.cpu.sampler import CPUSampler
from vllm.v1.worker.cpu.structured_outputs import CPUStructuredOutputsWorker
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_runner import BatchReqState, GPUModelRunner

logger = init_logger(__name__)

# Translate every MRV2 Triton kernel to numba, in place of the hand ports.
_TRANSLATE_TRITON = os.environ.get("VLLM_CPU_TRANSLATE_TRITON") == "1"


class CPUModelRunner(GPUModelRunner):
    @property
    def sampler_cls(self) -> type[CPUSampler]:
        return CPUSampler

    @property
    def structured_outputs_worker_cls(self) -> type[CPUStructuredOutputsWorker]:
        return CPUStructuredOutputsWorker

    def prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        batch_req_state: BatchReqState,
        batch_desc: BatchExecutionDescriptor,
        num_active_loras: int,
    ) -> InputBatch:
        # Spec decode, fast prefill and PCP keep the GPU path (Triton only).
        if (
            _TRANSLATE_TRITON
            or scheduler_output.scheduled_spec_decode_tokens
            or self.model_state.num_new_sampled_tokens_per_step != 1
            or self.fast_prefill is not None
            or self.pcp_manager is not None
        ):
            return super().prepare_inputs(
                scheduler_output, batch_req_state, batch_desc, num_active_loras
            )

        num_tokens = batch_req_state.num_tokens
        num_tokens_after_padding = max(num_tokens, batch_desc.num_tokens)
        is_padding = self.input_buffers.is_padding[:num_tokens_after_padding]
        if envs.VLLM_MOE_SKIP_PADDING:
            is_padding[:num_tokens].fill_(False)
            is_padding[num_tokens:].fill_(True)

        idx_mapping_np = batch_req_state.idx_mapping_np
        idx_mapping = torch.from_numpy(idx_mapping_np.astype(np.int32))
        num_reqs = len(idx_mapping_np)
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        query_start_loc_np = self.input_buffers.query_start_loc.numpy()
        query_start_loc_np[0] = 0
        np.cumsum(
            batch_req_state.num_scheduled_tokens,
            out=query_start_loc_np[1 : num_reqs + 1],
        )
        query_start_loc_np[num_reqs + 1 :] = num_tokens
        query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]

        kernels.prepare_inputs(
            self.input_buffers.input_ids.numpy(),
            self.input_buffers.positions.numpy(),
            self.input_buffers.seq_lens.numpy(),
            self.req_states.next_prefill_tokens.numpy(),
            idx_mapping.numpy(),
            query_start_loc_np,
            self.req_states.all_token_ids.gpu.numpy(),
            self.req_states.prefill_len.gpu.numpy(),
            self.req_states.num_computed_tokens.gpu.numpy(),
            self.req_states.last_sampled_tokens.numpy(),
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs_padded]
        cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)

        prompt_lens = None
        if self.model_config.rswa_window is not None:
            prompt_lens = self.req_states.prompt_len.gpu[idx_mapping]

        return InputBatch(
            req_ids=batch_req_state.req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs_padded,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=idx_mapping,
            expanded_local_pos=torch.zeros(num_reqs, dtype=torch.int32),
            num_scheduled_tokens=batch_req_state.num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=torch.from_numpy(query_start_loc_np),
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens,
            dcp_local_seq_lens=None,
            num_computed_tokens_np=self.req_states.num_computed_tokens_np[
                idx_mapping_np
            ],
            prefill_len_np=batch_req_state.prefill_len_np,
            num_computed_prefill_tokens_np=batch_req_state.num_computed_prefill_tokens_np,
            is_prefilling_np=batch_req_state.is_prefilling_np,
            has_prefill=batch_req_state.has_prefill,
            prefill_runs_as_decode_np=batch_req_state.prefill_runs_as_decode_np,
            decode_graph_eligible=batch_req_state.decode_graph_eligible,
            max_seq_len_np=batch_req_state.max_seq_len_np,
            input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
            positions=self.input_buffers.positions[:num_tokens_after_padding],
            is_padding=is_padding,
            logits_indices=torch.from_numpy(
                query_start_loc_np[1 : num_reqs + 1].astype(np.int64) - 1
            ),
            cu_num_logits=torch.from_numpy(cu_num_logits_np),
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
            prompt_lens=prompt_lens,
            fast_prefill=None,
            max_query_len=None,
        )

    # TBD: Whether need to move this to Worker?
    def warming_up_model(self) -> None:
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        self.profile_run()
        logger.info("Warming up done.")
