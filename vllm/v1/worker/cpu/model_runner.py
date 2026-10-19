# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.worker.cpu.block_table import CPUBlockTables
from vllm.v1.worker.cpu.input_batch import (
    combine_sampled_and_draft_tokens,
    post_update,
    post_update_num_computed_tokens,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from vllm.v1.worker.cpu.logprob import CPUPromptLogprobsWorker
from vllm.v1.worker.cpu.model_state import (
    CPUDefaultModelState,
    CPUEncoderOnlyModelState,
    CPUMambaHybridModelState,
)
from vllm.v1.worker.cpu.sampler import CPUSampler, CPUWatermarkSampler
from vllm.v1.worker.cpu.structured_outputs import CPUStructuredOutputsWorker
from vllm.v1.worker.cpu.utils import CPUKVBlockZeroer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_runner import BatchReqState, GPUModelRunner
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.encoder_decoder import EncoderDecoderModelState
from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker
from vllm.v1.worker.utils import KVBlockZeroer

logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):
    # TBD: Whether need to move this to Worker?
    def warming_up_model(self) -> None:
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        self.profile_run()
        logger.info("Warming up done.")


class NoTritonCPUModelRunner(CPUModelRunner):
    def prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        batch_req_state: BatchReqState,
        batch_desc: BatchExecutionDescriptor,
        num_active_loras: int,
    ) -> InputBatch:
        num_tokens = batch_req_state.num_tokens
        req_ids = batch_req_state.req_ids
        num_reqs = len(req_ids)

        is_padding = self.input_buffers.is_padding[:num_tokens]
        if envs.VLLM_MOE_SKIP_PADDING:
            is_padding.fill_(False)

        num_scheduled_tokens = batch_req_state.num_scheduled_tokens
        idx_mapping_np = batch_req_state.idx_mapping_np
        idx_mapping = torch.from_numpy(idx_mapping_np)

        cu_num_logits = torch.arange(
            num_reqs + 1, dtype=torch.int32, device=self.device
        )
        cu_num_logits_np = cu_num_logits.numpy()
        expanded_idx_mapping = idx_mapping
        expanded_local_pos = torch.zeros(
            num_reqs, dtype=torch.int32, device=self.device
        )

        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        query_start_loc_np = query_start_loc.numpy()
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])

        if batch_req_state.has_prefill:
            prepare_prefill_inputs(
                self.input_buffers.input_ids,
                self.req_states.next_prefill_tokens,
                idx_mapping,
                query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.prefill_len.gpu,
                self.req_states.num_computed_tokens.gpu,
            )

        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens.gpu,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs]
        logits_indices = combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            num_reqs,
        )

        fast_prefill = None
        if self.fast_prefill is not None:
            fast_prefill = self.fast_prefill.prepare(
                logits_indices,
                num_reqs,
                cu_num_logits_np,
                batch_req_state.has_prefill,
                batch_desc,
                num_active_loras=num_active_loras,
            )

        num_computed_tokens_np = self.req_states.num_computed_tokens_np[idx_mapping_np]

        prompt_lens = None
        if self.model_config.rswa_window is not None:
            prompt_lens = self.req_states.prompt_len.gpu[idx_mapping]

        return InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens,
            dcp_local_seq_lens=None,
            num_computed_tokens_np=num_computed_tokens_np,
            prefill_len_np=batch_req_state.prefill_len_np,
            num_computed_prefill_tokens_np=(
                batch_req_state.num_computed_prefill_tokens_np
            ),
            is_prefilling_np=batch_req_state.is_prefilling_np,
            has_prefill=batch_req_state.has_prefill,
            prefill_runs_as_decode_np=batch_req_state.prefill_runs_as_decode_np,
            decode_graph_eligible=batch_req_state.decode_graph_eligible,
            max_seq_len_np=batch_req_state.max_seq_len_np,
            input_ids=self.input_buffers.input_ids[:num_tokens],
            positions=self.input_buffers.positions[:num_tokens],
            is_padding=is_padding,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=(
                scheduler_output.has_structured_output_requests
            ),
            prompt_lens=prompt_lens,
            fast_prefill=fast_prefill,
        )

    def postprocess_sampled(
        self,
        idx_mapping: torch.Tensor,
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
    ) -> None:
        output_bin_counts = None
        if self.is_last_pp_rank:
            assert self.sampler is not None
            output_bin_counts = self.sampler.penalties_state.output_bin_counts

        post_update(
            idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.last_sampled_tokens,
            output_bin_counts,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
        )

        self.model_state.postprocess_state(
            idx_mapping, num_sampled, self.req_states.num_computed_tokens.gpu
        )

    def postprocess_num_computed_tokens(self, input_batch: InputBatch) -> None:
        post_update_num_computed_tokens(
            input_batch.idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            input_batch.query_start_loc,
        )

    @property
    def block_tables_cls(self) -> type[BlockTables]:
        return CPUBlockTables

    @property
    def sampler_cls(self) -> type[Sampler]:
        return CPUSampler

    @property
    def prompt_logprobs_worker_cls(self) -> type[PromptLogprobsWorker]:
        return CPUPromptLogprobsWorker

    @property
    def watermark_sampler_cls(self) -> type[GPUWatermarkSampler]:
        return CPUWatermarkSampler

    @property
    def model_state_cls(self) -> type[ModelState]:
        model_state_cls = super().model_state_cls
        if model_state_cls is DefaultModelState:
            return CPUDefaultModelState
        if model_state_cls is MambaHybridModelState:
            return CPUMambaHybridModelState
        if model_state_cls is EncoderOnlyModelState:
            return CPUEncoderOnlyModelState
        if model_state_cls is EncoderDecoderModelState:
            return model_state_cls
        raise ValueError(
            "CPU Model Runner V2 without Triton does not support model state "
            f"{model_state_cls.__name__}."
        )

    @property
    def structured_outputs_worker_cls(self) -> type[StructuredOutputsWorker]:
        return CPUStructuredOutputsWorker

    @property
    def kv_block_zeroer_cls(self) -> type[KVBlockZeroer]:
        return CPUKVBlockZeroer
