# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Optional

import torch
import ttnn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.tt_loader import TTModelLoader
from vllm.platforms.tt import TTPlatform
from vllm.sequence import IntermediateTensors
from vllm.utils import LayerBlockType
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsTensors,
                             ModelRunnerOutput)
from vllm.v1.worker.tt_input_batch import CachedRequestState, InputBatch
from vllm.worker.tt_model_runner import (TTModelInput, TTSamplingParams,
                                         sample_tokens)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

import numpy as np

logger = init_logger(__name__)


class TTModelRunner:

    def __init__(
        self,
        vllm_config: VllmConfig,
        mesh_device: ttnn.MeshDevice,
        trace_mode: bool,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        self.device_config = vllm_config.device_config

        # Because of multiprocessing, the config-dependent
        # class attributes might not have been set in this process,
        # so we need to call this again.
        TTPlatform.check_and_update_config(vllm_config)

        # Currently, TT model runner doesn't support chunked prefill.
        assert self.scheduler_config.chunked_prefill_enabled is False

        self.mesh_device = mesh_device
        self.trace_mode = trace_mode

        # Whether to sample on device
        self.sample_on_device_mode = TTPlatform.sample_on_device_mode

        logger.info(
            "TTModelRunner: trace_mode=%s, sample_on_device_mode=%s",
            self.trace_mode,
            self.sample_on_device_mode,
        )

        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: dict[str, dict[int, torch.Tensor]] = {}

        # Cached request states. Request states are tracked in the runner so
        # they don't need to be re-sent every scheduling step. For requests
        # that have been scheduled before, only the diff is received from
        # the scheduler output.
        self.requests: dict[str, CachedRequestState] = {}

    def load_model(self) -> None:
        loader = TTModelLoader(self.load_config)
        self.model = loader.load_model(vllm_config=self.vllm_config,
                                       model_config=self.model_config)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """

        kv_cache_groups = kv_cache_config.kv_cache_groups
        if len(kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")
        if isinstance(kv_cache_groups[0].kv_cache_spec, AttentionSpec):
            kv_cache_spec = kv_cache_groups[0].kv_cache_spec
        else:
            raise TypeError("Expected AttentionSpec")

        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            assert len(kv_cache_tensor.shared_by) == 1, (
                "KV cache shared by multiple layers is not supported for TT")

        # Initialize persistent input batch with block size from kv_cache_spec.
        # The persistent batch optimization reduces overhead between steps
        # when consecutive batches contain mostly the same requests.
        max_num_reqs = self.scheduler_config.max_num_seqs
        max_model_len = self.model_config.max_model_len
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        self.input_batch = InputBatch(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            block_sizes=[kv_cache_spec.block_size],
        )

        # Make the assumption that we are tensor parallel by
        # min(number of devices, number of KV heads).
        # TODO: move this into model.allocate_kv_cache.
        model_config = self.model_config
        data_parallel = 1
        if (self.model_config.override_tt_config
                and "data_parallel" in model_config.override_tt_config):
            data_parallel = model_config.override_tt_config["data_parallel"]
        num_devices = self.mesh_device.get_num_devices() // data_parallel
        total_kv_heads = kv_cache_spec.num_kv_heads
        num_kv_heads = total_kv_heads // min(num_devices, total_kv_heads)

        kv_cache_shape = (kv_cache_config.num_blocks, num_kv_heads,
                          kv_cache_spec.block_size, kv_cache_spec.head_size)
        dtype = kv_cache_spec.dtype
        num_layers = model_config.get_num_layers_by_block_type(
            self.parallel_config, LayerBlockType.attention)

        # Allocate KV cache tensors.
        self.kv_caches = self.model.allocate_kv_cache(kv_cache_shape, dtype,
                                                      num_layers)

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the 
        scheduler output.
        The updated states are used in `_prepare_model_inputs` to create the 
        input tensors for the model.
        Based on _update_states for GPU/TPU backends.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)

        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert new_req_data.sampling_params is not None,\
                "Pooling is not supported for TT yet"
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                pooling_params=None,
                generator=None,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens
            if not resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                for block_ids, new_ids in zip(req_state.block_ids,
                                              new_block_ids):
                    block_ids.extend(new_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)
            self.input_batch.block_table.append_row(new_block_ids, req_index)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

    def _prepare_model_inputs(
            self, scheduler_output: "SchedulerOutput") -> TTModelInput:

        assert scheduler_output.total_num_scheduled_tokens > 0
        input_batch = self.input_batch
        num_reqs = input_batch.num_reqs
        assert num_reqs > 0
        assert (len(input_batch.block_table.block_tables) == 1
                ), "Currently only supporting 1 KV cache group"

        # Second dim of block table kept as fixed size max_num_blocks_per_req
        # (ceil(max_model_len / block_size)) so ttnn tracing can work
        # (requires constant shape).
        block_tables = input_batch.block_table[0].get_cpu_tensor(
        )[:num_reqs, :]

        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = len(scheduler_output.scheduled_new_reqs) > 0
        if is_prompt:
            # Assert no running requests
            assert (
                len(scheduler_output.scheduled_cached_reqs.req_ids) == 0
            ), "Currently only supporting all prefills or all decodes in batch"

            input_positions = 0
            max_prompt_tokens = max(input_batch.num_prompt_tokens[:num_reqs])
            input_tokens = input_batch.token_ids_cpu_tensor[:num_reqs, :
                                                            max_prompt_tokens]
            prompt_lens = input_batch.num_prompt_tokens[:num_reqs]
        else:
            input_positions = torch.from_numpy(
                input_batch.num_tokens[:num_reqs] - 1)
            input_tokens = input_batch.token_ids_cpu_tensor[
                torch.arange(num_reqs), input_positions].view(-1, 1)
            prompt_lens = None

            # TODO: Remove once TT models can support arbitrary batch sizes.
            # Pad batch to max_num_reqs.
            if input_tokens.shape[0] < input_batch.max_num_reqs:
                batch_pad = input_batch.max_num_reqs - input_tokens.shape[0]
                input_tokens = torch.cat([
                    input_tokens,
                    torch.zeros(batch_pad, 1, dtype=torch.int32)
                ])
                # Pad positions with -1 to indicate no position
                input_positions = torch.cat([
                    input_positions,
                    torch.ones(batch_pad, dtype=torch.int32) * -1
                ])
                block_tables = torch.cat([
                    block_tables,
                    torch.zeros(batch_pad,
                                block_tables.shape[1],
                                dtype=torch.int32)
                ])

        # Sampling-related.
        temperature = input_batch.sampling.temperature_cpu[:num_reqs]
        top_p = input_batch.sampling.top_p_cpu[:num_reqs]
        top_k = input_batch.sampling.top_k_cpu[:num_reqs]
        if not np.all(temperature == temperature[0]):
            logger.warning(
                "Currently only supporting same temperature for all "
                "sequences in batch, falling back to first sequence's "
                "temperature (%s)", temperature[0])
        if not np.all(top_k == top_k[0]):
            logger.warning(
                "Currently only supporting same top_k"
                "for all sequences in batch, "
                "falling back to first sequence's top_k (%s)", top_k[0])
        if not np.all(top_p == top_p[0]):
            logger.warning(
                "Currently only supporting same top_p"
                "for all sequences in batch, "
                "falling back to first sequence's top_p (%s)", top_p[0])
        tt_sampling_params = TTSamplingParams(
            temperature=temperature[0],
            top_k=top_k[0],
            top_p=top_p[0],
        )

        assert not TTPlatform.compat_sampling_possible, (
            "Compatibility sampling is not yet supported in V1 TT backend")
        sampling_params_list: list[Any] = []
        compat_sampling_used = False
        sampling_metadata = None

        return TTModelInput(
            input_tokens=input_tokens,
            input_positions=input_positions,
            prompt_lens=prompt_lens,
            seq_groups=None,  # Not used in V1
            block_tables=block_tables,
            unpadded_batch_size=num_reqs,
            tt_sampling_params=tt_sampling_params,
            sampling_params_list=sampling_params_list,
            compat_sampling_used=compat_sampling_used,
            sampling_metadata=sampling_metadata,
            multi_modal_kwargs={},  # Not yet supported in V1
            cross_block_tables=None  # Not yet supported in V1
        )

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> ModelRunnerOutput:
        ''' Execute the model with the given scheduler output.
            Note: currently does not support chunked prefill.'''

        # Update cached state
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOutput if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT

        # Prepare model inputs
        model_input = self._prepare_model_inputs(scheduler_output)
        is_decode = model_input.prompt_lens is None
        execute_model_kwargs = {
            "tokens": model_input.input_tokens,
            "page_table": model_input.block_tables,
            "kv_cache": self.kv_caches,
            **(model_input.multi_modal_kwargs or {}),
        }
        if not is_decode:
            execute_model_kwargs["prompt_lens"] = model_input.prompt_lens
        else:
            execute_model_kwargs["start_pos"] = model_input.input_positions
        if self.sample_on_device_mode == "all" or (
                self.sample_on_device_mode == "decode_only" and is_decode):
            execute_model_kwargs[
                "sampling_params"] = model_input.tt_sampling_params

        # Execute model
        if not is_decode:
            outputs = self.model.prefill_forward(**execute_model_kwargs)
            tt_out = outputs  # [batch_size, seq_len, vocab_size]
        else:
            # TODO: Add encoder-decoder support
            enc_dec_kwargs: dict[str, Any] = {}
            tt_out = self.model.decode_forward(**execute_model_kwargs,
                                               **enc_dec_kwargs,
                                               enable_trace=self.trace_mode,
                                               read_from_device=True)

        if not self.sample_on_device_mode or (
                self.sample_on_device_mode == "decode_only" and not is_decode):
            next_logits = tt_out[:self.input_batch.num_reqs,
                                 -1, :]  # unpadded batch, vocab of last token
            next_token_ids = sample_tokens(next_logits,
                                           model_input.tt_sampling_params)
        else:
            next_token_ids = tt_out

        sampled_token_ids = [[int(next_token_ids[i])]
                             for i in range(self.input_batch.num_reqs)]
        output = self._generate_runner_output(sampled_token_ids)
        return output

    def _generate_runner_output(self, sampled_token_ids: list[list[int]]):
        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        for req_idx, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.model_config.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.model_config.max_model_len}")

            # Update persistent batch
            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens[req_idx] = end_idx

            # Update request state
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        # Empty prompt log probs
        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}
        for req_id in self.input_batch.req_ids[:self.input_batch.num_reqs]:
            prompt_logprobs_dict[req_id] = None

        # Note: currently does not support speculative decoding, log probs,
        # or pooling.
        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
        )
