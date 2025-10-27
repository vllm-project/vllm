# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import TYPE_CHECKING, Any, Optional

import torch
import ttnn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.tt_loader import TTModelLoader
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.platforms.tt import TTPlatform
from vllm.sequence import IntermediateTensors
from vllm.utils import LayerBlockType, cdiv
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

        # The block tables in the persistent input batch have
        # max_num_blocks_per_req = cdiv(max_model_len, block_size) but this
        # does not take into account num blocks in KV cache. Actual max is min
        # of these two. Used to slice block tables during input prep.
        self.max_num_blocks_per_req = min(
            cdiv(self.model_config.max_model_len,
                 self.cache_config.block_size), kv_cache_config.num_blocks)

        # Only DP rank 0 allocates KV cache
        if self.parallel_config.data_parallel_rank != 0:
            return

        # Make the assumption that we are tensor parallel by
        # min(number of devices, number of KV heads).
        # TODO: move this into model.allocate_kv_cache.
        model_config = self.model_config
        data_parallel = self.parallel_config.data_parallel_size
        num_devices = self.device_config.num_devices // data_parallel
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

    def _validate_mm_input(self, mm_input: MultiModalKwargs) -> None:
        """Validate multi-modal input supports only single images."""
        if list(mm_input.modalities) != ["image"]:
            raise NotImplementedError("Only images are supported for now")
        assert mm_input.get_item_count("image") == 1, (
            "Request can contain multiple inputs, \
            but each input can contain only one image!")

    def _gather_multi_modal_inputs(self, scheduler_output) -> dict:
        """
        Gather and batch multi-modal inputs from scheduled requests.
        #TODO: Currently only supports image inputs in the "pixel_values" field.

        Creates a list of pixel values for each request.
        Example:
        [
          None, # for requests without mm_inputs
          [pixel_values_1], # with single mm_input
          [pixel_values_2, pixel_values_3, ...], # with multiple mm_inputs
        ]
        """

        multi_modal_kwargs: MultiModalKwargs = {"pixel_values": []}

        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            req_state = self.requests[req_id]

            if not req_state.mm_inputs:
                multi_modal_kwargs["pixel_values"].append(None)
                continue

            pv_array = []
            for mm_input in req_state.mm_inputs:
                self._validate_mm_input(mm_input)
                pv_array.append(mm_input["pixel_values"])

            multi_modal_kwargs["pixel_values"].append(pv_array)

        return multi_modal_kwargs

    def _prepare_model_inputs(
            self, scheduler_output: "SchedulerOutput") -> TTModelInput:

        assert scheduler_output.total_num_scheduled_tokens > 0
        input_batch = self.input_batch
        num_reqs = input_batch.num_reqs
        assert num_reqs > 0
        assert (len(input_batch.block_table.block_tables) == 1
                ), "Currently only supporting 1 KV cache group"

        # Second dim of block table is (ceil(max_model_len / block_size)).
        # Slice to self.max_num_blocks_per_req which also takes into
        # account max num blocks in KV cache in case max KV blocks is smaller.
        # Constant shape is required for ttnn tracing to work.
        block_tables = input_batch.block_table[0].get_cpu_tensor(
        )[:num_reqs, :self.max_num_blocks_per_req]

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

        if self.model_config.is_multimodal_model and is_prompt:
            multi_modal_kwargs = self._gather_multi_modal_inputs(
                scheduler_output)
        else:
            multi_modal_kwargs = {}

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
            multi_modal_kwargs=multi_modal_kwargs,
            cross_block_tables=None  # Not yet supported in V1
        )

    def concat_model_inputs(
            self, inputs: list[Optional["TTModelInput"]]) -> "TTModelInput":
        """
        Concatenate a DP-sized list of TTModelInput (some may be None) into
        a single TTModelInput. For None slots, uses zeros for input_tokens and
        block_tables and -1 for input_positions.
        """
        assert inputs, "No inputs to concatenate"
        active_inputs: list[TTModelInput] = [mi for mi in inputs if mi]
        if not active_inputs:
            raise ValueError("All inputs are None; nothing to concatenate")

        batch_size_per_dp = [
            mi.unpadded_batch_size if mi else 0 for mi in inputs
        ]
        if os.environ.get("DP_GATHER_DEBUG") == "1":
            logger.info("batch_size_per_dp=%s", batch_size_per_dp)

        sampling_params_per_dp = [
            mi.tt_sampling_params if mi else None for mi in inputs
        ]

        is_decode = active_inputs[0].prompt_lens is None
        for mi in active_inputs:
            assert (
                mi.prompt_lens
                is None) == is_decode, "All inputs must be for the same mode"

        if not is_decode:
            # Determine max token width across slots.
            max_tok_width = 0
            for mi in active_inputs:
                assert mi.input_tokens.dim() == 2, "Input tokens must be 2D"
                max_tok_width = max(max_tok_width, mi.input_tokens.shape[1])
            assert max_tok_width > 0, "At least one input must have tokens"

        # For block tables, assume each slot is already padded to the max
        # number of blocks, so we do not pad widths further.
        max_bt_width = int(active_inputs[0].block_tables.shape[1])

        # Iterate over DP inputs and build segments for concatenation.
        toks_segments: list[torch.Tensor] = []  # input tokens
        bt_segments: list[torch.Tensor] = []  # block tables
        if is_decode:
            pos_segments: list[torch.Tensor] = []  # input positions
        else:
            pl_segments: list[torch.Tensor] = []  # prompt lengths
        for mi in inputs:
            if mi is None:
                # For decode, keep fixed stride by padding with max_batch.
                # For prefill, skip None slots entirely (do not append rows).
                if is_decode:
                    max_batch = self.scheduler_config.max_num_seqs
                    toks_segments.append(
                        torch.zeros((max_batch, 1), dtype=torch.int32))
                    bt_segments.append(
                        torch.zeros((max_batch, max_bt_width),
                                    dtype=torch.int32))
                    pos_segments.append(
                        torch.full((max_batch, ), -1, dtype=torch.int32))
            else:
                # Right-pad tokens and block tables to max widths across slots
                toks = mi.input_tokens
                if not is_decode and toks.shape[1] < max_tok_width:
                    pad_w = max_tok_width - toks.shape[1]
                    toks = torch.cat([
                        toks,
                        torch.zeros((toks.shape[0], pad_w), dtype=toks.dtype)
                    ],
                                     dim=1)
                toks_segments.append(toks)
                bt_segments.append(mi.block_tables)
                if is_decode:
                    pos_segments.append(mi.input_positions)
                else:
                    assert mi.prompt_lens is not None
                    pl_segments.append(mi.prompt_lens)

        input_tokens = torch.cat(toks_segments, dim=0)
        block_tables = torch.cat(bt_segments, dim=0)
        if is_decode:
            input_positions = torch.cat(pos_segments, dim=0)
            prompt_lens = None
        else:
            input_positions = 0
            prompt_lens = np.concatenate(pl_segments, axis=0)

        assert not TTPlatform.compat_sampling_possible, (
            "Compatibility sampling is not yet supported in V1 TT backend")
        sampling_params_list: list[Any] = []
        compat_sampling_used = False
        sampling_metadata = None

        if self.model_config.is_multimodal_model and not is_decode:
            # Gather multi-modal inputs from all DP ranks
            multi_modal_kwargs: MultiModalKwargs = {"pixel_values": []}
            for mi in inputs:
                multi_modal_kwargs["pixel_values"].append(
                    mi.multi_modal_kwargs["pixel_values"])
        else:
            multi_modal_kwargs = {}

        merged = TTModelInput(
            input_tokens=input_tokens,
            input_positions=input_positions,
            prompt_lens=prompt_lens,
            seq_groups=None,
            block_tables=block_tables,
            unpadded_batch_size=batch_size_per_dp,
            tt_sampling_params=sampling_params_per_dp,
            sampling_params_list=sampling_params_list,
            compat_sampling_used=compat_sampling_used,
            sampling_metadata=sampling_metadata,
            multi_modal_kwargs=multi_modal_kwargs,
            cross_block_tables=None  # Not yet supported in V1
        )
        return merged

    def build_model_input(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[TTModelInput]:
        """
        Update internal state with the scheduler output and build
        TTModelInput without executing the model.
        Returns None if there is no scheduled work in this step.
        
        For data parallel, this function is called by each DP rank to build
        TTModelInput from it's own scheduler output.
        """
        # Update cached state
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            return None

        # Prepare model inputs only
        model_input = self._prepare_model_inputs(scheduler_output)
        return model_input

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> ModelRunnerOutput:
        ''' Execution path for non-DP case.
            Execute the model with the given scheduler output.'''

        # Update cached state and prepare model inputs
        model_input = self.build_model_input(scheduler_output)
        if model_input is None:
            return EMPTY_MODEL_RUNNER_OUTPUT

        # Only 1 DP rank here
        sampled_token_ids = self.execute_with_model_input(model_input)[0]
        output = self.generate_runner_output(sampled_token_ids)
        return output

    def execute_with_model_input(
        self,
        model_input: TTModelInput,
    ) -> list[list[list[int]]]:
        """
        Execute with a prebuilt input and return per-DP sampled ids without
        mutating internal state. In DP case, called by DP rank 0 to run merged
        batch. Note: currently does not support chunked prefill.
        """
        is_decode = model_input.prompt_lens is None

        batch_size_per_dp = model_input.unpadded_batch_size
        if not isinstance(batch_size_per_dp, list):
            batch_size_per_dp = [batch_size_per_dp]

        sampling_params_per_dp = model_input.tt_sampling_params
        if not isinstance(sampling_params_per_dp, list):
            sampling_params_per_dp = [sampling_params_per_dp]

        kwargs = {
            "tokens": model_input.input_tokens,
            "page_table": model_input.block_tables,
            "kv_cache": self.kv_caches,
        }

        if not is_decode:
            kwargs["prompt_lens"] = model_input.prompt_lens
            kwargs.update(model_input.multi_modal_kwargs)
            if len(batch_size_per_dp) > 1:
                # TODO: the model should only require DP ranks, but passing
                # "global" user ids instead for backwards compatibility.
                stride = int(self.scheduler_config.max_num_seqs)
                empty_slots = []
                for dp_rank, sz in enumerate(batch_size_per_dp):
                    for i in range(int(sz)):
                        empty_slots.append(dp_rank * stride + i)
                kwargs["empty_slots"] = empty_slots
        else:
            kwargs["start_pos"] = model_input.input_positions
        if self.sample_on_device_mode == "all" or (
                self.sample_on_device_mode == "decode_only" and is_decode):
            # Check that sampling params are the same for all DP ranks.
            # TODO: Remove this restriction and concat sampling params in
            # concat_model_inputs once models can support mixed params.
            non_none_params = [
                sp for sp in sampling_params_per_dp if sp is not None
            ]
            assert all(sp == non_none_params[0] for sp in non_none_params), (
                "Sampling params must be the same for all active DP ranks")
            kwargs["sampling_params"] = non_none_params[0]

        # Execute model
        if not is_decode:
            tt_out = self.model.prefill_forward(**kwargs)
        else:
            # TODO: Add encoder-decoder support
            enc_dec_kwargs: dict[str, Any] = {}
            tt_out = self.model.decode_forward(**kwargs,
                                               **enc_dec_kwargs,
                                               enable_trace=self.trace_mode,
                                               read_from_device=True)

        sampled_token_ids_per_dp: list[list[list[int]]] = []
        start = 0
        for dp_rank, sz in enumerate(batch_size_per_dp):
            if sz <= 0:
                sampled_token_ids_per_dp.append([])
                continue
            if (not self.sample_on_device_mode
                    or (self.sample_on_device_mode == "decode_only"
                        and not is_decode)):
                logits = tt_out[start:start + sz, -1, :]
                next_token_ids = sample_tokens(logits,
                                               sampling_params_per_dp[dp_rank])
            else:
                next_token_ids = tt_out[start:start + sz]
            sampled_token_ids_per_dp.append([[int(t)] for t in next_token_ids])

            if is_decode:
                # Fixed stride segments per DP rank for decode
                start += self.scheduler_config.max_num_seqs
            else:
                # Prefill packed contiguously
                start += sz

        return sampled_token_ids_per_dp

    def generate_runner_output(self, sampled_token_ids: list[list[int]]):
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
