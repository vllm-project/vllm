# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union, cast

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
from vllm.worker.tt_model_runner import TTSamplingParams, sample_tokens

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

import numpy as np

logger = init_logger(__name__)


@dataclass(frozen=True)
class TTModelInput:
    """
    Used by the TTModelRunner.
    """
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    prompt_lens: Optional[list[int]]
    block_tables: torch.Tensor
    unpadded_batch_size: Union[int, list[int]]  # List is used for DP
    tt_sampling_params: Union[TTSamplingParams, list[TTSamplingParams]]
    multi_modal_kwargs: dict[str, Any]

    # always lists: single-element for non-DP, multi-element for DP
    # If not using structured outputs, [None]
    grammar_bitmask: list[Optional[torch.Tensor]]


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

        # Cache the arange needed for unpacking structured output bitmask
        self.structured_output_arange = torch.arange(0, 32)
        self.bitmask_size = cdiv(self.model_config.get_vocab_size(), 32)

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
        if self.parallel_config.data_parallel_rank_local != 0:
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
        # In DP, called on each rank
        # In non-DP, this is the only input preparation function

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

        # DP optimization: don't send padding blocks if possible to reduce
        # overhead from gathering inputs to rank 0 and rely on DP concat
        # function to pad to global max blocks.
        if self.parallel_config.data_parallel_size > 1:
            max_tokens_in_batch = max(input_batch.num_tokens[:num_reqs])
            max_blocks_in_batch = cdiv(max_tokens_in_batch,
                                       self.cache_config.block_size)
            block_tables = block_tables[:, :max_blocks_in_batch]

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
            temperature=float(temperature[0]),
            top_k=int(top_k[0]),
            top_p=float(top_p[0]),
        )

        if self.model_config.is_multimodal_model and is_prompt:
            multi_modal_kwargs = self._gather_multi_modal_inputs(
                scheduler_output)
        else:
            multi_modal_kwargs = {}

        # If we're not using structured outputs, grammar_bitmask is None
        bitmask = scheduler_output.grammar_bitmask
        if bitmask is not None:
            # Using torch tensor instead of numpy array for consistency
            # because we need it as tensor for gather.
            bitmask = torch.from_numpy(bitmask)
            # unpadded for prefill, padded for decode
            batch_length = input_tokens.shape[0]
            grammar_bitmask_length = bitmask.shape[1]
            # Ones in the compressed bitmask represent tokens that are allowed.
            reordered_bitmask = torch.zeros(
                (batch_length, grammar_bitmask_length), dtype=torch.int32)
            reordered_bitmask = torch.bitwise_not(reordered_bitmask)
            structured_request_ids = scheduler_output.structured_output_request_ids  # noqa: E501
            for req_id, persistent_batch_index in self.input_batch.req_id_to_index.items(  # noqa: E501
            ):
                if req_id in structured_request_ids:
                    scheduler_batch_index = structured_request_ids[req_id]
                    reordered_bitmask[persistent_batch_index, :] = bitmask[
                        scheduler_batch_index, :]
            bitmask = reordered_bitmask

        return TTModelInput(
            input_tokens=input_tokens,
            input_positions=input_positions,
            prompt_lens=prompt_lens,
            block_tables=block_tables,
            unpadded_batch_size=num_reqs,
            tt_sampling_params=tt_sampling_params,
            multi_modal_kwargs=multi_modal_kwargs,
            grammar_bitmask=[bitmask],  # wrap to match DP case
        )

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

    def build_dp_decode_gather_input(
            self, model_input: Optional[TTModelInput],
            max_blocks_decode_batch: int) -> dict[str, torch.Tensor]:
        """
        Called by each DP rank to build tensorized gather input for decode.
        max_blocks_decode_batch is the max blocks in the global DP batch.
        Returns dict[str, torch.Tensor] with keys:
          - "int_inputs": flattened int tensor of constant size.
          - "float_inputs": flattened float tensor of constant size.
        """

        max_batch = int(self.scheduler_config.max_num_seqs)
        if model_input is None:
            tokens = torch.zeros((max_batch, 1), dtype=torch.int32)
            positions = torch.full((max_batch, ), -1, dtype=torch.int32)
            block_tables = torch.zeros((max_batch, max_blocks_decode_batch),
                                       dtype=torch.int32)
            unpadded_batch_size = torch.tensor([0], dtype=torch.int32)
            temperature = torch.tensor([-1.0], dtype=torch.float32)
            top_k = torch.tensor([-1], dtype=torch.int32)
            top_p = torch.tensor([-1.0], dtype=torch.float32)
            has_structured_inputs = torch.tensor([0], dtype=torch.int32)

        else:
            tokens = model_input.input_tokens
            positions = model_input.input_positions
            block_tables = model_input.block_tables
            # Pad block tables to max_blocks_decode_batch
            if block_tables.shape[1] < max_blocks_decode_batch:
                pad_w = max_blocks_decode_batch - block_tables.shape[1]
                block_tables = torch.cat([
                    block_tables,
                    torch.zeros((block_tables.shape[0], pad_w),
                                dtype=block_tables.dtype)
                ],
                                         dim=1)
            # We know these are not a list here before concatenation
            unpadded_batch_size = torch.tensor(
                [cast(int, model_input.unpadded_batch_size)],
                dtype=torch.int32)
            sp: TTSamplingParams = model_input.tt_sampling_params

            temperature = torch.tensor([sp.temperature], dtype=torch.float32)
            top_k = torch.tensor([sp.top_k], dtype=torch.int32)
            top_p = torch.tensor([sp.top_p], dtype=torch.float32)
            has_structured_input = int(
                model_input.grammar_bitmask[0] is not None)
            has_structured_inputs = torch.tensor([has_structured_input],
                                                 dtype=torch.int32)

        # Pack into flattened tensors to reduce number of collectives.
        # B = max batch size, W = max_num_blocks_per_req.
        int_inputs = torch.cat(
            [
                tokens.contiguous().view(-1),  # B
                positions.contiguous().view(-1),  # B
                block_tables.contiguous().view(-1),  # B*W
                unpadded_batch_size.contiguous().view(-1),  # 1
                top_k.contiguous().view(-1),  # 1
                # This needs to stay at the end so that DPEngineCoreProc
                # can check it without doing the full unpacking
                has_structured_inputs.contiguous().view(-1),  # 1
            ],
            dim=0).contiguous()
        float_inputs = torch.cat(
            [
                temperature.contiguous().view(-1),  # 1
                top_p.contiguous().view(-1),  # 1
            ],
            dim=0).contiguous()

        return {
            "int_inputs": int_inputs,
            "float_inputs": float_inputs,
        }

    def build_padded_bitmasks(
            self, model_input: Optional[TTModelInput]) -> torch.Tensor:
        if model_input is None or model_input.grammar_bitmask[0] is None:
            max_batch = int(self.scheduler_config.max_num_seqs)
            return torch.zeros((max_batch, self.bitmask_size),
                               dtype=torch.int32)
        return model_input.grammar_bitmask[0]

    def concat_dp_model_inputs(
            self, inputs, is_decode: bool,
            max_blocks_decode_batch: Optional[int]) -> "TTModelInput":
        """
        Concatenate a DP-sized set of inputs into a single TTModelInput.
        inputs can be either:
        - For prefill: list[Optional[TTModelInput]]
        - For decode (optimized gather): dict[str, torch.Tensor] with keys:
          - "int_inputs": stacked int32 tensor of shape [world, -1]
          - "float_inputs": stacked float32 tensor of shape [world, -1]
          - if any of the batches has structured inputs, 
          "bitmasks": stacked int32 tensor of shape [world, -1]
        """

        input_tokens_list: list[torch.Tensor] = []
        block_tables_list: list[torch.Tensor] = []
        input_positions_list: list[torch.Tensor] = []  # (decode only)
        prompt_lens_list: list[np.ndarray] = []  # (prefill only)
        batch_size_per_dp: list[int] = []
        sampling_params_per_dp: list[Optional[TTSamplingParams]] = []
        grammar_bitmask_list = []

        # Need to pad block tables to global max num blocks for constant shape.
        def pad_block_tables(block_tables):
            max_bt_width = self.max_num_blocks_per_req
            if block_tables.shape[1] < max_bt_width:
                pad_w = max_bt_width - block_tables.shape[1]
                block_tables = torch.cat([
                    block_tables,
                    torch.zeros((block_tables.shape[0], pad_w),
                                dtype=block_tables.dtype)
                ],
                                         dim=1)
            return block_tables

        if is_decode:
            # For decode, given gathered flattened tensors from all DP ranks.
            # Ints: [toks(B), positions(B), block_tables(B*W), bs(1), top_k(1)]
            # Floats: [temperature(1), top_p(1)]
            assert max_blocks_decode_batch is not None, (
                "max_blocks_decode_batch must be provided for decode")
            has_structured_list = []
            B = int(self.scheduler_config.max_num_seqs)
            W = max_blocks_decode_batch
            for batch_num, (int_inputs, float_inputs) in enumerate(
                    zip(inputs["int_inputs"], inputs["float_inputs"])):
                # Slices
                off = 0
                stride = B
                tokens = int_inputs[off:off + stride].view(B, 1)
                off += stride
                stride = B
                positions = int_inputs[off:off + stride].view(B)
                off += stride
                stride = B * W
                block_tables = int_inputs[off:off + stride].view(B, W)
                off += stride
                batch_size = int(int_inputs[off].item())
                off += 1
                top_k = int(int_inputs[off].item())
                off += 1
                has_structured_input = int(int_inputs[off].item())
                off += 1

                temperature = float(float_inputs[0].item())
                top_p = float(float_inputs[1].item())

                input_tokens_list.append(tokens)
                input_positions_list.append(positions)
                block_tables_list.append(pad_block_tables(block_tables))
                batch_size_per_dp.append(batch_size)
                if batch_size > 0:
                    sampling_params_per_dp.append(
                        TTSamplingParams(temperature=temperature,
                                         top_k=top_k,
                                         top_p=top_p))
                else:
                    sampling_params_per_dp.append(None)
                has_structured_list.append(has_structured_input)

            input_positions = torch.cat(input_positions_list, dim=0)
            prompt_lens = None

            if any(has_structured_list):
                bitmasks = inputs["bitmasks"]
                for position, has_structured in enumerate(has_structured_list):
                    if has_structured > 0:
                        bitmask = bitmasks[position, :]
                        bitmask = bitmask.view(B, self.bitmask_size)
                        grammar_bitmask_list.append(bitmask)
                    else:
                        grammar_bitmask_list.append(None)
            else:
                grammar_bitmask_list = [None] * len(has_structured_list)
        else:
            active_inputs: list[TTModelInput] = [mi for mi in inputs if mi]
            if not active_inputs:
                raise ValueError("All inputs are None; nothing to concatenate")

            # Determine max token width across slots.
            max_tok_width = 0
            for mi in active_inputs:
                assert mi.input_tokens.dim() == 2, "Input tokens must be 2D"
                max_tok_width = max(max_tok_width, mi.input_tokens.shape[1])
            assert max_tok_width > 0, "At least one input must have tokens"

            # Iterate over DP inputs and build segments for concatenation.
            for mi in inputs:
                # Skip None slots entirely. Decode path reconstructs full
                # inputs, so None should not occur there anymore.
                if mi is not None:
                    # Right-pad tokens and block tables to max widths
                    toks = mi.input_tokens
                    if not is_decode and toks.shape[1] < max_tok_width:
                        pad_w = max_tok_width - toks.shape[1]
                        toks = torch.cat([
                            toks,
                            torch.zeros(
                                (toks.shape[0], pad_w), dtype=toks.dtype)
                        ],
                                         dim=1)
                    input_tokens_list.append(toks)
                    prompt_lens_list.append(mi.prompt_lens)
                    block_tables_list.append(pad_block_tables(mi.block_tables))

                # We know it's not a list here before concatenation
                unpadded_batch_size: int = cast(
                    int, mi.unpadded_batch_size) if mi else 0
                batch_size_per_dp.append(unpadded_batch_size)
                sampling_params_per_dp.append(
                    mi.tt_sampling_params if mi else None)
                grammar_bitmask_list.append(
                    mi.grammar_bitmask[0] if mi else None)

            input_positions = 0
            prompt_lens = np.concatenate(prompt_lens_list, axis=0)

        input_tokens = torch.cat(input_tokens_list, dim=0)
        block_tables = torch.cat(block_tables_list, dim=0)

        if self.model_config.is_multimodal_model and not is_decode:
            # Gather multi-modal inputs from all DP ranks
            multi_modal_kwargs: MultiModalKwargs = {"pixel_values": []}
            for mi in inputs:
                multi_modal_kwargs["pixel_values"].append(
                    mi.multi_modal_kwargs["pixel_values"])
        else:
            multi_modal_kwargs = {}

        if os.environ.get("DP_GATHER_DEBUG") == "1":
            logger.info("batch_size_per_dp=%s", batch_size_per_dp)
        merged = TTModelInput(
            input_tokens=input_tokens,
            input_positions=input_positions,
            prompt_lens=prompt_lens,
            block_tables=block_tables,
            unpadded_batch_size=batch_size_per_dp,
            tt_sampling_params=sampling_params_per_dp,
            multi_modal_kwargs=multi_modal_kwargs,
            grammar_bitmask=grammar_bitmask_list,
        )
        return merged

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> ModelRunnerOutput:
        ''' Execution path for non-DP case.
            Execute the model with the given scheduler output.'''
        # In the DP case, this function is skipped!
        # tt_worker.py directly calls execute_with_model_input
        # With DP, the actual model pass happens on a batch
        # produced by concatenating the inputs from all DP ranks.

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
    ) -> list[torch.Tensor]:
        """
        Execute with a prebuilt input and return per-DP sampled ids without
        mutating internal state. In DP case, called by DP rank 0 to run merged
        batch. Note: currently does not support chunked prefill.
        """
        is_decode = model_input.prompt_lens is None

        batch_size_per_dp = model_input.unpadded_batch_size
        if not isinstance(batch_size_per_dp, list):
            batch_size_per_dp = [batch_size_per_dp]
        if not any(bs > 0 for bs in batch_size_per_dp):
            return [torch.tensor([], dtype=torch.int32)
                    ] * len(batch_size_per_dp)

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
            # concat_dp_model_inputs once models can support mixed params.
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

        # The model input we got here may come from
        # concatenating multiple DP ranks.
        # We need to split the data back before sampling.
        sampled_token_ids_per_dp: list[torch.Tensor] = []

        start = 0
        for dp_rank, sz in enumerate(batch_size_per_dp):
            if sz <= 0:
                sampled_token_ids_per_dp.append(
                    torch.tensor([], dtype=torch.int32))
                continue
            if (not self.sample_on_device_mode
                    or (self.sample_on_device_mode == "decode_only"
                        and not is_decode)):
                logits = tt_out[start:start + sz, -1, :]

                grammar_bitmask = model_input.grammar_bitmask[dp_rank]

                if grammar_bitmask is not None:
                    # match shape of logits, which are now unpadded on batch dim
                    grammar_bitmask = grammar_bitmask[:sz, :]
                    self.apply_grammar_bitmask(logits, grammar_bitmask)

                next_token_ids = sample_tokens(logits,
                                               sampling_params_per_dp[dp_rank])
            else:  # sample on device
                # Grammar bitmask is applied on device
                next_token_ids = tt_out[start:start + sz]
            sampled_token_ids_per_dp.append(next_token_ids.view(sz, 1))

            if is_decode:
                # Fixed stride segments per DP rank for decode
                start += self.scheduler_config.max_num_seqs
            else:
                # Prefill packed contiguously
                start += sz

        return sampled_token_ids_per_dp

    def apply_grammar_bitmask(self, logits: torch.Tensor,
                              grammar_bitmask: torch.Tensor) -> None:
        """Apply structured output grammar constraints to logits in-place"""
        # The grammar bitmask is compressed as packed int32 values
        # where each bit represents one token. We need to unpack it
        # like the TPU model runner does.
        # Ones in the compressed bitmask represent tokens that are allowed.

        #TODO this is likely a quite inefficient way of doing it on host.

        # grammar_bitmask: (batch_size, bitmask_size)
        # logits: (batch_size, vocab_size)
        unpacked_bitmask = (torch.bitwise_right_shift(
            grammar_bitmask[:, :, None],
            self.structured_output_arange[None, None, :]) & 1) == 0
        unpacked_bitmask = unpacked_bitmask.reshape(grammar_bitmask.shape[0],
                                                    -1)[:, :logits.shape[-1]]
        logits.masked_fill_(unpacked_bitmask, -float("inf"))

    def generate_runner_output(self, sampled_token_ids: torch.Tensor):
        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        assert sampled_token_ids.shape[0] == self.input_batch.num_reqs, (
            f"Number of request outputs {sampled_token_ids.shape[0]} != "
            f"number of requests in input batch {self.input_batch.num_reqs}")
        num_out_tokens = sampled_token_ids.shape[1]
        assert num_out_tokens == 1, "Currently only supporting 1 output token"
        for req_idx, sampled_ids in enumerate(sampled_token_ids):
            start_idx = self.input_batch.num_tokens[req_idx]
            end_idx = start_idx + num_out_tokens
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
            sampled_token_ids=sampled_token_ids.tolist(),
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
        )
