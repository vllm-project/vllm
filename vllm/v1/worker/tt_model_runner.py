# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass, fields
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
from vllm.v1.sample.logits_processor import LogitsProcessorManager
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.tt_input_batch import (SEED_NONE_SENTINEL,
                                           CachedRequestState, InputBatch)
from vllm.worker.tt_model_runner import decode_warmup, prefill_warmup

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

import numpy as np

logger = init_logger(__name__)

# Maximum top_k value for on-device sampling
MAX_K = 32


@dataclass(frozen=True)
class TTSamplingParams:
    """Sampling parameters for TT model execution.
    
    Host sampling uses tensors, while on-device sampling uses lists.
    """
    temperature: Union[torch.Tensor, list[float]]
    top_k: Union[torch.Tensor, list[int]]
    top_p: Union[torch.Tensor, list[float]]
    presence_penalty: Union[torch.Tensor, list[float]]
    frequency_penalty: Union[torch.Tensor, list[float]]
    repetition_penalty: Union[torch.Tensor, list[float]]
    seed: Union[torch.Tensor, list[Optional[int]]]
    enable_log_probs: Optional[Union[torch.Tensor, list[bool]]] = None


@dataclass(frozen=True)
class TTModelInput:
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    prompt_lens: Optional[list[int]]
    block_tables: torch.Tensor
    unpadded_batch_size: Union[int, list[int]]  # List is used for DP
    tt_sampling_params: TTSamplingParams
    multi_modal_kwargs: dict[str, Any]

    # For DP gather, this is true only if all ranks can sample on device.
    perform_device_sampling: bool

    # always lists: single-element for non-DP, multi-element for DP
    # If not using structured outputs, [None]
    grammar_bitmask: list[Optional[torch.Tensor]]

    # Optional: tokens for sampling with penalties during decode
    prompt_tokens: Optional[torch.Tensor] = None
    output_tokens: Optional[torch.Tensor] = None

    # Decode-only: indicates the padded decode-batch layout changed since the
    # previous step (used by on-device sampling).
    reset_batch: bool = False


class TTModelRunner:

    def __init__(
        self,
        vllm_config: VllmConfig,
        mesh_device: ttnn.MeshDevice,
        trace_mode: str,
        enable_model_warmup: bool,
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

        # Detect if the model has "mrope" rope_scaling type.
        # mrope requires keeping "rope_deltas" between prefill/decode phases.
        self.request_specific_rope = bool(self.model_config.uses_mrope)
        if self.request_specific_rope:
            self.previous_req_ids: set[str] = set()

        # Because of multiprocessing, the config-dependent
        # class attributes might not have been set in this process,
        # so we need to call this again.
        TTPlatform.check_and_update_config(vllm_config)

        # Currently, TT model runner doesn't support chunked prefill.
        assert self.scheduler_config.chunked_prefill_enabled is False

        self.mesh_device = mesh_device
        self.trace_mode = trace_mode
        self.enable_model_warmup = enable_model_warmup
        # Whether to sample on device
        self.sample_on_device_mode = TTPlatform.sample_on_device_mode

        logger.info(
            "TTModelRunner: trace_mode=%s, "
            "sample_on_device_mode=%s, enable_model_warmup=%s",
            self.trace_mode,
            self.sample_on_device_mode,
            self.enable_model_warmup,
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
        self.vocab_size = self.model_config.get_vocab_size()
        self.bitmask_size = cdiv(self.vocab_size, 32)

        # For on-device decode sampling, we must signal if the padded decode
        # batch layout changed since the *previous decode step*. Layout can
        # change during prefill steps (e.g. new requests added), so we keep a
        # sticky flag and clear it only after a decode input consumes it.
        self._decode_layout_changed_since_last_decode: bool = True

        # Sampler for sampling on host when device sampling is not supported.
        # Only used by device ranks (local dp rank 0).
        if self.parallel_config.data_parallel_rank_local == 0:
            self.host_sampler = Sampler()

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
            vocab_size=self.vocab_size,
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
        persistent_batch_layout_changed = False

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
                persistent_batch_layout_changed = True

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
            persistent_batch_layout_changed = True

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
            persistent_batch_layout_changed = True

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)
            persistent_batch_layout_changed = True
        # Mark decode layout changed if persistent batch changed. This is
        # sticky across steps and will be consumed by the next decode batch.
        if persistent_batch_layout_changed:
            self._decode_layout_changed_since_last_decode = True

    def _validate_mm_input(self, mm_input: MultiModalKwargs) -> None:
        """Validate multi-modal input supports only single images."""
        if list(mm_input.modalities) != ["image"]:
            raise NotImplementedError("Only images are supported for now")
        assert mm_input.get_item_count("image") == 1, (
            "Request can contain multiple inputs, \
            but each input can contain only one image!")

    def _gather_multi_modal_inputs(self, scheduler_output) -> dict[str, Any]:
        """
        Gather and batch multi-modal inputs from scheduled requests.

        Currently only supports image inputs in the "pixel_values" and
        "image_grid_thw" fields.

        Returns dict, each value is a list of lists of tensors per-request:
        [
          # for requests w/o mm_inputs:
          {"pixel_values": None,
           "image_grid_thw": None},
          # for requests w/ single mm_input:
          {"pixel_values": [[pv_user_1],[pv_user_2]],
          "image_grid_thw": [[ig_user_1],[ig_user_2]]},
          # for requests w/ multiple mm_inputs:
          {"pixel_values": [[pv_user_1_image_1, pv_user_1_image_2],
                            [pv_user_2_image_1, pv_user_2_image_2]],
           "image_grid_thw":[[ig_user_1_image_1, ig_user_1_image_2],
                             [ig_user_2_image_1, ig_user_2_image_2]]},
        ]
        """

        multi_modal_kwargs: MultiModalKwargs = {
            "pixel_values": [],
            "image_grid_thw": []
        }

        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            req_state = self.requests[req_id]

            if not req_state.mm_inputs:
                multi_modal_kwargs["pixel_values"].append(None)
                multi_modal_kwargs["image_grid_thw"].append(None)
                continue

            pv_array = []
            image_grid_thw_array = []
            for mm_input in req_state.mm_inputs:
                self._validate_mm_input(mm_input)
                pv_array.append(mm_input["pixel_values"])
                image_grid_thw_array.append(
                    mm_input.get("image_grid_thw", None))

            multi_modal_kwargs["pixel_values"].append(pv_array)
            multi_modal_kwargs["image_grid_thw"].append(image_grid_thw_array)

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
        sample_params = input_batch.sampling
        if is_prompt:
            # NOTE: In SchedulerOutput, "cached" means "request data already
            # cached on the worker", not necessarily "decode". During a prefill
            # step we can legitimately see cached requests if they are resumed
            # from preemption (still prefill work).
            cached = scheduler_output.scheduled_cached_reqs
            if cached.num_reqs > 0:
                any_running = any(not x
                                  for x in cached.resumed_from_preemption)
                assert not any_running, (
                    "Prefill batch should not include decode/running cached "
                    "requests (resumed_from_preemption=False).")

            # num_computed_tokens for each request is the input position
            # (=computed previously and cached)
            input_positions = input_batch.num_computed_tokens_cpu[:num_reqs]
            max_prompt_tokens = max(input_batch.num_prompt_tokens[:num_reqs])
            input_tokens = input_batch.token_ids_cpu_tensor[:num_reqs, :
                                                            max_prompt_tokens]
            prompt_lens = input_batch.num_prompt_tokens[:num_reqs]
            reset_batch = False
        else:
            input_positions = torch.from_numpy(
                input_batch.num_tokens[:num_reqs] - 1)
            input_tokens = input_batch.token_ids_cpu_tensor[
                torch.arange(num_reqs), input_positions].view(-1, 1)
            prompt_lens = None
            # For on-device decode sampling, tell the backend if the padded
            # decode batch layout changed since the previous step.
            reset_batch = self._decode_layout_changed_since_last_decode
            self._decode_layout_changed_since_last_decode = False

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
                # Pad sampling parameters with default values
                sample_params.pad_with_defaults(num_reqs)

        if is_prompt:
            tt_sampling_params = TTSamplingParams(
                temperature=sample_params.temperature[:num_reqs],
                top_k=sample_params.top_k[:num_reqs],
                top_p=sample_params.top_p[:num_reqs],
                presence_penalty=sample_params.presence_penalty[:num_reqs],
                frequency_penalty=sample_params.frequency_penalty[:num_reqs],
                repetition_penalty=sample_params.repetition_penalty[:num_reqs],
                seed=sample_params.seed[:num_reqs],
                enable_log_probs=None,
            )
        else:
            tt_sampling_params = TTSamplingParams(
                temperature=sample_params.temperature,
                top_k=sample_params.top_k,
                top_p=sample_params.top_p,
                presence_penalty=sample_params.presence_penalty,
                frequency_penalty=sample_params.frequency_penalty,
                repetition_penalty=sample_params.repetition_penalty,
                seed=sample_params.seed,
                enable_log_probs=None,
            )
        perform_device_sampling = self.check_perform_device_sampling(
            is_decode=not is_prompt)

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
            for req_id, persistent_batch_index in (
                    input_batch.req_id_to_index.items()):
                if req_id in structured_request_ids:
                    scheduler_batch_index = structured_request_ids[req_id]
                    reordered_bitmask[persistent_batch_index, :] = bitmask[
                        scheduler_batch_index, :]
            bitmask = reordered_bitmask

        # Populate prompt_tokens and output_tokens if penalties are needed
        # (decode only).
        prompt_tokens = None
        output_tokens = None
        if (not input_batch.no_penalties) and not is_prompt:
            prompt_tokens = input_batch.make_prompt_token_ids_tensor()
            output_tokens = input_batch.make_output_token_ids_tensor()

            # Pad batch to max_num_reqs for non-DP case (don't send padding for
            # DP to reduce overhead from gathering inputs to rank 0).
            if (self.parallel_config.data_parallel_size == 1
                    and prompt_tokens.shape[0] < input_batch.max_num_reqs):
                batch_pad = (input_batch.max_num_reqs - prompt_tokens.shape[0])
                prompt_tokens = torch.cat([
                    prompt_tokens,
                    torch.full((batch_pad, prompt_tokens.shape[1]),
                               -1,
                               dtype=torch.int32)
                ])
                output_tokens = torch.cat([
                    output_tokens,
                    torch.full((batch_pad, output_tokens.shape[1]),
                               -1,
                               dtype=torch.int32)
                ])

        return TTModelInput(
            input_tokens=input_tokens,
            input_positions=input_positions,
            prompt_lens=prompt_lens,
            block_tables=block_tables,
            unpadded_batch_size=num_reqs,
            tt_sampling_params=tt_sampling_params,
            multi_modal_kwargs=multi_modal_kwargs,
            perform_device_sampling=perform_device_sampling,
            grammar_bitmask=[bitmask],  # wrap to match DP case
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            reset_batch=reset_batch,
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
            max_blocks_decode_batch: int, any_structured_inputs: bool,
            any_penalties_inputs: bool) -> dict[str, Any]:
        """
        Called by each DP rank to build tensorized gather input for decode.
        max_blocks_decode_batch: max blocks in the global DP batch.
        any_structured_inputs: whether the global batch has structured inputs.
        any_penalties_inputs: whether the global batch has penalties.
        Returns dict[str, Any] with keys:
          - "int_inputs": flattened int tensor of constant size.
          - "float_inputs": flattened float tensor of constant size.
          - "sampling_tokens_inputs": Optional[dict[str, torch.Tensor]] with
            keys "prompt_tokens" and "output_tokens", or None if not needed.
        """

        max_batch = int(self.scheduler_config.max_num_seqs)
        if model_input is None:
            tokens = torch.zeros((max_batch, 1), dtype=torch.int32)
            positions = torch.full((max_batch, ), -1, dtype=torch.int32)
            block_tables = torch.zeros((max_batch, max_blocks_decode_batch),
                                       dtype=torch.int32)
            unpadded_batch_size = torch.tensor([0], dtype=torch.int32)
            # Create default sampling parameter tensors (max_batch sized)
            sampling_default_tensors = (
                self.input_batch.sampling.create_default_tensors())
            temperature = sampling_default_tensors["temperature"]
            top_k = sampling_default_tensors["top_k"]
            top_p = sampling_default_tensors["top_p"]
            presence_penalty = sampling_default_tensors["presence_penalty"]
            frequency_penalty = sampling_default_tensors["frequency_penalty"]
            repetition_penalty = sampling_default_tensors["repetition_penalty"]
            seed = sampling_default_tensors["seed"]
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
            unpadded_batch_size = torch.tensor(
                [cast(int, model_input.unpadded_batch_size)],
                dtype=torch.int32)
            sampling_params: TTSamplingParams = model_input.tt_sampling_params
            temperature = sampling_params.temperature
            top_k = sampling_params.top_k
            top_p = sampling_params.top_p
            presence_penalty = sampling_params.presence_penalty
            frequency_penalty = sampling_params.frequency_penalty
            repetition_penalty = sampling_params.repetition_penalty
            seed = sampling_params.seed

        # Pack into flattened tensors to reduce number of collectives.
        # B = max batch size, W = max_num_blocks_per_req.
        int_inputs = torch.cat(
            [
                tokens.contiguous().view(-1),  # B
                positions.contiguous().view(-1),  # B
                block_tables.contiguous().view(-1),  # B*W
                unpadded_batch_size.contiguous().view(-1),  # 1
                top_k.contiguous().view(-1),  # B
                seed.contiguous().view(-1),  # B
            ],
            dim=0).contiguous()

        if any_structured_inputs:
            if model_input is None or model_input.grammar_bitmask[0] is None:
                has_structured_inputs = torch.tensor([0], dtype=torch.int32)
                bitmasks = torch.zeros((max_batch, self.bitmask_size),
                                       dtype=torch.int32)
            else:
                has_structured_inputs = torch.tensor([1], dtype=torch.int32)
                bitmasks = model_input.grammar_bitmask[0]
            bitmasks = bitmasks.contiguous().view(-1)  # B * bitmask_size
            int_inputs = torch.cat(
                [int_inputs, has_structured_inputs, bitmasks],
                dim=0).contiguous()

        float_inputs = torch.cat(
            [
                temperature.contiguous().view(-1),  # B
                top_p.contiguous().view(-1),  # B
                presence_penalty.contiguous().view(-1),  # B
                frequency_penalty.contiguous().view(-1),  # B
                repetition_penalty.contiguous().view(-1),  # B
            ],
            dim=0).contiguous()

        sampling_tokens_inputs = None
        if any_penalties_inputs and model_input is not None:
            sampling_tokens_inputs = {
                "prompt_tokens": model_input.prompt_tokens,
                "output_tokens": model_input.output_tokens,
            }

        result = {
            "int_inputs": int_inputs,
            "float_inputs": float_inputs,
            "sampling_tokens_inputs": sampling_tokens_inputs,
        }

        return result

    def concat_dp_model_inputs(self, inputs, is_decode: bool,
                               max_blocks_decode_batch: Optional[int],
                               any_structured_inputs: bool) -> "TTModelInput":
        """
        Concatenate a DP-sized set of inputs into a single TTModelInput.
        inputs can be either:
        - For prefill: list[Optional[TTModelInput]]
        - For decode (optimized gather): dict[str, torch.Tensor] with keys:
          - "int_inputs": stacked int32 tensor of shape [world, -1]
          - "float_inputs": stacked float32 tensor of shape [world, -1]
          - "sampling_tokens_inputs":
            Optional[list[dict[str, torch.Tensor]]]
            Only provided when there are requests with penalties.
            One dict per DP rank, each with keys "prompt_tokens" and
            "output_tokens" (tensors padded with -1).
          - "reset_batch": bool for if the batch layout changed
            since the previous step.
          - "all_sample_device": bool for if all ranks can sample on device.
        """

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
            # Ints: [toks(B), positions(B), block_tables(B*W),
            #        bs(1), top_k(B), seed(B)]
            #   - If any_structured_inputs, also has at the end of the list:
            #     [has_structured_inputs(1), bitmasks(B*bitmask_size)]
            # Floats: [temperature(B), top_p(B), presence_penalty(B),
            #          frequency_penalty(B), repetition_penalty(B)]
            assert max_blocks_decode_batch is not None, (
                "max_blocks_decode_batch must be provided for decode")
            B = int(self.scheduler_config.max_num_seqs)
            W = max_blocks_decode_batch
            reset_batch = inputs["reset_batch"]
            perform_device_sampling = inputs["all_sample_device"]
            stacked_int: torch.Tensor = inputs["int_inputs"]
            stacked_float: torch.Tensor = inputs["float_inputs"]
            assert isinstance(stacked_int, torch.Tensor) and stacked_int.dim(
            ) == 2, "decode expects stacked int_inputs of shape [world, -1]"
            assert isinstance(stacked_float,
                              torch.Tensor) and stacked_float.dim() == 2, (
                                  "decode expects stacked float_inputs "
                                  "of shape [world, -1]")
            world = int(stacked_int.shape[0])
            total_B = world * B

            # Slice views out of the stacked gather buffers (no per-rank
            # Python lists, no torch.cat). Layout is constant for fixed B.
            off = 0
            input_tokens = stacked_int[:, off:off + B].reshape(total_B, 1)
            off += B
            input_positions = stacked_int[:, off:off + B].reshape(total_B)
            off += B

            max_bt_width = self.max_num_blocks_per_req
            if max_bt_width < W:
                raise ValueError(f"max_blocks_decode_batch={W} exceeds "
                                 f"max_num_blocks_per_req={max_bt_width}")
            block_tables_raw = stacked_int[:, off:off + B * W].reshape(
                total_B, W)
            off += B * W
            if max_bt_width == W:
                block_tables = block_tables_raw
            else:
                # Pad to constant width expected by TT backend.
                # Use new_zeros to match dtype/device of gathered tensors.
                block_tables = block_tables_raw.new_zeros(
                    (total_B, max_bt_width))
                block_tables[:, :W] = block_tables_raw

            bs_tensor = stacked_int[:, off]
            off += 1
            batch_size_per_dp = bs_tensor.tolist()

            top_k = stacked_int[:, off:off + B].reshape(total_B)
            off += B
            seed = stacked_int[:, off:off + B].reshape(total_B)
            off += B

            # Optional structured inputs: keep as list[Optional[tensor]]
            # per DP rank to match prefill behavior.
            grammar_bitmask_list = []
            if any_structured_inputs:
                has_structured = stacked_int[:, off]
                off += 1
                bitmasks = stacked_int[:, off:off +
                                       (B * self.bitmask_size)].reshape(
                                           world, B, self.bitmask_size)
                off += B * self.bitmask_size
                for r in range(world):
                    if int(has_structured[r].item()) > 0:
                        grammar_bitmask_list.append(bitmasks[r])
                    else:
                        grammar_bitmask_list.append(None)
            else:
                grammar_bitmask_list = [None] * world

            off_f = 0
            temperature = stacked_float[:, off_f:off_f + B].reshape(total_B)
            off_f += B
            top_p = stacked_float[:, off_f:off_f + B].reshape(total_B)
            off_f += B
            presence_penalty = stacked_float[:,
                                             off_f:off_f + B].reshape(total_B)
            off_f += B
            frequency_penalty = stacked_float[:,
                                              off_f:off_f + B].reshape(total_B)
            off_f += B
            repetition_penalty = stacked_float[:, off_f:off_f +
                                               B].reshape(total_B)
            off_f += B

            prompt_lens = None
        else:
            input_tokens_list: list[torch.Tensor] = []
            block_tables_list: list[torch.Tensor] = []
            input_positions_list: list[torch.Tensor] = [
            ]  # (prefix cache positions for prefill)
            prompt_lens_list: list[np.ndarray] = []
            batch_size_per_dp = []
            grammar_bitmask_list = []
            # Sampling parameters
            temperature_list: list[torch.Tensor] = []
            top_k_list: list[torch.Tensor] = []
            top_p_list: list[torch.Tensor] = []
            presence_penalty_list: list[torch.Tensor] = []
            frequency_penalty_list: list[torch.Tensor] = []
            repetition_penalty_list: list[torch.Tensor] = []
            seed_list: list[torch.Tensor] = []
            reset_batch = False

            active_inputs: list[TTModelInput] = [mi for mi in inputs if mi]
            if not active_inputs:
                raise ValueError("All inputs are None; nothing to concatenate")

            # Check if all ranks can sample on device.
            perform_device_sampling = all(mi.perform_device_sampling
                                          for mi in active_inputs)

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
                    input_positions_list.append(mi.input_positions)

                    # Extract sampling parameter tensors from TTSamplingParams
                    sp = mi.tt_sampling_params
                    temperature_list.append(sp.temperature)
                    top_k_list.append(sp.top_k)
                    top_p_list.append(sp.top_p)
                    presence_penalty_list.append(sp.presence_penalty)
                    frequency_penalty_list.append(sp.frequency_penalty)
                    repetition_penalty_list.append(sp.repetition_penalty)
                    seed_list.append(sp.seed)

                # We know it's not a list here before concatenation
                unpadded_batch_size: int = cast(
                    int, mi.unpadded_batch_size) if mi else 0
                batch_size_per_dp.append(unpadded_batch_size)
                grammar_bitmask_list.append(
                    mi.grammar_bitmask[0] if mi else None)

            input_tokens = torch.cat(input_tokens_list, dim=0)
            input_positions = np.concatenate(input_positions_list, axis=0)
            prompt_lens = np.concatenate(prompt_lens_list, axis=0)
            block_tables = torch.cat(block_tables_list, dim=0)

            # Concatenate sampling parameter tensors across DP ranks
            temperature = torch.cat(temperature_list, dim=0)
            top_k = torch.cat(top_k_list, dim=0)
            top_p = torch.cat(top_p_list, dim=0)
            presence_penalty = torch.cat(presence_penalty_list, dim=0)
            frequency_penalty = torch.cat(frequency_penalty_list, dim=0)
            repetition_penalty = torch.cat(repetition_penalty_list, dim=0)
            seed = torch.cat(seed_list, dim=0)

        tt_sampling_params = TTSamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )

        if self.model_config.is_multimodal_model and not is_decode:
            # Gather multi-modal inputs from all DP ranks
            multi_modal_kwargs: MultiModalKwargs = {
                "pixel_values": [],
                "image_grid_thw": []
            }
            pixel_values = []
            image_grid_thw = []
            for mi in inputs:
                if mi is not None:
                    for pv in mi.multi_modal_kwargs["pixel_values"]:
                        pixel_values.append(pv)
                    for ig in mi.multi_modal_kwargs["image_grid_thw"]:
                        image_grid_thw.append(ig)
            multi_modal_kwargs["pixel_values"] = pixel_values
            multi_modal_kwargs["image_grid_thw"] = image_grid_thw
        else:
            multi_modal_kwargs = {}

        # Extract prompt and output tokens for decode with sampling penalties
        prompt_tokens = None
        output_tokens = None
        sampling_tokens_inputs = inputs.get(
            "sampling_tokens_inputs") if is_decode else None
        if sampling_tokens_inputs:
            # Find max shapes across all ranks
            max_prompt_len = 0
            max_output_len = 0
            for rank_tokens_dict in sampling_tokens_inputs:
                if rank_tokens_dict is not None:
                    rank_prompt_tokens = rank_tokens_dict.get("prompt_tokens")
                    rank_output_tokens = rank_tokens_dict.get("output_tokens")
                    if rank_prompt_tokens is not None:
                        assert rank_output_tokens is not None
                        max_prompt_len = max(max_prompt_len,
                                             rank_prompt_tokens.shape[1])
                        max_output_len = max(max_output_len,
                                             rank_output_tokens.shape[1])

            # Create tensors with shape (max_num_reqs * DP_size, max_len)
            max_num_reqs = int(self.scheduler_config.max_num_seqs)
            total_batch_size = max_num_reqs * len(sampling_tokens_inputs)

            # Create prompt and output tokens tensors
            prompt_tokens = torch.full((total_batch_size, max_prompt_len),
                                       -1,
                                       dtype=torch.int32)
            output_tokens = torch.full((total_batch_size, max_output_len),
                                       -1,
                                       dtype=torch.int32)
            for rank_idx, rank_tokens_dict in enumerate(
                    sampling_tokens_inputs):
                if rank_tokens_dict is not None:
                    start_idx = rank_idx * max_num_reqs
                    rank_prompt_tokens = rank_tokens_dict.get("prompt_tokens")
                    rank_output_tokens = rank_tokens_dict.get("output_tokens")
                    if rank_prompt_tokens is not None:
                        assert rank_output_tokens is not None
                        end_idx = start_idx + rank_prompt_tokens.shape[0]
                        prompt_padded_len = rank_prompt_tokens.shape[1]
                        output_padded_len = rank_output_tokens.shape[1]
                        prompt_tokens[start_idx:end_idx, :
                                      prompt_padded_len] = rank_prompt_tokens
                        output_tokens[start_idx:end_idx, :
                                      output_padded_len] = rank_output_tokens

        if os.environ.get("DP_GATHER_DEBUG") == "1":
            logger.info("batch_size_per_dp=%s", batch_size_per_dp)
        merged = TTModelInput(
            input_tokens=input_tokens,
            input_positions=input_positions,
            prompt_lens=prompt_lens,
            block_tables=block_tables,
            unpadded_batch_size=batch_size_per_dp,
            tt_sampling_params=tt_sampling_params,
            multi_modal_kwargs=multi_modal_kwargs,
            perform_device_sampling=perform_device_sampling,
            grammar_bitmask=grammar_bitmask_list,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            reset_batch=reset_batch,
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

    def check_perform_device_sampling(self, is_decode: bool) -> bool:
        want_device_sampling = self.sample_on_device_mode == "all" or (
            self.sample_on_device_mode == "decode_only" and is_decode)
        if not want_device_sampling:
            return False

        # Currently requests with logprobs fail on request
        # validation, but once supported, the TODOs below will be relevant.
        # TODO: Also if logprobs are not None,
        # TTPlatform.non_greedy_decoding_on_device must be True
        # (model limitations).
        # TODO: Also if logprobs is not None and devices_per_dp_cache == 1,
        # logprobs on device is not supported
        # (https://github.com/tenstorrent/tt-metal/issues/34077).
        params_device_supported = TTPlatform.non_greedy_decoding_on_device or (
            self.input_batch.all_greedy and self.input_batch.no_penalties)
        return params_device_supported

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

        kwargs = {
            "tokens": model_input.input_tokens,
            "page_table": model_input.block_tables,
            "kv_cache": self.kv_caches,
        }

        if not is_decode:
            kwargs["enable_trace"] = self.trace_mode in ["all"]
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

        kwargs["start_pos"] = model_input.input_positions

        # Sampling decision
        sampling_params = model_input.tt_sampling_params
        perform_device_sampling = model_input.perform_device_sampling
        if perform_device_sampling:
            # On-device sampling currently needs sampling param attributes to
            # be lists instead of tensors.
            sampling_param_dict = {
                field.name:
                (getattr(sampling_params, field.name).tolist() if getattr(
                    sampling_params, field.name) is not None else None)
                for field in fields(sampling_params)
            }
            # Convert seed sentinel value back to None
            # (vLLM treats -1 as equivalent to None for seeds)
            sampling_param_dict["seed"] = [
                None if s == SEED_NONE_SENTINEL else s
                for s in sampling_param_dict["seed"]
            ]
            kwargs["sampling_params"] = TTSamplingParams(**sampling_param_dict)

            # Pass prompt and output tokens for decode with sampling penalties
            if is_decode and model_input.prompt_tokens is not None:
                assert model_input.output_tokens is not None
                kwargs["prompt_tokens"] = model_input.prompt_tokens
                kwargs["output_tokens"] = model_input.output_tokens

            # For decode on-device sampling, signal whether the decode batch
            # layout changed since the previous step.
            if is_decode:
                # reset_batch is precomputed (for non-DP in
                # _prepare_model_inputs; for DP after inputs are gathered).
                kwargs["reset_batch"] = model_input.reset_batch

        # Execute model
        if not is_decode:
            if self.request_specific_rope:
                tt_out, rope_deltas = self.model.prefill_forward(**kwargs)
                # Store rope_deltas for each prefilled request
                for i, req_id in enumerate(self.input_batch.req_ids):
                    self.requests[req_id].mrope_position_delta = \
                        rope_deltas[i].item()
            else:
                tt_out = self.model.prefill_forward(**kwargs)
        else:
            # TODO: Add encoder-decoder support
            enc_dec_kwargs: dict[str, Any] = {}
            if self.request_specific_rope:
                if any(req_id not in self.previous_req_ids
                       for req_id in self.input_batch.req_ids):
                    # Gather and pass rope_deltas from prefill step to decode
                    enc_dec_kwargs = {
                        "rope_deltas_all_users": [
                            self.requests[req_id].mrope_position_delta
                            for req_id in self.input_batch.req_ids
                        ]
                    }
                else:
                    enc_dec_kwargs = {"rope_deltas_all_users": None}
                self.previous_req_ids = set(self.input_batch.req_ids)

            enable_trace = self.trace_mode in ["all", "decode_only"]
            # In the DP case, the model outputs for all ranks are concatenated.
            tt_out = self.model.decode_forward(**kwargs,
                                               **enc_dec_kwargs,
                                               enable_trace=enable_trace,
                                               read_from_device=True)
            # tt_out is a tuple of (logits, logprobs)
            # v1 currently doesn't handle logprobs from TT models
            if isinstance(tt_out, tuple):
                tt_out = tt_out[0]

        return self._get_output_tokens(
            tt_out=tt_out,
            sampling_params=sampling_params,
            model_input=model_input,
            batch_size_per_dp=batch_size_per_dp,
            perform_device_sampling=perform_device_sampling,
            is_decode=is_decode,
        )

    def _get_output_tokens(
        self,
        tt_out: torch.Tensor,
        sampling_params: TTSamplingParams,
        model_input: TTModelInput,
        batch_size_per_dp: list[int],
        perform_device_sampling: bool,
        is_decode: bool,
    ) -> list[torch.Tensor]:
        """Return sampled tokens per DP rank using concatenated model
        outputs.
        
        If perform_device_sampling is True, tokens are already sampled on
        device. Otherwise, sample on host using host_sampler.
        """
        sampled_token_ids_per_dp: list[torch.Tensor] = []

        start = 0
        for dp_rank, sz in enumerate(batch_size_per_dp):
            if sz <= 0:
                sampled_token_ids_per_dp.append(
                    torch.tensor([], dtype=torch.int32))
                if is_decode:
                    # Fixed stride segments per DP rank for decode
                    start += self.scheduler_config.max_num_seqs
                continue
            if not perform_device_sampling:
                logits = tt_out[start:start + sz, -1, :]

                grammar_bitmask = model_input.grammar_bitmask[dp_rank]

                if grammar_bitmask is not None:
                    # match shape of logits, which are now unpadded on batch dim
                    grammar_bitmask = grammar_bitmask[:sz, :]
                    self.apply_grammar_bitmask(logits, grammar_bitmask)

                # Extract sampling params for this DP rank from concatenated
                # tensors.
                assert isinstance(sampling_params.temperature, torch.Tensor)
                assert isinstance(sampling_params.top_k, torch.Tensor)
                assert isinstance(sampling_params.top_p, torch.Tensor)
                assert isinstance(sampling_params.presence_penalty,
                                  torch.Tensor)
                assert isinstance(sampling_params.frequency_penalty,
                                  torch.Tensor)
                assert isinstance(sampling_params.repetition_penalty,
                                  torch.Tensor)
                assert isinstance(sampling_params.seed, torch.Tensor)
                temperature = sampling_params.temperature[start:start + sz]
                top_k = sampling_params.top_k[start:start + sz]
                top_p = sampling_params.top_p[start:start + sz]
                presence_penalty = sampling_params.presence_penalty[
                    start:start + sz]
                frequency_penalty = sampling_params.frequency_penalty[
                    start:start + sz]
                repetition_penalty = sampling_params.repetition_penalty[
                    start:start + sz]
                seed = sampling_params.seed[start:start + sz]

                # Determine if all greedy (temperature == 0.0) or all random
                all_greedy = (temperature == 0.0).all().item()
                all_random = (temperature != 0.0).all().item()

                # Create generators from seeds for this DP rank
                # Generator keys are batch indices (0-based within current
                # slice).
                generators: dict[int, torch.Generator] = {}
                for i, seed_val in enumerate(seed):
                    if seed_val.item() != SEED_NONE_SENTINEL:
                        generators[i] = torch.Generator(
                            device="cpu").manual_seed(seed_val.item())

                # Determine if penalties are needed
                no_penalties = ((presence_penalty == 0.0).all().item()
                                and (frequency_penalty == 0.0).all().item()
                                and (repetition_penalty == 1.0).all().item())

                # Output history as list[list[int]] (filter TT -1 padding).
                output_token_ids: list[list[int]] = []
                if is_decode and model_input.output_tokens is not None:
                    output_tokens = model_input.output_tokens[start:start + sz]
                    for i in range(sz):
                        output_tokens_i = output_tokens[i].tolist()
                        output_token_ids.append(
                            [tok for tok in output_tokens_i if tok != -1])
                else:
                    output_token_ids = [[] for _ in range(sz)]

                # Prompt tokens for penalties: must be int64 and padded with a
                # valid index (vocab_size), not TT's -1 sentinel.
                prompt_token_ids: Optional[torch.Tensor] = None
                if not no_penalties:
                    if is_decode and model_input.prompt_tokens is not None:
                        prompt_token_ids = model_input.prompt_tokens[
                            start:start + sz].to(torch.int64)
                        prompt_token_ids = prompt_token_ids.masked_fill(
                            prompt_token_ids == -1, self.vocab_size)
                    elif not is_decode:
                        prompt_token_ids = model_input.input_tokens[
                            start:start + sz].to(torch.int64)
                        assert model_input.prompt_lens is not None
                        prompt_lens_t = torch.as_tensor(
                            model_input.prompt_lens[start:start + sz],
                            dtype=torch.int64,
                        )
                        positions = torch.arange(
                            prompt_token_ids.shape[1], ).unsqueeze(0)
                        pad_mask = positions >= prompt_lens_t.unsqueeze(1)
                        prompt_token_ids = prompt_token_ids.masked_fill(
                            pad_mask, self.vocab_size)

                # Create SamplingMetadata for this DP rank
                # TODO: support logprobs
                sampling_metadata = SamplingMetadata(
                    temperature=temperature if not all_greedy else None,
                    all_greedy=all_greedy,
                    all_random=all_random,
                    top_p=top_p,
                    top_k=top_k,
                    generators=generators,
                    max_num_logprobs=None,
                    no_penalties=no_penalties,
                    prompt_token_ids=prompt_token_ids,
                    frequency_penalties=frequency_penalty,
                    presence_penalties=presence_penalty,
                    repetition_penalties=repetition_penalty,
                    output_token_ids=output_token_ids,
                    allowed_token_ids_mask=None,
                    bad_words_token_ids={},
                    logitsprocs=LogitsProcessorManager(),
                )

                sampler_output = self.host_sampler(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                )
                next_token_ids = sampler_output.sampled_token_ids
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
        num_reqs = self.input_batch.num_reqs
        assert sampled_token_ids.shape[0] == num_reqs, (
            f"Number of request outputs {sampled_token_ids.shape[0]} != "
            f"number of requests in input batch {num_reqs}")
        num_out_tokens = sampled_token_ids.shape[1]
        assert num_out_tokens == 1, "Currently only supporting 1 output token"

        sampled_token_ids_np = sampled_token_ids.view(num_reqs).numpy()

        # Vectorized update of persistent batch token storage.
        start_idxs = self.input_batch.num_tokens[:num_reqs]
        end_idxs = start_idxs + 1
        max_end = int(end_idxs.max()) if num_reqs > 0 else 0
        assert max_end <= self.model_config.max_model_len, (
            "Sampled token IDs exceed the max model length. "
            f"Total number of tokens: {max_end} > max_model_len: "
            f"{self.model_config.max_model_len}")

        rows = np.arange(num_reqs)
        self.input_batch.token_ids_cpu[rows, start_idxs] = sampled_token_ids_np
        self.input_batch.num_tokens[:num_reqs] = end_idxs

        # Update request state (output token lists) without dict lookups.
        # NOTE: `InputBatch.req_output_token_ids[i]` is a direct reference to
        # the underlying `CachedRequestState.output_token_ids` list (stored in
        # `self.requests[req_id]`). Appending here updates request state too,
        # while avoiding a per-request dict lookup.
        sampled_token_ids_list_1d = sampled_token_ids_np.tolist()
        for req_idx in range(num_reqs):
            output_token_ids = self.input_batch.req_output_token_ids[req_idx]
            assert output_token_ids is not None
            output_token_ids.append(sampled_token_ids_list_1d[req_idx])

        # Empty prompt log probs
        prompt_logprobs_dict: dict[str,
                                   Optional[LogprobsTensors]] = (dict.fromkeys(
                                       self.input_batch.req_ids[:num_reqs],
                                       None))

        # Note: currently does not support speculative decoding, log probs,
        # or pooling.
        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[[t] for t in sampled_token_ids_list_1d],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
        )

    def warmup_model(self) -> None:
        trace_prefill_mode = self.trace_mode in ["all"]
        prefill_warmup(self.model, self.kv_caches, trace_prefill_mode,
                       self.scheduler_config.max_num_seqs,
                       self.parallel_config.data_parallel_size)

        trace_decode_mode = self.trace_mode in ["all", "decode_only"]
        decode_warmup(self.model, self.kv_caches, trace_decode_mode,
                      self.scheduler_config.max_num_seqs,
                      self.max_num_blocks_per_req, self.sample_on_device_mode,
                      self.parallel_config.data_parallel_size)
