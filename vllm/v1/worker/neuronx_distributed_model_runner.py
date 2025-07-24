# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from dataclasses import dataclass, field
from typing import Optional

import torch
from neuronx_distributed_inference.modules.generation.sampling import (
    prepare_sampling_params)

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import BatchedTensorInputs, MultiModalKwargs
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput,
                             SamplerOutput)
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.neuronx_distributed_model_loader import get_neuron_model

logger = init_logger(__name__)


@dataclass(frozen=True)
class ModelInputForNeuron:
    """
    Model input for NeuronX Distributed Inference model runner.
    """
    request_ids: Optional[list[str]] = None
    input_tokens: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    input_block_ids: Optional[torch.Tensor] = None
    slot_mapping: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None
    full_context_lens: Optional[torch.Tensor] = None
    computed_context_lens: Optional[torch.Tensor] = None
    sampling_params: Optional[torch.Tensor] = None
    multi_modal_kwargs: BatchedTensorInputs = None
    adapter_ids: Optional[str] = None
    # Boolean tensor to indicate if the request is ready
    # for sampling. Needed by chunked prefill.
    prefill_completion_state: Optional[torch.Tensor] = None


# This class is used for constructing ModelInputForNeuron and
# is not frozen.
@dataclass
class IntermediateInputData:
    request_ids: list[str] = field(default_factory=list)
    input_tokens: list[int] = field(default_factory=list)
    position_ids: list[int] = field(default_factory=list)
    input_block_ids: list[int] = field(default_factory=list)
    full_context_lens: list[int] = field(default_factory=list)
    computed_context_lens: list[int] = field(default_factory=list)
    slot_mapping: list[int] = field(default_factory=list)
    block_tables: list[list[int]] = field(default_factory=list)
    prefill_completion_state: list[bool] = field(default_factory=list)


class NeuronxDistributedModelRunner(LoRAModelRunnerMixin):
    # NEURON has an upper limit on the top_k
    _MAX_NEURON_SAMPLING_TOP_K = 256

    # NOTE: Padding table id for slot mapping, note that this will be
    # used as the block index to update KV cache, so we need to make
    # sure no real tokens are mapped to this block_id, we current
    # assume that block 0 will never be used.
    _SLOT_MAPPING_PAD = -1
    _BLOCK_TABLE_PAD = 0

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
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

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        self.device = device

        self.pin_memory = False
        self.block_size = cache_config.block_size
        self.max_num_reqs = scheduler_config.max_num_seqs
        self.max_model_len = model_config.max_model_len
        self.max_num_tokens = scheduler_config.max_num_batched_tokens

        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
        )

        self.requests: dict[str, CachedRequestState] = {}
        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: dict[str, dict[int, torch.Tensor]] = {}

        self.lora_checkpoint = None
        self.model = None
        self.lora_serving_config = None

        self.is_block_kv_layout = True

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig):
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        # Not required for NxD Inference.
        return

    def load_model(self) -> None:
        self.model = get_neuron_model(
            self.model_config,
            cache_config=self.cache_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            lora_serving_config=self.lora_serving_config)
        self.model.is_reorder_needed = not self.is_block_kv_layout

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> ModelRunnerOutput:
        logger.debug("scheduler_output: %s", scheduler_output)

        # Update cached state
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOutput if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT

        # _prepare_model_input converts the scheduler output to
        # ModelInputForNeuron
        model_input = self._prepare_model_input(scheduler_output)
        logger.debug("model_input: %s", model_input)

        is_mllama = self.model.architecture == "MllamaForConditionalGeneration"

        if is_mllama:
            sampler_outputs = self._execute_model_for_mllama(
                model_input,
                intermediate_tensors,
            )
        else:
            sampler_outputs = self._execute_model_for_text(
                model_input,
                intermediate_tensors,
            )

        return self._generate_model_runner_output(sampler_outputs)

    def _generate_model_runner_output(
            self, sampler_outputs: Optional[list[SamplerOutput]]
    ) -> ModelRunnerOutput:
        if sampler_outputs is None:
            return EMPTY_MODEL_RUNNER_OUTPUT

        sampled_token_ids = sampler_outputs[0].sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]

        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            raise NotImplementedError("spec decode is not supported yet")

        for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            # TODO: support the following fields. currently they
            # are hardcoded to None
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            finished_sending=None,
            finished_recving=None,
            pooler_output=[])

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        return {
            "layer":
            FullAttentionSpec(
                block_size=self.block_size,
                num_kv_heads=self.model.num_key_value_heads,
                head_size=self.model.head_dim,
                # TODO: take the following from the model config
                dtype=torch.bfloat16,
                use_mla=False,
                sliding_window=None,
            )
        }

    def _update_states(self, scheduler_output: "SchedulerOutput") -> bool:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        Returns:
            True if there is a new/resumed/paused/finished request.
            If False, we can skip copying SamplingMetadata to the GPU.
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
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

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
            self.input_batch.remove_request(req_id)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert new_req_data.sampling_params is not None,\
                "Pooling is not supported yet"
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
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            self.input_batch.add_request(req_state)

        self.input_batch.condense()
        self.input_batch.refresh_metadata()

        return len(unscheduled_req_ids) > 0 or len(req_ids_to_add) > 0

    def _execute_model_for_text(
        self,
        model_input: ModelInputForNeuron,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Optional[list[SamplerOutput]]:
        hidden_states = self.model(
            input_ids=model_input.input_tokens,
            position_ids=model_input.position_ids,
            input_block_ids=model_input.input_block_ids,
            slot_mapping=model_input.slot_mapping,
            block_tables=model_input.block_tables,
            full_context_lens=model_input.full_context_lens,
            computed_context_lens=model_input.computed_context_lens,
            sampling_params=model_input.sampling_params,
            adapter_ids=model_input.adapter_ids,
            prefill_completion_state=model_input.prefill_completion_state,
            **MultiModalKwargs.as_kwargs(model_input.multi_modal_kwargs or {},
                                         device=self.device),
        )

        sampled_output = self._sample(hidden_states, model_input)
        return [sampled_output]

    def _execute_model_for_mllama(
        self,
        model_input: ModelInputForNeuron,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Optional[list[SamplerOutput]]:

        raise NotImplementedError("MLLAMA is not supported yet")

    def _prepare_model_input(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelInputForNeuron:
        chunked_prefill_model_input = self._prepare_chunked_prefill_inputs(
            scheduler_output)

        multi_modal_kwargs = None
        lora_adapter_ids = None

        return self._finalize_chunked_prefill_inputs(
            chunked_prefill_model_input,
            multi_modal_kwargs,
            lora_adapter_ids,
        )

    def _prepare_chunked_prefill_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> IntermediateInputData:
        """
        This function is used to prepare the inputs for chunked prefill.
        It needs to treat prefill and decoding requests differently.
          *  For NewRequestData, it is guaranteed to be a prefill request.
          *  For CachedRequestData, it can be a prefill request or a 
          decoding request. 
        """
        data = IntermediateInputData()
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        logger.debug("num_scheduled_tokens: %s", num_scheduled_tokens)

        for request_data in scheduler_output.scheduled_new_reqs:
            self._process_new_request_for_chunked_prefill(
                request_data, num_scheduled_tokens[request_data.req_id], data)

        request_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(request_data.req_ids):
            self._process_cached_request_for_chunked_prefill(
                request_data, i, num_scheduled_tokens[req_id], data)

        return data

    def _process_new_request_for_chunked_prefill(
            self, request_data: NewRequestData, num_scheduled_tokens: int,
            data: IntermediateInputData) -> None:
        data.request_ids.append(request_data.req_id)
        assert len(request_data.block_ids) == 1
        block_table = copy.deepcopy(request_data.block_ids)[0]

        start = request_data.num_computed_tokens
        end = start + num_scheduled_tokens

        data.input_tokens.extend(request_data.prompt_token_ids[start:end])
        data.position_ids.extend(range(start, end))
        data.input_block_ids.append(0)

        for i in range(start, end):
            block_number = block_table[i // self.cache_config.block_size]
            offset = i % self.cache_config.block_size
            data.slot_mapping.append(block_number *
                                     self.cache_config.block_size + offset)

        data.block_tables.append(block_table)
        data.full_context_lens.append(end)
        data.computed_context_lens.append(start)
        data.prefill_completion_state.append(
            end >= len(request_data.prompt_token_ids))

    def _process_cached_request_for_chunked_prefill(
            self, request_data: CachedRequestData, index: int,
            num_scheduled_tokens: int, data: IntermediateInputData) -> None:
        req_id = request_data.req_ids[index]
        data.request_ids.append(req_id)
        state = self.requests[req_id]
        block_table = copy.deepcopy(state.block_ids)[0]

        start = request_data.num_computed_tokens[index]
        end = start + num_scheduled_tokens

        if num_scheduled_tokens > 1:
            resumed_prompt_tokens = state.prompt_token_ids[start:end - 1]
            data.input_tokens.extend(resumed_prompt_tokens)

        data.input_tokens.append(state.output_token_ids[-1])
        data.position_ids.extend(range(start, end))
        data.input_block_ids.append(0)

        for i in range(start, end):
            block_number = block_table[i // self.cache_config.block_size]
            offset = i % self.cache_config.block_size
            data.slot_mapping.append(block_number *
                                     self.cache_config.block_size + offset)

        data.block_tables.append(block_table)
        data.full_context_lens.append(end)
        data.computed_context_lens.append(start)
        data.prefill_completion_state.append(
            end >= len(state.prompt_token_ids))

    def _finalize_chunked_prefill_inputs(
        self,
        data: IntermediateInputData,
        multi_modal_kwargs: BatchedTensorInputs,
        lora_adapter_ids: Optional[str],
    ) -> ModelInputForNeuron:
        device = self.device

        input_tokens = torch.tensor(data.input_tokens,
                                    dtype=torch.long,
                                    device=device).reshape(1, -1)
        position_ids = torch.tensor(data.position_ids,
                                    dtype=torch.long,
                                    device=device).reshape(1, -1)
        input_block_ids = torch.tensor(data.input_block_ids[:1],
                                       dtype=torch.long,
                                       device=device)
        slot_mapping = torch.tensor(data.slot_mapping,
                                    dtype=torch.long,
                                    device=device)

        max_blocks = max(len(b) for b in data.block_tables)
        for b in data.block_tables:
            b.extend([self._BLOCK_TABLE_PAD] * (max_blocks - len(b)))

        block_tables = torch.tensor(data.block_tables,
                                    dtype=torch.long,
                                    device=device)
        full_context_lens = torch.tensor(data.full_context_lens,
                                         dtype=torch.long,
                                         device=device)
        computed_context_lens = torch.tensor(data.computed_context_lens,
                                             dtype=torch.long,
                                             device=device)
        prefill_completion_state = torch.tensor(data.prefill_completion_state,
                                                dtype=torch.bool,
                                                device=device)

        return ModelInputForNeuron(
            request_ids=data.request_ids,
            input_tokens=input_tokens,
            position_ids=position_ids,
            input_block_ids=input_block_ids,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            full_context_lens=full_context_lens,
            computed_context_lens=computed_context_lens,
            prefill_completion_state=prefill_completion_state,
            sampling_params=self.get_nxd_sampling_params(input_tokens),
            multi_modal_kwargs=multi_modal_kwargs,
            adapter_ids=lora_adapter_ids,
        )

    def _sample(
        self,
        hidden_states: torch.Tensor,
        model_input: ModelInputForNeuron,
    ):
        # The following logic reorders the model output to match the
        # incoming request order. First obtain the order of requests
        # processed by Neuron hardware
        request_id_order = {
            request_id: idx
            for idx, request_id in enumerate(model_input.request_ids)
        }

        # Identify the correct indices for each request in the original
        # input batch based on request ids
        reorder_indices = torch.tensor([
            request_id_order[request_id]
            for request_id in self.input_batch.req_ids
        ])

        # Reorder along the batch dimension to restore outputs into the
        # original request order
        hidden_states = hidden_states[reorder_indices]

        # Sample the next token.
        output = self.model.sample(logits=hidden_states, )
        return output

    def get_nxd_sampling_params(self, input_ids: torch.Tensor):
        if self.model.neuron_config.on_device_sampling_config:
            max_topk = (
                self.model.neuron_config.on_device_sampling_config.global_topk)
        else:
            max_topk = self.model.neuron_config.vocab_size

        max_topk = min(max_topk, self._MAX_NEURON_SAMPLING_TOP_K)

        top_k = [1] * self.scheduler_config.max_num_seqs
        top_p = [1.0] * self.scheduler_config.max_num_seqs
        temperature = [1.0] * self.scheduler_config.max_num_seqs

        for index, request in enumerate(self.requests.values()):
            top_k[index] = (request.sampling_params.top_k
                            if request.sampling_params.top_k > 0
                            and request.sampling_params.top_k < max_topk else
                            max_topk)
            top_p[index] = request.sampling_params.top_p
            temperature[index] = request.sampling_params.temperature
            if request.sampling_params.temperature == 0.0:
                top_k[index] = 1
                temperature[index] = 1.0

        sampling_params = prepare_sampling_params(
            batch_size=self.scheduler_config.max_num_seqs,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature)

        return sampling_params
