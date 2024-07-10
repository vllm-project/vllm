from typing import List, Optional

import torch

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, MultiModalConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig)
from vllm.logger import init_logger
from vllm.sequence import (IntermediateTensors, SamplerOutput,
                           SequenceGroupMetadata)
from vllm.worker.model_runner import (ModelInputForGPUWithSamplingMetadata,
                                      ModelRunner)

from vllm.attention.backends.flash_attn import FlashAttentionMetadata

from vllm import _custom_ops as ops

logger = init_logger(__name__)

log_advance_input = False

class TP1DraftModelRunner(ModelRunner):
    """Specialized model runner for speculative decoding draft model.
    Since the draft model always execute k forward passes consecutively to
    generate k speculative tokens in a single speculative decoding step,
    we could get rid of most CPU-GPU synchronization and data transfer
    overheads by keeping model input and output tensors on GPU all the time.

    This runner is still under development so there's no performance gain
    at this moment. Currently we adopt a temporary solution that caches the
    seq_group_metadata_list for multi-step execution, so that we can
    leverage existing prepare_model_input to be compatible with the current
    execution flow, but we plan to remove this cache and avoid calling
    prepare_model_input in execute_model at all.
    
    The detail development plan includes:
    1. Use "update_model_input" to update existing model_input without
       creating a new one.
    2. Improve the performance of "update_model_input" with a GPU kernel.
    3. Support TP > 1 (this requires some designs because we do not expect
       any broadcasting inside execute_model).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        multimodal_config: Optional[MultiModalConfig] = None,
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        return_hidden_states: bool = False,
    ):
        if return_hidden_states:
            raise ValueError(
                "return_hidden_states is not supported for TP1DraftModelRunner."
            )

        super().__init__(
            model_config=model_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            cache_config=cache_config,
            load_config=load_config,
            lora_config=lora_config,
            kv_cache_dtype=kv_cache_dtype,
            is_driver_worker=is_driver_worker,
            multimodal_config=multimodal_config,
            prompt_adapter_config=prompt_adapter_config,
            return_hidden_states=return_hidden_states,
        )

        # TODO: Remove this cache when we are able to update model_input
        # directly in advance_step.
        self.cached_seq_group_metadata_list: Optional[
            List[SequenceGroupMetadata]] = None

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForGPUWithSamplingMetadata:
        """A temporary solution that caches the seq_group_metadata_list
        for multi-step execution.
        TODO: In-place update model_input and remove this function.
        """
        self.cached_seq_group_metadata_list = seq_group_metadata_list
        return super().prepare_model_input(
            seq_group_metadata_list,
            finished_requests_ids=finished_requests_ids)

    def update_model_input(
            self, model_input: ModelInputForGPUWithSamplingMetadata,
            last_output: SamplerOutput
    ) -> ModelInputForGPUWithSamplingMetadata:
        """Prepare the model inputs for the next step.
        TODO: In-place update model_input instead of calling
        prepare_model_input.
        """

        # Append the output token to the sequence data.
        assert self.cached_seq_group_metadata_list is not None
        for seq_group_metadata, sequence_group_outputs in zip(
                self.cached_seq_group_metadata_list, last_output.outputs):
            seq_group_metadata.is_prompt = False

            for seq_output in sequence_group_outputs.samples:
                seq = seq_group_metadata.seq_data[seq_output.parent_seq_id]

                token_id = seq_output.output_token
                token_logprob = seq_output.logprobs[token_id]

                seq.append_token_id(token_id, token_logprob.logprob)
                seq.update_num_computed_tokens(1)

                if log_advance_input:
                    print("appended seq_id = {} token_id = {}".format(
                        seq_output.parent_seq_id, token_id))

        return self.prepare_model_input(self.cached_seq_group_metadata_list)

    def _update_seq_group_metadata(self, seq_group_metadata_list, last_output):
        for seq_group_metadata, sequence_group_outputs in zip(
                seq_group_metadata_list, last_output.outputs):
            seq_group_metadata.is_prompt = False

            for seq_output in sequence_group_outputs.samples:
                seq = seq_group_metadata.seq_data[seq_output.parent_seq_id]

                token_id = seq_output.output_token
                token_logprob = seq_output.logprobs[token_id]

                seq.append_token_id(token_id, token_logprob.logprob)
                seq.update_num_computed_tokens(1)

                if log_advance_input:
                    print("appended seq_id = {} token_id = {}".format(
                        seq_output.parent_seq_id, token_id))

    def _update_flash_attn_metadata(self, attn_metadata, num_seqs):
        assert isinstance(attn_metadata, FlashAttentionMetadata)

        assert attn_metadata.num_prefills == 0
        assert attn_metadata.num_prefill_tokens == 0
        assert attn_metadata.num_decode_tokens == num_seqs
        assert attn_metadata.slot_mapping.shape == (num_seqs, )

        assert len(attn_metadata.seq_lens) == num_seqs
        assert attn_metadata.seq_lens_tensor.shape == (num_seqs, )
        assert attn_metadata.max_query_len == 1
        assert attn_metadata.max_prefill_seq_len == 0
        assert attn_metadata.max_decode_seq_len == max(
            attn_metadata.seq_lens), "{} {}".format(
                attn_metadata.max_decode_seq_len, max(attn_metadata.seq_lens))

        assert attn_metadata.query_start_loc.shape == (num_seqs + 1, )
        assert attn_metadata.seq_start_loc.shape == (num_seqs + 1, )

        assert attn_metadata.context_lens_tensor.shape == (num_seqs, )

        assert attn_metadata.block_tables.shape[0] == num_seqs
        assert attn_metadata.use_cuda_graph == False

        # Update seq_lens
        for i in range(num_seqs):
            attn_metadata.seq_lens[i] += 1
        attn_metadata.max_decode_seq_len = max(attn_metadata.seq_lens)

    def _update_sampling_metadata(self, sampling_metadata, num_seqs):

        assert sampling_metadata.num_prompts == 0
        assert len(sampling_metadata.seq_groups) == num_seqs
        assert sampling_metadata.selected_token_indices.shape == (num_seqs, )
        # assert sampling_metadata.categorized_sample_indices == TODO: Add if needed

        for i in range(num_seqs):
            seq_group = sampling_metadata.seq_groups[i]

            assert seq_group.is_prompt == False  # No prompt
            assert seq_group.prompt_logprob_indices == []  # No prompt
            assert seq_group.sample_indices == [i]  # Simple
            assert seq_group.seq_len == None  # Decode
            assert seq_group.query_len == None  # Decode

    def _advance_step(
            self, model_input: ModelInputForGPUWithSamplingMetadata,
            last_output: SamplerOutput
    ) -> ModelInputForGPUWithSamplingMetadata:
        if log_advance_input:
            print("Inside _advance_step")

        # Append output tokens
        self._update_seq_group_metadata(self.cached_seq_group_metadata_list,
                                        last_output)

        # Get num_seqs
        num_seqs = len(model_input.seq_lens)

        # Get output tokens GPU tensor
        sampled_token_ids = last_output.sampled_token_ids.squeeze(dim=0)

        # Update attn_metadata
        attn_metadata = model_input.attn_metadata
        assert isinstance(attn_metadata, FlashAttentionMetadata)
        self._update_flash_attn_metadata(attn_metadata, num_seqs)

        # Update GPU tensors
        ops.advance_step(num_seqs=num_seqs,
                         block_size=self.block_size,
                         sampled_token_ids=sampled_token_ids,
                         input_positions=model_input.input_positions,
                         seq_lens=attn_metadata.seq_lens_tensor,
                         slot_mapping=attn_metadata.slot_mapping,
                         block_tables=attn_metadata.block_tables)

        # Update sampling_metadata
        sampling_metadata = model_input.sampling_metadata
        self._update_sampling_metadata(sampling_metadata, num_seqs)

        # Create new input
        new_model_input = self._model_input_cls(
            input_tokens=sampled_token_ids,
            input_positions=model_input.input_positions,
            attn_metadata=attn_metadata,
            seq_lens=attn_metadata.seq_lens,
            query_lens=model_input.query_lens,
            lora_mapping=model_input.lora_mapping,
            lora_requests=model_input.lora_requests,
            multi_modal_kwargs=model_input.multi_modal_kwargs,
            sampling_metadata=model_input.sampling_metadata,
            is_prompt=False,
        )

        if log_advance_input:
            print("NEW INPUT: ")
            print("  input_tokens = {}".format(new_model_input.input_tokens))
            print("  input_positions = {}".format(new_model_input.input_positions))
            print("  seq_lens = {}".format(new_model_input.seq_lens))
            print("  query_lens = {}".format(new_model_input.query_lens))
            print("  attn_metadata:")
            print("    seq_lens_tensor: {}".format(attn_metadata.seq_lens_tensor))
            print("    slot_mapping: {}".format(attn_metadata.slot_mapping))
            print("    block_tables: {}".format(attn_metadata.block_tables))

        return new_model_input

    def _can_use_advance_step(self):
        if not self.model_config.enforce_eager:
            return False

        if self.lora_config:
            return False

        return True

    @torch.inference_mode()
    def _execute_model_with_advance_step(
            self, model_input: ModelInputForGPUWithSamplingMetadata,
            kv_caches: List[torch.Tensor],
            num_steps: int) -> Optional[List[SamplerOutput]]:
        outputs: List[SamplerOutput] = []
        for step in range(num_steps):
            assert model_input.attn_metadata is not None
            model_executable = self.model

            multi_modal_kwargs = model_input.multi_modal_kwargs or {}

            hidden_states = model_executable(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                kv_caches=kv_caches,
                attn_metadata=model_input.attn_metadata,
                **multi_modal_kwargs,
            )

            # Compute the logits.
            logits = self.model.compute_logits(hidden_states,
                                               model_input.sampling_metadata)

            # Sample the next token.
            outputs.append(
                self.model.sample(
                    logits=logits,
                    sampling_metadata=model_input.sampling_metadata,
                ))

            # Prepare the inputs for the next step.
            if step != num_steps - 1:
                if step == 0 and model_input.is_prompt:
                    model_input = self.update_model_input(
                        model_input, outputs[-1])
                else:
                    assert not model_input.is_prompt
                    model_input = self._advance_step(model_input, outputs[-1])

        return outputs

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        # Since we do not broadcast data inside execute_model anymore,
        # we need to figure out the best way to support TP > 1 in this
        # case, because we will at least need to broadcast the sampled
        # tokens to all workers.
        if not self.is_driver_worker:
            raise ValueError("TP1DraftModelRunner only supports TP=1.")

        if self._can_use_advance_step():
            return self._execute_model_with_advance_step(
                model_input, kv_caches, num_steps)

        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests,
                                  model_input.lora_mapping)

        if self.prompt_adapter_config:
            assert model_input.prompt_adapter_requests is not None
            assert model_input.prompt_adapter_mapping is not None
            self.set_active_prompt_adapters(
                model_input.prompt_adapter_requests,
                model_input.prompt_adapter_mapping)

        virtual_engine = model_input.virtual_engine
        outputs: List[SamplerOutput] = []
        for step in range(num_steps):
            # Currently cuda graph is only supported by the decode phase.
            assert model_input.attn_metadata is not None
            prefill_meta = model_input.attn_metadata.prefill_metadata
            decode_meta = model_input.attn_metadata.decode_metadata
            if prefill_meta is None and decode_meta.use_cuda_graph:
                assert model_input.input_tokens is not None
                graph_batch_size = model_input.input_tokens.shape[0]
                model_executable = (
                    self.graph_runners[virtual_engine][graph_batch_size])
            else:
                model_executable = self.model

            multi_modal_kwargs = model_input.multi_modal_kwargs or {}
            hidden_states = model_executable(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                kv_caches=kv_caches,
                attn_metadata=model_input.attn_metadata,
                intermediate_tensors=intermediate_tensors,
                **multi_modal_kwargs,
            )

            # Compute the logits.
            logits = self.model.compute_logits(hidden_states,
                                               model_input.sampling_metadata)

            # Sample the next token.
            outputs.append(
                self.model.sample(
                    logits=logits,
                    sampling_metadata=model_input.sampling_metadata,
                ))

            # Prepare the inputs for the next step.
            if step != num_steps - 1:
                model_input = self.update_model_input(model_input, outputs[-1])

        return outputs
