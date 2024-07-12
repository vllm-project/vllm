from typing import List, Optional

import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, MultiModalConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig)
from vllm.logger import init_logger
from vllm.sequence import (IntermediateTensors, SamplerOutput,
                           SequenceGroupMetadata, ExecuteModelRequest)
from vllm.worker.model_runner import (ModelInputForGPUWithSamplingMetadata,
                                      ModelRunner)

logger = init_logger(__name__)

debug_advance_input = False
enable_gpu_advance_step = True


class TP1DraftModelRunner(ModelRunner):
    """Specialized model runner for speculative decoding draft model.
    Since the draft model always execute k forward passes consecutively to
    generate k speculative tokens in a single speculative decoding step,
    we could get rid of most CPU-GPU synchronization and data transfer
    overheads by keeping model input and output tensors on GPU all the time.

    TODOs:
    1. Currently supports only flash-attn, add support for other attn_backends.
    2. Support TP > 1 (this requires some designs because we do not expect
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

    def _update_flash_attn_metadata(self, attn_metadata, num_seqs,
                                    num_queries):
        assert isinstance(attn_metadata, FlashAttentionMetadata)

        if num_seqs != num_queries:
            assert num_seqs > num_queries
            assert attn_metadata.use_cuda_graph

        assert attn_metadata.num_prefills == 0
        assert attn_metadata.num_prefill_tokens == 0
        assert attn_metadata.num_decode_tokens == num_seqs
        assert attn_metadata.slot_mapping.shape == (num_seqs, )

        assert len(attn_metadata.seq_lens) == num_seqs
        assert attn_metadata.seq_lens_tensor.shape == (num_seqs, )
        assert attn_metadata.max_query_len == 1
        assert attn_metadata.max_prefill_seq_len == 0
        assert attn_metadata.max_decode_seq_len == max(attn_metadata.seq_lens)

        assert attn_metadata.query_start_loc.shape == (num_queries + 1, )
        assert attn_metadata.seq_start_loc.shape == (num_seqs + 1, )

        assert attn_metadata.context_lens_tensor.shape == (num_queries, )

        assert attn_metadata.block_tables.shape[0] == num_seqs

        # Update query lengths. Note that we update only queries and not seqs,
        # since tensors may be padded due to captured cuda graph batch size
        for i in range(num_queries):
            attn_metadata.seq_lens[i] += 1
        attn_metadata.max_decode_seq_len = max(attn_metadata.seq_lens)

    def _update_sampling_metadata(self, sampling_metadata, num_seqs,
                                  num_queries):

        assert sampling_metadata.num_prompts == 0
        assert len(sampling_metadata.seq_groups) == num_queries
        assert sampling_metadata.selected_token_indices.shape == (
            num_queries, )
        # assert sampling_metadata.categorized_sample_indices == TODO: Add if needed # noqa: E501

        for i in range(num_queries):
            seq_group = sampling_metadata.seq_groups[i]

            assert seq_group.is_prompt is False  # No prompt
            assert seq_group.prompt_logprob_indices == []  # No prompt
            assert seq_group.sample_indices == [i]  # Simple
            assert seq_group.seq_len is None  # Decode
            assert seq_group.query_len is None  # Decode

    def _gpu_advance_step(
            self, model_input: ModelInputForGPUWithSamplingMetadata,
            last_output: SamplerOutput
    ) -> ModelInputForGPUWithSamplingMetadata:
        # Currently, we expect "decode mode" only
        assert not model_input.is_prompt

        # Get num_seqs
        num_seqs = len(model_input.seq_lens)
        num_queries = len(model_input.query_lens)

        # Get output tokens GPU tensor
        sampled_token_ids = last_output.sampled_token_ids
        assert sampled_token_ids is not None

        # Update attn_metadata
        attn_metadata = model_input.attn_metadata
        assert isinstance(attn_metadata, FlashAttentionMetadata)
        self._update_flash_attn_metadata(attn_metadata, num_seqs, num_queries)

        # Update GPU tensors
        ops.advance_step(num_seqs=num_seqs,
                         num_queries=num_queries,
                         block_size=self.block_size,
                         input_tokens=model_input.input_tokens,
                         sampled_token_ids=sampled_token_ids,
                         input_positions=model_input.input_positions,
                         seq_lens=attn_metadata.seq_lens_tensor,
                         slot_mapping=attn_metadata.slot_mapping,
                         block_tables=attn_metadata.block_tables)

        # Update sampling_metadata
        sampling_metadata = model_input.sampling_metadata
        self._update_sampling_metadata(sampling_metadata, num_seqs,
                                       num_queries)

        # Create new input
        new_model_input = self._model_input_cls(
            input_tokens=model_input.input_tokens,
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

        model_input.sampling_metadata.reuse_sampling_tensors = True

        if debug_advance_input:
            print("NEW INPUT: ")
            print("  input_tokens = {}".format(new_model_input.input_tokens))
            print("  input_positions = {}".format(
                new_model_input.input_positions))
            print("  seq_lens = {}".format(new_model_input.seq_lens))
            print("  query_lens = {}".format(new_model_input.query_lens))
            print("  attn_metadata:")
            print("    seq_lens_tensor: {}".format(
                attn_metadata.seq_lens_tensor))
            print("    slot_mapping: {}".format(attn_metadata.slot_mapping))
            print("    block_tables: {}".format(attn_metadata.block_tables))

        return new_model_input

    def supports_gpu_multi_step(self, execute_model_req: ExecuteModelRequest):
        if not enable_gpu_advance_step:
            return False

        # We allow multi-step GPU only in decode mode
        for seq_group in execute_model_req.seq_group_metadata_list:
            if seq_group.is_prompt:
                return False

        # TODO: Add support for other attn backends
        if self.attn_backend.get_name() != "flash-attn":
            return False

        # TODO: Understand what is needed for LORA support
        if self.lora_config:
            return False

        # TODO: Ask Cade/Cody what is this
        if self.prompt_adapter_config:
            return False

        return True

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

        # Sanity
        assert self.lora_config is None
        assert self.prompt_adapter_config is None
        assert model_input.attn_metadata is not None

        # Detect exec mode
        use_cuda_graph = False
        if model_input.attn_metadata.num_prefills > 0:
            # In this case, execute_model(..) was called directly
            assert num_steps == 1, (
                "execute_model(..) of draft_model_runner can be called "
                "directly only with a single-step prefill")
        else:
            use_cuda_graph = model_input.attn_metadata.use_cuda_graph
            # We can skip CPU samples for spec token generation
            model_input.sampling_metadata.skip_sampler_cpu_output = True

        virtual_engine = model_input.virtual_engine
        outputs: List[SamplerOutput] = []
        for step in range(num_steps):
            # Get model
            if use_cuda_graph:
                graph_batch_size = model_input.input_tokens.shape[0]
                model_executable = (
                    self.graph_runners[virtual_engine][graph_batch_size])
            else:
                model_executable = self.model

            multi_modal_kwargs = model_input.multi_modal_kwargs or {}

            # Run model
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

            # Prepare the inputs for the next step on GPU.
            if step != num_steps - 1:
                model_input = self._gpu_advance_step(model_input, outputs[-1])

        return outputs
