from typing import List, Optional

import torch

from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.sampler import SamplerOutput

try:
    try:
        from vllm.attention.backends.flash_attn import FlashAttentionMetadata
    except (ModuleNotFoundError, ImportError):
        # vllm_flash_attn is not installed, try the ROCm FA metadata
        from vllm.attention.backends.rocm_flash_attn import (
            ROCmFlashAttentionMetadata as FlashAttentionMetadata)
except (ModuleNotFoundError, ImportError) as err:
    raise RuntimeError(
        "Draft model speculative decoding currently only supports"
        "CUDA and ROCm flash attention backend.") from err

from vllm.logger import init_logger
from vllm.multimodal import MultiModalKwargs
from vllm.sequence import ExecuteModelRequest, IntermediateTensors
from vllm.worker.model_runner import (ModelInputForGPUWithSamplingMetadata,
                                      ModelRunner)

logger = init_logger(__name__)

# A flag to enable debug prints for the updated input tensors
# before each step.
debug_advance_input = False
# A flag to allow GPU advance step for draft model runner.
# Set to False for debugging.
allow_gpu_advance_step = True


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

    def __init__(self, *args, **kwargs):
        if kwargs.get("return_hidden_states"):
            raise ValueError(
                "return_hidden_states is not supported for TP1DraftModelRunner."
            )

        super().__init__(*args, **kwargs)

        self.indices_of_seq_with_bonus_tokens = None

    def _update_sampling_metadata(self, sampling_metadata, num_seqs,
                                  num_queries):

        assert sampling_metadata.num_prompts == 0
        assert len(sampling_metadata.seq_groups) == num_queries
        assert sampling_metadata.selected_token_indices.shape == (
            num_queries, )
        # assert sampling_metadata.categorized_sample_indices == TODO: Add if needed # noqa: E501

        # Verify that all sequences are decodes
        for i in range(num_queries):
            seq_group = sampling_metadata.seq_groups[i]

            assert seq_group.is_prompt is False  # No prompt
            assert seq_group.prompt_logprob_indices == []  # No prompt
            assert seq_group.sample_indices == [i]  # Simple

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

        attn_metadata.advance_step(model_input, sampled_token_ids,
                                   self.block_size, num_seqs, num_queries)

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

        # Ensure we skip CPU samples
        assert new_model_input.sampling_metadata.skip_sampler_cpu_output is True
        # We can reuse sampling tensors since every decode iteration is the same
        new_model_input.sampling_metadata.reuse_sampling_tensors = True

        if debug_advance_input:
            logger.debug("NEW INPUT: ")
            logger.debug("  input_tokens = %s", new_model_input.input_tokens)
            logger.debug("  input_positions = %s",
                         new_model_input.input_positions)
            logger.debug("  seq_lens = %d", new_model_input.seq_lens)
            logger.debug("  query_lens = %d", new_model_input.query_lens)
            logger.debug("  attn_metadata:")
            logger.debug("    seq_lens_tensor: %s",
                         attn_metadata.seq_lens_tensor)
            logger.debug("    slot_mapping: %s", attn_metadata.slot_mapping)
            logger.debug("    block_tables: %s", attn_metadata.block_tables)

        return new_model_input

    def supports_gpu_multi_step(self, execute_model_req: ExecuteModelRequest):
        """Determines if draft_model_runner GPU multi-step can be used.
        Currently required conditions are:
            1. Only decodes 
            2. Only flash-attn
            3. No LORA
            4. No prompt_adapter_config
        """
        if not allow_gpu_advance_step:
            return False

        # We allow multi-step GPU only in decode mode
        for seq_group in execute_model_req.seq_group_metadata_list:
            if seq_group.is_prompt:
                return False

        # TODO: Add support for other attn backends
        if self.attn_backend.get_name() != "FLASH_ATTN":
            return False

        # TODO: Add support for LORA
        if self.lora_config:
            return False

        # TODO: Add soft-tuning prompt adapter support
        return not self.prompt_adapter_config

    def set_indices_of_seq_with_bonus_tokens(self,
                                             indices_of_seq_with_bonus_tokens):
        self.indices_of_seq_with_bonus_tokens = indices_of_seq_with_bonus_tokens

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        previous_hidden_states: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        """Executes num_steps forward passes with advacement of input tensors 
        on the GPU. Look at supports_gpu_multi_step(..) for pre-conditions.

        Optimizations used:
            1. Input tensors are updated on the GPU directly
            2. Skips GPU=>CPU serialization of sampler outputs (we don't need 
                them since we do batch expansion later that uses GPU outputs)
            3. Reuses sampling tensors (since we run only decodes and they have
                a repeating sampling logic)
        """

        # When num_steps == 1, we execute the fallback here for the GPU
        # advance_step, which runs prepare_inputs on CPU and for each spec
        # iteration invokes this function only once
        # (Look at multi-step-worker code)
        is_fallback = num_steps == 1
        if not is_fallback:
            # Since we do not broadcast data inside execute_model anymore,
            # we need to figure out the best way to support TP > 1 in this
            # case, because we will at least need to broadcast the sampled
            # tokens to all workers.
            if not self.is_driver_worker:
                raise ValueError("TP1DraftModelRunner only supports TP=1.")

            # Sanity
            if self.lora_config is not None:
                raise ValueError("TP1DraftModelRunner has no support for LORA")
            if self.prompt_adapter_config is not None:
                raise ValueError("TP1DraftModelRunner has no support for "
                                 "prompt_adapter_config")
            if model_input.multi_modal_kwargs:
                raise ValueError(
                    "TP1DraftModelRunner has no support for multi_modal_kwargs"
                )
        else:
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

            self.attn_state.begin_forward(model_input)

        # Detect exec mode
        assert model_input.attn_metadata is not None
        use_cuda_graph = False
        if model_input.attn_metadata.num_prefills > 0:
            # In this case, execute_model(..) was called directly
            if num_steps > 1:
                raise ValueError(
                    "execute_model(..) of draft_model_runner can be called "
                    "directly only with a single-step prefill")
        else:
            # We can skip CPU samples for spec token generation.
            # (We do allow CPU samples for num_steps == 1 to support the
            # fallback case, where supports_gpu_multi_step(..) does not pass)
            model_input.sampling_metadata.skip_sampler_cpu_output = (
                not is_fallback)

            # Attn attr defines if we use cuda graphs
            use_cuda_graph = model_input.attn_metadata.use_cuda_graph

        # Get model
        if use_cuda_graph:
            graph_batch_size = model_input.input_tokens.shape[0]
            model_executable = (self.graph_runners[model_input.virtual_engine]
                                [graph_batch_size])

            if previous_hidden_states is not None:
                hidden_states = torch.cat([
                    previous_hidden_states,
                    torch.empty([
                        graph_batch_size - previous_hidden_states.shape[0],
                        *previous_hidden_states.shape[1:]
                    ],
                                dtype=previous_hidden_states.dtype,
                                device=previous_hidden_states.device)
                ])
            else:
                hidden_states = None
        else:
            model_executable = self.model
            hidden_states = previous_hidden_states

        outputs: List[SamplerOutput] = []
        for step in range(num_steps):
            multi_modal_kwargs = model_input.multi_modal_kwargs or {}

            kwargs = {"previous_hidden_states": hidden_states} \
                if previous_hidden_states is not None else {}

            # Run model
            with set_forward_context(model_input.attn_metadata,
                                     self.vllm_config):
                hidden_states = model_executable(
                    input_ids=model_input.input_tokens,
                    positions=model_input.input_positions,
                    kv_caches=kv_caches,
                    attn_metadata=model_input.attn_metadata,
                    intermediate_tensors=intermediate_tensors,
                    **MultiModalKwargs.as_kwargs(multi_modal_kwargs,
                                                 device=self.device),
                    **kwargs,
                )

            # Compute the logits.
            logits = self.model.compute_logits(hidden_states,
                                               model_input.sampling_metadata)

            # Sample the next token.
            output = self.model.sample(
                logits=logits,
                sampling_metadata=model_input.sampling_metadata,
            )
            outputs.append(output)

            if model_input.attn_metadata.num_prefills == 0 \
                and self.indices_of_seq_with_bonus_tokens is not None:
                assert output.sampled_token_ids is not None
                # output.sampled_token_ids should be of shape (num_seqs, 1)
                nums_seqs, num_tokens_per_seq = output.sampled_token_ids.shape
                assert num_tokens_per_seq == 1
                count = 0
                for i in range(nums_seqs):
                    bonus_seq_idx = self.indices_of_seq_with_bonus_tokens[
                        count]
                    if i != bonus_seq_idx:
                        # The following might cause a cpu->gpu sync
                        # However, the performance impact is negligible as we
                        # benchmarked on H100.
                        output.sampled_token_ids[
                            i, :] = model_input.input_tokens[bonus_seq_idx]
                    else:
                        count += 1

            # Prepare inputs for the next step
            if step != num_steps - 1:
                model_input = self._gpu_advance_step(model_input, outputs[-1])

        return outputs
