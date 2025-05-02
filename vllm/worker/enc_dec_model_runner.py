# SPDX-License-Identifier: Apache-2.0

import dataclasses
import itertools
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import torch
import torch.distributed

from vllm.attention.backends.abstract import (AttentionBackend,
                                              AttentionMetadata)
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.attention.selector import (get_env_variable_attn_backend,
                                     get_global_forced_attn_backend)
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalKwargs,
                             MultiModalRegistry)
from vllm.platforms import _Backend
from vllm.sampling_params import SamplingParams
from vllm.sequence import (IntermediateTensors, PoolerOutput,
                           SequenceGroupMetadata)
from vllm.utils import STR_NOT_IMPL_ENC_DEC_BACKEND, make_tensor_with_pad
from vllm.worker.model_runner import (GPUModelRunnerBase,
                                      ModelInputForGPUBuilder,
                                      ModelInputForGPUWithSamplingMetadata)
from vllm.worker.model_runner_base import (
    _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict)
from vllm.worker.utils import assert_enc_dec_mr_supported_scenario

logger = init_logger(__name__)
LORA_WARMUP_RANK = 8


@dataclasses.dataclass(frozen=True)
class EncoderDecoderModelInput(ModelInputForGPUWithSamplingMetadata):
    """
    Used by the EncoderDecoderModelRunner.
    """
    encoder_input_tokens: Optional[torch.Tensor] = None
    encoder_input_positions: Optional[torch.Tensor] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "inputs_embeds": self.inputs_embeds,
            "input_positions": self.input_positions,
            "encoder_input_tokens": self.encoder_input_tokens,
            "encoder_input_positions": self.encoder_input_positions,
            "virtual_engine": self.virtual_engine,
            "request_ids_to_seq_ids": self.request_ids_to_seq_ids,
            "finished_requests_ids": self.finished_requests_ids,
            "multi_modal_kwargs": self.multi_modal_kwargs,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        _add_sampling_metadata_broadcastable_dict(tensor_dict,
                                                  self.sampling_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "EncoderDecoderModelInput":
        return cast(
            EncoderDecoderModelInput,
            super().from_broadcasted_tensor_dict(tensor_dict, attn_backend))


class EncoderDecoderModelRunner(GPUModelRunnerBase[EncoderDecoderModelInput]):
    _model_input_cls: Type[EncoderDecoderModelInput] = (
        EncoderDecoderModelInput)
    _builder_cls: Type[ModelInputForGPUBuilder] = (ModelInputForGPUBuilder)

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        input_registry: InputRegistry = INPUT_REGISTRY,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):
        '''
        EncoderDecoderModelRunner constructor.

        `lora_config` and `prompt_adapter_config` are
        unused (since these features are not yet supported for encoder/decoder
        models) but these arguments are present here for compatibility with 
        the base-class constructor.
        '''
        self._maybe_force_supported_attention_backend()

        super().__init__(
            vllm_config=vllm_config,
            kv_cache_dtype=kv_cache_dtype,
            is_driver_worker=is_driver_worker,
            input_registry=input_registry,
            mm_registry=mm_registry,
        )

        # Crash for unsupported encoder/scenarios
        assert_enc_dec_mr_supported_scenario(self)

    def _maybe_force_supported_attention_backend(self):
        '''
        Force vLLM to use the XFormers attention backend,
        which is currently the only supported option.
        '''

        def raise_backend_err():
            # The user has specified an attention backend override
            # which is invalid for encoder/decoder models
            raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_BACKEND)

        maybe_env_var_forced_backend = get_env_variable_attn_backend()
        maybe_global_forced_backend = get_global_forced_attn_backend()
        is_forced_by_global = maybe_global_forced_backend is not None
        is_forced_by_env_var = maybe_env_var_forced_backend is not None
        if is_forced_by_global:  # noqa: SIM102
            # Backend override enforced by global variable takes
            # precedence over vLLM backend environment variable.
            if maybe_global_forced_backend not in\
                 [_Backend.XFORMERS, _Backend.FLASH_ATTN]:
                raise_backend_err()
        elif is_forced_by_env_var:  # noqa: SIM102
            # Backend override enforced by vLLM backend
            # environment variable
            if maybe_env_var_forced_backend not in\
                 [_Backend.XFORMERS, _Backend.FLASH_ATTN]:
                raise_backend_err()

    def _list_to_int32_tensor(
        self,
        _list: List[int],
    ) -> torch.Tensor:
        return torch.tensor(_list, dtype=torch.int32, device=self.device)

    def _list_to_long_tensor(
        self,
        _list: List[int],
    ) -> torch.Tensor:
        return torch.tensor(_list, dtype=torch.long, device=self.device)

    def _empty_int32_tensor(self) -> torch.Tensor:
        return self._list_to_int32_tensor([])

    def _empty_long_tensor(self) -> torch.Tensor:
        return self._list_to_long_tensor([])

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: EncoderDecoderModelInput,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[PoolerOutput]]:
        if num_steps > 1:
            raise ValueError("num_steps > 1 is not supported in "
                             "EncoderDecoderModelRunner")
        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests,
                                  model_input.lora_mapping)
        if (model_input.attn_metadata is not None
                and model_input.attn_metadata.prefill_metadata is None
                and model_input.attn_metadata.decode_metadata.use_cuda_graph):
            if model_input.inputs_embeds is None:
                assert model_input.input_tokens is not None
                graph_batch_size = model_input.input_tokens.shape[0]
                model_executable = (
                    self.graph_runners[model_input.virtual_engine][(
                        graph_batch_size, False)])
            else:
                graph_batch_size = model_input.inputs_embeds.shape[0]
                model_executable = (
                    self.graph_runners[model_input.virtual_engine][(
                        graph_batch_size, True)])
        else:
            model_executable = self.model

        seqlen_agnostic_kwargs = {
            "finished_requests_ids": model_input.finished_requests_ids,
            "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
        } if self.has_inner_state else {}

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        with set_forward_context(model_input.attn_metadata, self.vllm_config,
                                 model_input.virtual_engine):
            hidden_or_intermediate_states = model_executable(
                input_ids=model_input.input_tokens,
                inputs_embeds=model_input.inputs_embeds,
                positions=model_input.input_positions,
                encoder_input_ids=model_input.encoder_input_tokens,
                encoder_positions=model_input.encoder_input_positions,
                intermediate_tensors=intermediate_tensors,
                **MultiModalKwargs.as_kwargs(multi_modal_kwargs,
                                             device=self.device),
                **seqlen_agnostic_kwargs)

        logits = self.model.compute_logits(hidden_or_intermediate_states,
                                           model_input.sampling_metadata)

        if not self.is_driver_worker:
            return []

        if model_input.async_callback is not None:
            model_input.async_callback()

        # Sample the next token.
        output: SamplerOutput = self.sampler(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )

        return [output]

    def make_model_input_from_broadcasted_tensor_dict(
            self, tensor_dict: Dict[str, Any]) -> EncoderDecoderModelInput:
        return EncoderDecoderModelInput.from_broadcasted_tensor_dict(
            tensor_dict,
            attn_backend=self.attn_backend,
        )

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> EncoderDecoderModelInput:
        """Prepare the model input based on a given sequence group, including
        metadata for the sampling step.

        Since chunked prefill is not supported for encoder/decoder models,
        `input_tokens` is assumed to be either entirely prefill tokens or
        entirely decode tokens.

        """
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids)
        (
            attn_metadata,
            encoder_input_tokens_tensor,
            encoder_input_positions_tensor,
        ) = (self._prepare_encoder_model_input_tensors(seq_group_metadata_list,
                                                       model_input))
        # Inject attn_metadata encoder/cross-attention fields &
        # encoder input tokens/positions into model_input.
        # Frozen dataclass fields cannot be modified, so use
        # dataclasses.replace to construct a new model input
        # instance.
        model_input = dataclasses.replace(
            model_input,
            attn_metadata=attn_metadata,
            encoder_input_tokens=encoder_input_tokens_tensor,
            encoder_input_positions=encoder_input_positions_tensor,
        )

        generators = self.get_generators(finished_requests_ids)
        sampling_metadata = SamplingMetadata.prepare(seq_group_metadata_list,
                                                     model_input.seq_lens,
                                                     model_input.query_lens,
                                                     self.device,
                                                     self.pin_memory,
                                                     generators=generators)
        is_prompt = (seq_group_metadata_list[0].is_prompt
                     if seq_group_metadata_list else None)
        return dataclasses.replace(model_input,
                                   sampling_metadata=sampling_metadata,
                                   is_prompt=is_prompt,
                                   virtual_engine=virtual_engine)

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs

        # This represents the maximum number of different requests
        # that will have unique loras, and therefore the max amount of
        # memory consumption. Create dummy lora request copies from the
        # lora request passed in, which contains a lora from the lora
        # warmup path.
        dummy_lora_requests: List[LoRARequest] = []
        dummy_lora_requests_per_seq: List[LoRARequest] = []
        if self.lora_config:
            dummy_lora_requests = self._add_dummy_loras(
                self.lora_config.max_loras)
            assert len(dummy_lora_requests) == self.lora_config.max_loras
            dummy_lora_requests_per_seq = [
                dummy_lora_requests[idx % len(dummy_lora_requests)]
                for idx in range(max_num_seqs)
            ]

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []

        max_mm_tokens = self.mm_registry.get_max_multimodal_tokens(
            self.model_config)
        if max_mm_tokens > 0:
            logger.info("Starting profile run for multi-modal models.")

        batch_size = 0
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            batch_size += seq_len

            decoder_dummy_data = self.input_registry \
                .dummy_data_for_profiling(self.model_config,
                                          seq_len,
                                          self.mm_registry,
                                          is_encoder_data=False)
            encoder_dummy_data = self.input_registry \
                .dummy_data_for_profiling(self.model_config,
                                          seq_len,
                                          self.mm_registry,
                                          is_encoder_data=True)

            # Having more tokens is over-conservative but otherwise fine
            assert len(
                decoder_dummy_data.seq_data.prompt_token_ids
            ) >= seq_len, (
                f"Expected at least {seq_len} dummy tokens for profiling, "
                f"but got: {len(decoder_dummy_data.seq_data.prompt_token_ids)}"
            )

            assert decoder_dummy_data.multi_modal_data is None or \
            encoder_dummy_data.multi_modal_data is None, (
                "Multi-modal data can't be provided in both encoder and decoder"
            )

            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: decoder_dummy_data.seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                encoder_seq_data=encoder_dummy_data.seq_data,
                cross_block_table=None,
                lora_request=dummy_lora_requests_per_seq[group_id]
                if dummy_lora_requests_per_seq else None,
                multi_modal_data=decoder_dummy_data.multi_modal_data
                or encoder_dummy_data.multi_modal_data,
                multi_modal_placeholders=decoder_dummy_data.
                multi_modal_placeholders
                or encoder_dummy_data.multi_modal_placeholders)
            seqs.append(seq)

        finished_requests_ids = [seq.request_id for seq in seqs]
        model_input = self.prepare_model_input(
            seqs, finished_requests_ids=finished_requests_ids)
        intermediate_tensors = None
        self.execute_model(model_input, None, intermediate_tensors)
        torch.cuda.synchronize()
        return

    def _prepare_encoder_model_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        model_input: EncoderDecoderModelInput,
    ) -> Tuple[AttentionMetadata, Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        """Helper method to prepare the encoder- and cross-attn-related
        model inputs based on a given sequence group. These additional inputs
        are used to augment an already-computed `EncoderDecoderModelInput`
        data structure which already has decoder-related model inputs
        populated.

        Sets the following attn_metadata fields:
        * `num_encoder_tokens`
        * `encoder_seq_lens`
        * `encoder_seq_lens_tensor`
        * `max_encoder_seq_len`
        * `cross_slot_mapping`
        * `cross_block_tables`

        Constructs a new model inputs data structure, based on
        (1) the existing fields in the `model_inputs` argument,
        and (2) the following additional fields which are
        computed (or in the case of `attn_metadata`, updated) 
        by this function:
        * attn_metadata
        * encoder_input_tokens
        * encoder_input_positions

        Arguments:

        * seq_group_metadata_list: list of sequence groups for which to
                                   compute inputs
        * model_inputs: model inputs data structure with decoder-oriented
                        fields already computed.

        Return:

        * Updated model inputs data structure
        """

        if len(seq_group_metadata_list) == 0:
            return (model_input.attn_metadata, None, None)

        # Since we are not supporting chunked prefill either the entire
        # batch is prefill or it is decode
        is_prompt = seq_group_metadata_list[0].is_prompt

        # Build encoder inputs
        encoder_seq_lens: List[int] = []
        if is_prompt:
            # Prefill phase.
            cross_block_tables = self._empty_int32_tensor().view(
                len(seq_group_metadata_list), -1)

            # Extract input tokens/positions, cross-attention slot-mapping,
            # & seq len from each sequence group metadata
            (
                encoder_input_tokens,
                encoder_input_positions,
                cross_slot_mapping,
            ) = (
                [],
                [],
                [],
            )
            for seq_group_metadata in seq_group_metadata_list:
                # Build seq lens
                seq_len = seq_group_metadata.encoder_seq_data.get_len()
                token_ids = seq_group_metadata.encoder_seq_data.get_token_ids()
                encoder_seq_lens.append(seq_len)

                # Build slot mapping
                is_profile_run = (seq_group_metadata.block_tables is None)
                if is_profile_run:
                    # During memory profiling, the block tables are not
                    # initialized yet. In this case, we just use a dummy
                    # slot mapping.
                    # In embeddings, the block tables are {seq_id: None}.
                    cross_slot_mapping.extend([PAD_SLOT_ID] * seq_len)
                else:
                    for i in range(0, seq_len):
                        block_number = seq_group_metadata.cross_block_table[
                            i // self.block_size]
                        block_offset = i % self.block_size
                        slot = block_number * self.block_size + block_offset
                        cross_slot_mapping.append(slot)

                # Build encoder input tokens
                encoder_input_tokens.extend(token_ids)
                encoder_input_positions.extend(list(range(0, seq_len)))

            # Convert tokens/positions & cross-attention
            # slot-mapping to encoder input tensors
            encoder_input_tokens_tensor = self._list_to_long_tensor(
                encoder_input_tokens)
            encoder_input_positions_tensor = self._list_to_long_tensor(
                encoder_input_positions)
            cross_slot_mapping_tensor = self._list_to_long_tensor(
                cross_slot_mapping)

        else:
            # Decode phase.
            encoder_input_tokens_tensor = self._empty_long_tensor()
            encoder_input_positions_tensor = self._empty_long_tensor()
            cross_slot_mapping_tensor = self._empty_long_tensor()
            # Extract cross-attention block tables &
            # seq len from each sequence group metadata.
            # Cross-attention block tables are empty
            # during vLLM memory profiling.
            cross_block_tables = []
            for seq_group_metadata in seq_group_metadata_list:
                for _ in range(len(seq_group_metadata.seq_data)):
                    encoder_seq_lens.append(
                        seq_group_metadata.encoder_seq_data.get_len())
                    cross_block_table = seq_group_metadata.cross_block_table
                    cross_block_tables.append([] if (
                        cross_block_table is None) else cross_block_table)

            if (model_input.attn_metadata is not None
                    and model_input.attn_metadata.use_cuda_graph):
                # We will be using CUDA graph replay for this decode.
                max_len_of_block_table = self.get_max_block_per_batch()
                batch_size = len(encoder_seq_lens)
                graph_batch_size = self.vllm_config.pad_for_cudagraph(
                    batch_size)
                assert graph_batch_size >= batch_size
                cuda_graph_pad_size = graph_batch_size - batch_size
                # extend the cross_block_tables and encoder_seq_lens to match
                # the graph_batch_size.
                cross_block_tables.extend([[]
                                           for _ in range(cuda_graph_pad_size)
                                           ])
                encoder_seq_lens.extend(
                    itertools.repeat(1, cuda_graph_pad_size))

            else:
                max_len_of_block_table = max(
                    len(block_table) for block_table in cross_block_tables)

            cross_block_tables = make_tensor_with_pad(
                cross_block_tables,
                max_len=max_len_of_block_table,
                pad=0,
                dtype=torch.int32,
                device=self.device,
            )

        # Compute encoder sequence lengths & encoder
        # sequence starting offset tensors
        max_encoder_seq_len = max(encoder_seq_lens, default=0)
        encoder_seq_lens_tensor = self._list_to_int32_tensor(encoder_seq_lens)
        encoder_seq_start_loc = torch.zeros(encoder_seq_lens_tensor.shape[0] +
                                            1,
                                            dtype=torch.int32,
                                            device=self.device)
        torch.cumsum(encoder_seq_lens_tensor,
                     dim=0,
                     dtype=encoder_seq_start_loc.dtype,
                     out=encoder_seq_start_loc[1:])

        # Update attention metadata with encoder-oriented attributes
        attn_metadata = model_input.attn_metadata
        assert attn_metadata is not None
        (
            attn_metadata.num_encoder_tokens,
            attn_metadata.encoder_seq_lens,
            attn_metadata.encoder_seq_lens_tensor,
            attn_metadata.max_encoder_seq_len,
            attn_metadata.encoder_seq_start_loc,
            attn_metadata.cross_slot_mapping,
            attn_metadata.cross_block_tables,
        ) = (
            sum(encoder_seq_lens),
            encoder_seq_lens,
            encoder_seq_lens_tensor,
            max_encoder_seq_len,
            encoder_seq_start_loc,
            cross_slot_mapping_tensor,
            cross_block_tables,
        )

        return (attn_metadata, encoder_input_tokens_tensor,
                encoder_input_positions_tensor)
