import dataclasses
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Type, cast

import torch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, MultiModalConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig)
from vllm.distributed import get_pp_group
from vllm.inputs import INPUT_REGISTRY
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import (IntermediateTensors, PoolerOutput, SamplerOutput,
                           SequenceGroupMetadata)
from vllm.utils import make_tensor_with_pad
from vllm.worker.model_runner import (_PAD_SLOT_ID, GPUModelRunnerBase,
                                      ModelInputForGPUBuilder,
                                      ModelInputForGPUWithSamplingMetadata)
from vllm.worker.model_runner_base import (
    _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)


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
            "input_positions": self.input_positions,
            "encoder_input_tokens": self.encoder_input_tokens,
            "encoder_input_positions": self.encoder_input_positions,
            "prompt_adapter_mapping": self.prompt_adapter_mapping,
            "prompt_adapter_requests": self.prompt_adapter_requests,
            "virtual_engine": self.virtual_engine,
            "request_ids_to_seq_ids": self.request_ids_to_seq_ids,
            "finished_requests_ids": self.finished_requests_ids,
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
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        multimodal_config: Optional[MultiModalConfig] = None,
    ):
        super().__init__(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config,
            lora_config=None,
            kv_cache_dtype=kv_cache_dtype,
            is_driver_worker=is_driver_worker,
            prompt_adapter_config=prompt_adapter_config,
        )

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

        if self.prompt_adapter_config:
            assert model_input.prompt_adapter_requests is not None
            assert model_input.prompt_adapter_mapping is not None
            self.set_active_prompt_adapters(
                model_input.prompt_adapter_requests,
                model_input.prompt_adapter_mapping)

        model_executable = self.model

        seqlen_agnostic_kwargs = {
            "finished_requests_ids": model_input.finished_requests_ids,
            "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
        } if self.has_seqlen_agnostic else {}
        hidden_or_intermediate_states = model_executable(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            encoder_input_ids=model_input.encoder_input_tokens,
            encoder_positions=model_input.encoder_input_positions,
            kv_caches=kv_caches,
            attn_metadata=model_input.attn_metadata,
            intermediate_tensors=intermediate_tensors,
            **seqlen_agnostic_kwargs)

        # Compute the logits in the last pipeline stage.
        if not get_pp_group().is_last_rank:
            return hidden_or_intermediate_states

        logits = self.model.compute_logits(hidden_or_intermediate_states,
                                           model_input.sampling_metadata)

        if not self.is_driver_worker:
            return []

        # Sample the next token.
        output: SamplerOutput = self.model.sample(
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

        model_input = self._prepare_encoder_model_input_tensors(
            seq_group_metadata_list, model_input)

        sampling_metadata = SamplingMetadata.prepare(seq_group_metadata_list,
                                                     model_input.seq_lens,
                                                     model_input.query_lens,
                                                     self.device,
                                                     self.pin_memory)
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

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []

        model_config = self.model_config

        batch_size = 0
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            batch_size += seq_len

            seq_data, _ = INPUT_REGISTRY \
                .dummy_data_for_profiling(model_config, seq_len)

            # Having more tokens is over-conservative but otherwise fine
            assert len(seq_data.prompt_token_ids) >= seq_len, (
                f"Expected at least {seq_len} dummy tokens for profiling, "
                f"but got: {len(seq_data.prompt_token_ids)}")

            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                encoder_seq_data=seq_data,
                cross_block_table=None,
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers
        finished_requests_ids = [seq.request_id for seq in seqs]
        model_input = self.prepare_model_input(
            seqs, finished_requests_ids=finished_requests_ids)
        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=batch_size,
                dtype=self.model_config.dtype,
                device=self.device)
        self.execute_model(model_input, kv_caches, intermediate_tensors)
        torch.cuda.synchronize()
        return

    def _prepare_encoder_model_input_tensors(
            self, seq_group_metadata_list: List[SequenceGroupMetadata],
            model_input: EncoderDecoderModelInput) -> EncoderDecoderModelInput:
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
        and (2) the following addition fields that are
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

        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        prompt_adapter_index_mapping: List[int] = []
        prompt_adapter_prompt_mapping: List[int] = []
        prompt_adapter_requests: Set[PromptAdapterRequest] = set()

        seq_lens: List[int] = []
        prefill_seq_lens: List[int] = []
        decode_seq_lens: List[int] = []
        context_lens: List[int] = []
        query_lens: List[int] = []
        block_tables: List[List[int]] = []
        num_prefills = 0
        num_prefill_tokens = 0
        num_decode_tokens = 0

        if len(seq_group_metadata_list) == 0:
            # Leave the encoder/cross-attention input
            # fields at default values if the seq group
            # metadata list arg is an empty list
            return model_input

        for seq_group_metadata in seq_group_metadata_list:
            is_prompt = seq_group_metadata.is_prompt

            encoder_seq_data = seq_group_metadata.encoder_seq_data
            cross_block_table = seq_group_metadata.cross_block_table
            if is_prompt:
                context_len = encoder_seq_data.get_num_computed_tokens()
            else:
                context_len = encoder_seq_data.get_len()

            seq_len = encoder_seq_data.get_len()

            if is_prompt:
                tokens = encoder_seq_data.get_token_ids()[context_len:seq_len]
            else:
                # Optimization. get_token_ids requires the entire copy of
                # tokens.
                tokens = [encoder_seq_data.get_last_token_id()]

            # These are seq_len/context_len capped to the sliding window.
            # They are passed to decode kernel.
            # We still need original seq_len/context_len to compute slot
            # mapping (and input position) below.
            curr_sliding_window_blocks = None
            sliding_seq_len = seq_len
            sliding_context_len = context_len

            if not is_prompt:
                if cross_block_table is not None:
                    # Decode
                    block_table = cross_block_table
                    if curr_sliding_window_blocks is not None:
                        block_table = block_table[-curr_sliding_window_blocks:]
                else:
                    # Only happens when memory profiling runs.
                    block_table = []
            else:
                # Prefill without memory profiling.
                block_table = []

            block_tables.append(block_table)

            seq_lens.append(sliding_seq_len)
            context_lens.append(sliding_context_len)
            query_len = sliding_seq_len - sliding_context_len
            query_lens.append(query_len)
            input_tokens.extend(tokens)
            input_positions.extend(list(range(context_len, seq_len)))
            prompt_adapter_id = seq_group_metadata.prompt_adapter_id

            if is_prompt:
                num_prefills += 1
                num_prefill_tokens += len(tokens)
                prefill_seq_lens.append(seq_len)
            else:
                num_decode_tokens += query_len
                decode_seq_lens.append(sliding_seq_len)

            if prompt_adapter_id > 0 and is_prompt:
                prompt_adapter_requests.add(
                    seq_group_metadata.prompt_adapter_request)

                num_tokens = seq_group_metadata.\
                                        prompt_adapter_num_virtual_tokens
                pm = [prompt_adapter_id
                      ] * num_tokens + [0] * (query_len - num_tokens)
                prompt_adapter_index_mapping += pm
                prompt_adapter_prompt_mapping.extend(
                    [prompt_adapter_id] *
                    (query_len if seq_group_metadata.sampling_params
                     and seq_group_metadata.sampling_params.prompt_logprobs
                     else 1))

            is_profile_run = _is_single_block_table_empty(
                seq_group_metadata.block_tables)
            if is_profile_run:
                # During memory profiling, the block tables are not
                # initialized yet. In this case, we just use a dummy
                # slot mapping.
                # In embeddings, the block tables are {seq_id: None}.
                slot_mapping.extend([_PAD_SLOT_ID] * seq_len)
                continue

            # Compute the slot mapping.
            block_table = cross_block_table

            for i in range(context_len, seq_len):

                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        max_query_len = max(query_lens)
        max_seq_len = (max(prefill_seq_lens, default=0)
                       if is_prompt else max(decode_seq_lens, default=0))

        max_block_table_len = max(
            len(block_table) for block_table in block_tables)
        block_tables = make_tensor_with_pad(
            block_tables,
            max_len=max_block_table_len,
            pad=0,
            dtype=torch.int,
            device=self.device,
        )
        assert (not is_prompt) or max_query_len > 0, (
            "Decode-phase query_lens: {}".format(query_lens))

        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.int,
                                       device=self.device)

        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=self.device)

        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])

        attn_metadata = model_input.attn_metadata
        assert attn_metadata is not None

        slot_mapping_tensor = torch.tensor(slot_mapping,
                                           dtype=torch.long,
                                           device=self.device)

        if seq_group_metadata.is_prompt:

            input_tokens_tensor = torch.tensor(input_tokens,
                                               dtype=torch.long,
                                               device=self.device)
            input_positions_tensor = torch.tensor(input_positions,
                                                  dtype=torch.long,
                                                  device=self.device)

        else:

            input_tokens_tensor = torch.tensor([],
                                               dtype=torch.long,
                                               device=self.device)
            input_positions_tensor = torch.tensor([],
                                                  dtype=torch.long,
                                                  device=self.device)

        # Set encoder-oriented attention metadata fields
        attn_metadata.num_encoder_tokens = sum(seq_lens)
        attn_metadata.encoder_seq_lens = seq_lens
        attn_metadata.encoder_seq_lens_tensor = seq_lens_tensor
        attn_metadata.max_encoder_seq_len = max_seq_len
        attn_metadata.cross_slot_mapping = slot_mapping_tensor
        attn_metadata.cross_block_tables = block_tables

        # Inject attn_metadata encoder/cross-attention fields &
        # encoder input tokens/positions into model_input.
        # Frozen dataclass fields cannot be modified, so use
        # dataclasses.replace to construct a new model input
        # instance.
        model_input = dataclasses.replace(
            model_input,
            attn_metadata=attn_metadata,
            encoder_input_tokens=input_tokens_tensor,
            encoder_input_positions=input_positions_tensor,
        )

        logits_soft_cap = getattr(self.model_config.hf_config,
                                  'attn_logit_softcapping', None)

        if logits_soft_cap is not None and self.attn_backend.get_name(
        ) != "flashinfer":
            raise ValueError("Models with logits_soft_cap"
                             " require FlashInfer backend, however vLLM"
                             " currently only supports xFormers backend"
                             " for encoder/decoder models.")

        return model_input


def _is_single_block_table_empty(block_table: Optional[List[int]]):
    """
    Check if a single block table has not been constructed
    """
    if block_table is None:
        return True
    return False
