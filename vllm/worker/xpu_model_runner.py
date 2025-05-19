# SPDX-License-Identifier: Apache-2.0

import dataclasses
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Type, TypeVar)

import torch
import torch.nn as nn

from vllm.attention import get_attn_backend
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadataCache
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalKwargs, MultiModalPlaceholderMap,
                             MultiModalRegistry)
from vllm.sampling_params import SamplingParams
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.utils import DeviceMemoryProfiler, GiB_bytes, make_tensor_with_pad
from vllm.worker.model_runner import AttentionMetadata, SamplingMetadata
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase, ModelRunnerInputBuilderBase,
    _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict,
    _init_sampling_metadata_from_tensor_dict)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)

_PAD_SLOT_ID = -1

TModelInputForXPU = TypeVar('TModelInputForXPU', bound="ModelInputForXPU")


@dataclass(frozen=True)
class ModelInputForXPU(ModelRunnerInputBase):
    """
    Used by the NeuronModelRunner.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    attn_metadata: Optional["AttentionMetadata"] = None
    multi_modal_kwargs: Optional[BatchedTensorInputs] = None
    virtual_engine: Optional[int] = None
    seq_lens: Optional[List[int]] = None
    query_lens: Optional[List[int]] = None
    async_callback: Optional[Callable] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)

        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type[TModelInputForXPU],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> TModelInputForXPU:
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


@dataclass(frozen=True)
class ModelInputForXPUWithSamplingMetadata(ModelInputForXPU):
    """
    Used by the ModelRunner.
    """
    sampling_metadata: Optional["SamplingMetadata"] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
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
    ) -> "ModelInputForXPUWithSamplingMetadata":
        tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


class ModelInputForXPUBuilder(ModelRunnerInputBuilderBase[ModelInputForXPU]):

    def __init__(self,
                 runner: "XPUModelRunner",
                 finished_requests_ids: Optional[List[str]] = None) -> None:
        super().__init__()
        self.runner = runner
        self.model_input_cls = self.runner._model_input_cls
        self.attn_backend = self.runner.attn_backend
        self.sliding_window = self.runner.sliding_window
        self.block_size = self.runner.block_size
        self.device = self.runner.device

    def prepare(self,
                finished_requests_ids: Optional[List[str]] = None) -> None:
        self.seq_group_metadata_list: List[SequenceGroupMetadata] = []

    def add_seq_group(self, seq_group_metadata: SequenceGroupMetadata):
        self.seq_group_metadata_list.append(seq_group_metadata)

    def build(self) -> ModelInputForXPU:
        is_prompt = self.seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        if is_prompt:
            (input_tokens, input_positions, attn_metadata, seq_lens,
             multi_modal_kwargs) = self._prepare_prompt(
                 self.seq_group_metadata_list)
        else:
            (input_tokens, input_positions,
             attn_metadata) = self._prepare_decode(
                 self.seq_group_metadata_list)
            seq_lens = None
            multi_modal_kwargs = None

        return self.model_input_cls(
            input_tokens=input_tokens,
            input_positions=input_positions,
            attn_metadata=attn_metadata,
            multi_modal_kwargs=multi_modal_kwargs,
            seq_lens=seq_lens,
            query_lens=seq_lens,
        )

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, List[int],
               BatchedTensorInputs]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        seq_lens: List[int] = []
        multi_modal_kwargs_list: List[MultiModalKwargs] = []
        multi_modal_placeholder_maps: Dict[
            str,
            MultiModalPlaceholderMap] = defaultdict(MultiModalPlaceholderMap)

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            computed_len = seq_data.get_num_computed_tokens()
            seq_len = len(prompt_tokens)

            seq_lens.append(seq_len)  # Prompt token num
            input_tokens.extend(prompt_tokens)  # Token ids

            # Token position ids
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            positions_range = range(computed_len, seq_len)
            input_positions.extend(list(positions_range))

            if seq_group_metadata.multi_modal_data:
                # NOTE: mm_kwargs only includes the subset of multi-modal items
                # that intersect with the current prefill positions.
                mm_kwargs, placeholder_maps = MultiModalPlaceholderMap \
                    .from_seq_group(seq_group_metadata, positions_range)

                multi_modal_kwargs_list.append(mm_kwargs)

                for modality, placeholder_map in placeholder_maps.items():
                    multi_modal_placeholder_maps[modality].extend(
                        placeholder_map)

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.extend([_PAD_SLOT_ID] * seq_len)
                continue

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, seq_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                start_idx = max(0, seq_len - self.sliding_window)

            for i in range(computed_len, seq_len):
                if i < start_idx:
                    slot_mapping.append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i //
                                           self.block_size]  # type: ignore
                block_offset = i % self.block_size  # type: ignore
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        num_prompt_tokens = len(input_tokens)

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device)  # type: ignore
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device)  # type: ignore
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=self.device)  # type: ignore
        placeholder_index_maps = {
            modality: placeholder_map.index_map()
            for modality, placeholder_map in
            multi_modal_placeholder_maps.items()
        }

        max_seqlen = max(seq_lens)
        tmp = [0]
        tmp.extend(seq_lens)
        seqlen = torch.tensor(tmp)
        seqlen_q = torch.cumsum(seqlen, dim=0).to(device=self.device)

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=True,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            enable_kv_scales_calculation=False,
            seq_lens=seq_lens,
            seqlen_q=seqlen_q,
            max_seqlen=max_seqlen,
            seq_lens_tensor=torch.tensor([]),
            max_decode_seq_len=0,
            num_prefills=len(seq_lens),
            num_prefill_tokens=num_prompt_tokens,
            num_decode_tokens=0,
            block_tables=torch.tensor([], device=self.device, dtype=torch.int),
        )

        multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_kwargs_list)

        return (input_tokens, input_positions, attn_metadata, seq_lens,
                multi_modal_kwargs)

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        seq_lens: List[int] = []
        block_tables: List[List[int]] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            assert seq_group_metadata.token_chunk_size == 1

            seq_ids = list(seq_group_metadata.seq_data.keys())

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append(position)

                seq_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                seq_lens.append(seq_len)

                block_table = seq_group_metadata.block_tables[seq_id]
                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                block_tables.append(block_table)

        max_decode_seq_len = max(seq_lens)

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device)
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=self.device)
        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.int,
                                       device=self.device)

        block_tables = make_tensor_with_pad(
            block_tables,
            pad=0,
            dtype=torch.int,
            device=self.device,
        )

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=False,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
            seq_lens=seq_lens,
            seqlen_q=torch.tensor([]),
            max_seqlen=0,
            seq_lens_tensor=seq_lens_tensor,
            max_decode_seq_len=max_decode_seq_len,
            num_prefill_tokens=0,
            num_decode_tokens=len(input_tokens),
            num_prefills=0,
            block_tables=block_tables,
        )
        return (
            input_tokens,
            input_positions,
            attn_metadata,
        )


class XPUModelRunner(ModelRunnerBase[ModelInputForXPUWithSamplingMetadata]):
    _model_input_cls: Type[ModelInputForXPUWithSamplingMetadata] = (
        ModelInputForXPUWithSamplingMetadata)
    _builder_cls: Type[ModelInputForXPUBuilder] = ModelInputForXPUBuilder

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        return_hidden_states: bool = False,
        input_registry: InputRegistry = INPUT_REGISTRY,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):

        ModelRunnerBase.__init__(self, vllm_config=vllm_config)
        model_config = self.model_config
        cache_config = self.cache_config
        self.is_driver_worker = is_driver_worker
        self.return_hidden_states = return_hidden_states

        self.device = self.device_config.device

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size

        self.attn_backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
        )

        # Multi-modal data support
        self.input_registry = input_registry
        self.mm_registry = mm_registry

        # Lazy initialization.
        self.model: nn.Module  # Set after init_Model
        self.sampler = get_sampler()

        self.sampling_metadata_cache: SamplingMetadataCache = \
              SamplingMetadataCache() \
                if self.parallel_config.pipeline_parallel_size == 1 else None

        self.builder = self._builder_cls(weakref.proxy(self))

    def load_model(self) -> None:
        with DeviceMemoryProfiler() as m:
            self.model = get_model(vllm_config=self.vllm_config)

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GiB",
                    self.model_memory_usage / GiB_bytes)

    def get_model(self) -> nn.Module:
        return self.model

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        # Additional GPU memory may be needed for multi-modal encoding, which
        # needs to be accounted for when calculating the GPU blocks for
        # vLLM blocker manager.
        # To exercise the worst scenario for GPU memory consumption,
        # the number of seqs (batch_size) is chosen to maximize the number
        # of images processed.
        max_mm_tokens = self.mm_registry.get_max_multimodal_tokens(
            self.model_config)
        if max_mm_tokens > 0:
            max_num_seqs_orig = max_num_seqs
            max_num_seqs = min(max_num_seqs,
                               max_num_batched_tokens // max_mm_tokens)
            if max_num_seqs < 1:
                expr = (f"min({max_num_seqs_orig}, "
                        f"{max_num_batched_tokens} // {max_mm_tokens})")
                logger.warning(
                    "Computed max_num_seqs (%s) to be less than 1. "
                    "Setting it to the minimum value of 1.", expr)
                max_num_seqs = 1

        batch_size = 0
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            batch_size += seq_len

            dummy_data = self.input_registry \
                .dummy_data_for_profiling(self.model_config,
                                          seq_len,
                                          self.mm_registry)

            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: dummy_data.seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                lora_request=None,
                multi_modal_data=dummy_data.multi_modal_data,
                multi_modal_placeholders=dummy_data.multi_modal_placeholders)
            seqs.append(seq)

        finished_requests_ids = [seq.request_id for seq in seqs]
        model_input = self.prepare_model_input(
            seqs, finished_requests_ids=finished_requests_ids)
        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=batch_size,
                dtype=self.model_config.dtype,
                device=self.device)
        self.execute_model(model_input, None, intermediate_tensors)
        torch.xpu.synchronize()
        return

    def make_model_input_from_broadcasted_tensor_dict(
            self,
            tensor_dict: Dict[str,
                              Any]) -> ModelInputForXPUWithSamplingMetadata:
        return (
            ModelInputForXPUWithSamplingMetadata.from_broadcasted_tensor_dict(
                tensor_dict,
                attn_backend=self.attn_backend,
            ))

    def _prepare_model_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForXPUWithSamplingMetadata:
        """Helper method to prepare the model input based on a given sequence
        group. Prepares metadata needed for the base model forward pass but not
        metadata for possible additional steps, e.g., sampling.

        """
        builder = self.builder
        builder.prepare(finished_requests_ids)
        for seq_group_metadata in seq_group_metadata_list:
            builder.add_seq_group(seq_group_metadata)

        return builder.build()  # type: ignore

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForXPUWithSamplingMetadata:
        """Prepare the model input based on a given sequence group, including
        metadata for the sampling step.

        """
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids)
        # Sampling metadata is only required for the final pp group
        generators = self.get_generators(finished_requests_ids)
        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            model_input.seq_lens,
            model_input.query_lens,
            self.device,
            pin_memory=False,
            generators=generators,
            cache=self.sampling_metadata_cache)

        return dataclasses.replace(model_input,
                                   sampling_metadata=sampling_metadata,
                                   virtual_engine=virtual_engine)

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForXPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        if num_steps > 1:
            raise ValueError(
                "XPUModelRunner does not support multi-step execution.")

        model_executable = self.model
        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_start_time = time.time()
        with set_forward_context(model_input.attn_metadata, self.vllm_config,
                                 model_input.virtual_engine):
            hidden_or_intermediate_states = model_executable(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                intermediate_tensors=intermediate_tensors,
                **MultiModalKwargs.as_kwargs(model_input.multi_modal_kwargs
                                             or {},
                                             device=self.device))
        # Compute the logits in the last pipeline stage.
        if not get_pp_group().is_last_rank:
            return hidden_or_intermediate_states

        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_end_time = time.time()

        # Compute the logits.
        logits = self.model.compute_logits(hidden_or_intermediate_states,
                                           model_input.sampling_metadata)

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return []

        if model_input.async_callback is not None:
            model_input.async_callback()

        # Sample the next token.
        output: SamplerOutput = self.sampler(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time
                and output is not None):
            model_forward_time = (model_forward_end_time -
                                  model_forward_start_time)
            # If there are multiple workers, we are still tracking the latency
            # from the start time of the driver worker to the end time of the
            # driver worker. The model forward time will then end up covering
            # the communication time as well.
            output.model_forward_time = model_forward_time

        return [output]
