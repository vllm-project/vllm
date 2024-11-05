import dataclasses
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn

from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalInputs, MultiModalPlaceholderMap)
from vllm.sequence import (IntermediateTensors, SequenceData,
                           SequenceGroupMetadata)
from vllm.transformers_utils.config import uses_mrope
from vllm.utils import make_tensor_with_pad
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


@dataclass(frozen=True)
class ModelInputForCPU(ModelRunnerInputBase):
    """
    Base class contains metadata needed for the base model forward pass on CPU
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    attn_metadata: Optional["AttentionMetadata"] = None
    multi_modal_kwargs: Optional[BatchedTensorInputs] = None
    virtual_engine: Optional[int] = None
    seq_lens: Optional[List[int]] = None
    query_lens: Optional[List[int]] = None

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "multi_modal_kwargs": self.multi_modal_kwargs,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)

        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type["ModelInputForCPU"],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None
    ) -> "ModelInputForCPU":
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


@dataclass(frozen=True)
class ModelInputForCPUWithSamplingMetadata(ModelInputForCPU):
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
    ) -> "ModelInputForCPUWithSamplingMetadata":
        tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


class ModelInputForCPUBuilder(ModelRunnerInputBuilderBase[ModelInputForCPU]):

    def __init__(self,
                 runner: "CPUModelRunner",
                 finished_requests_ids: Optional[List[str]] = None) -> None:
        super().__init__()
        self.seq_group_metadata_list: List[SequenceGroupMetadata] = []
        self.runner = runner
        self.model_input_cls = self.runner._model_input_cls
        self.attn_backend = self.runner.attn_backend
        self.sliding_window = self.runner.sliding_window
        self.block_size = self.runner.block_size
        self.device = self.runner.device
        self.multi_modal_input_mapper = self.runner.multi_modal_input_mapper

    def add_seq_group(self, seq_group_metadata: SequenceGroupMetadata):
        self.seq_group_metadata_list.append(seq_group_metadata)

    def build(self) -> ModelInputForCPU:
        multi_modal_kwargs = None
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
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

        return self.model_input_cls(
            input_tokens=input_tokens,
            input_positions=input_positions,
            attn_metadata=attn_metadata,
            multi_modal_kwargs=multi_modal_kwargs,
            # query_lens is not needed if chunked prefill is not
            # supported. Since CPU worker doesn't support chunked prefill
            # just use seq_lens instead.
            seq_lens=seq_lens,
            query_lens=seq_lens,
        )

    def _compute_multi_modal_input(self, seq_group: SequenceGroupMetadata,
                                   seq_data: SequenceData, computed_len: int,
                                   mm_processor_kwargs: Dict[str, Any]):

        # NOTE: mm_data only includes the subset of multi-modal items that
        # intersect with the current prefill positions.
        mm_data, placeholder_maps = MultiModalPlaceholderMap.from_seq_group(
            seq_group, range(computed_len, len(seq_data.get_token_ids())))

        if not mm_data:
            return

        mm_kwargs = self.multi_modal_input_mapper(mm_data, mm_processor_kwargs)

        # special processing for mrope position deltas.
        mrope_positions = None
        if self.runner.model_is_mrope:
            image_grid_thw = mm_kwargs.get("image_grid_thw", None)
            video_grid_thw = mm_kwargs.get("video_grid_thw", None)
            assert image_grid_thw is not None or video_grid_thw is not None, (
                "mrope embedding type requires multi-modal input mapper "
                "returns 'image_grid_thw' or 'video_grid_thw'.")

            hf_config = self.runner.model_config.hf_config
            token_ids = seq_data.get_token_ids()

            mrope_positions, mrope_position_delta = \
                MRotaryEmbedding.get_input_positions(
                    token_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    image_token_id=hf_config.image_token_id,
                    video_token_id=hf_config.video_token_id,
                    vision_start_token_id=hf_config.vision_start_token_id,
                    vision_end_token_id=hf_config.vision_end_token_id,
                    spatial_merge_size=hf_config.vision_config.
                    spatial_merge_size,
                    context_len=computed_len,
                )
            seq_data.mrope_position_delta = mrope_position_delta
        return mm_kwargs, placeholder_maps, mrope_positions

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, List[int],
               BatchedTensorInputs]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        input_mrope_positions: List[List[int]] = [[] for _ in range(3)]

        slot_mapping: List[int] = []
        seq_lens: List[int] = []
        multi_modal_inputs_list: List[MultiModalInputs] = []
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

            mrope_positions = None
            if seq_group_metadata.multi_modal_data:
                mm_kwargs, placeholder_maps, mrope_positions = self \
                    ._compute_multi_modal_input(
                        seq_group_metadata, seq_data, computed_len,
                    seq_group_metadata.mm_processor_kwargs)
                multi_modal_inputs_list.append(mm_kwargs)
                for modality, placeholder_map in placeholder_maps.items():
                    multi_modal_placeholder_maps[modality].extend(
                        placeholder_map)

            # Token position ids
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            if mrope_positions:
                for idx in range(3):
                    input_mrope_positions[idx].extend(mrope_positions[idx])
            else:
                input_positions.extend(list(range(computed_len, seq_len)))

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

        if any(input_mrope_positions):
            input_positions = None  # type: ignore
        else:
            input_mrope_positions = None  # type: ignore

        num_prompt_tokens = len(input_tokens)

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device)  # type: ignore
        input_positions = torch.tensor(input_positions
                                       or input_mrope_positions,
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

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=True,
            seq_lens=seq_lens,
            seq_lens_tensor=torch.tensor([]),
            max_decode_seq_len=0,
            num_prefills=len(seq_lens),
            num_prefill_tokens=num_prompt_tokens,
            num_decode_tokens=0,
            block_tables=torch.tensor([]),
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
        )

        multi_modal_kwargs = MultiModalInputs.batch(multi_modal_inputs_list)

        return (input_tokens, input_positions, attn_metadata, seq_lens,
                multi_modal_kwargs)

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        input_mrope_positions: List[List[int]] = [[] for _ in range(3)]
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
                if seq_data.mrope_position_delta is not None:
                    context_len = seq_data.get_num_computed_tokens()
                    next_pos = MRotaryEmbedding.get_next_input_positions(
                        seq_data.mrope_position_delta,
                        context_len,
                        seq_len,
                    )
                    for idx in range(3):
                        input_mrope_positions[idx].extend(next_pos[idx])
                else:
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

        if any(input_mrope_positions):
            input_positions = None  # type: ignore
        else:
            input_mrope_positions = None  # type: ignore

        max_decode_seq_len = max(seq_lens)

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device)
        input_positions = torch.tensor(input_positions
                                       or input_mrope_positions,
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
            seq_lens=seq_lens,
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


class CPUModelRunner(ModelRunnerBase[ModelInputForCPU]):
    _model_input_cls: Type[ModelInputForCPUWithSamplingMetadata] = (
        ModelInputForCPUWithSamplingMetadata)
    _builder_cls: Type[ModelInputForCPUBuilder] = ModelInputForCPUBuilder

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        *args,
        **kwargs,
    ):
        ModelRunnerBase.__init__(self, vllm_config)
        # Currently, CPU worker doesn't support chunked prefill.
        assert self.scheduler_config.chunked_prefill_enabled is False
        model_config = self.model_config
        cache_config = self.cache_config

        self.is_driver_worker = is_driver_worker

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
        self.mm_registry = MULTIMODAL_REGISTRY
        self.multi_modal_input_mapper = self.mm_registry \
            .create_input_mapper(self.model_config)
        self.mm_registry.init_mm_limits_per_prompt(self.model_config)

        # Lazy initialization.
        self.model: nn.Module  # Set after init_Model

    @property
    def model_is_mrope(self) -> bool:
        """Detect if the model has "mrope" rope_scaling type.
        mrope requires keep "rope_deltas" between prompt and decoding phases."""
        return uses_mrope(self.model_config.hf_config)

    def load_model(self) -> None:
        self.model = get_model(vllm_config=self.vllm_config)

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> ModelInputForCPUWithSamplingMetadata:
        return ModelInputForCPUWithSamplingMetadata.from_broadcasted_tensor_dict(  # noqa: E501
            tensor_dict,
            attn_backend=self.attn_backend,
        )

    def _prepare_model_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForCPUWithSamplingMetadata:
        """Helper method to prepare the model input based on a given sequence
        group. Prepares metadata needed for the base model forward pass but not
        metadata for possible additional steps, e.g., sampling.

        """
        builder = self._builder_cls(weakref.proxy(self), finished_requests_ids)
        for seq_group_metadata in seq_group_metadata_list:
            builder.add_seq_group(seq_group_metadata)

        return builder.build()  # type: ignore

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForCPUWithSamplingMetadata:
        """Prepare the model input based on a given sequence group, including
        metadata for the sampling step.

        """
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids)
        # Sampling metadata is only required for the final pp group
        generators = self.get_generators(finished_requests_ids)
        sampling_metadata = SamplingMetadata.prepare(seq_group_metadata_list,
                                                     model_input.seq_lens,
                                                     model_input.query_lens,
                                                     self.device,
                                                     pin_memory=False,
                                                     generators=generators)

        return dataclasses.replace(model_input,
                                   sampling_metadata=sampling_metadata,
                                   virtual_engine=virtual_engine)

    @torch.no_grad()
    def execute_model(
        self,
        model_input: ModelInputForCPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        if num_steps > 1:
            raise ValueError(
                "CPU worker does not support multi-step execution.")

        model_executable = self.model
        execute_model_kwargs = {
            "input_ids":
            model_input.input_tokens,
            "positions":
            model_input.input_positions,
            "kv_caches":
            kv_caches,
            "attn_metadata":
            model_input.attn_metadata,
            **MultiModalInputs.as_kwargs(model_input.multi_modal_kwargs or {},
                                         device=self.device),
            "intermediate_tensors":
            intermediate_tensors,
        }

        hidden_states = model_executable(**execute_model_kwargs)

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states,
                                           model_input.sampling_metadata)

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return []

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        return [output]
