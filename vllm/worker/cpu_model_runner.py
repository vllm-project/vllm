# SPDX-License-Identifier: Apache-2.0

import dataclasses
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Dict, List, Optional, Set, Type,
                    TypeVar, Union)

import torch
from torch import nn

from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_lora, supports_multimodal
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalKwargs, MultiModalPlaceholderMap)
from vllm.sequence import (IntermediateTensors, SequenceData,
                           SequenceGroupMetadata)
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase, ModelRunnerInputBuilderBase,
    _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict,
    _init_sampling_metadata_from_tensor_dict)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)

TModelInputForCPU = TypeVar('TModelInputForCPU', bound="ModelInputForCPU")
_PAD_SLOT_ID = -1


@dataclass(frozen=True)
class ModelInputForCPU(ModelRunnerInputBase):
    """
    Base class contains metadata needed for the base model forward pass on CPU
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    token_type_ids: Optional[torch.Tensor] = None
    attn_metadata: Optional["AttentionMetadata"] = None
    multi_modal_kwargs: Optional[BatchedTensorInputs] = None
    virtual_engine: Optional[int] = None
    seq_lens: Optional[List[int]] = None
    query_lens: Optional[List[int]] = None
    lora_mapping: Optional["LoRAMapping"] = None
    lora_requests: Optional[Set[LoRARequest]] = None

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "token_type_ids": self.token_type_ids,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "lora_requests": self.lora_requests,
            "lora_mapping": self.lora_mapping,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)

        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type[TModelInputForCPU],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None
    ) -> TModelInputForCPU:
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
    is_prompt: Optional[bool] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "token_type_ids": self.token_type_ids,
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
    ) -> "ModelInputForCPUWithSamplingMetadata":
        tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


class ModelInputForCPUBuilder(ModelRunnerInputBuilderBase[ModelInputForCPU]):

    class ModelInputData:

        def __init__(self, use_mrope: bool):
            self.use_mrope = use_mrope
            self.input_tokens: List[int] = []
            self.input_positions: List[int] = []
            self.token_type_ids: Optional[List[int]] = []
            self.seq_lens: List[int] = []
            self.query_lens: List[int] = []
            self.prefill_block_tables: List[List[int]] = []
            self.decode_block_tables: List[List[int]] = []
            self.max_decode_seq_len: int = 0
            self.num_prefills: int = 0
            self.num_prefill_tokens: int = 0
            self.num_decode_tokens: int = 0
            self.slot_mapping: List[int] = []
            self.multi_modal_inputs_list: List[MultiModalKwargs] = []
            self.multi_modal_placeholder_maps: Dict[
                str, MultiModalPlaceholderMap] = defaultdict(
                    MultiModalPlaceholderMap)
            self.input_mrope_positions: List[List[int]] = [[]
                                                           for _ in range(3)]

    def __init__(self,
                 runner: "CPUModelRunner",
                 finished_requests_ids: Optional[List[str]] = None) -> None:
        super().__init__()
        self.runner = runner
        self.chunked_prefill = (runner.scheduler_config.chunked_prefill_enabled
                                or runner.cache_config.enable_prefix_caching)
        self.model_input_cls = self.runner._model_input_cls
        self.attn_backend = self.runner.attn_backend
        self.sliding_window = self.runner.sliding_window
        self.block_size = self.runner.block_size
        self.device = self.runner.device
        self.multi_modal_input_mapper = self.runner.multi_modal_input_mapper
        self.enable_lora = self.runner.lora_config is not None
        if self.runner.attn_backend is not None:
            # spec decode (e.g. Medusa) does not have atten backend
            attn_backend = self.runner.attn_backend
            self.att_metadata_builder = attn_backend.get_builder_cls()(self)

    def prepare(self,
                finished_requests_ids: Optional[List[str]] = None) -> None:
        self.seq_group_metadata_list: List[SequenceGroupMetadata] = []
        self.input_data = ModelInputForCPUBuilder.ModelInputData(
            self.runner.model_config.uses_mrope)
        self.att_metadata_builder.prepare()

    def add_seq_group(self, seq_group_metadata: SequenceGroupMetadata):
        self.seq_group_metadata_list.append(seq_group_metadata)

    def set_seq_group_list(
            self, seq_group_metadata_list: List[SequenceGroupMetadata]):
        self.seq_group_metadata_list = seq_group_metadata_list

    def build(self) -> ModelInputForCPU:
        self._build_input_data()

        input_data = self.input_data
        input_tokens = torch.tensor(input_data.input_tokens,
                                    dtype=torch.long,
                                    device="cpu")
        input_positions = torch.tensor(
            input_data.input_positions
            if not any(input_data.input_mrope_positions) else
            input_data.input_mrope_positions,
            dtype=torch.long,
            device="cpu")
        token_type_ids = torch.tensor(input_data.token_type_ids,
                                    dtype=torch.long,
                                    device="cpu") \
                                    if input_data.token_type_ids else None

        # For multi-modal models
        multi_modal_kwargs = None
        if len(input_data.multi_modal_inputs_list) != 0:
            multi_modal_kwargs = MultiModalKwargs.batch(
                input_data.multi_modal_inputs_list)

        attn_metadata = self.att_metadata_builder.build(
            input_data.seq_lens, input_data.query_lens, -1, -1)

        is_prompt = (self.seq_group_metadata_list[0].is_prompt
                     if self.seq_group_metadata_list else None)
        # LoRA data.
        lora_requests = set()
        lora_mapping = None
        if self.enable_lora:
            lora_requests = set(seq.lora_request
                                for seq in self.seq_group_metadata_list
                                if seq.lora_request is not None)

            lora_mapping = self._prepare_lora_input(
                self.seq_group_metadata_list, is_prompt)

        return self.model_input_cls(input_tokens=input_tokens,
                                    input_positions=input_positions,
                                    token_type_ids=token_type_ids,
                                    seq_lens=input_data.seq_lens,
                                    query_lens=input_data.query_lens,
                                    attn_metadata=attn_metadata,
                                    multi_modal_kwargs=multi_modal_kwargs,
                                    lora_mapping=lora_mapping,
                                    lora_requests=lora_requests)

    def _build_input_data(self):
        for seq_group_metadata in self.seq_group_metadata_list:
            for seq_id, seq_data in seq_group_metadata.seq_data.items():
                if seq_group_metadata.is_prompt:
                    self._compute_prompt_input_tokens(self.input_data,
                                                      seq_group_metadata,
                                                      seq_data, seq_id)
                    if seq_group_metadata.multi_modal_data:
                        self._compute_multi_modal_input(
                            seq_group_metadata, seq_data)
                else:
                    self._compute_decode_input_tokens(self.input_data,
                                                      seq_group_metadata,
                                                      seq_data, seq_id)

    def _compute_decode_input_tokens(self, data: ModelInputData,
                                     seq_group_metadata: SequenceGroupMetadata,
                                     seq_data: SequenceData, seq_id: int):
        """
        Compute decode input tokens, positions, block table and slot mapping.
        """
        block_size = self.runner.block_size

        block_table = seq_group_metadata.block_tables[seq_id]
        seq_len = seq_data.get_len()
        context_len = seq_data.get_num_computed_tokens()

        tokens = seq_data.get_last_token_id()
        token_positions = seq_len - 1
        block_number = block_table[token_positions // block_size]
        block_offset = token_positions % block_size
        slot = block_number * block_size + block_offset

        # For paged_attention kernel
        if self.runner.sliding_window:
            start_idx = max(0, seq_len - self.runner.sliding_window)
            start_block = start_idx // block_size
            start_idx = start_block * block_size
            seq_len = seq_len - start_idx
            block_table = block_table[start_block:]

        # For MRotaryEmbedding
        if seq_data.mrope_position_delta is not None:
            next_pos = MRotaryEmbedding.get_next_input_positions(
                seq_data.mrope_position_delta,
                context_len,
                seq_len,
            )
            for idx in range(3):
                data.input_mrope_positions[idx].extend(  # type: ignore
                    next_pos[idx])
        else:
            data.input_positions.append(token_positions)  # type: ignore

        # Update fields
        data.input_tokens.append(tokens)
        data.max_decode_seq_len = max(data.max_decode_seq_len, seq_len)
        data.num_decode_tokens += 1
        data.slot_mapping.append(slot)
        data.decode_block_tables.append(block_table)
        data.query_lens.append(1)
        data.seq_lens.append(seq_len)

    def _compute_prompt_input_tokens(self, data: ModelInputData,
                                     seq_group_metadata: SequenceGroupMetadata,
                                     seq_data: SequenceData, seq_id: int):
        """
        Compute prompt input tokens, positions, block table and slot mapping.
        """
        token_chunk_size = seq_group_metadata.token_chunk_size
        block_size = self.runner.block_size

        block_table = seq_group_metadata.block_tables[seq_id]
        seq_len = seq_data.get_len()
        context_len = seq_data.get_num_computed_tokens()
        seq_len = min(seq_len, context_len + token_chunk_size)

        # For prefix caching
        prefix_cache_block_num = len(seq_group_metadata.computed_block_nums)
        if prefix_cache_block_num > 0:
            prefix_cache_len = (prefix_cache_block_num *
                                self.runner.block_size)
            if prefix_cache_len <= context_len:
                # We already passed the cache hit region,
                # so do normal computation.
                pass
            elif context_len < prefix_cache_len < seq_len:
                # Partial hit. Compute the missing part.
                context_len = prefix_cache_len
                token_chunk_size = seq_len - context_len
            elif seq_len <= prefix_cache_len:
                # Full hit. Only compute the last token to avoid
                # erroneous behavior. FIXME: Ideally we should directly
                # mark all tokens as computed in the scheduler and do not
                # schedule this sequence, so this case should not happen.
                context_len = seq_len - 1
                token_chunk_size = 1

        tokens = seq_data.get_token_ids()
        tokens = tokens[context_len:seq_len]
        token_positions = range(context_len, seq_len)
        token_types = seq_group_metadata.token_type_ids

        # For encoder-only models, the block_table is None,
        # and there is no need to initialize the slot_mapping.
        if block_table is not None:
            slot_mapping = [_PAD_SLOT_ID] * len(token_positions)
            for i, pos in enumerate(token_positions):
                block_number = block_table[pos // block_size]
                block_offset = pos % block_size
                slot = block_number * block_size + block_offset
                slot_mapping[i] = slot
            data.slot_mapping.extend(slot_mapping)

        # The MROPE positions are prepared in _compute_multi_modal_input
        data.input_positions.extend(token_positions)

        if data.token_type_ids is not None:
            data.token_type_ids.extend(token_types if token_types else [])

        # Update fields
        data.input_tokens.extend(tokens)
        data.num_prefills += 1
        data.num_prefill_tokens += len(tokens)
        data.query_lens.append(len(tokens))
        data.prefill_block_tables.append(block_table)
        data.seq_lens.append(seq_len)

    def _compute_multi_modal_input(self,
                                   seq_group_metadata: SequenceGroupMetadata,
                                   seq_data: SequenceData):
        computed_len = seq_data.get_num_computed_tokens()
        seq_len = self.input_data.seq_lens[-1]

        # NOTE: mm_data only includes the subset of multi-modal items that
        # intersect with the current prefill positions.
        mm_data, placeholder_maps = MultiModalPlaceholderMap.from_seq_group(
            seq_group_metadata, range(computed_len, seq_len))

        if not mm_data:
            return

        if self.runner.mm_registry.has_processor(self.runner.model_config):
            mm_kwargs = mm_data
        else:
            mm_kwargs = self.multi_modal_input_mapper(
                mm_data,
                seq_group_metadata.mm_processor_kwargs,
            )

        # special processing for mrope position deltas.
        if self.runner.model_config.uses_mrope:
            assert not self.chunked_prefill, \
                "MROPE on CPU does not support chunked-prefill."

            image_grid_thw = mm_kwargs.get("image_grid_thw", None)
            video_grid_thw = mm_kwargs.get("video_grid_thw", None)
            assert image_grid_thw is not None or video_grid_thw is not None, (
                "mrope embedding type requires multi-modal input mapper "
                "returns 'image_grid_thw' or 'video_grid_thw'.")

            second_per_grid_ts = mm_kwargs.get("second_per_grid_ts", None)
            hf_config = self.runner.model_config.hf_config
            token_ids = seq_data.get_token_ids()

            mrope_positions, mrope_position_delta = \
                MRotaryEmbedding.get_input_positions(
                    token_ids,
                    hf_config=hf_config,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    context_len=computed_len,
                )
            seq_data.mrope_position_delta = mrope_position_delta

            for i in range(3):
                self.input_data.input_mrope_positions[  # type: ignore
                    i].extend(mrope_positions[i])

        self.input_data.multi_modal_inputs_list.append(mm_kwargs)
        for modality, placeholder_map in placeholder_maps.items():
            self.input_data.multi_modal_placeholder_maps[modality].extend(
                placeholder_map)

    def _prepare_lora_input(
            self, seq_group_metadata_list: List[SequenceGroupMetadata],
            is_prefill: bool) -> LoRAMapping:
        index_mapping = []
        prompt_mapping = []
        for seq in seq_group_metadata_list:
            lora_id = seq.lora_int_id
            query_len = seq.token_chunk_size

            index_mapping += [lora_id] * query_len
            prompt_mapping += [lora_id] * (
                query_len if seq.sampling_params
                and seq.sampling_params.prompt_logprobs is not None else 1)

        return LoRAMapping(index_mapping=tuple(index_mapping),
                           prompt_mapping=tuple(prompt_mapping),
                           is_prefill=is_prefill)


class CPUModelRunnerBase(ModelRunnerBase[TModelInputForCPU]):
    """
    Helper class for shared methods between CPU model runners.
    """
    _model_input_cls: Type[TModelInputForCPU]
    _builder_cls: Type[ModelInputForCPUBuilder]
    builder: ModelInputForCPUBuilder

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        return_hidden_states: bool = False,
        *args,
        **kwargs,
    ):
        ModelRunnerBase.__init__(self, vllm_config)
        model_config = self.model_config
        cache_config = self.cache_config

        self.is_driver_worker = is_driver_worker
        self.return_hidden_states = return_hidden_states

        self.device = self.device_config.device
        self.pin_memory = False

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        num_attn_heads = self.model_config.get_num_attention_heads(
            self.parallel_config)
        needs_attn_backend = (num_attn_heads != 0
                              or self.model_config.is_attention_free)
        self.attn_backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
        ) if needs_attn_backend else None

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.multi_modal_input_mapper = self.mm_registry \
            .create_input_mapper(self.model_config)
        self.mm_registry.init_mm_limits_per_prompt(self.model_config)

        # Lazy initialization.
        self.model: nn.Module  # Set after init_Model
        # Set after load_model.
        self.lora_manager: Optional[LRUCacheWorkerLoRAManager] = None

        if hasattr(self, "_builder_cls"):
            # multi-step model runner does not have `_builder_cls`
            self.builder = self._builder_cls(weakref.proxy(self))

    def load_model(self) -> None:
        self.model = get_model(vllm_config=self.vllm_config)

        if self.lora_config:
            assert supports_lora(
                self.model
            ), f"{self.model.__class__.__name__} does not support LoRA yet."

            if supports_multimodal(self.model):
                logger.warning("Regarding multimodal models, vLLM currently "
                               "only supports adding LoRA to language model.")

            # It's necessary to distinguish between the max_position_embeddings
            # of VLMs and LLMs.
            if hasattr(self.model.config, "max_position_embeddings"):
                max_pos_embeddings = self.model.config.max_position_embeddings
            else:
                max_pos_embeddings = (
                    self.model.config.text_config.max_position_embeddings)

            self.lora_manager = LRUCacheWorkerLoRAManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
                self.vocab_size,
                self.lora_config,
                self.device,
                self.model.embedding_modules,
                self.model.embedding_padding_modules,
                max_position_embeddings=max_pos_embeddings,
            )
            self.model = self.lora_manager.create_lora_manager(self.model)

    def get_model(self) -> nn.Module:
        return self.model

    def _prepare_model_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        finished_requests_ids: Optional[List[str]] = None
    ) -> TModelInputForCPU:
        """Helper method to prepare the model input based on a given sequence
        group. Prepares metadata needed for the base model forward pass but not
        metadata for possible additional steps, e.g., sampling.

        """
        self.builder.prepare(finished_requests_ids)
        self.builder.set_seq_group_list(seq_group_metadata_list)

        return self.builder.build()  # type: ignore

    # sampler property will be used by spec_decode_worker
    @property
    def sampler(self):
        return self.model.sampler

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()

    def remove_all_loras(self):
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.remove_all_adapters()

    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.set_active_adapters(lora_requests, lora_mapping)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.add_adapter(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.remove_adapter(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.pin_adapter(lora_id)

    def list_loras(self) -> Set[int]:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.list_adapters()


class CPUModelRunner(CPUModelRunnerBase[ModelInputForCPUWithSamplingMetadata]):
    _model_input_cls: Type[ModelInputForCPUWithSamplingMetadata] = (
        ModelInputForCPUWithSamplingMetadata)
    _builder_cls: Type[ModelInputForCPUBuilder] = ModelInputForCPUBuilder

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> ModelInputForCPUWithSamplingMetadata:
        return ModelInputForCPUWithSamplingMetadata.from_broadcasted_tensor_dict(  # noqa: E501
            tensor_dict,
            attn_backend=self.attn_backend,
        )

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

        is_prompt = (seq_group_metadata_list[0].is_prompt
                     if seq_group_metadata_list else None)
        return dataclasses.replace(model_input,
                                   sampling_metadata=sampling_metadata,
                                   virtual_engine=virtual_engine,
                                   is_prompt=is_prompt)

    @torch.no_grad()
    def execute_model(
        self,
        model_input: ModelInputForCPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
        previous_hidden_states: Optional[torch.Tensor] = None,
    ) -> Optional[List[SamplerOutput]]:
        if num_steps > 1:
            raise ValueError(
                "CPU worker does not support multi-step execution.")

        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests,
                                  model_input.lora_mapping)

        model_executable = self.model

        multimodal_kwargs = {}
        if model_input.multi_modal_kwargs is not None:
            multimodal_kwargs = MultiModalKwargs.as_kwargs(
                model_input.multi_modal_kwargs, device=self.device)
        execute_model_kwargs = {}
        if previous_hidden_states is not None:
            execute_model_kwargs.update(
                {"previous_hidden_states": previous_hidden_states})

        with set_forward_context(model_input.attn_metadata, self.vllm_config,
                                 model_input.virtual_engine):
            hidden_states = model_executable(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                intermediate_tensors=intermediate_tensors,
                **execute_model_kwargs,
                **multimodal_kwargs,
            )

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
        if self.return_hidden_states:
            # we only need to pass hidden states of most recent token
            if model_input.is_prompt:
                output.prefill_hidden_states = hidden_states
            output.hidden_states = hidden_states
        return [output]

    def generate_proposals(self, *args, **kwargs):
        return self.model.generate_proposals(*args, **kwargs)
