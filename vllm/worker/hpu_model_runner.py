# SPDX-License-Identifier: Apache-2.0

###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

import collections
import contextlib
import dataclasses
import functools
import gc
import itertools
import math
import os
import time
from array import array
from enum import Enum, IntEnum
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple,
                    Optional, Set, Tuple, Type, TypeVar, Union)

import habana_frameworks.torch as htorch
import habana_frameworks.torch.internal.bridge_config as bc
import torch
import torch.nn as nn
import vllm_hpu_extension.environment as environment
from vllm_hpu_extension.bucketing import HPUBucketingContext
from vllm_hpu_extension.flags import enabled_flags
from vllm_hpu_extension.ops import LoraMask as LoraMask
from vllm_hpu_extension.ops import batch2block, block2batch
from vllm_hpu_extension.profiler import (HabanaHighLevelProfiler,
                                         HabanaMemoryProfiler, format_bytes)

from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.backends.hpu_attn import HPUAttentionImpl
from vllm.config import DeviceConfig, VllmConfig
from vllm.distributed import broadcast_tensor_dict
from vllm.distributed.parallel_state import get_world_group
from vllm.forward_context import set_forward_context
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.sampling_metadata import SequenceGroupToSample
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalKwargs, MultiModalPlaceholderMap,
                             MultiModalRegistry)
from vllm.sampling_params import SamplingParams
from vllm.sequence import (CompletionSequenceGroupOutput, IntermediateTensors,
                           Logprob, SequenceData, SequenceGroupMetadata,
                           SequenceOutput)
from vllm.utils import (bind_kv_cache, is_fake_hpu, is_pin_memory_available,
                        make_tensor_with_pad)
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase,
    _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict,
    _init_sampling_metadata_from_tensor_dict)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)

_TYPE_CACHE = {}
# These values are assumed to be zero in several places.
# Use caution when updating them!
_PAD_SLOT_ID = 0
_PAD_BLOCK_ID = 0

LORA_WARMUP_RANK = 8

VLLM_DELAYED_SAMPLING = os.environ.get('VLLM_DELAYED_SAMPLING',
                                       'false').lower() == 'true'
DUMMY_TOKEN_ID = -1


class PhaseType(Enum):
    PREFILL = 'prefill'
    PREFIX_PREFILL = 'prefix_prefill'
    DECODE = 'decode'


def subtuple(obj: object,
             typename: str,
             to_copy: List[str],
             to_override: Optional[Dict[str, object]] = None):
    if obj is None:
        return None
    if to_override is None:
        to_override = {}
    fields = set(to_copy) | set(to_override.keys())
    if type(obj) is dict:
        values = {key: obj[key] for key in fields if key in obj}
    else:
        values = {f: to_override.get(f, getattr(obj, f)) for f in fields}
    if typename not in _TYPE_CACHE:
        _TYPE_CACHE[typename] = collections.namedtuple(typename,
                                                       ' '.join(fields))
    return _TYPE_CACHE[typename](**values)


def align_workers(value, op):
    group = get_world_group().cpu_group
    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        return value
    value_t = torch.tensor(value, device='cpu')
    torch.distributed.all_reduce(value_t, op=op, group=group)
    return value_t.item()


def setup_profiler():
    schedule = torch.profiler.schedule(wait=0, warmup=2, active=1, repeat=1)
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.HPU
    ]
    profiler = torch.profiler.profile(
        schedule=schedule,
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('.',
                                                                use_gzip=True),
        record_shapes=False,
        with_stack=True)
    return profiler


def round_up(value: int, k: int) -> int:
    return (value + k - 1) // k * k


def pad_list(input, k, v):
    input_len = len(input)
    target_len = round_up(input_len, k)
    padding = target_len - input_len
    return input + [v] * padding


def gather_list(input, indices, v):
    return [input[i] if i is not None else v for i in indices]


def flatten(in_list):
    return list(itertools.chain(*in_list))


def get_target_layer_suffix_list(model_type) -> list[str]:
    # This sets the suffix for the hidden layer name, which is controlled by
    # VLLM_CONFIG_HIDDEN_LAYERS. The default suffix is "DecoderLayer," which is
    # applicable for most language models such as LLaMA, Qwen, and BART. If the
    # model's decoder layer name differs from the default, it will need to
    # be specified here.
    decoder_layer_table = {
        "gpt_bigcode": "BigCodeBlock",
    }

    return [
        decoder_layer_table.get(model_type, "DecoderLayer"), "EncoderLayer"
    ]


def modify_model_layers(module: torch.nn.Module,
                        suffix_list: list[str],
                        n=1,
                        counter=None):
    """Currently add mark_step at the end of specified layers.
    """

    def forward_hook(module, args, output):
        htorch.core.mark_step()
        return output

    if counter is None:
        counter = [0]

    for child_name, child_module in module.named_children():
        if any(
                child_module.__class__.__name__.endswith(layer)
                for layer in suffix_list):
            counter[0] += 1
            if counter[0] % n == 0:
                child_module.register_forward_hook(forward_hook)
        else:
            modify_model_layers(child_module, suffix_list, n, counter)


def get_path_to_rope(model: torch.nn.Module):
    """Dynamically get the path to the RotaryEmbedding layer in the model.
    This function will recursively search through the module hierarchy to find
    a RotaryEmbedding layer and return the full path to that layer as a list
    of names.
    If no such layer is found, it returns None.
    """

    def find_rope_layer(parent, path):
        # Base case: check if this parent is None
        if parent is None:
            return None

        # Check if the current layer is a RotaryEmbedding
        if hasattr(parent, 'named_children'):
            for child_name, child_module in parent.named_children():
                # If the current child is of type RotaryEmbedding,
                # return the full path
                if child_module.__class__.__name__.endswith("RotaryEmbedding"):
                    return path + [child_name]
                # Otherwise, recurse into this child to check its children
                result = find_rope_layer(child_module, path + [child_name])
                if result is not None:
                    return result
        return None

    # Start the search from the top level model
    path_to_rope = find_rope_layer(model, [])

    # Return the result if found, otherwise None
    return path_to_rope


class HpuModelAdapter:

    def __init__(self, model, vllm_config, layer_names):
        self.model = model
        self.prefill_use_fusedsdpa = "fsdpa" in enabled_flags()
        self.recompute_cos_sin = os.getenv('VLLM_COS_SIN_RECOMPUTE',
                                           'false').lower() in ['1', 'true']
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.dtype = vllm_config.model_config.dtype
        self.layer_names = layer_names
        enforce_eager = vllm_config.model_config.enforce_eager
        self.is_pooler = hasattr(self.model, "_pooler")
        self.is_causal = True
        if self.is_pooler:
            self.set_causal_option(self.model)
        if not is_fake_hpu() and not htorch.utils.internal.is_lazy(
        ) and not enforce_eager:
            if os.getenv('VLLM_REGIONAL_COMPILATION',
                         'true').lower() == 'true':
                self.regional_compilation_layers_list = [
                    RMSNorm, VocabParallelEmbedding
                ]
                self._regional_compilation(self.model)
            else:
                self.model = torch.compile(self.model,
                                           backend='hpu_backend',
                                           dynamic=False)

    def _regional_compilation(self,
                              module,
                              parent_module=None,
                              module_name=None):
        if isinstance(module, torch.nn.ModuleList):
            for children_name, children_module in module.named_children():
                self._compile_region(module, children_name, children_module)
        elif any(
                isinstance(module, layer)
                for layer in self.regional_compilation_layers_list):
            self._compile_region(parent_module, module_name, module)
        else:
            for children_name, children_module in module.named_children():
                self._regional_compilation(children_module, module,
                                           children_name)

    def _compile_region(self, model, name, module):
        module = torch.compile(module, backend='hpu_backend', dynamic=False)
        setattr(model, name, module)

    def _set_attn_bias(self, attn_metadata, batch_size, seq_len, device,
                       dtype):
        if (attn_metadata is None
                or (self.prefill_use_fusedsdpa and self.is_causal
                    and attn_metadata.block_list is None)
                or not attn_metadata.is_prompt):
            return attn_metadata

        prefill_metadata = attn_metadata

        seq_lens_t = prefill_metadata.seq_lens_tensor
        context_lens_t = prefill_metadata.context_lens_tensor
        query_lens_t = seq_lens_t - context_lens_t

        block_list = attn_metadata.block_list
        max_context_len = (block_list.size(-1) //
                           batch_size if block_list is not None else 0)
        max_context_len = max_context_len * self.block_size
        past_mask = torch.arange(0,
                                 max_context_len,
                                 dtype=torch.int32,
                                 device=device)
        past_mask = (past_mask.view(1, -1).expand(batch_size, -1).ge(
            context_lens_t.view(-1, 1)).view(batch_size, 1, -1).expand(
                batch_size, seq_len, -1).view(batch_size, 1, seq_len, -1))

        len_mask = (torch.arange(0, seq_len, device=device,
                                 dtype=torch.int32).view(1, seq_len).ge(
                                     query_lens_t.unsqueeze(-1)).view(
                                         batch_size, 1, 1, seq_len))
        if self.is_causal:
            attn_mask = torch.triu(torch.ones(
                (batch_size, 1, seq_len, seq_len),
                device=device,
                dtype=torch.bool),
                                   diagonal=1)
        else:
            attn_mask = torch.zeros((batch_size, 1, seq_len, seq_len),
                                    device=device,
                                    dtype=torch.bool)
        if self.is_pooler:
            len_mask_v = len_mask.view(batch_size, 1, seq_len, 1)
            mask = attn_mask.logical_or(len_mask).logical_or(len_mask_v)
            off_value = -3E38  #small number, avoid nan and overflow
        else:
            mask = attn_mask.logical_or(
                len_mask)  #no need for len_mask_v as decode overwrites it
            off_value = -math.inf

        mask = torch.concat((past_mask, mask), dim=-1)
        attn_bias = (torch.zeros_like(mask, dtype=dtype).masked_fill_(
            mask, off_value))
        attn_metadata = prefill_metadata._replace(attn_bias=attn_bias)
        return attn_metadata

    def _set_block_mapping(self, metadata, batch_size, device, dtype):
        mask = torch.arange(0,
                            self.block_size,
                            device=device,
                            dtype=torch.int32).unsqueeze(0)
        mask = mask >= metadata.block_usage.unsqueeze(-1)
        attn_bias = (torch.zeros_like(mask, dtype=dtype).masked_fill_(
            mask, -math.inf))

        if not is_fake_hpu():
            block_mapping = torch.nn.functional.one_hot(metadata.block_groups,
                                                        num_classes=batch_size)
        else:
            # Unfortunately one_hot on CPU
            # doesn't handle out of bounds classes so we need to convert
            # all negative values to 0 (block_mapping) or bs (block_groups)
            block_groups = metadata.block_groups.to(torch.long)
            block_mapping = torch.nn.functional.relu(block_groups)
            block_mapping = torch.nn.functional.one_hot(block_mapping,
                                                        num_classes=batch_size)
            oob_values = block_groups.lt(0)
            block_mapping.masked_fill_(oob_values.unsqueeze(-1), 0)
            block_groups.masked_fill_(oob_values, batch_size)
            metadata = metadata._replace(block_groups=block_groups)
        block_mapping = block_mapping.to(dtype)
        metadata = metadata._replace(block_mapping=block_mapping,
                                     attn_bias=attn_bias)
        return metadata

    def _set_block_scales(self, metadata, device):
        block_mapping = metadata.block_mapping
        ones = torch.ones((block_mapping.size(0), ),
                          device=device,
                          dtype=block_mapping.dtype)
        sums = batch2block(block2batch(ones, block_mapping), block_mapping)
        block_scales = torch.reciprocal(torch.maximum(ones, sums))
        metadata = metadata._replace(block_scales=block_scales)
        return metadata

    def _set_indices_and_offsets(self, metadata, block_size, is_prompt):
        slot_mapping = metadata.slot_mapping.flatten()
        indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
        if is_prompt:
            indices = indices.unflatten(0, (-1, block_size))[:, 0]
            offsets = None
        else:
            offsets = torch.fmod(slot_mapping, block_size)
        metadata = metadata._replace(block_offsets=offsets,
                                     block_indices=indices)
        return metadata

    def _update_metadata(self, attn_metadata, batch_size, seq_len, device,
                         dtype):

        if attn_metadata.is_prompt:
            attn_metadata = self._set_attn_bias(attn_metadata, batch_size,
                                                seq_len, device, dtype)
        else:
            attn_metadata = self._set_block_mapping(attn_metadata, batch_size,
                                                    device, dtype)
            attn_metadata = self._set_block_scales(attn_metadata, device)
        attn_metadata = self._set_indices_and_offsets(attn_metadata,
                                                      self.block_size,
                                                      attn_metadata.is_prompt)
        return attn_metadata

    def _prepare_cos_sin(self, positions):
        """Navigate through the model using the provided path and call
        the prepare_cos_sin method on the 'RotaryEmbedding' layer."""

        current_module = self.model  # Start from the top level of the model

        for layer in self.layer_names:
            if layer.isdigit():  # Check if the layer is an index
                layer = int(layer)

            # Check if the current layer is a name in a module
            if isinstance(
                    layer,
                    str) and not isinstance(layer, int):  # Name-based access
                current_module = getattr(current_module, layer)
            elif isinstance(layer,
                            int):  # Indexed-based access (like ModuleList)
                current_module = list(current_module._modules.values())[layer]

        # At the end, we should be at the RotaryEmbedding layer.
        if hasattr(current_module, 'prepare_cos_sin'):
            current_module.prepare_cos_sin(
                positions, recompute_cos_sin=self.recompute_cos_sin)
        else:
            raise AttributeError(
                "The module at the end of the path does not have \
                a 'prepare_cos_sin' method.")

    def forward(self, *args, **kwargs):
        kwargs = kwargs.copy()
        selected_token_indices = kwargs.pop('selected_token_indices')
        if 'warmup_mode' in kwargs:
            kwargs.pop('warmup_mode')
        virtual_engine = 0
        if 'virtual_engine' in kwargs:
            virtual_engine = kwargs.pop('virtual_engine')
        input_ids = kwargs['input_ids']
        kwargs['attn_metadata'] = self._update_metadata(
            kwargs['attn_metadata'], input_ids.size(0), input_ids.size(1),
            input_ids.device, self.dtype)
        if 'lora_mask' in kwargs:
            LoraMask.setLoraMask(kwargs.pop('lora_mask'))
        if self.layer_names is not None:
            self._prepare_cos_sin(kwargs['positions'])

        with set_forward_context(kwargs['attn_metadata'], self.vllm_config,
                                 virtual_engine):
            hidden_states = self.model(*args, **kwargs)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            if selected_token_indices is not None:
                hidden_states = hidden_states.index_select(
                    0, selected_token_indices)
        return hidden_states

    def compute_logits(self, *args, **kwargs):
        return self.model.compute_logits(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)

    def generate_proposals(self, *args, **kwargs):
        return self.model.generate_proposals(*args, **kwargs)

    def set_causal_option(self, module):
        if isinstance(module, HPUAttentionImpl) and hasattr(
                module, 'attn_type'):
            self.is_causal = not (
                module.attn_type == AttentionType.ENCODER
                or module.attn_type == AttentionType.ENCODER_ONLY
                or module.attn_type == AttentionType.ENCODER_DECODER)
            return
        else:
            for child_name, child_module in module.named_children():
                self.set_causal_option(child_module)

    # sampler property will be used by spec_decode_worker
    # don't rename
    @property
    def sampler(self):
        return self.model.sampler


class PreparePromptMetadata(NamedTuple):
    input_tokens: torch.Tensor
    input_positions: List[List[int]]
    attn_metadata: Optional[AttentionMetadata]
    seq_lens: List[int]
    query_lens: List[int]
    lora_index_mapping: List[List[int]]
    lora_prompt_mapping: List[List[int]]
    lora_requests: Set[LoRARequest]
    multi_modal_kwargs: Optional[Dict[str, BatchedTensorInputs]]
    slot_mapping: List[List[int]]
    lora_ids: List[int]

    @classmethod
    def empty(cls):
        return PreparePromptMetadata(input_tokens=[],
                                     input_positions=[],
                                     attn_metadata=None,
                                     seq_lens=[],
                                     query_lens=[],
                                     lora_index_mapping=[],
                                     lora_prompt_mapping=[],
                                     lora_requests=set(),
                                     multi_modal_kwargs=None,
                                     slot_mapping=[],
                                     lora_ids=[])


class PrepareDecodeMetadata(NamedTuple):
    input_tokens: torch.Tensor
    input_positions: List[List[int]]
    attn_metadata: Optional[AttentionMetadata]
    lora_index_mapping: List[List[int]]
    lora_prompt_mapping: List[List[int]]
    lora_requests: Set[LoRARequest]
    slot_mapping: List[List[int]]
    lora_ids: List[int]

    @classmethod
    def empty(cls):
        return PrepareDecodeMetadata(input_tokens=[],
                                     input_positions=[],
                                     attn_metadata=None,
                                     lora_index_mapping=[],
                                     lora_prompt_mapping=[],
                                     lora_requests=set(),
                                     slot_mapping=[],
                                     lora_ids=[])


# How batches are constructed.
class BatchType(IntEnum):
    # Every batch is prefill.
    PREFILL = 0
    # Every batch is decode.
    DECODE = 1
    # Batch is a mixture of prefill and decode.
    MIXED = 2


TModelInputForHPU = TypeVar('TModelInputForHPU', bound="ModelInputForHPU")


@dataclasses.dataclass(frozen=True)
class ModelInputForHPU(ModelRunnerInputBase):
    """
    This base class contains metadata needed for the base model forward pass
    but not metadata for possible additional steps, e.g., sampling. Model
    runners that run additional steps should subclass this method to add
    additional fields.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    seq_lens: Optional[List[int]] = None
    query_lens: Optional[List[int]] = None
    lora_mapping: Optional["LoRAMapping"] = None
    lora_requests: Optional[Set[LoRARequest]] = None
    attn_metadata: Optional["AttentionMetadata"] = None
    multi_modal_kwargs: Optional[Dict[str, torch.Tensor]] = None
    real_batch_size: Optional[int] = None
    batch_size_padded: Optional[int] = None
    virtual_engine: int = 0
    lora_ids: Optional[List[int]] = None
    async_callback: Optional[Callable] = None
    is_first_multi_step: bool = True
    is_last_step: bool = True

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "lora_requests": self.lora_requests,
            "lora_mapping": self.lora_mapping,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "real_batch_size": self.real_batch_size,
            "batch_size_padded": self.batch_size_padded,
            "virtual_engine": self.virtual_engine,
            "lora_ids": self.lora_ids,
            "is_first_multi_step": self.is_first_multi_step,
            "is_last_step": self.is_last_step,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type[TModelInputForHPU],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> TModelInputForHPU:
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


@dataclasses.dataclass(frozen=True)
class ModelInputForHPUWithSamplingMetadata(ModelInputForHPU):
    """
    Used by the ModelRunner.
    """
    sampling_metadata: Optional["SamplingMetadata"] = None
    # Used for speculative decoding. We do not broadcast it because it is only
    # used by the driver worker.
    is_prompt: Optional[bool] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "lora_requests": self.lora_requests,
            "lora_mapping": self.lora_mapping,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "lora_ids": self.lora_ids,
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
    ) -> "ModelInputForHPUWithSamplingMetadata":
        tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        # FIXME(kzawora): this fails for whatever reason - why?
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


class HPUModelRunnerBase(ModelRunnerBase[TModelInputForHPU]):
    """
    Helper class for shared methods between GPU model runners.
    """
    _model_input_cls: Type[TModelInputForHPU]

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
        environment.set_model_config(self.model_config)
        self.is_driver_worker = is_driver_worker
        self.return_hidden_states = return_hidden_states

        self.sliding_window = (self.model_config.get_sliding_window()
                               if self.model_config is not None else None)
        self.device_config = (self.device_config if self.device_config
                              is not None else DeviceConfig())
        if is_fake_hpu():
            self.device_config.device = torch.device('cpu')
            self.device_config.device_type = 'cpu'
            self.load_config.device = None
        self.device = self.device_config.device
        self.enforce_eager = self.model_config.enforce_eager
        self.max_num_seqs = self.scheduler_config.max_num_seqs
        self.max_num_prefill_seqs = self.scheduler_config.max_num_prefill_seqs \
            if self.scheduler_config.max_num_prefill_seqs is not None \
                else self.max_num_seqs
        self.max_model_len = self.scheduler_config.max_model_len
        self.max_num_batched_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.block_size = self.cache_config.block_size

        self.pin_memory = is_pin_memory_available()
        self.kv_cache_dtype = self.cache_config.cache_dtype

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
        self.input_registry = input_registry
        self.mm_registry = mm_registry
        self.mm_registry = MULTIMODAL_REGISTRY
        self.multi_modal_input_mapper = self.mm_registry \
            .create_input_mapper(self.model_config)
        self.mm_registry.init_mm_limits_per_prompt(self.model_config)

        # Lazy initialization
        self.lora_manager: LRUCacheWorkerLoRAManager = None
        self.model: torch.nn.Module = None
        self.inc_initialized_successfully = False

        # Profiler stats
        self.profiler = HabanaHighLevelProfiler()
        self.profiler_counter_helper = HabanaProfilerCounterHelper()
        self.seen_configs: set = set()
        self._mem_margin: Optional[int] = None
        self.bucketing_ctx = HPUBucketingContext(self.max_num_seqs,
                                                 self.max_num_prefill_seqs,
                                                 self.block_size,
                                                 self.max_num_batched_tokens)
        self.graphed_buckets: Set[Any] = set()

        self._set_gc_threshold()
        self.use_contiguous_pa = os.environ.get('VLLM_CONTIGUOUS_PA',
                                                'true').lower() == 'true'
        if vllm_config.speculative_config is not None \
            and self.use_contiguous_pa:
            raise ValueError(
                "Speculative decoding is not supported with "
                "contiguous PA, please set VLLM_CONTIGUOUS_PA=false")
        # For both multi-step scheduling and delayed sampling
        self.cached_step_outputs: List[torch.Tensor] = []
        self.is_pooler = False
        # For delayed sampling
        self.cached_step_inputs: List[
            ModelInputForHPUWithSamplingMetadata] = []

    def _set_gc_threshold(self) -> None:
        """
        Read https://docs.python.org/3/library/gc.html#gc.set_threshold
        for comprehensive description of gc generations.
        We can either use VLLM_GC_THR_GEN[0-2] (this has higher priority)
        to set particular generation threshold or use simpler
        VLLM_GC_THR_MULTIPLIER to multiply default values.
        """

        # gc.get_threshold default, avoiding potential overflow due to
        # multiplier and set later (get->mult->set->repeat->...->overflow)
        default_gc_thrs = [700, 10, 10]

        requested_gc_thrs = [0] * len(default_gc_thrs)
        for i in range(len(default_gc_thrs)):
            requested_gc_thrs[i] = int(
                os.environ.get(f'VLLM_GC_THR_GEN{i}', default_gc_thrs[i]))
        if requested_gc_thrs == default_gc_thrs:
            # 16*threshold is rare enough for gc to not cause perf issues
            gc_thr_multiplier = int(
                os.environ.get('VLLM_GC_THR_MULTIPLIER', 16))
            requested_gc_thrs = [
                t * gc_thr_multiplier for t in default_gc_thrs
            ]
        gc.set_threshold(*requested_gc_thrs)

        # Multi-modal data support
        self.multi_modal_input_mapper = MULTIMODAL_REGISTRY \
            .create_input_mapper(self.model_config)

        self.skip_warmup = os.environ.get('VLLM_SKIP_WARMUP',
                                          'false').lower() == 'true'

    def load_model(self) -> None:
        import habana_frameworks.torch.core as htcore
        if self.model_config.quantization == 'inc' or \
           self.model_config.quantization == 'fp8':
            htcore.hpu_set_env()
        with HabanaMemoryProfiler() as m:
            with HabanaMemoryProfiler() as m_getmodel:
                self.model = get_model(vllm_config=self.vllm_config)
            msg = ("Pre-loading model weights on "
                   f"{next(self.model.parameters()).device} "
                   f"took {m_getmodel.get_summary_string()}")
            logger.info(msg)
            self.is_pooler = hasattr(self.model, "_pooler")
            if self.lora_config:
                assert hasattr(self.model, "supported_lora_modules"
                               ) and self.model.supported_lora_modules, (
                                   "Model does not support LoRA")
                assert hasattr(self.model, "embedding_modules"
                               ), "Model does not have embedding_modules"
                assert hasattr(
                    self.model, "embedding_padding_modules"
                ), "Model does not have embedding_padding_modules"
                assert not self.lora_config.bias_enabled, \
                    "Bias support in LoRA is not enabled in HPU yet."
                assert not self.lora_config.fully_sharded_loras, \
                    "Fully sharded LoRAs is not enabled in HPU yet."
                if supports_multimodal(self.model):
                    logger.warning(
                        "Regarding multimodal models, vLLM currently "
                        "only supports adding LoRA to language model.")
                # It's necessary to distinguish between the
                # max_position_embeddings of VLMs and LLMs.
                if hasattr(self.model.config, "max_position_embeddings"):
                    max_pos_embeddings = (
                        self.model.config.max_position_embeddings)
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

            if self.model_config.quantization == 'inc':
                logger.info("Preparing model with INC..")
                with HabanaMemoryProfiler() as m_inc:
                    from neural_compressor.torch.quantization import (
                        FP8Config, convert, prepare)
                    config = FP8Config.from_json_file(
                        os.getenv("QUANT_CONFIG", ""))
                    if config.measure:
                        self.model = prepare(self.model, config)
                    elif config.quantize:
                        self.model = convert(self.model, config)
                    htcore.hpu_initialize(self.model,
                                          mark_only_scales_as_const=True)
                self.inc_initialized_successfully = True
                logger.info("Preparing model with INC took %s",
                            m_inc.get_summary_string())
            elif not is_fake_hpu():
                self.model = self.model.to("hpu")
                htcore.mark_step()

            hidden_layer_markstep_interval = int(
                os.getenv('VLLM_CONFIG_HIDDEN_LAYERS', '1'))
            model_config = getattr(self.model, "config", None)
            modify_model_layers(
                self.model,
                get_target_layer_suffix_list(
                    model_config.
                    model_type if model_config is not None else None),
                hidden_layer_markstep_interval)
            path_to_rope = get_path_to_rope(self.model)
            torch.hpu.synchronize()

            with HabanaMemoryProfiler() as m_wrap:
                self.model = self._maybe_wrap_in_hpu_graph(
                    self.model,
                    vllm_config=self.vllm_config,
                    layer_names=path_to_rope)
            msg = f"Wrapping in HPU Graph took {m_wrap.get_summary_string()}"
            logger.info(msg)

        self.model_memory_usage = m.consumed_device_memory
        msg = f"Loading model weights took in total {m.get_summary_string()}"
        logger.info(msg)

    def _add_dummy_seq(self, seq_group_metadata_list, is_prompt):
        real_batch_size = len(seq_group_metadata_list)
        batch_size_padded = self.bucketing_ctx.get_padded_batch_size(
            real_batch_size, is_prompt)
        batch_size_padding = batch_size_padded - real_batch_size

        seq_group_metadata_list = seq_group_metadata_list.copy()

        if batch_size_padding > 0:
            if self.is_pooler:
                temperature = None
            else:
                has_greedy_samples = any(
                    seq_group_metadata.sampling_params.temperature == 0.0
                    for seq_group_metadata in seq_group_metadata_list)
                temperature = 0.0 if has_greedy_samples else 1.0
            dummy_seq_group_metadata = self.create_dummy_seq_group_metadata(
                -1, 0, is_prompt, temperature=temperature)
            seq_group_metadata_list.extend(dummy_seq_group_metadata
                                           for _ in range(batch_size_padding))
        return seq_group_metadata_list, real_batch_size, batch_size_padded

    def _maybe_wrap_in_hpu_graph(self, *args, **kwargs):
        return htorch.hpu.wrap_in_hpu_graph(
            HpuModelAdapter(*args, **kwargs), disable_tensor_cache=True
        ) if htorch.utils.internal.is_lazy() else HpuModelAdapter(
            *args, **kwargs)

    def get_model(self) -> nn.Module:
        if isinstance(self.model, HpuModelAdapter):
            return self.model.model
        return self.model

    def _use_graphs(self, batch_size, seq_len, is_prompt):
        if self.enforce_eager:
            return False
        if self.skip_warmup:
            return True
        return (batch_size, seq_len, is_prompt) in self.graphed_buckets

    def _is_valid_bucket(self, bucket):
        return bucket[0] * bucket[1] <= self.max_num_batched_tokens

    def _num_blocks(self, attn_metadata):
        if attn_metadata.block_list is None:
            return 0
        return attn_metadata.block_list.numel()

    def _phase(self, attn_metadata):
        phase_type: PhaseType
        is_prompt = attn_metadata.is_prompt
        is_prefix_prefill = is_prompt and attn_metadata.block_list is not None
        if is_prompt and is_prefix_prefill:
            phase_type = PhaseType.PREFIX_PREFILL
        elif is_prompt and not is_prefix_prefill:
            phase_type = PhaseType.PREFILL
        elif not is_prompt:
            phase_type = PhaseType.DECODE
        else:
            raise ValueError("Unrecognized pass type, likely due to malformed "
                             "attention metadata")
        return phase_type

    def _check_config(self, batch_size, seq_len, attn_metadata, warmup_mode):
        is_prefix_caching = self.vllm_config.cache_config.enable_prefix_caching
        cfg: Optional[tuple] = None
        assert cfg is None, "Configs changed between 2D and 3D"
        if is_prefix_caching:
            phase = self._phase(attn_metadata)
            num_blocks = self._num_blocks(attn_metadata)
            cfg = (batch_size, seq_len, num_blocks, phase)
            phase = phase.value
        else:
            phase = 'prompt' if attn_metadata.is_prompt else 'decode'
            cfg = (batch_size, seq_len, phase)
        seen = cfg in self.seen_configs
        self.seen_configs.add(cfg)
        if not seen and not warmup_mode:
            logger.warning("Configuration: %s was not warmed-up!",
                           (phase.value, batch_size, seq_len,
                            num_blocks) if is_prefix_caching else
                           (phase, batch_size, seq_len))

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> PreparePromptMetadata:
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []
        lora_index_mapping: List[List[int]] = []
        lora_prompt_mapping: List[List[int]] = []
        lora_requests: Set[LoRARequest] = set()

        seq_lens: List[int] = []
        context_lens: List[int] = []
        query_lens: List[int] = []
        prefix_block_tables: List[List[int]] = []
        multi_modal_kwargs_list: List[MultiModalKwargs] = []
        multi_modal_placeholder_maps: Dict[
            str, MultiModalPlaceholderMap] = collections.defaultdict(
                MultiModalPlaceholderMap)

        if len(seq_group_metadata_list) == 0:
            return PreparePromptMetadata.empty()

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            computed_block_nums = seq_group_metadata.computed_block_nums
            if (self.scheduler_config is not None
                    and self.scheduler_config.chunked_prefill_enabled
                    and not (computed_block_nums is None
                             or computed_block_nums == [])):
                raise RuntimeError(
                    "chunked prefill cannot be used with prefix caching "
                    "now.")

            token_chunk_size = seq_group_metadata.token_chunk_size
            seq_data = seq_group_metadata.seq_data[seq_id]
            context_len = seq_data.get_num_computed_tokens()
            # We should use get_len here because in case of preemption
            # it contains output tokens.
            seq_len = min(seq_data.get_len(), context_len + token_chunk_size)
            prompt_tokens = seq_data.get_token_ids()[context_len:seq_len]
            seq_lens.append(seq_len)

            # NOTE: This only works for oooooooxxx style attention.
            if computed_block_nums is not None and len(
                    computed_block_nums) > 0 and self.sliding_window is None:
                # Prefix is not supported with sliding_window
                context_len = len(computed_block_nums) * self.block_size
                prompt_tokens = prompt_tokens[context_len:]
                prefix_block_tables.append(computed_block_nums)
            elif self.scheduler_config.chunked_prefill_enabled:
                if seq_group_metadata.block_tables is not None:
                    # Prefill has chunked before.
                    block_table = seq_group_metadata.block_tables[seq_id]
                    prefix_block_tables.append(block_table)
                else:
                    # The first prefill.
                    prefix_block_tables.append([])
            else:
                prefix_block_tables.append([])
                # Right now, prefill start is always 0. However, this
                # assumption can be changed once chunked prefill is introduced.
                assert context_len == 0

            # actual prompt lens
            context_lens.append(context_len)
            query_lens.append(seq_len - context_len)
            input_tokens.append(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.append(list(range(context_len, seq_len)))

            if seq_group_metadata.multi_modal_data:
                positions = input_positions[0]
                mm_data, placeholder_maps = MultiModalPlaceholderMap \
                    .from_seq_group(seq_group_metadata,
                      range(positions[0], positions[0] + len(positions)))

                if self.mm_registry.has_processor(self.model_config):
                    mm_kwargs = mm_data
                else:
                    mm_kwargs = self.multi_modal_input_mapper(
                        mm_data,
                        seq_group_metadata.mm_processor_kwargs,
                    )

                multi_modal_kwargs_list.append(mm_kwargs)

                for modality, placeholder_map in placeholder_maps.items():
                    multi_modal_placeholder_maps[modality].extend(
                        placeholder_map)

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.append([_PAD_SLOT_ID] * seq_len)
                continue

            # Compute the slot mapping.
            slot_mapping.append([])
            block_table = seq_group_metadata.block_tables[seq_id]

            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, seq_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                assert context_len == 0, (
                    "Prefix caching is currently not supported with "
                    "sliding window attention")
                start_idx = max(0, seq_len - self.sliding_window)
            for i in range(context_len, seq_len):
                if i < start_idx:
                    slot_mapping[-1].append(_PAD_SLOT_ID)
                    continue
                # For encoder-only models, the block_table is None,
                # and there is no need to initialize the slot_mapping.
                if block_table is not None:
                    block_number = block_table[i // self.block_size]
                    block_offset = i % self.block_size
                    slot = block_number * self.block_size + block_offset
                    slot_mapping[-1].append(slot)

        max_query_len = max(query_lens)
        real_num_seqs = len(query_lens)

        assert max_query_len > 0

        max_prompt_len = max(
            self.bucketing_ctx.get_padded_prompt_seq_len(max_query_len),
            self.block_size)

        lora_ids: List[int] = []
        for seq_group_metadata, context_len in zip(seq_group_metadata_list,
                                                   context_lens):
            lora_id = seq_group_metadata.lora_int_id
            lora_ids.append(lora_id)

            if lora_id > 0:
                lora_requests.add(seq_group_metadata.lora_request)

            lora_index_mapping += [lora_id] * max_prompt_len
            lora_prompt_mapping.extend(
                [lora_id] *
                (max_prompt_len if seq_group_metadata.sampling_params and
                 seq_group_metadata.sampling_params.prompt_logprobs else 1))

        if any(context_lens):
            assert not self.scheduler_config.chunked_prefill_enabled
            # prefix caching

            max_num_block = max(len(bt) for bt in prefix_block_tables)
            prefix_block_list = list(
                itertools.chain.from_iterable(
                    bt if len(bt) == max_num_block else bt +
                    ([_PAD_BLOCK_ID] * (max_num_block - len(bt)))
                    for bt in prefix_block_tables))

            # TODO: pad to proper len
            pad_len = len(prefix_block_list)
            prefix_block_list = pad_list(prefix_block_list, pad_len,
                                         _PAD_BLOCK_ID)

            prefix_block_list_tensor = torch.tensor(prefix_block_list,
                                                    dtype=torch.long,
                                                    device='cpu')
        else:
            prefix_block_list_tensor = None

        input_tokens_tensor = make_tensor_with_pad(input_tokens,
                                                   max_len=max_prompt_len,
                                                   pad=0,
                                                   dtype=torch.long,
                                                   device='cpu')

        input_positions = make_tensor_with_pad(input_positions,
                                               max_len=max_prompt_len,
                                               pad=0,
                                               dtype=torch.long,
                                               device='cpu')

        slot_mapping = make_tensor_with_pad(slot_mapping,
                                            max_len=max_prompt_len,
                                            pad=_PAD_SLOT_ID,
                                            dtype=torch.long,
                                            device='cpu')
        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.long,
                                       device='cpu')

        context_lens_tensor = torch.tensor(context_lens,
                                           dtype=torch.long,
                                           device='cpu')

        placeholder_index_maps = {
            modality: placeholder_map.index_map()
            for modality, placeholder_map in
            multi_modal_placeholder_maps.items()
        }

        # Note: num_prefill_tokens is calculated using the length of
        # input_tokens after padding.
        num_prefill_tokens = input_tokens_tensor.numel()
        if prefix_block_list_tensor is not None:
            prefix_block_list_tensor = prefix_block_list_tensor.to(
                self.device, non_blocking=True)
        input_tokens_tensor = input_tokens_tensor.to(  # type: ignore
            self.device, non_blocking=True)
        input_positions = input_positions.to(  # type: ignore
            self.device, non_blocking=True)
        slot_mapping = slot_mapping.to(  # type: ignore
            self.device, non_blocking=True)
        seq_lens_tensor = seq_lens_tensor.to(self.device, non_blocking=True)
        context_lens_tensor = context_lens_tensor.to(self.device,
                                                     non_blocking=True)

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=True,
            block_list=prefix_block_list_tensor,
            block_mapping=None,
            block_usage=None,
            block_indices=None,
            block_offsets=None,
            block_scales=None,
            block_groups=None,
            attn_bias=None,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            context_lens_tensor=context_lens_tensor,
            num_prefills=real_num_seqs,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            enable_kv_scales_calculation=False,
        )
        multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_kwargs_list)
        for t in multi_modal_kwargs:
            if torch.is_tensor(multi_modal_kwargs[t]):
                multi_modal_kwargs[t] = multi_modal_kwargs[t].to(
                    self.device, non_blocking=True)

        return PreparePromptMetadata(input_tokens=input_tokens_tensor,
                                     input_positions=input_positions,
                                     attn_metadata=attn_metadata,
                                     seq_lens=seq_lens,
                                     query_lens=query_lens,
                                     lora_index_mapping=lora_index_mapping,
                                     lora_prompt_mapping=lora_prompt_mapping,
                                     lora_requests=lora_requests,
                                     multi_modal_kwargs=multi_modal_kwargs,
                                     slot_mapping=slot_mapping,
                                     lora_ids=lora_ids)

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        output=None,
    ) -> PrepareDecodeMetadata:
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []
        seq_lens: List[int] = []
        encoder_seq_lens: List[int] = []
        cross_block_tables: List[List[int]] = []
        block_tables: List[List[int]] = []
        lora_index_mapping: List[List[int]] = []
        lora_prompt_mapping: List[List[int]] = []
        lora_requests: Set[LoRARequest] = set()

        is_enc_dec_model = self.model_config.is_encoder_decoder
        if len(seq_group_metadata_list) == 0:
            return PrepareDecodeMetadata.empty()
        lora_ids: List[int] = []

        dummy_slots = itertools.cycle(
            range(_PAD_SLOT_ID, _PAD_SLOT_ID + self.block_size))

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            assert seq_group_metadata.token_chunk_size == 1

            seq_ids = list(seq_group_metadata.seq_data.keys())
            lora_id = seq_group_metadata.lora_int_id
            lora_ids.append(lora_id)
            if is_enc_dec_model:
                for _ in range(len(seq_group_metadata.seq_data)):
                    encoder_seq_len = (
                        seq_group_metadata.encoder_seq_data.get_len()
                        if seq_group_metadata.encoder_seq_data else 0)
                    encoder_seq_lens.append(encoder_seq_len)
                    cross_block_table = seq_group_metadata.cross_block_table
                    cross_block_tables.append([] if (
                        cross_block_table is None) else cross_block_table)

            if lora_id > 0:
                lora_requests.add(seq_group_metadata.lora_request)

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                if output is None:
                    generation_token = seq_data.get_last_token_id()
                    input_tokens.append([generation_token])

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append([position])

                seq_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                seq_lens.append(seq_len)

                block_table = seq_group_metadata.block_tables[seq_id]
                num_fully_occupied_blocks = position // self.block_size
                block_table = block_table[:num_fully_occupied_blocks + 1]

                if len(block_table) == 0:
                    block_number = _PAD_BLOCK_ID
                else:
                    block_number = block_table[position // self.block_size]
                if block_number == _PAD_BLOCK_ID:
                    slot = next(dummy_slots)
                else:
                    block_offset = position % self.block_size
                    slot = block_number * self.block_size + block_offset
                slot_mapping.append([slot])
                lora_index_mapping.append(lora_id)
                lora_prompt_mapping.append(lora_id)

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                block_tables.append(block_table)

        if output is None:
            input_tokens = torch.tensor(input_tokens,
                                        dtype=torch.long,
                                        device='cpu')
        else:
            real_batch_size = len(seq_group_metadata_list)
            input_tokens = output[:real_batch_size].clone()

        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device='cpu')

        num_decode_tokens = len(seq_lens)

        last_block_usage = [
            slot[0] % self.block_size + 1 for slot in slot_mapping
        ]
        block_groups = [[i] * len(bt) for i, bt in enumerate(block_tables)]
        block_usage = [[self.block_size] * (len(bt) - 1) + [lbu]
                       for bt, lbu in zip(block_tables, last_block_usage)
                       if bt]

        block_list = flatten(block_tables)
        block_groups = flatten(block_groups)
        block_usage = flatten(block_usage)

        assert len(block_list) == len(block_groups)
        assert len(block_list) == len(block_usage)

        if is_enc_dec_model:
            last_cross_block_usage = [
                (encoder_seq_len - 1) % self.block_size + 1
                for encoder_seq_len in encoder_seq_lens
            ]
            cross_block_groups = [[i] * len(bt)
                                  for i, bt in enumerate(cross_block_tables)]
            cross_block_usage = [
                [self.block_size] * (len(bt) - 1) + [lbu]
                for bt, lbu in zip(cross_block_tables, last_cross_block_usage)
                if bt
            ]
            cross_block_list = flatten(cross_block_tables)
            cross_block_groups = flatten(cross_block_groups)
            cross_block_usage = flatten(cross_block_usage)
            assert len(cross_block_list) == len(cross_block_groups)
            assert len(cross_block_list) == len(cross_block_usage)

        else:
            cross_block_list = None
            cross_block_groups = None
            cross_block_usage = None
            encoder_seq_lens_tensor = None

        padding_fn = None
        if self.use_contiguous_pa:
            block_bucket_size = max(max(block_list) + 1, len(block_list))
            block_bucket_size = self.bucketing_ctx.get_padded_decode_num_blocks(
                block_bucket_size)
            indices: List[Any]
            indices = [None] * block_bucket_size
            for i, bid in enumerate(block_list):
                indices[bid] = i
            padding_fn = lambda tensor, pad_value: gather_list(
                tensor, indices, pad_value)
        else:
            block_bucket_size = self.bucketing_ctx.get_padded_decode_num_blocks(
                len(block_list))
            padding_fn = lambda tensor, pad_value: pad_list(
                tensor, block_bucket_size, pad_value)

        block_list = padding_fn(block_list, _PAD_BLOCK_ID)
        block_groups = padding_fn(block_groups, -1)
        block_usage = padding_fn(block_usage, 1)

        if is_enc_dec_model:
            if self.use_contiguous_pa:
                cross_block_bucket_size = max(
                    max(cross_block_list) +
                    1, len(cross_block_list)) if cross_block_list else 0
                cross_block_bucket_size = \
                    self.bucketing_ctx.get_padded_decode_num_blocks(
                    cross_block_bucket_size)
                indices = [None] * cross_block_bucket_size
                for i, bid in enumerate(cross_block_list):
                    indices[bid] = i
                padding_fn = lambda tensor, pad_value: gather_list(
                    tensor, indices, pad_value)
            else:
                cross_block_bucket_size = \
                    self.bucketing_ctx.get_padded_decode_num_blocks(
                    len(cross_block_list))
                padding_fn = lambda tensor, pad_value: pad_list(
                    tensor, cross_block_bucket_size, pad_value)

            real_batch_size = len(seq_group_metadata_list)
            batch_size_padded = self.bucketing_ctx.get_padded_batch_size(
                real_batch_size, False)
            batch_size_padding = batch_size_padded - real_batch_size
            if batch_size_padding > 0:
                encoder_seq_lens.extend(encoder_seq_lens[0]
                                        for _ in range(batch_size_padding))
            cross_block_list = padding_fn(cross_block_list, _PAD_BLOCK_ID)
            cross_block_groups = padding_fn(cross_block_groups, -1)
            cross_block_usage = padding_fn(cross_block_usage, 1)

            cross_block_list = torch.tensor(cross_block_list,
                                            dtype=torch.int,
                                            device='cpu')
            cross_block_groups = torch.tensor(cross_block_groups,
                                              dtype=torch.int,
                                              device='cpu')
            cross_block_usage = torch.tensor(cross_block_usage,
                                             dtype=self.model_config.dtype,
                                             device='cpu')
            encoder_seq_lens_tensor = torch.tensor(encoder_seq_lens,
                                                   dtype=torch.long,
                                                   device='cpu')

        block_list = torch.tensor(block_list, dtype=torch.int, device='cpu')
        block_groups = torch.tensor(block_groups,
                                    dtype=torch.int,
                                    device='cpu')
        block_usage = torch.tensor(block_usage,
                                   dtype=self.model_config.dtype,
                                   device='cpu')
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device='cpu')

        input_tokens = input_tokens.to(  # type: ignore
            self.device, non_blocking=True)
        input_positions = input_positions.to(  # type: ignore
            self.device, non_blocking=True)
        block_list = block_list.to(  # type: ignore
            self.device, non_blocking=True)
        block_groups = block_groups.to(  # type: ignore
            self.device, non_blocking=True)
        block_usage = block_usage.to(  # type: ignore
            self.device, non_blocking=True)
        slot_mapping = slot_mapping.to(  # type: ignore
            self.device, non_blocking=True)
        if is_enc_dec_model:
            cross_block_list = cross_block_list.to(  # type: ignore
                self.device, non_blocking=True)
            cross_block_groups = cross_block_groups.to(  # type: ignore
                self.device, non_blocking=True)
            cross_block_usage = cross_block_usage.to(  # type: ignore
                self.device, non_blocking=True)
            encoder_seq_lens_tensor = encoder_seq_lens_tensor.to(  # type: ignore
                self.device, non_blocking=True)

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=False,
            block_list=block_list,
            block_mapping=None,
            block_usage=block_usage,
            block_indices=None,
            block_offsets=None,
            block_scales=None,
            block_groups=block_groups,
            attn_bias=None,
            seq_lens_tensor=None,
            encoder_seq_lens=encoder_seq_lens,
            encoder_seq_lens_tensor=encoder_seq_lens_tensor,
            cross_block_list=cross_block_list,
            cross_block_groups=cross_block_groups,
            cross_block_usage=cross_block_usage,
            context_lens_tensor=None,
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=num_decode_tokens,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
        )
        return PrepareDecodeMetadata(input_tokens=input_tokens,
                                     input_positions=input_positions,
                                     attn_metadata=attn_metadata,
                                     lora_index_mapping=lora_index_mapping,
                                     lora_prompt_mapping=lora_prompt_mapping,
                                     lora_requests=lora_requests,
                                     slot_mapping=slot_mapping,
                                     lora_ids=lora_ids)

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[TModelInputForHPU, SamplingMetadata]:
        if len(seq_group_metadata_list) == 0:
            return self._model_input_cls(), None

        input_tokens = None
        input_positions = None
        lora_mapping = None
        lora_requests = None
        multi_modal_kwargs = None
        batch_type = None
        seq_lens = None
        query_lens = None
        real_batch_size = None
        batch_size_padded = None

        self.event_start = self.profiler.get_timestamp_us()
        is_prompt = seq_group_metadata_list[0].is_prompt
        base_event_name = 'prompt' if is_prompt else 'decode'
        self.profiler.start('internal', base_event_name)

        seq_group_metadata_list, real_batch_size, batch_size_padded = (
            self._add_dummy_seq(seq_group_metadata_list, is_prompt))

        prefill_reqs = []
        decode_reqs = []
        for seq_group_meta in seq_group_metadata_list:
            if seq_group_meta.is_prompt:
                prefill_reqs.append(seq_group_meta)
            else:
                decode_reqs.append(seq_group_meta)

        # Prepare input tensors.
        (
            input_tokens,
            input_positions,
            prefill_attn_metadata,
            seq_lens,
            query_lens,
            lora_index_mapping,
            lora_prompt_mapping,
            lora_requests,
            multi_modal_kwargs,
            slot_mapping,
            lora_ids,
        ) = self._prepare_prompt(prefill_reqs)
        (
            decode_input_tokens,
            decode_input_positions,
            decode_attn_metadata,
            decode_lora_index_mapping,
            decode_lora_prompt_mapping,
            decode_lora_requests,
            decode_slot_mapping,
            decode_lora_ids,
        ) = self._prepare_decode(decode_reqs)

        if not self.is_pooler:
            sampling_metadata = SamplingMetadata.prepare(
                seq_group_metadata_list, seq_lens, query_lens, self.device,
                self.pin_memory)

        if not self.scheduler_config.chunked_prefill_enabled:
            assert (len(prefill_reqs) and len(decode_reqs)) == 0

        num_prefills = len(seq_lens)
        num_prefill_tokens = len(input_tokens)
        num_decode_tokens = len(decode_input_tokens)

        # NOTE(kzawora): Here we diverge from GPU code - we don't
        # support mixed batches, so we either use decode or prefill
        # inputs, without coalescing.
        assert (num_prefills == 0 and num_decode_tokens > 0) or (
            num_prefills > 0
            and num_decode_tokens == 0), "HPU does not support mixed batches!"
        if num_decode_tokens > 0:
            input_tokens = decode_input_tokens
            input_positions = decode_input_positions
            slot_mapping = decode_slot_mapping
            lora_index_mapping = decode_lora_index_mapping
            lora_prompt_mapping = decode_lora_prompt_mapping
            lora_requests = decode_lora_requests
            lora_ids = decode_lora_ids

        # FIXME: We need to adjust selected_token_indices to accommodate
        # for padding
        max_len = input_tokens.size(1)
        paddings = [max_len - q for q in query_lens]
        paddings = [0] + paddings[:-1]
        paddings = list(itertools.accumulate(paddings))
        paddings_prompt_logprobs = []

        if not self.is_pooler:
            for i, seq_group_metadata in enumerate(seq_group_metadata_list):
                if seq_group_metadata.sampling_params \
                    and seq_group_metadata.sampling_params.prompt_logprobs \
                        is not None and seq_group_metadata.is_prompt:
                    paddings_prompt_logprobs += ([paddings[i]] * seq_lens[i])

            paddings = torch.tensor(
                paddings_prompt_logprobs
                if paddings_prompt_logprobs else paddings,
                dtype=sampling_metadata.selected_token_indices.dtype,
                device=sampling_metadata.selected_token_indices.device)
            sampling_metadata.selected_token_indices.add_(paddings)
        else:
            sampling_metadata = None

        if self.lora_config:
            lora_mapping = LoRAMapping(
                **dict(index_mapping=lora_index_mapping,
                       prompt_mapping=lora_prompt_mapping,
                       is_prefill=(num_prefills > 0)))
        else:
            lora_mapping = None

        if (prefill_attn_metadata is not None
                and decode_attn_metadata is not None):
            batch_type = BatchType.MIXED
            raise NotImplementedError("Mixed batch is not supported on HPU")
        elif prefill_attn_metadata is not None:
            batch_type = BatchType.PREFILL
        else:
            batch_type = BatchType.DECODE

        metadata_dict = {
            "input_tokens":
            input_tokens,
            "input_positions":
            input_positions,
            "selected_token_indices":
            sampling_metadata.selected_token_indices
            if sampling_metadata else None,
            "lora_requests":
            lora_requests,
            "lora_mapping":
            lora_mapping,
            "multi_modal_kwargs":
            multi_modal_kwargs,
            "num_prefill_tokens":
            num_prefill_tokens,
            "num_decode_tokens":
            num_decode_tokens,
            "slot_mapping":
            slot_mapping,
            "num_prefills":
            num_prefills,
            "batch_type":
            batch_type,
            "seq_lens":
            seq_lens,
            "query_lens":
            query_lens
        }
        if prefill_attn_metadata is not None:
            metadata_dict.update(prefill_attn_metadata.asdict_zerocopy())
        else:
            assert decode_attn_metadata is not None
            metadata_dict.update(decode_attn_metadata.asdict_zerocopy())

        attn_metadata = prefill_attn_metadata if \
            prefill_attn_metadata is not None else decode_attn_metadata

        return self._model_input_cls(input_tokens=input_tokens,
                                     seq_lens=seq_lens,
                                     query_lens=query_lens,
                                     input_positions=input_positions,
                                     attn_metadata=attn_metadata,
                                     lora_requests=lora_requests,
                                     lora_mapping=lora_mapping,
                                     multi_modal_kwargs=multi_modal_kwargs,
                                     real_batch_size=real_batch_size,
                                     batch_size_padded=batch_size_padded,
                                     lora_ids=lora_ids), \
                                        sampling_metadata

    def _seq_len(self, attn_metadata):
        if attn_metadata.num_prefills != 0:
            return attn_metadata.slot_mapping.size(1)
        else:
            return attn_metadata.block_list.numel()

    def trim_attn_metadata(self, metadata: AttentionMetadata) -> object:
        # NOTE(kzawora): To anyone working on this in the future:
        # Trimming metadata is required when using HPUGraphs.
        # Attention metadata is going to be hashed by PT bridge, and
        # appropriate HPUGraphs will be matched based on all inputs' hash.

        # Before you put more keys in here, make sure you know their
        # value type and make sure you know how it's going to be hashed.
        # You can find that information in input_hash function
        # in habana_frameworks/torch/hpu/graphs.py. You can also hash
        # it manually with torch.hpu.graphs.input_hash(attention_metadata)

        # If you use primitive types here - they will get hashed based
        # on their value. You *will* get lots of excessive graph captures
        # (and an OOM eventually) if you decide to put something like
        # seq_len int here.
        # If you absolutely need a scalar, put it in a tensor. Tensors
        # get hashed using their metadata, not their values:
        # input_hash(torch.tensor(123)) == input_hash(torch.tensor(321))
        # input_hash(123) != input_hash(321)
        # input_hash("abc") != input_hash("cba")
        attention_metadata = subtuple(metadata, 'TrimmedAttentionMetadata', [
            'attn_bias',
            'seq_lens_tensor',
            'context_lens_tensor',
            'block_list',
            'block_mapping',
            'block_usage',
            'slot_mapping',
            'is_prompt',
            'block_indices',
            'block_offsets',
            'block_scales',
            'block_groups',
        ])
        return attention_metadata

    def create_dummy_seq_group_metadata(self,
                                        group_id,
                                        seq_len,
                                        is_prompt,
                                        lora_request=None,
                                        temperature=0):
        if self.is_pooler:
            sampling_params = None
        else:
            sampling_params = SamplingParams(temperature=temperature)
            num_blocks = math.ceil(seq_len / self.block_size)
        seq_len = max(seq_len, 1)
        if is_prompt:
            input_len = seq_len
            output_len = 0
            block_tables = None
        else:
            input_len = seq_len - 1
            output_len = 1
            block_tables = {group_id: [_PAD_BLOCK_ID] * num_blocks}
        prompt_token_ids = [0] * input_len
        output_token_ids = [1] * output_len
        prompt_token_ids_array = array('l', prompt_token_ids)  # noqa: F821
        seq_data = SequenceData(prompt_token_ids_array)
        seq_data.output_token_ids = output_token_ids
        return SequenceGroupMetadata(request_id=str(group_id),
                                     is_prompt=(output_len == 0),
                                     seq_data={group_id: seq_data},
                                     sampling_params=sampling_params,
                                     block_tables=block_tables,
                                     lora_request=lora_request)

    def profile_run(self) -> None:
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers
        bind_kv_cache(
            self.vllm_config.compilation_config.static_forward_context,
            [kv_caches])
        _, max_seq_len = self.bucketing_ctx.get_max_prompt_shape()
        max_batch_size = min(self.max_num_seqs,
                             self.max_num_batched_tokens // max_seq_len)

        self.warmup_scenario(max_batch_size, max_seq_len, True, kv_caches,
                             False, True)
        return

    def warmup_scenario(self,
                        batch_size,
                        seq_len,
                        is_prompt,
                        kv_caches,
                        is_pt_profiler_run=False,
                        is_lora_profile_run=False,
                        temperature=0) -> None:
        use_graphs = self._use_graphs(batch_size, seq_len, is_prompt)
        scenario_name = ("warmup_"
                         f"{'prompt' if is_prompt else 'decode'}_"
                         f"bs{batch_size}_"
                         f"seq{seq_len}_"
                         f"graphs{'T' if use_graphs else 'F'}")
        # This represents the maximum number of different requests
        # that will have unique loras, an therefore the max amount of memory
        # consumption create dummy lora request copies from the lora request
        # passed in, which contains a lora from the lora warmup path.
        dummy_lora_requests: List[LoRARequest] = []
        dummy_lora_requests_per_seq: List[LoRARequest] = []
        if self.lora_config and is_lora_profile_run:
            assert self.lora_manager is not None
            with self.lora_manager.dummy_lora_cache():
                for idx in range(self.lora_config.max_loras):
                    lora_id = idx + 1
                    dummy_lora_request = LoRARequest(
                        lora_name=f"warmup_{lora_id}",
                        lora_int_id=lora_id,
                        lora_local_path="/not/a/real/path",
                    )
                    self.lora_manager.add_dummy_lora(dummy_lora_request,
                                                     rank=LORA_WARMUP_RANK)
                    dummy_lora_requests.append(dummy_lora_request)
                dummy_lora_requests_per_seq = [
                    dummy_lora_requests[idx % len(dummy_lora_requests)]
                    for idx in range(batch_size)
                ]
        self.profiler.start('internal', scenario_name)
        times = 3 if use_graphs or is_pt_profiler_run else 1
        if is_prompt:
            seqs = [
                self.create_dummy_seq_group_metadata(
                    i,
                    seq_len,
                    is_prompt,
                    lora_request=dummy_lora_requests_per_seq[i]
                    if dummy_lora_requests_per_seq else None,
                    temperature=temperature) for i in range(batch_size)
            ]
        else:
            # FIXME: seq_len is actually number of blocks
            blocks = [seq_len // batch_size for _ in range(batch_size)]
            blocks[0] += seq_len % batch_size
            seqs = [
                self.create_dummy_seq_group_metadata(
                    i,
                    b * self.block_size - 1,
                    is_prompt,
                    lora_request=dummy_lora_requests_per_seq[i]
                    if dummy_lora_requests_per_seq else None,
                    temperature=temperature) for i, b in enumerate(blocks)
            ]
        torch.hpu.synchronize()
        profiler = None
        if is_pt_profiler_run and self.is_driver_worker:
            profiler = setup_profiler()
            profiler.start()
        for _ in range(times):
            inputs = self.prepare_model_input(seqs)
            is_single_step = \
                self.vllm_config.scheduler_config.num_scheduler_steps == 1
            if is_prompt or is_single_step:
                self.execute_model(inputs, kv_caches, warmup_mode=True)
            else:  # decode with multi-step
                inputs = dataclasses.replace(inputs,
                                             is_first_multi_step=True,
                                             is_last_step=False)
                self.execute_model(inputs,
                                   kv_caches,
                                   warmup_mode=True,
                                   num_steps=2,
                                   seqs=seqs)
                inputs = dataclasses.replace(inputs,
                                             is_first_multi_step=False,
                                             is_last_step=True)
                self.execute_model(inputs,
                                   kv_caches,
                                   warmup_mode=True,
                                   num_steps=2,
                                   seqs=seqs)
            torch.hpu.synchronize()
            if profiler:
                profiler.step()
        if profiler:
            profiler.stop()
        self.profiler.end()
        gc.collect()

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

    def log_warmup(self, phase, i, max_i, batch_size, seq_len):
        free_mem = format_bytes(
            HabanaMemoryProfiler.current_free_device_memory())
        dim = "num_blocks"
        if "Prompt" in phase:
            dim = "seq_len"
        msg = (f"[Warmup][{phase}][{i+1}/{max_i}] "
               f"batch_size:{batch_size} "
               f"{dim}:{seq_len} "
               f"free_mem:{free_mem}")
        logger.info(msg)

    def warmup_all_buckets(self, buckets, is_prompt, kv_caches):
        for i, (batch_size, seq_len) in enumerate(reversed(buckets)):
            self.log_warmup('Prompt' if is_prompt else 'Decode', i,
                            len(buckets), batch_size, seq_len)
            self.warmup_scenario(batch_size, seq_len, is_prompt, kv_caches)

    def warmup_graphs(self,
                      strategy,
                      buckets,
                      is_prompt,
                      kv_caches,
                      available_mem,
                      starting_mem=0,
                      total_batch_seq=0.001):
        total_mem = starting_mem
        idx = 0
        phase = f'Graph/{"Prompt" if is_prompt else "Decode"}'
        num_candidates = len(buckets)
        ordering : Union[Callable[[Any], Tuple[Any, Any]], \
            Callable[[Any], Tuple[Any, Any, Any]]]
        if strategy == 'min_tokens':
            ordering = lambda b: (b[0] * b[1], b[1], b[0])
        elif strategy == 'max_bs':
            ordering = lambda b: (-b[0], b[1])
        else:
            raise NotImplementedError(
                f'Unsupported graph allocation strategy: {strategy}')
        buckets = list(sorted(buckets, key=ordering))
        captured_all = True
        warmed_random_sampler_bs: Set[int] = set()
        for idx, (batch_size, seq_len) in enumerate(buckets):
            # Graph memory usage is proportional to seq dimension in a batch
            batch_seq = batch_size * seq_len if is_prompt else batch_size
            mem_estimate = batch_seq / total_batch_seq * total_mem
            if mem_estimate >= available_mem:
                captured_all = False
                continue
            graphed_bucket = (batch_size, seq_len, is_prompt)
            if graphed_bucket in self.graphed_buckets:
                continue
            self.graphed_buckets.add(graphed_bucket)
            self.log_warmup(phase, idx, num_candidates, batch_size, seq_len)
            with HabanaMemoryProfiler() as mem_prof:
                self.warmup_scenario(batch_size,
                                     seq_len,
                                     is_prompt,
                                     kv_caches,
                                     temperature=1.0 if batch_size
                                     not in warmed_random_sampler_bs else 0)
            warmed_random_sampler_bs.add(batch_size)
            used_mem = align_workers(mem_prof.consumed_device_memory,
                                     torch.distributed.ReduceOp.MAX)
            available_mem -= used_mem
            total_mem += used_mem
            total_batch_seq += batch_seq

        return total_mem, total_batch_seq, captured_all

    def log_graph_warmup_summary(self, buckets, is_prompt, total_mem):
        num_candidates = len(buckets)
        phase = f'Graph/{"Prompt" if is_prompt else "Decode"}'
        graphed = list(c[:2] for c in self.graphed_buckets
                       if c[2] == is_prompt)
        if num_candidates == 0:
            num_candidates = 1
        msg = (f'{phase} captured:{len(graphed)} '
               f'({100 * len(graphed) / num_candidates:.1f}%) '
               f'used_mem:{format_bytes(total_mem)} '
               f'buckets:{sorted(list(graphed))}')
        logger.info(msg)

    @torch.inference_mode()
    def warmup_model(self, kv_caches: List[torch.Tensor]) -> None:
        if profile := os.environ.get('VLLM_PT_PROFILE', None):
            phase, bs, seq_len, graph = profile.split('_')
            is_prompt = phase == 'prompt'
            graphs = graph == 't'
            if graphs:
                self.graphed_buckets.add((int(bs), int(seq_len), is_prompt))
            self.warmup_scenario(int(bs), int(seq_len), is_prompt, kv_caches,
                                 True)
            raise AssertionError("Finished profiling")
        if not self.is_pooler:
            max_blocks = kv_caches[0][0].size(0)
        self.bucketing_ctx.generate_prompt_buckets()
        if not self.is_pooler:
            self.bucketing_ctx.generate_decode_buckets(max_blocks)
        if not htorch.utils.internal.is_lazy() and not self.enforce_eager:
            multiplier = 3 if os.getenv('VLLM_REGIONAL_COMPILATION',
                                        'true').lower() == 'true' else 1
            cache_size_limit = 1 + multiplier * (
                len(self.bucketing_ctx.prompt_buckets) +
                len(self.bucketing_ctx.decode_buckets))
            torch._dynamo.config.cache_size_limit = max(
                cache_size_limit, torch._dynamo.config.cache_size_limit)
            # Multiply by 8 to follow the original default ratio between
            # the cache_size_limit and accumulated_cache_size_limit
            torch._dynamo.config.accumulated_cache_size_limit = max(
                cache_size_limit * 8,
                torch._dynamo.config.accumulated_cache_size_limit)
        if self.skip_warmup:
            logger.info("Skipping warmup...")
            return
        self.profiler.start('internal', 'warmup')
        start_mem = HabanaMemoryProfiler.current_device_memory_usage()
        start_time = time.perf_counter()

        compile_only_mode_context = functools.partial(bc.env_setting,
                                                      "PT_COMPILE_ONLY_MODE",
                                                      True)
        can_use_compile_only_mode = True
        try:
            with compile_only_mode_context():
                pass
            logger.debug("Using PT_COMPILE_ONLY_MODE.")
        except KeyError:
            can_use_compile_only_mode = False
            logger.warning('Cannot use PT_COMPILE_ONLY_MODE. '
                           'Warmup time will be negatively impacted. '
                           'Please update Gaudi Software Suite.')
        with compile_only_mode_context(
        ) if can_use_compile_only_mode else contextlib.nullcontext():
            self.warmup_all_buckets(self.bucketing_ctx.prompt_buckets, True,
                                    kv_caches)
            if not self.is_pooler:
                self.warmup_all_buckets(self.bucketing_ctx.decode_buckets,
                                        False, kv_caches)

            if not self.enforce_eager and htorch.utils.internal.is_lazy():
                if not self.is_pooler:
                    assert self.mem_margin is not None, \
                        ("HabanaWorker.determine_num_available_blocks needs "
                        "to be called before warming up the model.")

                free_mem = HabanaMemoryProfiler.current_free_device_memory()
                graph_free_mem = free_mem - self.mem_margin
                graph_free_mem = align_workers(graph_free_mem,
                                               torch.distributed.ReduceOp.MIN)
                prompt_strategy = os.environ.get('VLLM_GRAPH_PROMPT_STRATEGY',
                                                 'min_tokens')
                if not self.is_pooler:
                    prompt_graph_mem_ratio = float(
                        os.environ.get('VLLM_GRAPH_PROMPT_RATIO', '0.3'))
                    prompt_available_memory = (prompt_graph_mem_ratio *
                                               graph_free_mem)
                    decode_available_memory = (graph_free_mem -
                                               prompt_available_memory)
                    msg = (
                        f"Using {format_bytes(graph_free_mem)}"
                        f"/{format_bytes(free_mem)} "
                        "of free device memory for HPUGraphs, "
                        f"{format_bytes(prompt_available_memory)} \
                            for prompt and "
                        f"{format_bytes(decode_available_memory)} for decode "
                        f"(VLLM_GRAPH_PROMPT_RATIO={prompt_graph_mem_ratio})")
                    logger.info(msg)
                    mem_post_prompt, prompt_batch_seq, prompt_captured_all = \
                        self.warmup_graphs(
                        prompt_strategy, self.bucketing_ctx.prompt_buckets,
                        True, kv_caches, prompt_available_memory)

                    decode_strategy = os.environ.get(
                        'VLLM_GRAPH_DECODE_STRATEGY', 'max_bs')
                    mem_post_decode, decode_batch_seq, decode_captured_all = \
                        self.warmup_graphs(
                        decode_strategy, self.bucketing_ctx.decode_buckets,
                        False, kv_caches, decode_available_memory)

                    # Not all prompt buckets were captured, but all decode
                    # buckets were captured and we have some free
                    # graph-allocated space left. Let's try to use it for
                    # capturing more prompt buckets.
                    if (mem_post_decode + mem_post_prompt < graph_free_mem
                            and not prompt_captured_all
                            and decode_captured_all):
                        mem_post_prompt, _, prompt_captured_all = (
                            self.warmup_graphs(
                                prompt_strategy,
                                self.bucketing_ctx.prompt_buckets, True,
                                kv_caches, graph_free_mem - mem_post_prompt -
                                mem_post_decode, mem_post_prompt,
                                prompt_batch_seq))
                        # Not all decode buckets were captured, but all prompt
                        # buckets were captured and we have some free
                        # graph-allocated space left. Let's try to use it for
                        # capturing more decode buckets.
                        if mem_post_decode + mem_post_prompt < graph_free_mem \
                            and not decode_captured_all \
                                and prompt_captured_all:
                            mem_post_decode, _, _ = self.warmup_graphs(
                                decode_strategy,
                                self.bucketing_ctx.decode_buckets, False,
                                kv_caches, graph_free_mem - mem_post_prompt -
                                mem_post_decode, mem_post_decode,
                                decode_batch_seq)
                else:
                    prompt_available_memory = graph_free_mem
                    msg = (
                        f"Using {format_bytes(graph_free_mem)}"
                        f"/{format_bytes(free_mem)} "
                        "of free device memory for HPUGraphs, "
                        f"{format_bytes(prompt_available_memory)} for prompt")
                    logger.info(msg)
                    prompt_strategy = os.environ.get(
                        'VLLM_GRAPH_PROMPT_STRATEGY', 'min_tokens')

                    mem_post_prompt, prompt_batch_seq, prompt_captured_all = \
                        self.warmup_graphs(
                        prompt_strategy, self.bucketing_ctx.prompt_buckets,
                        True, kv_caches, prompt_available_memory)
                    if mem_post_prompt < graph_free_mem \
                        and not prompt_captured_all:
                        mem_post_prompt, _, prompt_captured_all = (
                            self.warmup_graphs(
                                prompt_strategy,
                                self.bucketing_ctx.prompt_buckets, True,
                                kv_caches, graph_free_mem - mem_post_prompt,
                                mem_post_prompt, prompt_batch_seq))

                self.log_graph_warmup_summary(
                    self.bucketing_ctx.prompt_buckets, True, mem_post_prompt)
                if not self.is_pooler:
                    self.log_graph_warmup_summary(
                        self.bucketing_ctx.decode_buckets, False,
                        mem_post_decode)

        end_time = time.perf_counter()
        end_mem = HabanaMemoryProfiler.current_device_memory_usage()
        elapsed_time = end_time - start_time
        msg = (
            f"Warmup finished in {elapsed_time:.0f} secs, "
            f"allocated {format_bytes(end_mem - start_mem)} of device memory")
        logger.info(msg)
        self.profiler.end()

    def finish_measurements(self):
        from neural_compressor.torch.quantization import finalize_calibration
        finalize_calibration(self.model.model)

    def shutdown_inc(self):
        can_finalize_inc = (self.model_config.quantization == 'inc') and \
            (self.model.model is not None) and \
            self.inc_initialized_successfully and \
            not getattr(self, "_is_inc_finalized", False)
        if can_finalize_inc:
            from neural_compressor.torch.quantization import (
                finalize_calibration)
            finalize_calibration(self.model.model)
            self._is_inc_finalized = True

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()

    @property
    def mem_margin(self) -> Optional[int]:
        return self._mem_margin

    @mem_margin.setter
    def mem_margin(self, value):
        self._mem_margin = value


class HabanaProfilerCounterHelper:

    def __init__(self):
        self.niter = 0
        self.average_real_throughput = None
        self.logged_once = False
        self.real_seq_lens = []
        self.prompt_seq_lens = []

    def capture_seq_group_metadata_stats(self, seq_group_metadata_list):
        self.real_seq_lens = [
            len(seq_data.prompt_token_ids) + len(seq_data.output_token_ids)
            for seq_group_metadata in seq_group_metadata_list
            for seq_data in seq_group_metadata.seq_data.values()
        ]
        self.prompt_seq_lens = [
            len(seq_data.prompt_token_ids)
            for seq_group_metadata in seq_group_metadata_list
            for seq_data in seq_group_metadata.seq_data.values()
        ]

    def get_counter_dict(self, cache_config, duration, seq_len,
                         batch_size_padded, real_batch_size, is_prompt):
        throughput = batch_size_padded / (duration / 1e6)
        throughput_effective = real_batch_size / (duration / 1e6)

        real_max_seq_len = max(self.real_seq_lens)
        real_num_tokens = sum(self.real_seq_lens)
        padded_num_tokens = batch_size_padded * seq_len
        batch_token_utilization = real_num_tokens / padded_num_tokens
        if self.average_real_throughput is None:
            self.average_real_throughput = throughput_effective
        else:  # https://www.heikohoffmann.de/htmlthesis/node134.html
            self.average_real_throughput = self.average_real_throughput + 1 / (
                self.niter + 1) * (throughput_effective -
                                   self.average_real_throughput)
        phase = "prompt" if is_prompt else "decode"
        counters = {
            f'{phase}_bucket_batch_size': batch_size_padded,
            f'{phase}_batch_size': real_batch_size,
            f'{phase}_bucket_seq_len': seq_len,
            f'{phase}_seq_len': real_max_seq_len,
            f'{phase}_bucket_gen_throughput': throughput,
            f'{phase}_real_gen_throughput': throughput_effective,
            f'{phase}_batch_token_utilization': batch_token_utilization,
            'average_real_throughput': self.average_real_throughput,
            'engine_iteration': self.niter,
        }
        self.niter += 1
        if is_prompt:
            prompt_bucket_in_throughput = (seq_len * batch_size_padded) / (
                duration / 1e6)
            prompt_real_in_throughput = sum(
                self.prompt_seq_lens) / (duration / 1e6)
            counters[
                f'{phase}_bucket_in_throughput'] = prompt_bucket_in_throughput
            counters[f'{phase}_real_in_throughput'] = prompt_real_in_throughput

        # KV cache might not be created yet (e.g. for profiling run)
        if cache_config.num_gpu_blocks is not None and \
            cache_config.num_gpu_blocks != 0:
            cache_num_blocks_used = [
                math.ceil(sl / cache_config.block_size)
                for sl in self.real_seq_lens
            ]
            cache_total_num_blocks_used = sum(cache_num_blocks_used)
            num_cache_blocks = cache_config.num_gpu_blocks
            cache_total_num_free_blocks = \
                num_cache_blocks - cache_total_num_blocks_used
            cache_computed_utilization = \
                cache_total_num_blocks_used / num_cache_blocks
            max_blocks_per_seq = math.ceil(seq_len / cache_config.block_size)
            batch_block_utilization = cache_total_num_blocks_used / (
                batch_size_padded * max_blocks_per_seq)
            counters['cache_num_blocks_used'] = cache_total_num_blocks_used
            counters['cache_num_free_blocks'] = cache_total_num_free_blocks
            counters['cache_computed_utilization'] = cache_computed_utilization
            counters[
                f'{phase}_batch_block_utilization'] = batch_block_utilization
        if not self.logged_once:
            counters['const_cache_num_blocks'] = cache_config.num_gpu_blocks
            counters[
                'const_gpu_memory_utilization'] = \
                    cache_config.gpu_memory_utilization
            counters['const_block_size'] = cache_config.block_size
            self.logged_once = True
        return counters


class HPUModelRunner(HPUModelRunnerBase[ModelInputForHPUWithSamplingMetadata]):
    """
    GPU model runner with sampling step.
    """
    _model_input_cls: Type[ModelInputForHPUWithSamplingMetadata] = (
        ModelInputForHPUWithSamplingMetadata)

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> ModelInputForHPUWithSamplingMetadata:
        return (
            ModelInputForHPUWithSamplingMetadata.from_broadcasted_tensor_dict(
                tensor_dict,
                attn_backend=self.attn_backend,
            ))

    @torch.inference_mode()
    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForHPUWithSamplingMetadata:
        """Prepare the model input based on a given sequence group, including
        metadata for the sampling step.
        The API assumes seq_group_metadata_list is sorted by prefill -> decode.
        The result tensors and data structure also batches input in prefill
        -> decode order. For example,
        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.
        If cuda graph is required, this API automatically pads inputs.
        """
        with self.profiler.record_event('internal', 'prepare_input_tensors'):
            assert seq_group_metadata_list is not None
            if self.profiler.enabled:
                self.profiler_counter_helper.capture_seq_group_metadata_stats(
                    seq_group_metadata_list=seq_group_metadata_list)
            model_input, sampling_metadata = self.prepare_input_tensors(
                seq_group_metadata_list)
            assert model_input.attn_metadata is not None
            is_prompt = model_input.attn_metadata.is_prompt

        return dataclasses.replace(model_input,
                                   sampling_metadata=sampling_metadata,
                                   is_prompt=is_prompt,
                                   virtual_engine=virtual_engine)

    def create_lora_mask(self, input_tokens: torch.Tensor, lora_ids: List[int],
                         is_prompt: bool):
        '''
        This is a helper function to create the mask for lora computations.
        Lora Mask is needed to ensure we match the correct lora weights for the
        for the request.
        For Prompt phase we have
        lora_mask with shape (batch_size * seq_len, max_loras * max_rank)
        lora_logits_mask with shape (batch_size, max_loras * max_rank)
        For Decode phase we have both
        lora_mask and lora_logits_mask with shape
        (batch_size, max_loras * max_rank)
        '''
        lora_mask: torch.Tensor = None
        lora_logits_mask: torch.Tensor = None
        lora_index = 0

        if self.lora_config:
            if is_prompt:
                lora_mask = torch.zeros(
                    input_tokens.shape[0] * input_tokens.shape[1],
                    (self.lora_config.max_loras) *\
                        self.lora_config.max_lora_rank,
                    dtype=self.lora_config.lora_dtype)
                lora_logits_mask = torch.zeros(
                    input_tokens.shape[0], (self.lora_config.max_loras) *
                    self.lora_config.max_lora_rank,
                    dtype=self.lora_config.lora_dtype)

                ones = torch.ones(input_tokens.shape[1],
                                  self.lora_config.max_lora_rank,
                                  dtype=self.lora_config.lora_dtype)
                logit_ones = torch.ones(1,
                                        self.lora_config.max_lora_rank,
                                        dtype=self.lora_config.lora_dtype)

                for i in range(len(lora_ids)):
                    if lora_ids[i] == 0:
                        continue
                    lora_index = self.lora_manager._adapter_manager.\
                        lora_index_to_id.index(lora_ids[i])
                    start_row = i * input_tokens.shape[1]
                    end_row = start_row + input_tokens.shape[1]
                    start_col = lora_index * self.lora_config.max_lora_rank
                    end_col = start_col + self.lora_config.max_lora_rank
                    lora_mask[start_row:end_row, start_col:end_col] = ones
                    lora_logits_mask[i, start_col:end_col] = logit_ones
                lora_mask = lora_mask.to('hpu')
                lora_logits_mask = lora_logits_mask.to('hpu')
            else:
                lora_mask = torch.zeros(input_tokens.shape[0],
                                        (self.lora_config.max_loras) *
                                        self.lora_config.max_lora_rank,
                                        dtype=self.lora_config.lora_dtype)
                ones = torch.ones(1,
                                  self.lora_config.max_lora_rank,
                                  dtype=self.lora_config.lora_dtype)
                for i in range(len(lora_ids)):
                    if lora_ids[i] == 0:
                        continue
                    lora_index = self.lora_manager._adapter_manager.\
                        lora_index_to_id.index(lora_ids[i])
                    start_pos = lora_index * self.lora_config.max_lora_rank
                    end_pos = start_pos + self.lora_config.max_lora_rank
                    lora_mask[i, start_pos:end_pos] = ones
                lora_mask = lora_mask.to('hpu')
                lora_logits_mask = lora_mask

        return lora_mask, lora_logits_mask

    def _get_seq_ids(self, model_input):
        return ([
            sg.seq_ids[0] for sg in model_input.sampling_metadata.seq_groups
        ])

    def _pad_to_max_num_seqs(self, tensor, value):
        padding_needed = self.max_num_seqs - tensor.size(0)
        if padding_needed:
            padding = torch.full((padding_needed, *tensor.shape[1:]),
                                 value,
                                 device=tensor.device,
                                 dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding])
        return tensor

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForHPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
        warmup_mode=False,
        previous_hidden_states: Optional[torch.Tensor] = None,
        seqs=None,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        use_delayed_sampling = VLLM_DELAYED_SAMPLING and not warmup_mode
        assert not (use_delayed_sampling and num_steps != 1), \
            'Delayed sampling is not compatible with MSS!'
        assert model_input.input_tokens is not None
        if use_delayed_sampling and not model_input.is_prompt and \
                self.is_driver_worker:
            num_cached = len(self.cached_step_outputs)
            assert num_cached > 0
            cur_seq_ids = self._get_seq_ids(model_input)
            cur_seq_id_pos = {
                sid: idx
                for idx, sid in enumerate(cur_seq_ids) if sid >= 0
            }
            htorch.core.mark_step()
            for i in range(num_cached):
                prev_seq_ids = self._get_seq_ids(self.cached_step_inputs[i])
                target_indices = [
                    cur_seq_id_pos.get(psi, -1) for psi in prev_seq_ids
                ]
                padding = self.cached_step_outputs[i].size(0) - len(
                    target_indices)
                target_indices.extend([-1] * padding)
                target_indices = torch.tensor(
                    target_indices,
                    device=model_input.input_tokens.device,
                    dtype=model_input.input_tokens.dtype)
                model_input.input_tokens.index_copy_(
                    0, target_indices, self.cached_step_outputs[i])
                htorch.core.mark_step()

        if not model_input.is_first_multi_step:
            if not model_input.is_last_step:
                # not first or last multi-step
                return []
            # last multi-step
            output = self._decode_sampler_outputs(
                model_input) if self.is_driver_worker else []
            torch.hpu.synchronize()
        if model_input.is_first_multi_step:
            # first multi-step
            if self.lora_config:
                assert model_input.lora_requests is not None
                assert model_input.lora_mapping is not None
                self.set_active_loras(model_input.lora_requests,
                                      model_input.lora_mapping)
            # Rank!=0 workers has is_prompt==None
            if use_delayed_sampling and not model_input.is_prompt and \
                    model_input.input_tokens.size(1) == 1:
                if self.is_driver_worker:
                    model_kwargs_broadcast_data = {
                        "input_tokens": model_input.input_tokens
                    }
                    broadcast_tensor_dict(model_kwargs_broadcast_data, src=0)
                    input_tokens = model_input.input_tokens

                else:
                    model_kwargs_broadcast_data = broadcast_tensor_dict(src=0)
                    input_tokens = model_kwargs_broadcast_data["input_tokens"]
            else:
                input_tokens = model_input.input_tokens
            input_positions = model_input.input_positions
            attn_metadata = model_input.attn_metadata
            sampling_metadata = model_input.sampling_metadata
            real_batch_size = model_input.real_batch_size
            batch_size_padded = model_input.batch_size_padded
            assert input_tokens is not None
            assert input_positions is not None
            assert sampling_metadata is not None
            assert attn_metadata is not None
            is_prompt = attn_metadata.is_prompt
            assert is_prompt is not None
            batch_size = input_tokens.size(0)
            seq_len = self._seq_len(attn_metadata)
            use_graphs = self._use_graphs(batch_size, seq_len, is_prompt)
            self._check_config(batch_size, seq_len, attn_metadata, warmup_mode)

            lora_mask: torch.Tensor = None
            lora_logits_mask: torch.Tensor = None
            if self.lora_config:
                assert model_input.lora_ids is not None
                lora_mask, lora_logits_mask = self.create_lora_mask(
                    input_tokens, model_input.lora_ids,
                    attn_metadata.is_prompt)

            execute_model_kwargs = {
                "input_ids": input_tokens,
                "positions": input_positions,
                "kv_caches": kv_caches,
                "attn_metadata": self.trim_attn_metadata(attn_metadata),
                "intermediate_tensors": intermediate_tensors,
                "lora_mask": lora_mask,
                "virtual_engine": model_input.virtual_engine,
                **(model_input.multi_modal_kwargs or {}),
            }
            if previous_hidden_states is not None:
                execute_model_kwargs.update(
                    {"previous_hidden_states": previous_hidden_states})
            if htorch.utils.internal.is_lazy():
                execute_model_kwargs.update(
                    {"bypass_hpu_graphs": not use_graphs})

            htorch.core.mark_step()
            if self.is_driver_worker:
                model_event_name = ("model_"
                                    f"{'prompt' if is_prompt else 'decode'}_"
                                    f"bs{batch_size}_"
                                    f"seq{seq_len}_"
                                    f"graphs{'T' if use_graphs else 'F'}")
            else:
                model_event_name = 'model_executable'
            if num_steps > 1 or use_delayed_sampling:
                # in case of multi-step scheduling
                # we only want to pythonize in the last step
                sampling_metadata.skip_sampler_cpu_output = True
                self.model.model.sampler.include_gpu_probs_tensor = True
            cache_orig_output_tokens_len: List[Dict] = []

            def try_revert_dummy_output_tokens():
                if len(cache_orig_output_tokens_len) > 0:
                    # Reuse the original output token ids length
                    for i in range(len(cache_orig_output_tokens_len)):
                        seq_group_metadata = seq_group_metadata_list[i]
                        for j, data in seq_group_metadata.seq_data.items():
                            orig_output_tokens_len = \
                                cache_orig_output_tokens_len[i][j]
                            data.output_token_ids = \
                                data.output_token_ids[:orig_output_tokens_len]

            for i in range(num_steps):
                if i != 0 and not self.is_driver_worker:
                    broadcast_data = broadcast_tensor_dict(src=0)
                    if 'early_exit' in broadcast_data and broadcast_data[
                            'early_exit']:
                        return [output] if num_steps == 1 else []
                    execute_model_kwargs.update({
                        "input_ids":
                        broadcast_data["input_ids"],
                        "positions":
                        broadcast_data["positions"],
                        "attn_metadata":
                        self.trim_attn_metadata(
                            broadcast_data["attn_metadata"])
                    })
                profiler_args = {
                    'real_seq_len': model_input.seq_lens,
                    'real_batch_size': real_batch_size
                }

                with self.profiler.record_event('internal',
                                                model_event_name,
                                                args=profiler_args):
                    hidden_states = self.model.forward(
                        **execute_model_kwargs,
                        selected_token_indices=sampling_metadata.
                        selected_token_indices)

                if self.lora_config:
                    LoraMask.setLoraMask(
                        lora_logits_mask.index_select(
                            0, sampling_metadata.selected_token_indices))

                # Compute the logits.
                with self.profiler.record_event(
                        'internal',
                    ('compute_logits_'
                     f'{"prompt" if is_prompt else "decode"}_bs'
                     f'{batch_size}_'
                     f'seq{seq_len}'),
                        args=profiler_args):
                    if num_steps == 1:
                        sampling_metadata.selected_token_indices = None
                    logits = self.model.compute_logits(hidden_states,
                                                       sampling_metadata)
                htorch.core.mark_step()
                # Only perform sampling in the driver worker.
                if not self.is_driver_worker:
                    continue

                if use_delayed_sampling:
                    fake_output = self._delayed_sampler_outputs(model_input)

                with self.profiler.record_event(
                        'internal', ('sample_'
                                     f'{"prompt" if is_prompt else "decode"}_'
                                     f'bs{batch_size}_'
                                     f'seq{seq_len}'),
                        args=profiler_args):
                    output = self.model.sample(
                        logits=logits,
                        sampling_metadata=sampling_metadata,
                    )
                    if num_steps > 1:
                        output = output.sampled_token_ids
                        self.cached_step_outputs.append(output)
                    if use_delayed_sampling and self.is_driver_worker:
                        self._patch_prev_output()
                        output = self._pad_to_max_num_seqs(
                            output.sampled_token_ids, DUMMY_TOKEN_ID)
                        self.cached_step_outputs.append(output)
                        self.cached_step_inputs.append(model_input)
                htorch.core.mark_step()
                if model_input.async_callback is not None:
                    model_input.async_callback()
                if i < num_steps - 1:
                    if i == 0:
                        if model_input.async_callback is not None:
                            ctx = model_input.async_callback.keywords[  # type: ignore
                                "ctx"]
                            seq_group_metadata_list = \
                                ctx.seq_group_metadata_list
                        elif seqs is not None:
                            seq_group_metadata_list = seqs
                        else:
                            raise RuntimeError(
                                "seq_group_metadata_list is uninitialized")
                        for seq_idx, seq_group_metadata in enumerate(
                                seq_group_metadata_list):
                            # Skip empty steps
                            seq_group_metadata.state.current_step += (
                                num_steps - 2)
                            # Cache the original output token ids
                            cache_orig_output_tokens_len.append({})
                            for j, data in seq_group_metadata.seq_data.items():
                                cache_orig_output_tokens_len[seq_idx][j] = \
                                    len(data.output_token_ids)
                    seq_group_metadata_list, _, _ = self._add_dummy_seq(
                        seq_group_metadata_list, is_prompt=False)
                    for seq_group_metadata in seq_group_metadata_list:
                        for data in seq_group_metadata.seq_data.values():
                            max_output_len = sampling_metadata.seq_groups[
                                0].sampling_params.max_tokens
                            if len(data.output_token_ids) < max_output_len - 1:
                                # add a place holder for prepare_decode
                                # arbitrary value, this could be any token
                                dummy_token = (540, )
                                data.output_token_ids += (dummy_token)
                            else:
                                broadcast_tensor_dict({'early_exit': True},
                                                      src=0)
                                if num_steps == 1:
                                    return [output]
                                else:
                                    try_revert_dummy_output_tokens()
                                    return []

                    result = self._prepare_decode(seq_group_metadata_list,
                                                  output=output)
                    if self.lora_config:
                        lora_mapping = LoRAMapping(
                            **dict(index_mapping=result.lora_index_mapping,
                                   prompt_mapping=result.lora_prompt_mapping,
                                   is_prefill=False))
                        self.set_active_loras(result.lora_requests,
                                              lora_mapping)
                        lora_mask, lora_logits_mask = self.create_lora_mask(
                            result.input_tokens, result.lora_ids, False)

                    execute_model_kwargs.update({
                        "input_ids":
                        result.input_tokens,
                        "positions":
                        result.input_positions,
                        "attn_metadata":
                        self.trim_attn_metadata(result.attn_metadata),
                        "lora_mask":
                        lora_mask,
                    })
                    model_kwargs_broadcast_data = {
                        "input_ids": result.input_tokens,
                        "positions": result.input_positions,
                        "attn_metadata": vars(result.attn_metadata),
                        "lora_mask": lora_mask,
                    }
                    broadcast_tensor_dict(model_kwargs_broadcast_data, src=0)
                else:
                    try_revert_dummy_output_tokens()

            if self.is_driver_worker and self.profiler.enabled:
                # Stop recording 'execute_model' event
                self.profiler.end()
                event_end = self.profiler.get_timestamp_us()
                counters = self.profiler_counter_helper.get_counter_dict(
                    cache_config=self.cache_config,
                    duration=event_end - self.event_start,
                    seq_len=seq_len,
                    batch_size_padded=batch_size_padded,
                    real_batch_size=real_batch_size,
                    is_prompt=is_prompt)
                self.profiler.record_counter(self.event_start, counters)
            if num_steps == 1:
                if self.return_hidden_states:
                    # we only need to pass hidden states of most recent token
                    assert model_input.sampling_metadata is not None
                    if model_input.is_prompt:
                        output.prefill_hidden_states = hidden_states
                    output.hidden_states = hidden_states
                if use_delayed_sampling:
                    if self.is_driver_worker:
                        return [fake_output]
                    else:
                        return []

                return [output] if self.is_driver_worker else []
            else:
                return []

        return output if type(output) is list else [output]

    def _delayed_sampler_outputs(self, model_input):
        next_token_ids = [[DUMMY_TOKEN_ID]] * len(
            model_input.sampling_metadata.seq_groups)
        sampler_output = self._make_decode_output(
            next_token_ids, model_input.sampling_metadata.seq_groups)
        return sampler_output

    def _decode_sampler_outputs(self, model_input):
        use_async_out_proc = model_input.async_callback is not None
        sampler_outputs = []
        num_outputs = len(self.cached_step_outputs)
        for i in range(num_outputs):
            next_token_ids = self.cached_step_outputs.pop(0)
            next_token_ids = next_token_ids.cpu().tolist()
            sampler_output = self._make_decode_output(
                next_token_ids, model_input.sampling_metadata.seq_groups)
            sampler_outputs.append(sampler_output)

            if i < num_outputs - 1 and use_async_out_proc:
                assert model_input.async_callback is not None
                ctx = model_input.async_callback.keywords[  # type: ignore
                    "ctx"]
                ctx.append_output(
                    outputs=[sampler_output],
                    seq_group_metadata_list=ctx.seq_group_metadata_list,
                    scheduler_outputs=ctx.scheduler_outputs,
                    is_async=False,
                    is_last_step=False,
                    is_first_step_output=False)
                model_input.async_callback()

        if use_async_out_proc:
            return [sampler_outputs[-1]]
        else:
            return sampler_outputs

    def _make_decode_output(
        self,
        next_token_ids: List[List[int]],
        seq_groups: List[SequenceGroupToSample],
    ) -> SamplerOutput:
        zero_logprob = Logprob(0.0)
        sampler_outputs = []
        batch_idx = 0
        for seq_group in seq_groups:
            seq_ids = seq_group.seq_ids
            seq_outputs = []
            for seq_id in seq_ids:
                next_token_id = next_token_ids[batch_idx][0]
                seq_outputs.append(
                    SequenceOutput(seq_id, next_token_id,
                                   {next_token_id: zero_logprob}))
                batch_idx += 1
            sampler_outputs.append(
                CompletionSequenceGroupOutput(seq_outputs, None))
        return SamplerOutput(sampler_outputs)

    def _patch_prev_output(self):
        assert len(self.cached_step_inputs) == len(self.cached_step_outputs), \
            f'''Inputs and outputs are out of sync!
            {len(self.cached_step_inputs)} vs {len(self.cached_step_outputs)}'''
        if len(self.cached_step_inputs) == 0:
            return
        model_input = self.cached_step_inputs.pop(0)
        delayed_output = self.cached_step_outputs.pop(0).cpu().squeeze(
            -1).tolist()
        ctx = model_input.async_callback.keywords["ctx"]  # type: ignore
        # If there's no output to patch with, which is usually the case when
        # we're starting a new request after all requests are completed.
        if len(ctx.output_queue) == 0:
            return
        assert len(
            ctx.output_queue) == 1, 'There should be exactly 1 output waiting!'
        output_data = ctx.output_queue[0]
        assert len(output_data.outputs) == 1
        for fake_out, real_out in zip(output_data.outputs[0], delayed_output):
            fake_out.samples[0].output_token = real_out
        for sg, real_out in zip(output_data.seq_group_metadata_list,
                                delayed_output):
            assert len(sg.seq_data) == 1
            seq_data = list(sg.seq_data.values())[0]
            # This is a hack. Assigning output_token_ids triggers
            # a cache recomputation and we only need to update the last token
            seq_data.output_token_ids_array[-1] = real_out
            seq_data._cached_all_token_ids[-1] = real_out
