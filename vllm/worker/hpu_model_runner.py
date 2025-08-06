# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
from contextlib import suppress
from enum import Enum, IntEnum

if os.getenv("QUANT_CONFIG", None) is not None:
    from neural_compressor.torch.quantization import finalize_calibration
else:
    finalize_calibration = None
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple,
                    Optional, Set, Tuple, Type, TypeVar, Union)

import habana_frameworks.torch as htorch
import habana_frameworks.torch.internal.bridge_config as bc
import torch
import vllm_hpu_extension.environment as environment
from attr import dataclass
from vllm_hpu_extension.bucketing.common import HPUBucketingManager
from vllm_hpu_extension.ops import LoraMask as LoraMask
from vllm_hpu_extension.profiler import (HabanaHighLevelProfiler,
                                         HabanaMemoryProfiler,
                                         HabanaProfilerCounterHelper,
                                         format_bytes)
from vllm_hpu_extension.runtime import finalize_config, get_config

import vllm.envs as envs
from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.backends.hpu_attn import HPUAttentionImpl
from vllm.config import DeviceConfig, VllmConfig
from vllm.distributed import broadcast_tensor_dict, get_pp_group
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.parallel_state import (get_dp_group, get_tp_group,
                                             get_world_group)
from vllm.forward_context import set_forward_context
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.layers.sampler import (SampleResultArgsType,
                                                SamplerOutput, get_logprobs,
                                                get_pythonized_sample_results,
                                                get_sampler)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.sampling_metadata import SequenceGroupToSample
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalKwargs, MultiModalPlaceholderMap,
                             MultiModalRegistry)
from vllm.multimodal.inputs import PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.sequence import (CompletionSequenceGroupOutput, IntermediateTensors,
                           Logprob, SequenceData, SequenceGroupMetadata,
                           SequenceOutput)
from vllm.transformers_utils.config import uses_mrope
from vllm.utils import (bind_kv_cache, is_fake_hpu, is_pin_memory_available,
                        make_mrope_positions_tensor_with_pad,
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

DUMMY_TOKEN_ID = -1
UNSET_IMG_ARGS = 9999999
shutdown_inc_called = False


class PhaseType(Enum):
    PREFILL = 'prefill'
    PREFIX_PREFILL = 'prefix_prefill'
    DECODE = 'decode'


class VisionBuckets:
    '''
    This class is used to bucket image tokens
    '''

    def __init__(self, is_batch_based):
        self.is_batch_based = is_batch_based
        envvar = os.environ.get('VLLM_MULTIMODAL_BUCKETS', "")
        if envvar == 'None':
            self.multimodal_buckets = None
        else:
            if envvar == "":
                if is_batch_based:
                    multimodal_buckets = [1, 2, 4, 8]  # batch sizes for gemma3
                else:
                    multimodal_buckets = [
                        1600, 3136, 4096, 6400, 7744, 9216, 12544
                    ]
            else:
                multimodal_buckets = [int(i) for i in envvar.split(',')]
            self.multimodal_buckets = self._process_buckets(multimodal_buckets)

    def _process_buckets(self, buckets):
        if not self.is_batch_based:
            for bucket in buckets:
                assert bucket % 8 == 0, (
                    'Buckets needs to be multiples 8 (slices of 64)')
        return sorted(buckets)

    def get_multimodal_bucket(self, curr_num_image_patches):
        if self.multimodal_buckets is not None:
            for mm_bucket in self.multimodal_buckets:
                if curr_num_image_patches <= mm_bucket:
                    return mm_bucket
            return curr_num_image_patches
        else:
            return 0

    def __repr__(self):
        return str(self.multimodal_buckets)


class Singleton(type):
    _instances: Dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def is_mm_optimized(model):
    return 'Gemma3ForConditionalGeneration' in str(type(model.model)) \
        if hasattr(model, 'model') else \
        'Gemma3ForConditionalGeneration' in str(type(model))


def pad_flat_tensor(tensor, desired_size):
    assert tensor.dim() == 1, 'Only flat tensors are supported'
    padding_needed = desired_size - tensor.size(0)
    if padding_needed > 0 and tensor.size(0) > 0:
        padding = torch.zeros((padding_needed, ),
                              dtype=tensor.dtype,
                              device=tensor.device)
        tensor = torch.cat([tensor, padding])
    return tensor


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
        _TYPE_CACHE[typename] = {
            'type': collections.namedtuple(typename, ' '.join(fields)),
            'fields': fields
        }
    return _TYPE_CACHE[typename]['type'](**values)  # type: ignore


def custom_tuple_replace(obj: object, typename: str, **to_override):
    # Torch compile dynamo doesn't support calling any named tuple
    # dynamic methods other than len and get_attr. This function is
    # a torch.compile friendly version of tuple._replace

    cached_type = _TYPE_CACHE[typename]['type']
    fields = _TYPE_CACHE[typename]['fields']
    values = {
        field: getattr(obj, field)
        for field in fields  # type: ignore
    }
    values.update(to_override)
    return cached_type(**values)  # type: ignore


def align_dp_groups(value, op):
    group = get_dp_group().cpu_group
    value_t = torch.tensor(value, device="cpu", dtype=torch.int32)
    torch.distributed.all_reduce(value_t, op=op, group=group)
    return value_t.item()


def align_tp_groups(value, op):
    group = get_tp_group().cpu_group
    world_size = get_tp_group().world_size
    if world_size <= 1:
        return value
    value_t = torch.tensor(value, device='cpu')
    torch.distributed.all_reduce(value_t, op=op, group=group)
    return value_t.item()


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


def make_cpu_tensor(data, max_len, pad, dtype, flat) -> torch.Tensor:
    if flat:
        data = [flatten(data)]
    return make_tensor_with_pad(data,
                                max_len=max_len,
                                pad=pad,
                                dtype=dtype,
                                device='cpu')


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


class HpuModelAdapter(torch.nn.Module):

    def __init__(self, model, vllm_config, is_causal, sampler):
        super().__init__()
        self.model = model
        self.prefill_use_fusedsdpa = get_config(
        ).prompt_attn_impl == 'fsdpa_impl'
        self.recompute_cos_sin = os.getenv('VLLM_COS_SIN_RECOMPUTE',
                                           'false').lower() in ['1', 'true']
        self.sampler = sampler
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.dtype = vllm_config.model_config.dtype
        self.is_pooler = hasattr(self.model, "_pooler")
        self.is_causal = is_causal
        self.use_merged_prefill = get_config().merged_prefill
        self.dp_awared_padding = \
            self.vllm_config.parallel_config.data_parallel_size > 1

        model_config = getattr(self.model, "config", None)

        self.model_is_mrope = uses_mrope(model_config)
        self.is_mm_optimized = is_mm_optimized(self.model)
        text_config = vllm_config.model_config.hf_config.get_text_config()
        self.interleaved_sliding_window = getattr(
            text_config, "interleaved_sliding_window",
            None) if text_config else None

        self.use_window_sdpa = os.getenv("PT_HPU_SDPA_QKV_SLICE_MODE_FWD",
                                         "false").strip().lower() in ("1",
                                                                      "true")
        if self.use_window_sdpa:
            self.slice_size = int(os.getenv("PT_HPU_SDPA_BC_FACTOR", "1024"))
            self.sliding_window_thld = int(
                os.environ.get('VLLM_FUSEDSDPA_SLIDE_THLD', '8192'))

        # This applies exclusively to Qwen2/2.5-VL models
        # both use mrope. We wrap the visual and language
        # models separately with HPU graph.
        # This is to ensure that we keeps
        # the static and dynamic parts distinct.
        if htorch.utils.internal.is_lazy():
            if self.model_is_mrope and hasattr(self.model, 'visual'):
                logger.info("[Multimodal] Wrapping Visual Model")
                self.model.visual = htorch.hpu.wrap_in_hpu_graph(
                    self.model.visual, disable_tensor_cache=True)

            if self.is_mm_optimized:
                if hasattr(self.model, 'vision_tower'):
                    self.model.vision_tower = htorch.hpu.wrap_in_hpu_graph(
                        self.model.vision_tower, disable_tensor_cache=True)
                if hasattr(self.model, 'multi_modal_projector'):
                    self.model.multi_modal_projector = \
                            htorch.hpu.wrap_in_hpu_graph( \
                            self.model.multi_modal_projector, \
                            disable_tensor_cache=True)

        self._rotary_embed_module = self._get_rotary_embedding_module(
            self.model)
        self._rotary_prepare_cos_sin = self._get_prepare_cos_sin()

    def _get_rotary_embedding_module(self, model: torch.nn.Module):
        """
        Dynamically get the RotaryEmbedding layer in the model.
        This function will recursively search through the module 
        hierarchy to find and return a RotaryEmbedding layer.
        If no such layer is found, it returns None.
        """
        if model is None:
            return None

        if model.__class__.__name__.endswith("RotaryEmbedding"):
            return model

        if hasattr(model, 'children'):
            for child in model.children():
                result = self._get_rotary_embedding_module(child)
                if result is not None:
                    return result
        return None

    def _get_prepare_cos_sin(self):
        if self._rotary_embed_module is not None and hasattr(
                self._rotary_embed_module, 'prepare_cos_sin'):
            return self._rotary_embed_module.prepare_cos_sin
        return None

    def _reset_rotary_cos_sin(self):
        if hasattr(self._rotary_embed_module, "cos"):
            delattr(self._rotary_embed_module, "cos")
        if hasattr(self._rotary_embed_module, "sin"):
            delattr(self._rotary_embed_module, "sin")

    def _set_attn_bias(self, attn_metadata, batch_size, seq_len, device,
                       dtype):
        if (attn_metadata is None
                or (self.prefill_use_fusedsdpa and self.is_causal
                    and attn_metadata.block_list is None)
                or not attn_metadata.is_prompt):
            return attn_metadata

        if attn_metadata.attn_bias is not None:
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
        attn_metadata = custom_tuple_replace(prefill_metadata,
                                             "TrimmedAttentionMetadata",
                                             attn_bias=attn_bias)
        return attn_metadata

    def _set_attn_bias_for_sliding_window(self, attn_metadata, batch_size,
                                          seq_len, window_size, device, dtype):

        if (seq_len <= window_size) or (not attn_metadata.is_prompt) or (
                attn_metadata.use_window_sdpa):
            # no need to set sliding window mask, just use built-in sdpa
            return attn_metadata

        prefill_metadata = attn_metadata
        shift = 0

        #causal + window size
        tensor = torch.full((batch_size, 1, seq_len, seq_len),
                            device=device,
                            dtype=dtype,
                            fill_value=1)
        mask = torch.tril(tensor, diagonal=shift)
        mask = torch.triu(mask, diagonal=shift - window_size + 1)
        attn_bias = torch.log(mask)

        attn_metadata = prefill_metadata._replace(window_attn_bias=attn_bias)
        return attn_metadata

    def _set_block_mapping(self, metadata, batch_size, device, dtype,
                           is_window_block):
        if is_window_block:
            block_usage = metadata.window_block_usage
            block_groups = metadata.window_block_groups
        else:
            block_usage = metadata.block_usage
            block_groups = metadata.block_groups

        mask = torch.arange(0,
                            self.block_size,
                            device=device,
                            dtype=torch.int32).unsqueeze(0)
        mask = mask >= block_usage.unsqueeze(-1)
        attn_bias = (torch.zeros_like(mask, dtype=dtype).masked_fill_(
            mask, -math.inf))

        if not is_fake_hpu():
            block_mapping = torch.nn.functional.one_hot(block_groups,
                                                        num_classes=batch_size)
        else:
            # Unfortunately one_hot on CPU
            # doesn't handle out of bounds classes so we need to convert
            # all negative values to 0 (block_mapping) or bs (block_groups)
            block_groups = block_groups.to(torch.long)
            block_mapping = torch.nn.functional.relu(block_groups)
            block_mapping = torch.nn.functional.one_hot(block_mapping,
                                                        num_classes=batch_size)
            oob_values = block_groups.lt(0)
            block_mapping.masked_fill_(oob_values.unsqueeze(-1), 0)
            block_groups.masked_fill_(oob_values, batch_size)
            if is_window_block:
                metadata = custom_tuple_replace(
                    metadata,
                    "TrimmedAttentionMetadata",
                    window_block_groups=block_groups)
            else:
                metadata = custom_tuple_replace(metadata,
                                                "TrimmedAttentionMetadata",
                                                block_groups=block_groups)
        block_mapping = block_mapping.to(dtype)
        if is_window_block:
            metadata = custom_tuple_replace(metadata,
                                            "TrimmedAttentionMetadata",
                                            window_block_mapping=block_mapping,
                                            window_attn_bias=attn_bias)
        else:
            metadata = custom_tuple_replace(metadata,
                                            "TrimmedAttentionMetadata",
                                            block_mapping=block_mapping,
                                            attn_bias=attn_bias)
        return metadata

    def forward_update_meta_only(self, *args, **kwargs):
        kwargs = kwargs.copy()
        if 'warmup_mode' in kwargs:
            kwargs.pop('warmup_mode')
        input_ids = kwargs['input_ids']
        attn_metadata = self._update_metadata(kwargs['attn_metadata'],
                                              input_ids.size(0),
                                              input_ids.size(1),
                                              input_ids.device, self.dtype)
        kwargs['attn_metadata'] = attn_metadata
        return attn_metadata

    def _update_use_window_sdpa(self, attn_metadata, seq_len, is_img):
        use_window_sdpa = False
        if self.use_window_sdpa and seq_len >= self.sliding_window_thld and \
           self.prefill_use_fusedsdpa:
            if self.slice_size != 0 and (seq_len % self.slice_size == 0):
                use_window_sdpa = True
            else:
                raise AssertionError(
                    f"input token length {seq_len} is not multiple "
                    f"of SLICE_SIZE {self.slice_size}. Please adjust "
                    f"VLLM_EXPONENTIAL_BUCKETING: False "
                    f"VLLM_PROMPT_SEQ_BUCKET_MIN: 1024 "
                    f"VLLM_PROMPT_SEQ_BUCKET_STEP: 1024 ")

        attn_metadata = attn_metadata._replace(use_window_sdpa=use_window_sdpa)
        return attn_metadata

    def _update_metadata(self,
                         attn_metadata,
                         batch_size,
                         seq_len,
                         device,
                         dtype,
                         global_attn_masks=None,
                         local_attn_masks=None):

        if attn_metadata.is_prompt:
            attn_metadata = self._set_attn_bias(attn_metadata, batch_size,
                                                seq_len, device, dtype)

            #For Gemma3, we need to override attn_mask with these sliding_window
            #mask which are updated during prepare_attn_mask()
            if global_attn_masks is not None:
                attn_metadata = attn_metadata._replace(
                    attn_bias=global_attn_masks[0])

            if self.interleaved_sliding_window:
                if local_attn_masks is not None:
                    attn_metadata = attn_metadata._replace(
                        window_attn_bias=local_attn_masks[0])
                elif global_attn_masks is None:
                    attn_metadata = self._set_attn_bias_for_sliding_window(
                        attn_metadata, batch_size, seq_len,
                        self.interleaved_sliding_window, device, dtype)

        else:
            attn_metadata = self._set_block_mapping(attn_metadata, batch_size,
                                                    device, dtype, False)
        if hasattr(attn_metadata, 'window_block_list'
                   ) and attn_metadata.window_block_list is not None:

            attn_metadata = self._set_block_mapping(attn_metadata, batch_size,
                                                    device, dtype, True)
        return attn_metadata

    def compute_input_embeddings_for_mm_optimized(self, **kwargs):
        input_ids = kwargs['input_ids']
        vision_embeddings = self.model.get_multimodal_embeddings(**kwargs)
        inputs_embeds = self.model.get_input_embeddings(
            input_ids, vision_embeddings)

        if vision_embeddings is not None:
            input_ids = kwargs['input_ids']
            positions = kwargs['positions']
            kwargs = self.model.prepare_attn_masks(
                mask_dtype=self.dtype,
                **kwargs,
            )
            kwargs['input_ids'] = input_ids
            kwargs['positions'] = positions
            #input_ids = None

        kwargs.update({'inputs_embeds': inputs_embeds})
        # done compute the visual tokens
        kwargs.pop('pixel_values', None)
        return kwargs

    def compute_input_embeddings_for_mrope_mm_optimized(self, **kwargs):

        if 'inputs_embeds' in kwargs:
            return kwargs
        if not self.model_is_mrope and not self.is_mm_optimized:
            return None
        # For Qwen2.5-VL/Gemma3 VL multimodal embedding,
        # this embedding part should be executed
        # with PT_COMPILE_ONLY_MODE off at all times
        # due to it's dynamicity.
        # During warmup, this is ON by default, so we
        # are turning it off here.
        # Also, we moved this code block from
        # model.forward() since we want to get
        # embedding before pass it to model which is also
        # aligned with VLLM V1.
        compile_only_mode_context_false = functools.partial(
            bc.env_setting, "PT_COMPILE_ONLY_MODE", False)

        input_ids = kwargs['input_ids']
        with compile_only_mode_context_false():
            if self.model_is_mrope:
                image_input = self.model._parse_and_validate_image_input(
                    **kwargs)
                video_input = self.model._parse_and_validate_video_input(
                    **kwargs)
                inputs_embeds = self.model.get_input_embeddings_v0(
                    input_ids,
                    image_input=image_input,
                    video_input=video_input)
                input_ids = None
                kwargs.update({
                    'inputs_embeds': inputs_embeds,
                })
                # done compute the visual tokens
                kwargs.pop('pixel_values', None)
                kwargs.pop('image_grid_thw', None)
                return kwargs
            else:
                return self.compute_input_embeddings_for_mm_optimized(**kwargs)

    def forward(self, *args, **kwargs):
        kwargs = kwargs.copy()
        selected_token_indices = kwargs.pop('selected_token_indices')
        if 'warmup_mode' in kwargs:
            kwargs.pop('warmup_mode')
        virtual_engine = 0
        if 'virtual_engine' in kwargs:
            virtual_engine = kwargs.pop('virtual_engine')

        input_ids = kwargs['input_ids']
        global_attn_masks = kwargs.get("global_attn_masks") \
                if kwargs.get("global_attn_masks") else None
        local_attn_masks = kwargs.get("local_attn_masks") \
                if kwargs.get("local_attn_masks") else None

        kwargs['attn_metadata'] = self._update_metadata(
            kwargs['attn_metadata'], input_ids.size(0), input_ids.size(1),
            input_ids.device, self.dtype, global_attn_masks, local_attn_masks)

        if 'lora_mask' in kwargs:
            LoraMask.setLoraMask(kwargs.pop('lora_mask'))
        if self._rotary_prepare_cos_sin is not None and not self.model_is_mrope:
            self._rotary_prepare_cos_sin(
                kwargs['positions'], recompute_cos_sin=self.recompute_cos_sin)
        if self.model_is_mrope or self.is_mm_optimized:
            # inputs_embeds was computed on execute_model
            # now we always want to use the inputs_embeds
            # even if the prompt is text only
            # that keeps all the shapes consistent with warmup
            kwargs.update({
                'input_ids': None,
            })
        attn_meta = kwargs.pop('attn_metadata')
        if 'kv_caches' in kwargs:
            kwargs.pop('kv_caches')
        with set_forward_context(attn_meta,
                                 self.vllm_config,
                                 virtual_engine,
                                 dp_awared_padding=self.dp_awared_padding):
            hidden_states = self.model(*args, **kwargs)
            if self._rotary_prepare_cos_sin is not None and \
                not self.model_is_mrope:
                self._reset_rotary_cos_sin()
            if not get_pp_group().is_last_rank:
                return hidden_states
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            if selected_token_indices is not None:
                hidden_states = hidden_states.index_select(
                    0, selected_token_indices)

        return hidden_states

    def compute_logits(self, *args, **kwargs):
        return self.model.compute_logits(*args, **kwargs)

    # def sample(self, *args, **kwargs):
    #    return self.sampler(*args, **kwargs)

    def make_empty_intermediate_tensors(self, *args, **kwargs):
        return self.model.make_empty_intermediate_tensors(*args, **kwargs)

    def generate_proposals(self, *args, **kwargs):
        if hasattr(self.model, "sampler"):
            # Speculative decoding
            self.model.sampler = self.sampler
        return self.model.generate_proposals(*args, **kwargs)

    # sampler property will be used by spec_decode_worker
    # don't rename
    # @property
    # def sampler(self):
    #    return self.model.sampler

    # lm_head property will be used by spec_decode_worker
    # don't rename
    @property
    def lm_head(self):
        return self.model.lm_head


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
    previous_hidden_states: Optional[torch.Tensor] = None

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


@dataclass
class CachedStepOutput:
    token_ids: torch.Tensor
    logprobs: Optional[torch.Tensor] = None
    deffered_sample_results: Optional[SampleResultArgsType] = None
    sampling_metadata: Optional[SamplingMetadata] = None
    is_prompt: Optional[bool] = False

    def __init__(
            self,
            token_ids: torch.Tensor,
            logprobs: Optional[torch.Tensor] = None,
            deffered_sample_results: Optional[SampleResultArgsType] = None,
            sampling_metadata: Optional[SamplingMetadata] = None,
            is_prompt: Optional[bool] = False):
        self.token_ids = token_ids
        self.logprobs = logprobs
        self.deffered_sample_results = deffered_sample_results
        self.sampling_metadata = sampling_metadata
        self.is_prompt = is_prompt


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
        is_causal: bool = True,
    ):
        ModelRunnerBase.__init__(self, vllm_config=vllm_config)

        environment.set_vllm_config(vllm_config)
        finalize_config()

        self.is_driver_worker = is_driver_worker
        self.return_hidden_states = return_hidden_states

        self.sliding_window = (self.model_config.get_sliding_window()
                               if self.model_config is not None else None)

        self.interleaved_sliding_window = getattr(
            self.model_config.hf_text_config, "interleaved_sliding_window",
            None)

        self.device_config = (self.device_config if self.device_config
                              is not None else DeviceConfig())
        if is_fake_hpu():
            self.device_config.device = torch.device('cpu')
            self.device_config.device_type = 'cpu'
            self.load_config.device = None
        self.device = self.device_config.device
        self.enforce_eager = self.model_config.enforce_eager
        self.max_num_seqs = self.scheduler_config.max_num_seqs
        self.max_num_prefill_seqs = \
            self.scheduler_config.max_num_prefill_seqs \
            if self.scheduler_config.max_num_prefill_seqs is not None \
                else self.max_num_seqs
        self.max_model_len = self.scheduler_config.max_model_len
        self.max_num_batched_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.block_size = self.cache_config.block_size
        self.use_merged_prefill = get_config().merged_prefill
        assert not (self.scheduler_config.use_padding_aware_scheduling
                    and self.use_merged_prefill), \
            'Merged prefill is not compatible with padding aware scheduling!'

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
            use_mla=self.model_config.use_mla,
        ) if needs_attn_backend else None

        # Multi-modal data support
        self.input_registry = input_registry
        self.mm_registry = mm_registry
        self.mm_registry = MULTIMODAL_REGISTRY
        self.multi_modal_input_mapper = self.mm_registry \
            .create_input_mapper(self.model_config)
        self.mm_registry.init_mm_limits_per_prompt(self.model_config)
        self.is_mm_optimized = False
        # Lazy initialization
        self.lora_manager: LRUCacheWorkerLoRAManager = None
        self.model: torch.nn.Module = None
        self.inc_initialized_successfully = False

        # Profiler stats
        self.profiler = HabanaHighLevelProfiler()
        self.profiler_counter_helper = HabanaProfilerCounterHelper(is_v1=False)
        self.seen_configs: set = set()
        self._mem_margin: Optional[int] = None
        self.use_prefix_caching = (
            self.vllm_config.cache_config.enable_prefix_caching)
        self.bucketing_manager = HPUBucketingManager()
        self.bucketing_manager.initialize(
            max_num_seqs=self.max_num_seqs,
            max_num_prefill_seqs=self.max_num_prefill_seqs,
            block_size=self.block_size,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_model_len=self.max_model_len)
        self.graphed_buckets: Set[Any] = set()
        self.multimodal_buckets: List[int] = [
        ]  #TODO: Move to HPUBucketingContext
        self.graphed_multimodal_buckets: Set[Any] = set()
        self.use_contiguous_pa = envs.VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH

        # Data Parallel
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_awared_padding = self.dp_size > 1

        self._set_gc_threshold()
        self.use_contiguous_pa = get_config().use_contiguous_pa
        if vllm_config.speculative_config is not None \
           and self.use_contiguous_pa:
            raise ValueError(
                "Speculative decoding is not supported with "
                "contiguous PA, please set VLLM_CONTIGUOUS_PA=false")
        # For both multi-step scheduling and delayed sampling
        self.is_single_step = \
            self.vllm_config.scheduler_config.num_scheduler_steps == 1
        self.cached_step_outputs: List[CachedStepOutput] = []
        self.is_pooler = False
        self.is_causal = is_causal
        # For delayed sampling
        self.cached_step_inputs: List[
            ModelInputForHPUWithSamplingMetadata] = []
        self.spec_decode_enabled = \
            self.vllm_config.speculative_config is not None
        self.sampler = get_sampler()
        can_use_delayed_sampling = (not self.spec_decode_enabled
                                    and not is_fake_hpu()
                                    and self.is_single_step
                                    and not self.lora_config)
        self.use_delayed_sampling = get_config(
        ).use_delayed_sampling and can_use_delayed_sampling

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

        self.skip_warmup = get_config().skip_warmup

    @property
    def model_is_mrope(self) -> bool:
        config = self.model_config.hf_config
        return uses_mrope(config)

    def _is_quant_with_inc(self):
        quant_config = os.getenv("QUANT_CONFIG", None) is not None
        return (self.model_config.quantization == "inc" or quant_config)

    def _maybe_init_alibi_biases(self) -> None:
        layers = None
        layer_alibi_config = None
        if (not hasattr(self.model, "config")
                or not hasattr(self.model.config, "architectures")):
            pass
        elif "BaichuanForCausalLM" in self.model.config.architectures:
            if self.model.config.hidden_size != 4096:
                layers = self.model.model.layers
                layer_alibi_config = lambda layer: \
                    layer.self_attn.attn \
                        if hasattr(layer, 'self_attn') else None
        elif "JAISLMHeadModel" in self.model.config.architectures:
            if self.model.config.position_embedding_type == "alibi":
                layers = self.model.transformer.h
                layer_alibi_config = lambda layer: \
                    layer.attn.attn \
                        if hasattr(layer, 'attn') else None
        elif "FalconForCausalLM" in self.model.config.architectures:
            if self.model.config.alibi:
                layers = self.model.transformer.h
                layer_alibi_config = lambda layer: \
                    layer.self_attention.attn \
                        if hasattr(layer, 'self_attention') else None
        elif "MPTForCausalLM" in self.model.config.architectures:
            if self.model.config.attn_config['alibi']:
                layers = self.model.transformer.blocks
                layer_alibi_config = lambda layer: \
                    layer.attn.attn \
                        if hasattr(layer, 'attn') else None
        elif "BloomForCausalLM" in self.model.config.architectures:
            layers = self.model.transformer.h
            layer_alibi_config = lambda layer: \
                layer.self_attention.attn \
                    if hasattr(layer, 'self_attention') else None

        if (layers is not None and layer_alibi_config is not None):
            max_seq_len = self.bucketing_manager.get_max_prompt_shape()
            self.use_alibi = True
            prev_attn = None
            for layer in layers:
                attn = layer_alibi_config(layer)
                if attn is None or not hasattr(attn, "impl"):
                    continue
                if (hasattr(attn.impl, "_maybe_init_alibi_biases")):
                    attn.impl._maybe_init_alibi_biases(
                        max_seq_len=max_seq_len,
                        prev_attn=prev_attn,
                    )
                prev_attn = attn
        else:
            self.use_alibi = False

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

                # Use get_text_config() in case of multimodal models
                text_config = self.model_config.hf_config.get_text_config()

                self.lora_manager = LRUCacheWorkerLoRAManager(
                    self.scheduler_config.max_num_seqs,
                    self.scheduler_config.max_num_batched_tokens,
                    self.vocab_size,
                    self.lora_config,
                    self.device,
                    self.model.embedding_modules,
                    self.model.embedding_padding_modules,
                    max_position_embeddings=text_config.
                    max_position_embeddings,
                )
                self.model = self.lora_manager.create_lora_manager(self.model)

            if self._is_quant_with_inc():
                logger.info("Preparing model with INC..")
                with HabanaMemoryProfiler() as m_inc:
                    from neural_compressor.torch.quantization import (
                        FP8Config, convert, prepare)

                    disable_mark_scales_as_const = os.getenv(
                        "VLLM_DISABLE_MARK_SCALES_AS_CONST",
                        "false") in ("1", "true")
                    config = FP8Config.from_json_file(
                        os.getenv("QUANT_CONFIG", ""))
                    self._inc_preprocess()
                    if config.measure:
                        self.model = prepare(self.model, config)
                    elif config.quantize:
                        self.model = convert(self.model, config)
                    if not disable_mark_scales_as_const:
                        htcore.hpu_initialize(self.model,
                                              mark_only_scales_as_const=True)
                    if torch.distributed.is_initialized():
                        torch.distributed.barrier()
                self.inc_initialized_successfully = True
                logger.info("Preparing model with INC took %s",
                            m_inc.get_summary_string())
            elif not is_fake_hpu():
                self.model = self.model.to("hpu")
                htcore.mark_step()

            self._maybe_init_alibi_biases()
            hidden_layer_markstep_interval = int(
                os.getenv('VLLM_CONFIG_HIDDEN_LAYERS', '1'))
            model_config = getattr(self.model, "config", None)
            modify_model_layers(
                self.model,
                get_target_layer_suffix_list(
                    model_config.
                    model_type if model_config is not None else None),
                hidden_layer_markstep_interval)
            torch.hpu.synchronize()

            if self.is_pooler:
                self.set_causal_option(self.model)
            with HabanaMemoryProfiler() as m_wrap:
                self.model = self._maybe_wrap_in_hpu_graph(
                    self.model,
                    vllm_config=self.vllm_config,
                    is_causal=self.is_causal,
                    sampler=self.sampler)
            msg = f"Wrapping in HPU Graph took {m_wrap.get_summary_string()}"
            logger.info(msg)
            with HabanaMemoryProfiler() as m_wrap:
                self._maybe_compile(self.model)
            msg = f"Compiling took {m_wrap.get_summary_string()}"
            logger.info(msg)

        self.model_memory_usage = m.consumed_device_memory
        msg = f"Loading model weights took in total {m.get_summary_string()}"
        logger.info(msg)

        # Models that process images at different resolutions
        # need to be warmed up. Current tested for MRoPE models only.
        self.add_vision_buckets_to_mrope_mm_optimized()

    def _add_dummy_seq(self,
                       seq_group_metadata_list,
                       is_prompt,
                       align_worker=False):
        real_batch_size = len(seq_group_metadata_list)
        ctx = seq_group_metadata_list[0].computed_block_nums
        ctx = 0 if ctx is None else sum(ctx)
        batch_size_padded = real_batch_size
        if is_prompt:
            first_key = next(iter(seq_group_metadata_list[0].seq_data))
            seq_len = len(seq_group_metadata_list[0].seq_data[first_key].
                          prompt_token_ids)
            query_len = seq_len - ctx * self.block_size
            batch_size_padded = self.bucketing_manager.find_prompt_bucket(
                real_batch_size, query_len, ctx)[0]
        else:
            batch_size_padded = self.bucketing_manager.find_decode_bucket(
                real_batch_size, ctx)[0]
        if self.dp_awared_padding and (self.vllm_config.kv_transfer_config
                                       is None or not is_prompt):
            if self.is_driver_worker:
                batch_size_padded = align_dp_groups(
                    batch_size_padded, torch.distributed.ReduceOp.MAX)
            if align_worker:
                batch_size_padded = align_tp_groups(
                    batch_size_padded, torch.distributed.ReduceOp.MAX)
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
        if htorch.utils.internal.is_lazy():
            return htorch.hpu.wrap_in_hpu_graph(HpuModelAdapter(
                *args, **kwargs),
                                                disable_tensor_cache=True)
        else:
            return HpuModelAdapter(*args, **kwargs)

    def _maybe_compile(self, *args, **kwargs):
        if not is_fake_hpu() and not htorch.utils.internal.is_lazy(
        ) and not self.vllm_config.model_config.enforce_eager:
            if os.getenv('VLLM_REGIONAL_COMPILATION',
                         'true').strip().lower() in ("1", "true"):
                compiled_methods = [
                    '_update_metadata', '_rotary_prepare_cos_sin'
                ]
                for method_name in compiled_methods:
                    method = getattr(self.model, method_name)
                    if method is not None:
                        self._compile_region(self.model, method_name, method)

                self.regional_compilation_layers_list = [
                    RMSNorm, VocabParallelEmbedding
                ]
                self._regional_compilation(self.model)
            else:
                self.model = self._compile(self.model)

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
            self._compile_region(
                parent_module,
                module_name,
                module,
            )
        else:
            for children_name, children_module in module.named_children():
                self._regional_compilation(children_module, module,
                                           children_name)

    def _compile_region(self, model, name, module):
        module = self._compile(module)
        setattr(model, name, module)

    def _compile(self, module):
        if not hasattr(self, '_compile_config'):
            fullgraph = os.getenv('VLLM_T_COMPILE_FULLGRAPH',
                                  'false').strip().lower() in ("1", "true")
            dynamic = os.getenv('VLLM_T_COMPILE_DYNAMIC_SHAPES',
                                'false').strip().lower() in ("1", "true")
            self._compile_config = {'fullgraph': fullgraph, 'dynamic': dynamic}
        fullgraph = self._compile_config['fullgraph']
        dynamic = self._compile_config['dynamic']
        if dynamic:
            return torch.compile(module,
                                 backend='hpu_backend',
                                 fullgraph=fullgraph,
                                 options={"force_static_compile": True})
        else:
            return torch.compile(module,
                                 backend='hpu_backend',
                                 fullgraph=fullgraph,
                                 dynamic=False)

    def get_model(self) -> torch.nn.Module:
        if isinstance(self.model, HpuModelAdapter):
            return self.model.model
        return self.model

    def _use_graphs(self, img_args=None):
        if not img_args:
            return not self.enforce_eager
        #TODO: We might need to check both language bucket and multimodal bucket
        # and return True only it's avialble, or return separately.
        return (img_args) in self.graphed_multimodal_buckets

    def _is_valid_bucket(self, bucket):
        return bucket[0] * bucket[1] <= self.max_num_batched_tokens

    def _num_blocks(self, attn_metadata):
        if attn_metadata.block_list is None:
            return 0
        return attn_metadata.block_list.numel()

    def _check_config(self, batch_size, seq_len, ctx, attn_metadata,
                      warmup_mode):
        phase = 'prompt' if attn_metadata.is_prompt else 'decode'
        num_blocks = ctx if warmup_mode else self._num_blocks(attn_metadata)
        cfg: Optional[tuple] = (batch_size, seq_len, num_blocks, phase)
        seen = cfg in self.seen_configs
        self.seen_configs.add(cfg)
        if not seen and not warmup_mode:
            logger.warning("Configuration: %s was not warmed-up!",
                           (phase, batch_size, seq_len, num_blocks))

    def _get_mrope_positions_and_delta(self, seq_data, mm_kwargs, context_len):
        image_grid_thw = mm_kwargs.get("image_grid_thw", None)
        video_grid_thw = mm_kwargs.get("video_grid_thw", None)
        second_per_grid_ts = mm_kwargs.get("second_per_grid_ts", None)
        assert image_grid_thw is not None or video_grid_thw is not None, (
            "mrope embedding type requires multi-modal input mapper "
            "returns 'image_grid_thw' or 'video_grid_thw'.")
        hf_config = self.model_config.hf_config
        token_ids = seq_data.get_token_ids()
        mrope_positions, mrope_position_delta = \
            MRotaryEmbedding.get_input_positions(
                token_ids,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                context_len=context_len,
            )
        assert mrope_positions is not None
        return mrope_positions, mrope_position_delta

    def make_attn_bias(self, seq_lens, max_prompt_len, dtype):
        seq_pos = [list(range(sl)) for sl in seq_lens]
        seq_idx = [[i] * sl for i, sl in enumerate(seq_lens)]

        seq_pos_t = make_cpu_tensor(seq_pos,
                                    max_len=max_prompt_len,
                                    pad=-1,
                                    dtype=torch.long,
                                    flat=self.use_merged_prefill)
        seq_idx_t = make_cpu_tensor(seq_idx,
                                    max_len=max_prompt_len,
                                    pad=-1,
                                    dtype=torch.long,
                                    flat=self.use_merged_prefill)

        q_seq_idx_t = seq_idx_t.unsqueeze(-1)
        kv_seq_idx_t = seq_idx_t.unsqueeze(-2)
        q_seq_pos_t = seq_pos_t.unsqueeze(-1)
        kv_seq_pos_t = seq_pos_t.unsqueeze(-2)
        seq_idx_t = q_seq_idx_t != kv_seq_idx_t
        seq_pos_t = kv_seq_pos_t > q_seq_pos_t
        attn_mask = (seq_idx_t | seq_pos_t) if self.is_causal else seq_idx_t
        if self.is_pooler:
            mask_v = torch.where(q_seq_pos_t < 0, True, False)
            attn_mask = attn_mask | mask_v
            off_value = -3E38  #small number, avoid nan and overflow
        else:
            off_value = -math.inf
        attn_bias = torch.zeros_like(attn_mask, dtype=dtype)
        attn_bias.masked_fill_(attn_mask, off_value)
        return attn_bias.unsqueeze(1)

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

    def move_to_device(self, tensor):
        return tensor if tensor is None else tensor.to(self.device,
                                                       non_blocking=True)

    def add_vision_buckets_to_mrope_mm_optimized(self):
        if self.mm_registry is not None:
            model = self.get_model()
            self.is_mm_optimized = is_mm_optimized(model)
            if self.model_is_mrope or self.is_mm_optimized:
                model.vision_buckets = VisionBuckets(self.is_mm_optimized)

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        align_worker=False,
    ) -> PreparePromptMetadata:
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        input_mrope_positions: List[List[List[int]]] = []
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
        encoder_seq_lens: List[int] = []
        cross_slot_mapping: List[int] = []

        if len(seq_group_metadata_list) == 0:
            return PreparePromptMetadata.empty()

        is_enc_dec_model = self.model_config.is_encoder_decoder
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
                if context_len == seq_len \
                and self.use_prefix_caching:
                    # Fully cached prompt - compute only last token
                    context_len = context_len - 1
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

            seq_data_mrope_positions: Optional[List[List[int]]] = None

            if is_enc_dec_model:
                encoder_seq_len = seq_group_metadata.encoder_seq_data.get_len(
                ) if seq_group_metadata.encoder_seq_data else 0
                encoder_seq_lens.append(encoder_seq_len)
                # Build slot mapping
                if seq_group_metadata.cross_block_table is None:
                    cross_slot_mapping.extend([_PAD_SLOT_ID] * encoder_seq_len)
                else:
                    for i in range(0, encoder_seq_len):
                        block_number = seq_group_metadata.cross_block_table[
                            i // self.block_size]
                        block_offset = i % self.block_size
                        slot = block_number * self.block_size + block_offset
                        cross_slot_mapping.append(slot)

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

                # special processing for mrope position deltas.
                if self.model_is_mrope:
                    mrope_positions, mrope_position_delta = \
                        self._get_mrope_positions_and_delta(
                            seq_data=seq_data,
                            mm_kwargs=mm_kwargs,
                            context_len=context_len)
                    assert mrope_positions is not None
                    seq_data.mrope_position_delta = mrope_position_delta
                    seq_data_mrope_positions = [[] for _ in range(3)]
                    for idx in range(3):
                        seq_data_mrope_positions[idx] \
                            .extend(mrope_positions[idx])

                multi_modal_kwargs_list.append(mm_kwargs)

                for modality, placeholder_map in placeholder_maps.items():
                    multi_modal_placeholder_maps[modality].extend(
                        placeholder_map)

            input_mrope_positions.append(
                seq_data_mrope_positions)  # type: ignore

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

        if self.use_merged_prefill:
            target_query_len = sum(query_lens)
        else:
            target_query_len = max(query_lens)
        ctx = len(computed_block_nums) if computed_block_nums else 0

        if is_enc_dec_model:
            real_batch_size = len(seq_group_metadata_list)
            batch_size_padded = self.bucketing_manager.find_prompt_bucket(
                real_batch_size, target_query_len, ctx)[0]
            batch_size_padding = batch_size_padded - real_batch_size
            if batch_size_padding > 0:
                encoder_seq_lens.extend(encoder_seq_lens[0]
                                        for _ in range(batch_size_padding))

        real_num_seqs = len(query_lens)
        max_prompt_len = max(
            self.bucketing_manager.find_prompt_bucket(
                len(seq_group_metadata_list), target_query_len, ctx)[1],
            self.block_size)

        if self.dp_awared_padding and\
            self.vllm_config.kv_transfer_config is None:
            if self.is_driver_worker:
                max_prompt_len = align_dp_groups(
                    max_prompt_len, torch.distributed.ReduceOp.MAX)
            if align_worker:
                max_prompt_len = align_tp_groups(
                    max_prompt_len, torch.distributed.ReduceOp.MAX)

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

            pad_len = len(prefix_block_list)
            prefix_block_list = pad_list(prefix_block_list, pad_len,
                                         _PAD_BLOCK_ID)

            prefix_block_list_tensor = torch.tensor(prefix_block_list,
                                                    dtype=torch.long,
                                                    device=self.device)
        else:
            prefix_block_list_tensor = None

        input_tokens_tensor = make_cpu_tensor(input_tokens,
                                              max_len=max_prompt_len,
                                              pad=0,
                                              dtype=torch.long,
                                              flat=self.use_merged_prefill)
        if self.model_is_mrope:
            input_positions = \
                make_mrope_positions_tensor_with_pad(input_positions=input_positions,
                                                     input_mrope_positions=input_mrope_positions,
                                                     max_prompt_len=max_prompt_len,
                                                     pad=0)
        else:
            input_positions = make_cpu_tensor(input_positions,
                                              max_len=max_prompt_len,
                                              pad=0,
                                              dtype=torch.long,
                                              flat=self.use_merged_prefill)

        slot_mapping = make_cpu_tensor(slot_mapping,
                                       max_len=max_prompt_len,
                                       pad=_PAD_SLOT_ID,
                                       dtype=torch.long,
                                       flat=self.use_merged_prefill)

        if is_enc_dec_model:
            encoder_seq_lens_tensor = torch.tensor(encoder_seq_lens,
                                                   dtype=torch.int32,
                                                   device='cpu')
            cross_slot_mapping = torch.tensor(cross_slot_mapping,
                                              dtype=torch.long,
                                              device='cpu')
        else:
            encoder_seq_lens = []
            encoder_seq_lens_tensor = None
            cross_slot_mapping = []

        attn_bias = None
        seq_lens_tensor = None
        context_lens_tensor = None

        if self.use_merged_prefill:
            attn_bias = self.make_attn_bias(seq_lens, max_prompt_len,
                                            self.model_config.dtype)

        num_seqs = self.max_num_prefill_seqs \
            if self.use_merged_prefill else real_num_seqs
        seq_lens_tensor = make_cpu_tensor([seq_lens],
                                          max_len=num_seqs,
                                          pad=0,
                                          dtype=torch.long,
                                          flat=True).flatten()
        context_lens_tensor = make_cpu_tensor([context_lens],
                                              max_len=num_seqs,
                                              pad=0,
                                              dtype=torch.long,
                                              flat=True).flatten()

        placeholder_index_maps = {
            modality: placeholder_map.index_map()
            for modality, placeholder_map in
            multi_modal_placeholder_maps.items()
        }

        # Note: num_prefill_tokens is calculated using the length of
        # input_tokens after padding.
        num_prefill_tokens = input_tokens_tensor.numel()

        prefix_block_list_tensor = self.move_to_device(
            prefix_block_list_tensor)
        input_tokens_tensor = self.move_to_device(input_tokens_tensor)
        input_positions = self.move_to_device(input_positions)
        seq_lens_tensor = self.move_to_device(seq_lens_tensor)
        slot_mapping = self.move_to_device(slot_mapping)
        context_lens_tensor = self.move_to_device(context_lens_tensor)
        attn_bias = self.move_to_device(attn_bias)
        if is_enc_dec_model:
            cross_slot_mapping = self.move_to_device(cross_slot_mapping)
            encoder_seq_lens_tensor = self.move_to_device(
                encoder_seq_lens_tensor)

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=True,
            block_size=self.block_size,
            block_list=prefix_block_list_tensor,
            block_mapping=None,
            block_usage=None,
            block_groups=None,
            attn_bias=attn_bias,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            encoder_seq_lens=encoder_seq_lens,
            encoder_seq_lens_tensor=encoder_seq_lens_tensor,
            max_encoder_seq_len=max(encoder_seq_lens, default=0),
            cross_slot_mapping=cross_slot_mapping,
            context_lens_tensor=context_lens_tensor,
            num_prefills=real_num_seqs,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            alibi_blocks=None,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            enable_kv_scales_calculation=False,
            input_positions=input_positions,
        )
        multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_kwargs_list)
        multi_modal_kwargs = MultiModalKwargs.as_kwargs(multi_modal_kwargs,
                                                        device=self.device)

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
        align_worker=False,
    ) -> PrepareDecodeMetadata:
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        input_mrope_positions: List[List[int]] = [[] for _ in range(3)]
        slot_mapping: List[List[int]] = []
        seq_lens: List[int] = []
        encoder_seq_lens: List[int] = []
        cross_block_tables: List[List[int]] = []
        block_tables: List[List[int]] = []
        window_block_tables: List[List[int]] = []
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

                if self.model_is_mrope:
                    if seq_data.mrope_position_delta is not None:
                        pos_for_mrope = MRotaryEmbedding \
                            .get_next_input_positions(
                                seq_data.mrope_position_delta,
                                seq_data.get_num_computed_tokens(),
                                seq_len)
                    else:
                        pos_for_mrope = [[position]] * 3
                    for idx in range(3):
                        input_mrope_positions[idx].extend(pos_for_mrope[idx])

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

                if self.interleaved_sliding_window is not None:
                    sliding_window_blocks = (self.interleaved_sliding_window //
                                             self.block_size)
                    window_block_table = block_table[-sliding_window_blocks:]
                    window_block_tables.append(window_block_table)

        if output is None:
            input_tokens = torch.tensor(input_tokens,
                                        dtype=torch.long,
                                        device='cpu')
        else:
            real_batch_size = len(seq_group_metadata_list)
            input_tokens = output[:real_batch_size].clone()

        input_positions = torch.tensor(
            input_mrope_positions if self.model_is_mrope else input_positions,
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

        if self.interleaved_sliding_window is not None:
            window_block_groups = [[i] * len(bt)
                                   for i, bt in enumerate(window_block_tables)]
            window_block_usage = [
                [self.block_size] * (len(bt) - 1) + [lbu]
                for bt, lbu in zip(block_tables, last_block_usage) if bt
            ]

            window_block_list = flatten(window_block_tables)
            window_block_groups = flatten(window_block_groups)
            window_block_usage = flatten(window_block_usage)

            assert len(window_block_list) == len(window_block_groups)
            assert len(window_block_list) == len(window_block_list)
        else:
            window_block_list = None
            window_block_groups = None
            window_block_usage = None

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
            block_bucket_size = self.bucketing_manager.find_decode_bucket(
                len(seq_group_metadata_list), block_bucket_size)[2]
            if self.dp_awared_padding:
                if self.is_driver_worker:
                    block_bucket_size = align_dp_groups(
                        block_bucket_size, torch.distributed.ReduceOp.MAX)
                if align_worker:
                    block_bucket_size = align_tp_groups(
                        block_bucket_size, torch.distributed.ReduceOp.MAX)
            indices: List[Any]
            indices = [None] * block_bucket_size
            for i, bid in enumerate(block_list):
                indices[bid] = i
            padding_fn = lambda tensor, pad_value: gather_list(
                tensor, indices, pad_value)
            if self.interleaved_sliding_window is not None:
                window_indices: List[Any]
                window_indices = [None] * block_bucket_size
                for i, bid in enumerate(window_block_list):
                    window_indices[bid] = i
                window_padding_fn = lambda tensor, pad_value: gather_list(
                    tensor, window_indices, pad_value)
        else:
            block_bucket_size = self.bucketing_manager.find_decode_bucket(
                len(seq_group_metadata_list), len(block_list))[2]
            if self.dp_awared_padding:
                if self.is_driver_worker:
                    block_bucket_size = align_dp_groups(
                        block_bucket_size, torch.distributed.ReduceOp.MAX)
                if align_worker:
                    block_bucket_size = align_tp_groups(
                        block_bucket_size, torch.distributed.ReduceOp.MAX)
            padding_fn = lambda tensor, pad_value: pad_list(
                tensor, block_bucket_size, pad_value)

        block_list = padding_fn(block_list, _PAD_BLOCK_ID)
        block_groups = padding_fn(block_groups, -1)
        block_usage = padding_fn(block_usage, 1)

        if self.interleaved_sliding_window is not None:
            window_block_list = window_padding_fn(window_block_list,
                                                  _PAD_BLOCK_ID)
            window_block_groups = window_padding_fn(window_block_groups, -1)
            #window_block_usage = window_padding_fn(window_block_usage, 1)
            window_block_usage = [
                [1] if i == 0 else [block_usage[idx]]
                for idx, (i,
                          j) in enumerate(zip(window_block_list, block_usage))
            ]

        if is_enc_dec_model:
            if self.use_contiguous_pa:
                cross_block_bucket_size = max(
                    max(cross_block_list) +
                    1, len(cross_block_list)) if cross_block_list else 0
                cross_block_bucket_size = \
                    self.bucketing_manager.find_decode_bucket(
                        len(seq_group_metadata_list),
                        cross_block_bucket_size)[2]
                indices = [None] * cross_block_bucket_size
                for i, bid in enumerate(cross_block_list):
                    indices[bid] = i
                padding_fn = lambda tensor, pad_value: gather_list(
                    tensor, indices, pad_value)
            else:
                cross_block_bucket_size = \
                    self.bucketing_manager.find_decode_bucket(
                        len(seq_group_metadata_list),
                        len(cross_block_list))[2]
                padding_fn = lambda tensor, pad_value: pad_list(
                    tensor, cross_block_bucket_size, pad_value)

            real_batch_size = len(seq_group_metadata_list)
            batch_size_padded = \
                self.bucketing_manager.find_decode_bucket(
                        real_batch_size,
                        cross_block_bucket_size)[0]
            if self.dp_awared_padding:
                if self.is_driver_worker:
                    batch_size_padded = align_dp_groups(
                        batch_size_padded, torch.distributed.ReduceOp.MAX)
                if align_worker:
                    batch_size_padded = align_tp_groups(
                        batch_size_padded, torch.distributed.ReduceOp.MAX)
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

        alibi_blocks = None
        if self.use_alibi:
            alibi_blocks = self._compute_alibi_block(block_tables, seq_lens,
                                                     len(block_groups))
            alibi_blocks = alibi_blocks.to(  # type: ignore
                self.device, non_blocking=True)

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

            encoder_seq_lens_tensor = \
                encoder_seq_lens_tensor.to(  # type: ignore
                    self.device, non_blocking=True)

        if self.interleaved_sliding_window is not None:
            window_block_list = torch.tensor(window_block_list,
                                             dtype=torch.int,
                                             device='cpu')
            window_block_groups = torch.tensor(window_block_groups,
                                               dtype=torch.int,
                                               device='cpu')
            window_block_usage = torch.tensor(window_block_usage,
                                              dtype=self.model_config.dtype,
                                              device='cpu')

            window_block_list = window_block_list.to(  # type: ignore
                self.device, non_blocking=True)
            window_block_groups = window_block_groups.to(  # type: ignore
                self.device, non_blocking=True)
            window_block_usage = window_block_usage.to(  # type: ignore
                self.device, non_blocking=True)

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=False,
            block_size=self.block_size,
            block_list=block_list,
            block_mapping=None,
            block_usage=block_usage,
            block_groups=block_groups,
            window_block_list=window_block_list,
            window_block_mapping=None,
            window_block_usage=window_block_usage,
            window_block_groups=window_block_groups,
            attn_bias=None,
            seq_lens_tensor=None,
            encoder_seq_lens=encoder_seq_lens,
            encoder_seq_lens_tensor=encoder_seq_lens_tensor,
            max_encoder_seq_len=max(encoder_seq_lens, default=0),
            cross_block_list=cross_block_list,
            cross_block_groups=cross_block_groups,
            cross_block_usage=cross_block_usage,
            context_lens_tensor=None,
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=num_decode_tokens,
            slot_mapping=slot_mapping,
            alibi_blocks=alibi_blocks,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
            input_positions=input_positions)
        return PrepareDecodeMetadata(input_tokens=input_tokens,
                                     input_positions=input_positions,
                                     attn_metadata=attn_metadata,
                                     lora_index_mapping=lora_index_mapping,
                                     lora_prompt_mapping=lora_prompt_mapping,
                                     lora_requests=lora_requests,
                                     slot_mapping=slot_mapping,
                                     lora_ids=lora_ids)

    def _compute_alibi_block(self, block_tables, seq_lens, num_blocks):
        """
        Compute the ALiBi offsets for each block during decoding.

        For each block in each sequence, this function assigns position-based
        offsets according to ALiBi logic. It returns a tensor that captures
        these offsets for all sequences and blocks, which is then used for
        decode-time ALiBi bias creation.

        Args:
            block_tables:
                A list of lists, where each inner list contains block indices
                assigned to a particular sequence.
            seq_lens:
                A list of sequence lengths corresponding to each sequence.
            num_blocks:
                The total number of blocks across all sequences for which
                ALiBi offsets need to be computed.

        Returns:
            A torch.Tensor of shape [num_blocks, block_size], containing ALiBi
            offsets for each block.
        """
        # Create intermediary and output structures on the CPU
        max_block_table_len = max(
            len(block_table) for block_table in block_tables)
        alibi_offsets = torch.arange(
            -max_block_table_len * self.block_size + 1,
            1,
            dtype=torch.long,
            device='cpu',
        )
        alibi_blocks = torch.zeros(
            (num_blocks, self.block_size),
            dtype=torch.long,
            device='cpu',
        )

        # Use lists to accumulate data for each block
        block_data: List[List[int]] = [[] for _ in range(num_blocks)]

        # Assign biases per token
        for batch_idx, block_table in enumerate(block_tables):
            seq_len = seq_lens[batch_idx]
            for seq_idx, block_idx in enumerate(block_table):
                # Calculate the number of valid positions in the current block
                valid_length = seq_len - seq_idx * self.block_size
                if valid_length > 0:
                    current_block_length = min(valid_length, self.block_size)
                    offset_end = current_block_length - valid_length
                    if offset_end == 0:
                        block_data[block_idx] = alibi_offsets[
                            -valid_length:].tolist()
                    else:
                        block_data[block_idx] = alibi_offsets[
                            -valid_length:offset_end].tolist()

        # Populate the alibi_blocks tensor from the accumulated data
        for block_idx, data in enumerate(block_data):
            alibi_blocks[block_idx, :len(data)] = torch.tensor(
                data, dtype=torch.long)

        return alibi_blocks

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        finished_requests_ids: Optional[List[str]] = None,
        align_worker=False,
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
            self._add_dummy_seq(seq_group_metadata_list, is_prompt,
                                align_worker))

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
        ) = self._prepare_prompt(prefill_reqs, align_worker=align_worker)
        (
            decode_input_tokens,
            decode_input_positions,
            decode_attn_metadata,
            decode_lora_index_mapping,
            decode_lora_prompt_mapping,
            decode_lora_requests,
            decode_slot_mapping,
            decode_lora_ids,
        ) = self._prepare_decode(decode_reqs, align_worker=align_worker)

        selected_token_indices = None
        if not self.is_pooler:
            generators = self.get_generators(finished_requests_ids)
            sampling_metadata = SamplingMetadata.prepare(
                seq_group_metadata_list,
                seq_lens,
                query_lens,
                'cpu',
                self.pin_memory,
                generators=generators)
            selected_token_indices = \
                sampling_metadata.selected_token_indices
            categorized_sample_indices = \
                sampling_metadata.categorized_sample_indices
            if self.use_merged_prefill and len(seq_lens) > 0:
                selected_token_indices = pad_flat_tensor(
                    selected_token_indices, self.max_num_prefill_seqs)
                categorized_sample_indices = {
                    k: pad_flat_tensor(v, self.max_num_prefill_seqs)
                    for k, v in categorized_sample_indices.items()
                }
                padding_groups = self.max_num_prefill_seqs - len(
                    sampling_metadata.seq_groups)
                import copy
                dummy_seq_group = copy.deepcopy(
                    sampling_metadata.seq_groups[0])
                sampling_metadata.seq_groups.extend(
                    dummy_seq_group for _ in range(padding_groups))
            sampling_metadata.selected_token_indices = \
                self.move_to_device(selected_token_indices)
            sampling_metadata.categorized_sample_indices = \
                {k: self.move_to_device(v)
                 for k, v in categorized_sample_indices.items()}

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

        if self.is_pooler:
            sampling_metadata = None
        elif not self.use_merged_prefill:
            # FIXME: We need to adjust selected_token_indices to accommodate
            # for padding
            max_len = input_tokens.size(1)
            paddings = [max_len - q for q in query_lens]
            paddings = [0] + paddings[:-1]
            paddings = list(itertools.accumulate(paddings))
            paddings_prompt_logprobs = []

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
            "input_tokens": input_tokens,
            "input_positions": input_positions,
            "selected_token_indices": selected_token_indices,
            "lora_requests": lora_requests,
            "lora_mapping": lora_mapping,
            "multi_modal_kwargs": multi_modal_kwargs,
            "num_prefill_tokens": num_prefill_tokens,
            "num_decode_tokens": num_decode_tokens,
            "slot_mapping": slot_mapping,
            "num_prefills": num_prefills,
            "batch_type": batch_type,
            "seq_lens": seq_lens,
            "query_lens": query_lens
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

    @torch.inference_mode()
    def prepare_model_input_align_worker(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
        align_worker: bool = False,
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
                seq_group_metadata_list, finished_requests_ids, align_worker)
            assert model_input.attn_metadata is not None
            is_prompt = model_input.attn_metadata.is_prompt

        return ModelInputForHPUWithSamplingMetadata(
            input_tokens=model_input.input_tokens,
            input_positions=model_input.input_positions,
            seq_lens=model_input.seq_lens,
            query_lens=model_input.query_lens,
            lora_mapping=model_input.lora_mapping,
            lora_requests=model_input.lora_requests,
            attn_metadata=model_input.attn_metadata,
            multi_modal_kwargs=model_input.multi_modal_kwargs,
            real_batch_size=model_input.real_batch_size,
            batch_size_padded=model_input.batch_size_padded,
            virtual_engine=virtual_engine,
            lora_ids=model_input.lora_ids,
            async_callback=model_input.async_callback,
            is_first_multi_step=model_input.is_first_multi_step,
            is_last_step=model_input.is_last_step,
            previous_hidden_states=model_input.previous_hidden_states,
            sampling_metadata=sampling_metadata,
            is_prompt=is_prompt,
        )

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
            'block_size',
            'block_groups',
            'input_positions',
            'alibi_blocks',
            'window_block_list',
            'window_block_mapping',
            'window_block_usage',
            'window_block_groups',
            'window_attn_bias',
            'use_window_sdpa',
            'sliding_window_right',
        ])
        return attention_metadata

    def create_dummy_multi_modal_seq_group_metadata(self, group_id, img_args,
                                                    sampling_params,
                                                    lora_request):
        assert self.model_is_mrope or self.is_mm_optimized, \
            ("Warmup compatible with Qwen2vl/Gemma3 models")
        if img_args == UNSET_IMG_ARGS:
            # Using the largest bucket
            img_args = self.get_model().vision_buckets.multimodal_buckets[-1]

        if self.model_is_mrope:
            if not hasattr(self.get_model().config, "vision_config"):
                raise ValueError("Expect mrope model to have vision_config")
            vision_config = self.get_model().config.vision_config
            if not hasattr(vision_config, "spatial_merge_size"):
                raise ValueError(
                    "Expect mrope model to have spatial_merge_size")

            spatial_merge_unit = vision_config.spatial_merge_size**2
            num_image_tokens = img_args // spatial_merge_unit
            assert img_args % 8 == 0, (
                f"Expects img_args to be multiples of 8, got: {img_args}")
            image_h = img_args // 8
            image_grid_thw = torch.tensor(
                [[1, image_h, int(img_args / image_h)]])
            pixel_values = torch.randn(
                image_grid_thw[0].prod(),
                1176)  # TODO: figure out the variable name

            assert pixel_values.shape[0] % 64 == 0, (
                f"pixel_values must be sliced in 64 chunks, "
                f"got: {pixel_values.shape}")

            multi_modal_data = {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }
        else:
            s = self.model.model.config.vision_config.image_size
            pixel_values = torch.randn([img_args, 3, s, s])
            num_image_tokens = self.model.model.config.mm_tokens_per_image \
                    * img_args
            multi_modal_data = {
                "pixel_values": pixel_values,
                "num_crops": torch.zeros([img_args], dtype=torch.int32)
            }

        image_token_id = self.get_model().config.image_token_id
        prompt_token_ids = [image_token_id] * num_image_tokens
        prompt_token_ids_array = array('l', prompt_token_ids)  # noqa: F821
        placeholders_by_modality = {
            'image':
            [PlaceholderRange(offset=0, length=len(prompt_token_ids))]
        }
        seq_data = SequenceData.from_seqs(prompt_token_ids)
        seq_data = SequenceData(prompt_token_ids_array)
        multi_modal_data = MultiModalKwargs(multi_modal_data)

        seq_group = SequenceGroupMetadata(
            request_id=str(group_id),
            is_prompt=True,
            seq_data={group_id: seq_data},
            sampling_params=sampling_params,
            block_tables=None,
            lora_request=lora_request[group_id] if lora_request else None,
            multi_modal_data=multi_modal_data,
            multi_modal_placeholders=placeholders_by_modality,
        )
        return seq_group

    def create_dummy_seq_group_metadata(self,
                                        group_id,
                                        seq_len,
                                        is_prompt,
                                        lora_request=None,
                                        img_args=None,
                                        temperature=0,
                                        ctx=0):
        if self.is_pooler:
            sampling_params = None
        else:
            sampling_params = SamplingParams(temperature=temperature)
        num_blocks = math.ceil(seq_len / self.block_size)
        seq_len = max(seq_len, 1)
        computed_block_nums = None
        if is_prompt:
            if self.is_mm_run() and img_args is not None:
                return self.create_dummy_multi_modal_seq_group_metadata(
                    group_id=group_id,
                    img_args=img_args,
                    sampling_params=sampling_params,
                    lora_request=lora_request,
                )
            else:
                input_len = seq_len
                output_len = 0
                block_tables = None
                if ctx:
                    block_tables = {
                        group_id: [_PAD_BLOCK_ID] * ctx * self.block_size
                    }
                    computed_block_nums = ([1] * ctx)
        else:
            input_len = seq_len - 1
            output_len = 1
            block_tables = {group_id: [_PAD_BLOCK_ID] * num_blocks}
            computed_block_nums = ([1] * ctx)
        prompt_token_ids = [0] * input_len
        output_token_ids = [1] * output_len
        prompt_token_ids_array = array('l', prompt_token_ids)  # noqa: F821
        seq_data = SequenceData(prompt_token_ids_array)
        seq_data.output_token_ids = output_token_ids
        return SequenceGroupMetadata(request_id=str(group_id),
                                     is_prompt=(output_len == 0),
                                     seq_data={group_id: seq_data},
                                     sampling_params=sampling_params,
                                     computed_block_nums=computed_block_nums,
                                     block_tables=block_tables,
                                     lora_request=lora_request)

    def is_mm_run(self) -> bool:
        return (self.is_mm_optimized or self.model_is_mrope) and \
            (self.multimodal_buckets is not None)

    def profile_run(self) -> None:
        # Skip profile run on decode instances
        if self.vllm_config.kv_transfer_config is not None and\
            self.vllm_config.kv_transfer_config.is_kv_consumer:
            return

        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers
        bind_kv_cache(
            self.vllm_config.compilation_config.static_forward_context,
            [kv_caches] * self.parallel_config.pipeline_parallel_size)
        max_seq_len = self.bucketing_manager.get_max_prompt_shape()
        max_batch_size = min(self.max_num_seqs,
                             self.max_num_batched_tokens // max_seq_len)
        # Using batch_size 1 is profile multimodal models
        max_batch_size = max_batch_size if self.mm_registry is None else 1

        if self.model_is_mrope or self.is_mm_optimized:
            model = self.get_model()
            self.multimodal_buckets = model.vision_buckets.multimodal_buckets
            logger_msg = "Multimodal bucket : " + str(self.multimodal_buckets)
            logger.info(logger_msg)

        self.warmup_scenario(
            batch_size=max_batch_size,
            seq_len=max_seq_len,
            ctx=0,
            is_prompt=True,
            kv_caches=kv_caches,
            is_pt_profiler_run=False,
            img_args=UNSET_IMG_ARGS if self.is_mm_run() else None,
            is_lora_profile_run=True,
        )

        return

    def _dummy_run(self, max_num_batched_tokens: int) -> None:
        assert max_num_batched_tokens == 1
        self.warmup_scenario(
            batch_size=max_num_batched_tokens,
            seq_len=1,
            ctx=1,
            is_prompt=False,
            kv_caches=None,
            is_pt_profiler_run=False,
            img_args=UNSET_IMG_ARGS if self.is_mm_run() else None,
            is_lora_profile_run=True,
            num_iters=1,
            align_worker=True,
            is_dummy_run=True)
        return

    def _remove_duplicate_submodules(self):
        model = self.get_model()
        if hasattr(model, "model"):
            for layer in self.get_model().model.layers:
                self_attn = layer.self_attn
                # delete attr kv_b_proj in self_attn,
                # as they have been transferred to the MLAImpl.
                if hasattr(self_attn, "mla_attn") and hasattr(
                        self_attn, "kv_b_proj"):
                    delattr(self_attn, "kv_b_proj")

    def _inc_preprocess(self):
        self._remove_duplicate_submodules()

    def warmup_scenario(self,
                        batch_size,
                        seq_len,
                        ctx,
                        is_prompt,
                        kv_caches,
                        is_pt_profiler_run=False,
                        is_lora_profile_run=False,
                        temperature=0,
                        img_args=None,
                        num_iters=3,
                        align_worker=False,
                        is_dummy_run=False) -> None:
        phase = 'prompt' if is_prompt else 'decode'
        use_graphs = is_dummy_run or self._use_graphs(img_args)

        scenario_name = ("warmup_"
                         f"{phase}_"
                         f"bs{batch_size}_"
                         f"seq{seq_len}_"
                         f"ctx{ctx}_"
                         f"multimodal{img_args if img_args else 'F'}_"
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
        times = num_iters if use_graphs or is_pt_profiler_run else 1
        if is_prompt:
            seqs = [
                self.create_dummy_seq_group_metadata(
                    i,
                    seq_len + ctx * self.block_size,
                    is_prompt,
                    lora_request=dummy_lora_requests_per_seq[i]
                    if dummy_lora_requests_per_seq else None,
                    img_args=img_args,
                    temperature=temperature,
                    ctx=ctx) for i in range(batch_size)
            ]
        else:
            blocks = [ctx // batch_size for _ in range(batch_size)]
            blocks[0] += ctx % batch_size
            seqs = [
                self.create_dummy_seq_group_metadata(
                    i,
                    b * self.block_size - 1,
                    is_prompt,
                    lora_request=dummy_lora_requests_per_seq[i]
                    if dummy_lora_requests_per_seq else None,
                    temperature=temperature,
                    ctx=ctx) for i, b in enumerate(blocks)
            ]
        if not is_dummy_run:
            torch.hpu.synchronize()
        profiler = None
        if is_pt_profiler_run and self.is_driver_worker:
            profiler = setup_profiler()
            profiler.start()
        for time_index in range(times):
            inputs = self.prepare_model_input_align_worker(
                seqs, align_worker=align_worker)
            # Chendi: Necessary fix for warmup with TP>1
            if time_index == 0:
                if self.is_driver_worker:
                    broadcast_tensor_dict(
                        {"input_tokens": inputs.input_tokens}, src=0)
                else:
                    broadcast_tensor_dict(src=0)
            if is_prompt or self.is_single_step:
                intermediate_tensors = None
                if not get_pp_group().is_first_rank:
                    intermediate_tensors = \
                        self.model.make_empty_intermediate_tensors(
                            batch_size=batch_size,
                            context_size=seq_len if is_prompt else 1,
                            dtype=self.model_config.dtype,
                            device=self.device)
                self.execute_model(inputs,
                                   kv_caches,
                                   intermediate_tensors=intermediate_tensors,
                                   warmup_mode=True,
                                   ctx_blocks=ctx,
                                   is_dummy_run=is_dummy_run,
                                   is_pt_profiler_run=is_pt_profiler_run)
            else:  # decode with multi-step
                inputs = dataclasses.replace(inputs,
                                             is_first_multi_step=True,
                                             is_last_step=False)
                self.execute_model(inputs,
                                   kv_caches,
                                   warmup_mode=True,
                                   num_steps=2,
                                   seqs=seqs,
                                   ctx_blocks=ctx)
                inputs = dataclasses.replace(inputs,
                                             is_first_multi_step=False,
                                             is_last_step=True)
                self.execute_model(inputs,
                                   kv_caches,
                                   warmup_mode=True,
                                   num_steps=2,
                                   seqs=seqs,
                                   ctx_blocks=ctx)
            if not is_dummy_run:
                torch.hpu.synchronize()
            if profiler:
                profiler.step()
        if profiler:
            profiler.stop()
        self.profiler.end()
        if not is_dummy_run:
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

    def log_warmup(self, phase, i, max_i, batch_size, seq_len, ctx):
        free_mem = format_bytes(
            HabanaMemoryProfiler.current_free_device_memory())
        msg = (f"[Warmup][{phase}][{i+1}/{max_i}] "
               f"batch_size:{batch_size} "
               f"query_len:{seq_len} "
               f"num_blocks:{ctx} "
               f"free_mem:{free_mem}")
        logger.info(msg)

    def log_warmup_multimodal(self, phase, i, max_i, batch_size, seq_len,
                              img_args):
        free_mem = format_bytes(
            HabanaMemoryProfiler.current_free_device_memory())
        dim = "seq_len"
        msg = (f"[Warmup][{phase}][{i+1}/{max_i}] "
               f"batch_size:{batch_size} "
               f"{dim}:{seq_len} "
               f"img_args:{img_args} "
               f"free_mem:{free_mem}")
        logger.info(msg)

    def warmup_graphs(self,
                      buckets,
                      is_prompt,
                      kv_caches,
                      starting_mem=0,
                      total_batch_seq=0.001):
        total_mem = starting_mem
        idx = 0
        num_candidates = len(buckets)
        captured_all = True
        warmed_random_sampler_bs: Set[int] = set()
        for idx, (batch_size, query_len, ctx) in enumerate(reversed(buckets)):
            # Graph memory usage is proportional to seq dimension in a batch
            phase = f"Graph/{'prompt' if is_prompt else 'decode'}"
            if is_prompt:
                seq_len = query_len + ctx * self.block_size
                batch_seq = batch_size * seq_len
            else:
                batch_seq = batch_size
            graphed_bucket = (batch_size, query_len, ctx, is_prompt)
            if graphed_bucket in self.graphed_buckets:
                continue
            self.graphed_buckets.add(graphed_bucket)
            self.log_warmup(phase, idx, num_candidates, batch_size, query_len,
                            ctx)
            with HabanaMemoryProfiler() as mem_prof:
                self.warmup_scenario(
                    batch_size,
                    query_len,
                    ctx,
                    is_prompt,
                    kv_caches,
                    temperature=1.0
                    if batch_size not in warmed_random_sampler_bs else 0,
                )
            warmed_random_sampler_bs.add(batch_size)
            used_mem = align_workers(mem_prof.consumed_device_memory,
                                     torch.distributed.ReduceOp.MAX)
            total_mem += used_mem
            total_batch_seq += batch_seq

        if is_prompt and self.is_mm_run():
            #For multimodal total_batch_seq and total_mem, we store it in the
            #attribute for now.
            mm_outputs = self._warmup_multimodal_graph(
                kv_caches=kv_caches,
                starting_mem=0
                if not hasattr(self, "mm_total_mem") \
                    else self.mm_total_mem, # type: ignore
                total_batch_seq=0.001
                if not hasattr(self, "mm_total_batch_seq") else
                self.mm_total_batch_seq) # type: ignore

            if mm_outputs is not None:
                mm_total_mem, total_mm_batch_seq, mm_captured_all = mm_outputs
                total_mem = total_mem + mm_total_mem
                captured_all = captured_all and mm_captured_all
                self.mm_total_mem = mm_total_mem
                self.mm_total_batch_seq = total_mm_batch_seq

        return total_mem, total_batch_seq, captured_all

    def _warmup_multimodal_graph(self,
                                 kv_caches,
                                 starting_mem=0,
                                 total_batch_seq=0.001):

        total_mem = starting_mem
        idx = 0
        phase = 'Graph/Multimodal'
        num_candidates = len(self.multimodal_buckets)
        captured_all = True

        for idx, img_args in enumerate(self.multimodal_buckets):
            batch_size = 1  # Note: Multimodal buckets do not change with bs
            max_seq_len = self.bucketing_manager.get_max_prompt_shape()
            seq_len = max_seq_len
            batch_seq = 1 * img_args
            graphed_multimodal_bucket = img_args
            if graphed_multimodal_bucket in self.graphed_multimodal_buckets:
                continue
            self.graphed_multimodal_buckets.add(graphed_multimodal_bucket)
            self.log_warmup_multimodal(phase, idx, num_candidates, batch_size,
                                       seq_len, img_args)

            with HabanaMemoryProfiler() as mem_prof:
                self.warmup_scenario(batch_size=batch_size,
                                     seq_len=seq_len,
                                     ctx=0,
                                     is_prompt=True,
                                     kv_caches=kv_caches,
                                     img_args=img_args)

            used_mem = align_workers(mem_prof.consumed_device_memory,
                                     torch.distributed.ReduceOp.MAX)
            total_mem += used_mem
            total_batch_seq += batch_seq

        return total_mem, total_batch_seq, captured_all

    def log_graph_warmup_summary(self, buckets, is_prompt, total_mem):
        num_candidates = len(buckets)
        phase = 'Prompt' if is_prompt else 'Decode'
        graphed = buckets
        if num_candidates == 0:
            num_candidates = 1
        msg = (f'{phase} captured:{len(graphed)} '
               f'({100 * len(graphed) / num_candidates:.1f}%) '
               f'used_mem:{format_bytes(total_mem)}')
        logger.info(msg)
        if "Prompt" in phase and len(self.multimodal_buckets) > 0:
            phase = "Graph/Multimodal"
            num_candidates = len(self.multimodal_buckets)
            mm_graphed = self.graphed_multimodal_buckets
            msg = (f'{phase} captured:{len(mm_graphed)} '
                   f'({100 * len(mm_graphed) / num_candidates:.1f}%) '
                   f'buckets:{sorted(list(mm_graphed))}')
            logger.info(msg)

    @torch.inference_mode()
    def warmup_model(self, kv_caches: List[torch.Tensor]) -> None:
        prompt_buckets = len(self.bucketing_manager.prompt_buckets)
        if not self.is_pooler:
            decode_buckets = len(self.bucketing_manager.decode_buckets)
        else:
            # When pooling we're not using decode phase
            decode_buckets = 0

        if profile := os.environ.get('VLLM_PT_PROFILE', None):
            phase, bs, seq_len, graph = profile.split('_')
            is_prompt = phase == 'prompt'
            ctx = 0
            if not is_prompt:
                ctx = int(seq_len)
                seq_len = '1'
            cfg = (int(bs), int(seq_len), ctx, is_prompt)
            graphs = graph == 't'
            if graphs:
                self.graphed_buckets.add(cfg)
            if self.is_mm_run():
                img_args = (int(seq_len) //
                            self.model.model.config.mm_tokens_per_image
                            if self.is_mm_optimized else int(seq_len))
            self.warmup_scenario(
                int(bs),
                int(seq_len),
                ctx,
                is_prompt,
                kv_caches,
                is_pt_profiler_run=True,
                img_args=img_args if self.is_mm_run() else None)
            raise AssertionError("Finished profiling")
        if not htorch.utils.internal.is_lazy() and not self.enforce_eager:
            multiplier = 3 if os.getenv('VLLM_REGIONAL_COMPILATION',
                                        'true').lower() == 'true' else 1
            cache_size_limit = 1 + multiplier * (prompt_buckets +
                                                 decode_buckets)
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
            if not self.enforce_eager:
                if not self.is_pooler:
                    assert self.mem_margin is not None, \
                        ("HabanaWorker.determine_num_available_blocks needs "
                        "to be called before warming up the model.")

                free_mem = HabanaMemoryProfiler.current_free_device_memory()
                graph_free_mem = free_mem - self.mem_margin
                graph_free_mem = align_workers(graph_free_mem,
                                               torch.distributed.ReduceOp.MIN)

                if not self.is_pooler:
                    mem_post_prompt, prompt_batch_seq, prompt_captured_all = \
                        self.warmup_graphs(
                        self.bucketing_manager.prompt_buckets,
                        True, kv_caches)

                    mem_post_decode, decode_batch_seq, decode_captured_all = \
                        self.warmup_graphs(
                        self.bucketing_manager.decode_buckets,
                        False, kv_caches)
                else:
                    msg = (f"Using {format_bytes(graph_free_mem)}"
                           f"/{format_bytes(free_mem)} "
                           "of free device memory for HPUGraphs")
                    logger.info(msg)

                    mem_post_prompt, prompt_batch_seq, prompt_captured_all = \
                        self.warmup_graphs(
                        self.bucketing_manager.prompt_buckets,
                        True, kv_caches)
                    if mem_post_prompt < graph_free_mem \
                        and not prompt_captured_all:
                        mem_post_prompt, _, prompt_captured_all = (
                            self.warmup_graphs(
                                self.bucketing_manager.prompt_buckets, True,
                                kv_caches))

                self.log_graph_warmup_summary(
                    self.bucketing_manager.prompt_buckets, True,
                    mem_post_prompt)
                if not self.is_pooler:
                    self.log_graph_warmup_summary(
                        self.bucketing_manager.decode_buckets, False,
                        mem_post_decode)

        end_time = time.perf_counter()
        end_mem = HabanaMemoryProfiler.current_device_memory_usage()
        if os.getenv('VLLM_FULL_WARMUP',
                     'false').strip().lower() in ("1", "true"):
            # Since the model is warmed up for all possible tensor sizes,
            # Dynamo can skip checking the guards
            torch.compiler.set_stance(skip_guard_eval_unsafe=True)
        elapsed_time = end_time - start_time
        msg = (
            f"Warmup finished in {elapsed_time:.0f} secs, "
            f"allocated {format_bytes(end_mem - start_mem)} of device memory")
        logger.info(msg)
        self.profiler.end()

    def finish_measurements(self):
        from neural_compressor.torch.quantization import finalize_calibration
        finalize_calibration(self.model.model)

    def shutdown_inc(self,
                     suppress=suppress,
                     finalize_calibration=finalize_calibration):
        global shutdown_inc_called
        if shutdown_inc_called:
            return
        shutdown_inc_called = True
        can_finalize_inc = False
        with suppress(AttributeError):
            can_finalize_inc = (self._is_quant_with_inc()
                                and (self.model.model is not None)
                                and self.inc_initialized_successfully and
                                not getattr(self, "_is_inc_finalized", False))
        if can_finalize_inc:
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

    def need_recv_kv(self, model_input, kv_caches, warmup_mode) -> bool:
        """Check if we need to receive kv-cache from the other worker.
        We need to receive KV when
            1. current vLLM instance is KV cache consumer/decode vLLM instance
            2. this batch is not a profiling run
            3. this batch is a prefill run
        Args:
            model_input: input to the model executable
            kv_caches: vLLM's paged memory
        """
        if warmup_mode:
            return False

        if self.vllm_config.kv_transfer_config is None:
            return False

        is_prefill_run = model_input.attn_metadata.is_prompt

        # check if the current run is profiling
        is_profile_run = kv_caches is None or kv_caches[0] is None or (
            kv_caches[0][0].numel() == 0)
        # check if the current run is prefill
        return self.vllm_config.kv_transfer_config.is_kv_consumer and (
            not is_profile_run) and is_prefill_run

    def need_send_kv(self, model_input, kv_caches, warmup_mode) -> bool:
        """Check if we need to send kv-cache to the other worker.
        We need to send KV when
            1. current vLLM instance is KV cache producer/prefill vLLM instance
            2. this batch is not a profiling run or a warmup run.
            3. this batch is a prefill run
        Args:
            model_input: input to the model executable
            kv_caches: vLLM's paged memory
        """
        if warmup_mode:
            return False

        if self.vllm_config.kv_transfer_config is None:
            return False

        is_prefill_run = model_input.attn_metadata.is_prompt

        # check if the current run is profiling
        is_profile_run = kv_caches is None or kv_caches[0] is None or (
            kv_caches[0][0].numel() == 0)
        # check if the current run is prefill

        return self.vllm_config.kv_transfer_config.is_kv_producer and (
            not is_profile_run) and is_prefill_run

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
        return self.prepare_model_input_align_worker(seq_group_metadata_list,
                                                     virtual_engine,
                                                     finished_requests_ids,
                                                     False)

    def finish_measurements(self):
        from neural_compressor.torch.quantization import finalize_calibration
        finalize_calibration(self.model.model)

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

    def _check_config(self, batch_size, seq_len, ctx, attn_metadata,
                      warmup_mode):
        is_prefix_caching = self.vllm_config.cache_config.enable_prefix_caching
        cfg: Optional[tuple] = None
        assert cfg is None, "Configs changed between 2D and 3D"
        if is_prefix_caching:
            phase = self._phase(attn_metadata)
            num_blocks = self._num_blocks(attn_metadata)
            cfg = (batch_size, seq_len, num_blocks, phase)
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

    def _get_img_args_from_model_input(self, model_input):
        if (not self.model_is_mrope and not self.is_mm_optimized) or \
            not model_input.multi_modal_kwargs or \
            'pixel_values' not in model_input.multi_modal_kwargs:
            return None
        if self.model_is_mrope:
            pixel_values_list = model_input.multi_modal_kwargs['pixel_values']
            if isinstance(pixel_values_list, torch.Tensor):
                pixel_values_list = [pixel_values_list]
            assert isinstance(pixel_values_list, list)
            model = self.get_model()
            max_bucket_size = 0
            for pixel_values in pixel_values_list:
                assert isinstance(pixel_values, torch.Tensor)
                curr_num_pixels = pixel_values.shape[-2]
                bucket_size = model.vision_buckets.get_multimodal_bucket(
                    curr_num_pixels)
                max_bucket_size = max(max_bucket_size, bucket_size)
        else:
            max_bucket_size = self.get_model(
            ).vision_buckets.multimodal_buckets[-1]
        return max_bucket_size

    def _pad_to_max_num_seqs(self, tensor, value):
        padding_needed = self.max_num_seqs - tensor.size(0)
        if padding_needed > 0:
            padding = torch.full((padding_needed, *tensor.shape[1:]),
                                 value,
                                 device=tensor.device,
                                 dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding])
        return tensor

    def has_logits_processors(self, sampling_metadata):
        return any(seq_group.sampling_params.logits_processors
                   for seq_group in sampling_metadata.seq_groups)

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
        ctx_blocks: int = 1,
        is_dummy_run: bool = False,
        is_pt_profiler_run: bool = False,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        self.has_patched_prev_output = False
        use_delayed_sampling = self.use_delayed_sampling and not warmup_mode
        assert not (use_delayed_sampling and num_steps != 1), \
            'Delayed sampling is not compatible with MSS!'
        assert not (use_delayed_sampling and
            self.parallel_config.pipeline_parallel_size != 1), \
            'Delayed sampling is not compatible with Pipeline Parallelism!'
        assert not (use_delayed_sampling and self.spec_decode_enabled), \
            'Delayed sampling is not compatible with speculative decoding!'
        assert model_input.input_tokens is not None
        output = None
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
                padding = self.cached_step_outputs[i].token_ids.size(0) - len(
                    target_indices)
                target_indices.extend([-1] * padding)
                target_indices = torch.tensor(
                    target_indices,
                    device=model_input.input_tokens.device,
                    dtype=model_input.input_tokens.dtype)
                model_input.input_tokens.index_copy_(
                    0, target_indices, self.cached_step_outputs[i].token_ids)
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
            phase = 'prompt' if is_prompt else 'decode'
            if phase == 'decode':
                if not warmup_mode:
                    ctx_blocks = seq_len
                seq_len = 1
            img_args = self._get_img_args_from_model_input(model_input)
            use_graphs = self._use_graphs(img_args=img_args)
            self._check_config(batch_size, seq_len, ctx_blocks, attn_metadata,
                               warmup_mode)
            lora_mask: torch.Tensor = None
            lora_logits_mask: torch.Tensor = None
            if self.lora_config:
                assert model_input.lora_ids is not None
                lora_mask, lora_logits_mask = self.create_lora_mask(
                    input_tokens, model_input.lora_ids,
                    attn_metadata.is_prompt)
            if model_input.multi_modal_kwargs is not None \
                and 'embed_is_patch' in model_input.multi_modal_kwargs:

                def fix_embed_is_patch(embed_is_patch):
                    if isinstance(embed_is_patch, torch.Tensor):
                        if embed_is_patch.dim() == 3:
                            result = []
                            if embed_is_patch.size(1) > 1:
                                embed_is_patch = embed_is_patch.transpose(0, 1)
                            for i in range(embed_is_patch.size(0)):
                                result.append(embed_is_patch[i])
                            return result
                        elif embed_is_patch.dim() == 2:
                            result = []
                            result.append(embed_is_patch)
                            return result
                    elif isinstance(embed_is_patch, (list, tuple)):
                        # Apply only once per item, avoid repeated recursion
                        result = []
                        for item in embed_is_patch:
                            fixed = fix_embed_is_patch(item)
                            if isinstance(fixed, list):
                                result.extend(fixed)
                            else:
                                result.append(fixed)
                        return result
                    else:
                        return None

                model_input.multi_modal_kwargs[
                    'embed_is_patch'] = fix_embed_is_patch(
                        model_input.multi_modal_kwargs['embed_is_patch'])

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
                # HPU will pad up to block_size,
                # pad previous_hidden_states as well
                previous_hidden_states = previous_hidden_states.unsqueeze(
                    1).expand(-1, input_tokens.shape[-1], -1)
                batch_size_padding = batch_size - previous_hidden_states.shape[
                    0]
                if batch_size_padding > 0:
                    dummy_previous_hidden_states = torch.zeros(
                        batch_size_padding,
                        *previous_hidden_states.shape[1:],
                        dtype=previous_hidden_states.dtype,
                        device=previous_hidden_states.device)
                    previous_hidden_states = torch.cat(
                        [previous_hidden_states, dummy_previous_hidden_states],
                        dim=0)
                execute_model_kwargs.update(
                    {"previous_hidden_states": previous_hidden_states})
            if htorch.utils.internal.is_lazy():
                execute_model_kwargs.update(
                    {"bypass_hpu_graphs": not use_graphs})

            htorch.core.mark_step()
            if self.is_driver_worker:
                model_event_name = ("model_"
                                    f"{phase}_"
                                    f"bs{batch_size}_"
                                    f"seq{seq_len}_"
                                    f"ctx{ctx_blocks}_"
                                    f"graphs{'T' if use_graphs else 'F'}")
            else:
                model_event_name = 'model_executable'
            if num_steps > 1 or use_delayed_sampling:
                # in case of multi-step scheduling
                # we only want to pythonize in the last step
                sampling_metadata.skip_sampler_cpu_output = True
                self.sampler.include_gpu_probs_tensor = True
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
                # Receive KV cache in distributed KV cache transfer setting
                # In disagg prefill setting, it will also recv hidden states
                # and bypass model forwarding. In KV cache database setting,
                # it will change the model input so that we can skip prefilling
                # on tokens that successfully received KV caches
                # NOTE: The receive operation is blocking
                bypass_model_exec = False
                if self.need_recv_kv(model_input, kv_caches, warmup_mode):
                    attn_metadata = self.model.forward_update_meta_only(
                        **execute_model_kwargs,
                        selected_token_indices=sampling_metadata.
                        selected_token_indices)
                    hidden_states, bypass_model_exec, model_input = \
                    get_kv_transfer_group().recv_kv_caches_and_hidden_states_hpu(
                        # model is used to know which layer the current worker
                        # is working on, so that we can receive KV for
                        # only those layers.
                        self.get_model(),
                        model_input,
                        attn_metadata,
                        kv_caches=kv_caches
                    )
                profiler_args = {
                    'real_seq_len': model_input.seq_lens,
                    'real_batch_size': real_batch_size
                }

                #Need to set the window_slide mask at this point to decide
                if is_prompt:
                    attn_metadata = self.model._update_use_window_sdpa(
                        execute_model_kwargs['attn_metadata'], seq_len,
                        bool(model_input.multi_modal_kwargs and \
                       'pixel_values' in model_input.multi_modal_kwargs))
                    execute_model_kwargs['attn_metadata'] = attn_metadata

                if not bypass_model_exec:
                    if self.model_is_mrope or self.is_mm_optimized:
                        if 'pixel_values' in execute_model_kwargs and \
                                self.is_mm_optimized:
                            if warmup_mode and not is_pt_profiler_run:
                                bypass_model_exec = True
                            execute_model_kwargs[
                                    'graphed_multimodal_buckets'] = \
                                list(self.graphed_multimodal_buckets)
                            # set is unhasable and causes friction with
                            # hpu graphs, hence turning it to a list
                        execute_model_kwargs = \
                            self.model.compute_input_embeddings_for_mrope_mm_optimized(
                                **execute_model_kwargs
                            )
                        if warmup_mode and bypass_model_exec:
                            return []

                    with self.profiler.record_event('internal',
                                                    model_event_name,
                                                    args=profiler_args):
                        hidden_states = self.model.forward(
                            **execute_model_kwargs,
                            selected_token_indices=sampling_metadata.
                            selected_token_indices)
                        if warmup_mode and not is_dummy_run:
                            torch.hpu.synchronize()
                            import torch.distributed as dist
                            if dist.is_initialized():
                                get_tp_group().barrier()
                else:
                    logger.debug("Bypassing model execution")

                # Sending KV cache in distributed KV cache transfer setting
                # TODO: update send operation to blocking one.
                if self.need_send_kv(model_input, kv_caches, warmup_mode):
                    get_kv_transfer_group(
                    ).send_kv_caches_and_hidden_states_hpu(
                        # model_executable is used to know which layer the
                        # current worker is working on, so that we can send KV
                        # for only those layers.
                        self.get_model(),
                        model_input,
                        kv_caches,
                        hidden_states,
                    )

                if self.lora_config:
                    LoraMask.setLoraMask(
                        lora_logits_mask.index_select(
                            0, sampling_metadata.selected_token_indices))

                if is_dummy_run:
                    fake_output = self._delayed_sampler_outputs(model_input)
                    return [fake_output]

                if not get_pp_group().is_last_rank:
                    return hidden_states

                # In case there are any logits processors pending
                # we need to sync with host earlier
                if use_delayed_sampling \
                   and self.is_driver_worker:
                    self._patch_prev_output()

                if (use_delayed_sampling and self.is_driver_worker
                        and self.has_logits_processors(sampling_metadata)):
                    # when use_delayed_sampling if the computation
                    # of logits depends on the sampled results
                    # we obtain the actual sampled results in advance
                    self._patch_prev_output()
                # Compute the logits.
                with self.profiler.record_event('internal',
                                                ('compute_logits_'
                                                 f'{phase}_bs'
                                                 f'{batch_size}_'
                                                 f'seq{seq_len}_ctx'
                                                 f'{ctx_blocks}'),
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
                elif model_input.async_callback is not None:
                    model_input.async_callback()

                with self.profiler.record_event('internal',
                                                ('sample_'
                                                 f'{phase}_'
                                                 f'bs{batch_size}_'
                                                 f'seq{seq_len}_'
                                                 f'ctx{ctx_blocks}'),
                                                args=profiler_args):
                    output = self.sampler(
                        logits=logits,
                        sampling_metadata=sampling_metadata,
                    )
                    if num_steps > 1:
                        output = output.sampled_token_ids
                        self.cached_step_outputs.append(
                            CachedStepOutput(output))
                    if use_delayed_sampling and self.is_driver_worker:
                        token_ids = self._pad_to_max_num_seqs(
                            output.sampled_token_ids, DUMMY_TOKEN_ID)
                        self.cached_step_outputs.append(
                            CachedStepOutput(
                                token_ids, output.logprobs,
                                output.deferred_sample_results_args,
                                sampling_metadata, is_prompt))
                        self.cached_step_inputs.append(model_input)
                htorch.core.mark_step()
                if use_delayed_sampling \
                   and model_input.async_callback is not None:
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
                    prompt_batch_idx=0,
                    is_prompt=is_prompt)
                self.profiler.record_counter(self.event_start, counters)
            if num_steps == 1:
                if self.spec_decode_enabled and isinstance(
                        output, SamplerOutput):
                    output.sampled_token_ids = output.sampled_token_ids[:
                                                                        real_batch_size]
                    output.sampled_token_probs = output.sampled_token_probs[:
                                                                            real_batch_size]
                    output.logprobs = output.logprobs[:real_batch_size]
                if self.return_hidden_states and isinstance(
                        output, SamplerOutput):
                    # we only need to pass hidden states of most recent token
                    assert model_input.sampling_metadata is not None
                    hidden_states = hidden_states[:real_batch_size]
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
            next_token_ids = self.cached_step_outputs.pop(0).token_ids
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

    def __del__(self):
        self.shutdown_inc()

    def _patch_prev_output(self):
        if self.has_patched_prev_output:
            return
        assert len(self.cached_step_inputs) == len(self.cached_step_outputs), \
            f'''Inputs and outputs are out of sync!
            {len(self.cached_step_inputs)} vs {len(self.cached_step_outputs)}'''
        if len(self.cached_step_inputs) == 0:
            return
        model_input = self.cached_step_inputs.pop(0)
        model_output = self.cached_step_outputs.pop(0)

        assert model_output.sampling_metadata is not None, \
            'Sampling metadata is required to patch the output!'
        seq_groups = model_output.sampling_metadata.seq_groups
        logprobs_required = any(seq_group.sampling_params.logprobs is not None
                                for seq_group in seq_groups)
        prompt_logprobs_required = any(
            seq_group.sampling_params.prompt_logprobs is not None
            for seq_group in seq_groups)

        if model_output.is_prompt and prompt_logprobs_required:
            sample_idx_tensor = torch.tensor(
                [sdx for sg in seq_groups for sdx in sg.sample_indices])

            sampled_tokens = model_output.token_ids[sample_idx_tensor, :]
            delayed_tokens = sampled_tokens.cpu().squeeze(-1).tolist()
        else:
            delayed_tokens = model_output.token_ids.cpu().squeeze(-1).tolist()

        ctx = model_input.async_callback.keywords["ctx"]  # type: ignore
        # If there's no output to patch with, which is usually the case when
        # we're starting a new request after all requests are completed.
        if len(ctx.output_queue) == 0:
            return
        assert len(
            ctx.output_queue) == 1, 'There should be exactly 1 output waiting!'
        output_data = ctx.output_queue[0]
        assert len(output_data.outputs) == 1
        for fake_out, real_out in zip(output_data.outputs[0], delayed_tokens):
            fake_out.samples[0].output_token = real_out
        for sg, real_out in zip(output_data.seq_group_metadata_list,
                                delayed_tokens):
            assert len(sg.seq_data) == 1
            seq_data = list(sg.seq_data.values())[0]
            # This is a hack. Assigning output_token_ids triggers
            # a cache recomputation and we only need to update the last token
            seq_data.output_token_ids_array[-1] = real_out
            seq_data._cached_all_token_ids[-1] = real_out
        delayed_logprobs = None
        delayed_prompt_logprobs = None
        if logprobs_required or prompt_logprobs_required:
            # We are one step ahead, so prompt is already marked as a computed.
            # We need to reset the computed tokens count to 0,
            # so that we can recompute the prompt logprobs.
            computed_tokens = []
            if model_output.is_prompt:
                for seq_group in seq_groups:
                    seq_ids = seq_group.seq_ids
                    assert len(seq_ids) == 1  # prompt has only 1 seq id.
                    seq_data = seq_group.seq_data[seq_ids[0]]
                    computed_tokens.append(seq_data.get_num_computed_tokens())
                    seq_data._num_computed_tokens = 0
            sampling_results = get_pythonized_sample_results(
                model_output.deffered_sample_results)
            delayed_prompt_logprobs, delayed_logprobs = get_logprobs(
                model_output.logprobs, model_output.sampling_metadata,
                sampling_results)

            # Reset the computed tokens count to the original value.
            if model_output.is_prompt:
                for seq_group in seq_groups:
                    seq_ids = seq_group.seq_ids
                    seq_data = seq_group.seq_data[seq_ids[0]]
                    seq_data.update_num_computed_tokens(computed_tokens.pop(0))

        # Another hack. We need to pass the logprobs to the output data,
        # which are part of scheduler output.
        if logprobs_required and delayed_logprobs is not None:
            for sg, real_logprobs in zip(
                    output_data.scheduler_outputs.scheduled_seq_groups,
                    delayed_logprobs):
                assert len(sg.seq_group.seqs) == 1
                assert len(real_logprobs) == 1
                sg.seq_group.first_seq.output_logprobs[-1] = real_logprobs[0]

        # If prompt logprobs are available, we need to patch them
        # as well.
        if prompt_logprobs_required and delayed_prompt_logprobs is not None:
            seq_groups = output_data.scheduler_outputs.scheduled_seq_groups
            assert len(seq_groups) == len(delayed_prompt_logprobs), \
                f'''Output data has {len(seq_groups)} seq groups, but prompt
                logprobs has {len(delayed_prompt_logprobs)} entries!'''
            for sg, real_logprobs in zip(seq_groups, delayed_prompt_logprobs):
                if real_logprobs is not None:
                    # Prepending None just like in vllm.engine.output_processor
                    # .single_step.single_step_process_prompt_logprob, but
                    # hence we are not going through async output processor
                    # with data from prompt in delayed sampling scenario we
                    # need to do that manually.
                    sg.seq_group.prompt_logprobs = [None] + real_logprobs
        self.has_patched_prev_output = True
