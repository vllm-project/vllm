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
import operator
import os
import time
from array import array
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple,
                    Optional, Set, Tuple, Type, TypeVar, Union)

import habana_frameworks.torch as htorch
import habana_frameworks.torch.internal.bridge_config as bc
import torch
import torch.nn as nn
from vllm_hpu_extension.ops import LoraMask as LoraMask
from vllm_hpu_extension.ops import batch2block, block2batch
from vllm_hpu_extension.profiler import (HabanaHighLevelProfiler,
                                         HabanaMemoryProfiler, format_bytes)

import vllm.envs as envs
from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.config import DeviceConfig, VllmConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.forward_context import set_forward_context
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
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalKwargs)
from vllm.sampling_params import SamplingParams
from vllm.sequence import (IntermediateTensors, SequenceData,
                           SequenceGroupMetadata)
from vllm.utils import (bind_kv_cache, is_pin_memory_available,
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


class Singleton(type):
    _instances: Dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass
class HPUBucketingGlobalState(metaclass=Singleton):
    prompt_bs_bucket_cfg: Tuple[int, int, int] = field(init=False)
    decode_bs_bucket_cfg: Tuple[int, int, int] = field(init=False)
    prompt_seq_bucket_cfg: Tuple[int, int, int] = field(init=False)
    decode_block_bucket_cfg: Tuple[int, int, int] = field(init=False)
    prompt_buckets: List[Tuple[int, int]] = field(init=False)
    decode_buckets: List[Tuple[int, int]] = field(init=False)


def subtuple(obj: object,
             typename: str,
             to_copy: List[str],
             to_override: Optional[Dict[str, object]] = None):
    if obj is None:
        return None
    if to_override is None:
        to_override = {}
    fields = set(to_copy) | set(to_override.keys())
    values = {f: to_override.get(f, getattr(obj, f)) for f in fields}
    if typename not in _TYPE_CACHE:
        _TYPE_CACHE[typename] = collections.namedtuple(typename,
                                                       ' '.join(fields))
    return _TYPE_CACHE[typename](**values)


def read_bucket_settings(phase: str, dim: str, **defaults):
    """Read bucketing configuration from env variables.

    phase is either 'prompt' or 'decode'
    dim is either 'bs', 'seq' or 'block'
    param is either 'min', 'step' or 'max'
    example env variable: VLLM_DECODE_BS_BUCKET_STEP=128
    """
    params = ['min', 'step', 'max']
    env_vars = [f'VLLM_{phase}_{dim}_BUCKET_{p}'.upper() for p in params]
    default_values = [defaults[p] for p in params]
    values = [
        int(os.environ.get(e, d)) for e, d in zip(env_vars, default_values)
    ]
    for e, v, d in zip(env_vars, values, default_values):
        logger.info('%s=%s (default:%s)', e, v, d)
    return values


def warmup_range(config: Tuple[int, int, int]):
    """Generate a warmup range.

    Start from bmin and multiply by 2 until you reach bstep.
    Then, increase the values in the range by the value of bstep until you 
    reach bmax.

    Example:
    bmin = 2, bstep = 32, bmax = 64
    => ramp_up = (2, 4, 8, 16)
    => stable = (32, 64)
    => return ramp_up + stable => (2, 4, 8, 16, 32, 64)
    """
    bmin, bstep, bmax = config
    assert bmin <= bmax, ("Min. batch size cannot be greater than max. "
                          "batch size. If you want to skip warmup, "
                          "set VLLM_SKIP_WARMUP=true")
    base = itertools.repeat(2)
    ramp_up_acc = itertools.accumulate(base, func=operator.mul, initial=bmin)
    ramp_up_tw = itertools.takewhile(lambda x: x < bstep and x <= bmax, \
        ramp_up_acc)
    stable = range(bstep, bmax + 1, bstep)
    buckets = list(ramp_up_tw) + list(stable)
    return list(filter(lambda bucket: bucket >= bmin, buckets))


def generate_prompt_buckets(bs_bucket_config,
                            seq_bucket_config,
                            max_num_batched_tokens=None):
    buckets = list(
        itertools.product(warmup_range(bs_bucket_config),
                          warmup_range(seq_bucket_config)))
    if len(buckets) == 0:
        msg = ("No buckets could be captured with following config "
               f"(min, step, max_warmup): "
               f"bs:{bs_bucket_config}, "
               f"seq:{seq_bucket_config}")
        raise ValueError(msg)

    filtered_buckets = buckets
    if max_num_batched_tokens is not None:
        # Remove buckets exceeding batch token budget
        filtered_buckets = list(
            filter(
                lambda bucket: bucket[0] * bucket[1] <= max_num_batched_tokens,
                buckets))

        if len(filtered_buckets) == 0:
            # we can handle this if we ignore max_num_batched_tokens
            min_bucket_bs, min_bucket_seq = min(buckets,
                                                key=lambda b: (b[0] * b[1]))
            min_reqd_budget = min_bucket_bs * min_bucket_seq
            msg = (
                "The current bucketing configuration "
                f"(min, step, max_warmup): "
                f"bs:{bs_bucket_config}, "
                f"seq:{seq_bucket_config} cannot be used with specified "
                f"max_num_batched_tokens ({max_num_batched_tokens}), as the "
                f"smallest bucket ({min_reqd_budget}) would exceed token "
                "budget. Please increase max_num_batched_tokens or decrease "
                "bucket minimum Ignoring max_num_batched_tokens at risk of "
                "out-of-memory errors.")
            logger.error(msg)
            return list(
                sorted(buckets, key=lambda b: (b[0] * b[1], b[1], b[0]))), []

    captured_buckets = list(
        sorted(filtered_buckets, key=lambda b: (b[0] * b[1], b[1], b[0])))
    omitted_buckets = list(
        sorted([x for x in buckets if x not in filtered_buckets]))
    return captured_buckets, omitted_buckets


def generate_decode_buckets(bs_bucket_config, blocks_bucket_config,
                            max_blocks):
    buckets = []
    bs_buckets = warmup_range(bs_bucket_config)
    block_buckets = warmup_range(blocks_bucket_config)
    bmin, bstep, bmax = blocks_bucket_config
    last_bucket = round_up(max_blocks, bstep)
    for bs in bs_buckets:
        for blocks in block_buckets:
            if blocks < bs:
                continue
            if blocks > last_bucket:
                break
            buckets.append((bs, blocks))
    return list(sorted(buckets, key=lambda b: (b[0] * b[1], b[1], b[0])))


def next_pow2(value: int, base: int):
    res = base
    while value > 1:
        value = (value + 1) // 2
        res *= 2
    return res


def round_up(value: int, k: int):
    return (value + k - 1) // k * k


def find_bucket(value: int, config: Tuple[int, int, int]):
    bmin, bstep, _ = config
    next_step = round_up(value, bstep)
    next_pow = next_pow2(value, bmin)
    return max(bmin, min(next_step, next_pow))


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
    DEVICE = 'hpu'
    activities = [torch.profiler.ProfilerActivity.CPU]
    activities.extend([torch.profiler.ProfilerActivity.HPU] if DEVICE ==
                      'hpu' else [])
    #from habana_frameworks.torch.activity_profiler import DebugActivity
    #debug_activities=[DebugActivity.BRIDGE_FUNCTION_CALLS]

    profiler = torch.profiler.profile(
        schedule=schedule,
        activities=activities,
        #debug_activities=debug_activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('.',
                                                                use_gzip=True),
        record_shapes=False,
        with_stack=True)
    return profiler


def pad_list(input, k, v):
    input_len = len(input)
    target_len = round_up(input_len, k)
    padding = target_len - input_len
    return input + [v] * padding


def gather_list(input, indices, v):
    return [input[i] if i is not None else v for i in indices]


def flatten(in_list):
    return list(itertools.chain(*in_list))


def precompute_indices_and_offsets(block_size, slot_mapping, is_prompt):
    slot_mapping = slot_mapping.flatten()
    indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    if is_prompt:
        indices = indices.unflatten(0, (-1, block_size))[:, 0]
        offsets = None
    else:
        offsets = torch.fmod(slot_mapping, block_size)
    return indices, offsets


def modify_decoder_layer(module: torch.nn.Module, suffix="DecoderLayer"):
    if module.__class__.__name__.endswith(suffix):

        def forward_hook(module, args, output):
            htorch.core.mark_step()
            return output

        module.register_forward_hook(forward_hook)

    for child_name, child_module in module.named_children():
        modify_decoder_layer(child_module)


class HpuModelAdapter:

    def __init__(self, model, vllm_config):
        self.model = model
        self.prefill_use_fusedsdpa = os.getenv('VLLM_PROMPT_USE_FUSEDSDPA',
                                               '0').lower() in ['1', 'true']
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.dtype = vllm_config.model_config.dtype
        enforce_eager = vllm_config.model_config.enforce_eager

        if not htorch.utils.internal.is_lazy() and not enforce_eager:
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
        prefill_metadata = attn_metadata
        if prefill_metadata is None or self.prefill_use_fusedsdpa:
            return attn_metadata

        seq_lens_t = prefill_metadata.seq_lens_tensor
        len_mask = (torch.arange(0, seq_len, device=device,
                                 dtype=torch.int32).view(1, seq_len).ge(
                                     seq_lens_t.unsqueeze(-1)).view(
                                         batch_size, 1, 1, seq_len))
        causal_mask = torch.triu(torch.ones((batch_size, 1, seq_len, seq_len),
                                            device=device,
                                            dtype=torch.bool),
                                 diagonal=1)
        mask = causal_mask.logical_or(len_mask)
        attn_bias = (torch.zeros_like(mask, dtype=dtype).masked_fill_(
            mask, -math.inf))
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
        block_mapping = torch.nn.functional.one_hot(metadata.block_groups,
                                                    num_classes=batch_size)
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

    def _update_metadata(self, attn_metadata, batch_size, seq_len, device,
                         dtype):
        if attn_metadata.is_prompt:
            meta = attn_metadata
            attn_metadata = self._set_attn_bias(meta, batch_size, seq_len,
                                                device, dtype)
        else:
            meta = attn_metadata
            attn_metadata = self._set_block_mapping(meta, batch_size, device,
                                                    dtype)
            attn_metadata = self._set_block_scales(attn_metadata, device)
        return attn_metadata

    def forward(self, *args, **kwargs):
        kwargs = kwargs.copy()
        selected_token_indices = kwargs.pop('selected_token_indices')
        if 'warmup_mode' in kwargs:
            kwargs.pop('warmup_mode')
        virtual_engine = 0
        if 'virtual_engine' in kwargs:
            virtual_engine = kwargs.pop('virtual_engine')
        input_ids = kwargs['input_ids']
        attn_metadata = self._update_metadata(kwargs.pop('attn_metadata'),
                                              input_ids.size(0),
                                              input_ids.size(1),
                                              input_ids.device, self.dtype)
        LoraMask.setLoraMask(kwargs.pop('lora_mask'))
        with set_forward_context(attn_metadata, self.vllm_config,
                                 virtual_engine):
            hidden_states = self.model(*args, **kwargs)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            hidden_states = hidden_states.index_select(0,
                                                       selected_token_indices)
        return hidden_states

    def compute_logits(self, *args, **kwargs):
        return self.model.compute_logits(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)


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
        is_driver_worker: bool = False,
        return_hidden_states: bool = False,
    ):
        ModelRunnerBase.__init__(self, vllm_config=vllm_config)
        self.is_driver_worker = is_driver_worker
        self.return_hidden_states = return_hidden_states

        self.sliding_window = (self.model_config.get_sliding_window()
                               if self.model_config is not None else None)
        self.device_config = (self.device_config if self.device_config
                              is not None else DeviceConfig())
        self.device = self.device_config.device
        self.enforce_eager = self.model_config.enforce_eager
        self.max_num_seqs = self.scheduler_config.max_num_seqs
        # NOTE(kzawora): Change that to scheduler_config.max_num_prefill_seqs
        # once padding-aware scheduling gets merged
        self.max_num_prefill_seqs = 64
        self.max_model_len = self.scheduler_config.max_model_len
        self.max_num_batched_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.block_size = self.cache_config.block_size

        self.pin_memory = is_pin_memory_available()
        self.kv_cache_dtype = self.cache_config.cache_dtype

        self.attn_backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
        )

        # Lazy initialization
        self.lora_manager: LRUCacheWorkerLoRAManager = None
        self.model: torch.nn.Module = None
        self.inc_initialized_successfully = False

        # Profiler stats
        self.profiler = HabanaHighLevelProfiler()
        self.profiler_counter_helper = HabanaProfilerCounterHelper()
        self.seen_configs: set = set()
        self._mem_margin: Optional[int] = None
        self.bucketing_global_state = HPUBucketingGlobalState()
        self._setup_buckets()
        self._set_gc_threshold()
        self.use_contiguous_pa = envs.VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH

    def _set_gc_threshold(self) -> None:
        # Read https://docs.python.org/3/library/gc.html#gc.set_threshold
        # for comprehensive description of gc generations.
        # We can either use VLLM_GC_THR_GEN[0-2] (this has higher priority)
        # to set particular generation threshold or use simpler
        # VLLM_GC_THR_MULTIPLIER to multiply default values.
        default_gc_thrs = list(gc.get_threshold())
        requested_gc_thrs = [0] * len(default_gc_thrs)
        for i in range(len(default_gc_thrs)):
            requested_gc_thrs[i] = int(
                os.environ.get(f'VLLM_GC_THR_GEN{i}', default_gc_thrs[i]))
        if requested_gc_thrs == default_gc_thrs:
            gc_thr_multiplier = int(os.environ.get('VLLM_GC_THR_MULTIPLIER',
                                                   2))
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
            else:
                self.model = self.model.to("hpu")
                htcore.mark_step()
            modify_decoder_layer(self.model)
            torch.hpu.synchronize()

            with HabanaMemoryProfiler() as m_wrap:
                self.model = _maybe_wrap_in_hpu_graph(
                    self.model, vllm_config=self.vllm_config)
            msg = f"Wrapping in HPU Graph took {m_wrap.get_summary_string()}"
            logger.info(msg)

        self.model_memory_usage = m.consumed_device_memory
        msg = f"Loading model weights took in total {m.get_summary_string()}"
        logger.info(msg)

    def get_model(self) -> nn.Module:
        return self.model

    def _use_graphs(self, batch_size, seq_len, is_prompt):
        if self.enforce_eager:
            return False
        if self.skip_warmup:
            return True
        return (batch_size, seq_len, is_prompt) in self.graphed_buckets

    def _is_valid_bucket(self, bucket):
        return bucket[0] * bucket[1] <= self.max_num_batched_tokens

    def _setup_buckets(self) -> None:
        align_bs = lambda x: min(self.max_num_seqs, x)
        #FIXME: The default values should be max_model_len
        max_prompt_seq = 1024
        max_decode_seq = 2048
        self.bucketing_global_state.prompt_bs_bucket_cfg = read_bucket_settings(
            'prompt',
            'bs',
            min=1,
            step=align_bs(32),
            max=self.max_num_prefill_seqs)
        self.bucketing_global_state.decode_bs_bucket_cfg = read_bucket_settings(
            'decode', 'bs', min=1, step=align_bs(32), max=self.max_num_seqs)
        self.bucketing_global_state.prompt_seq_bucket_cfg = \
            read_bucket_settings(
            'prompt',
            'seq',
            min=self.block_size,
            step=self.block_size,
            max=max_prompt_seq)
        self.bucketing_global_state.decode_block_bucket_cfg = \
            read_bucket_settings(
            'decode',
            'block',
            min=self.block_size,
            step=self.block_size,
            max=max(self.block_size,
                    self.max_num_seqs * max_decode_seq // self.block_size))
        self.graphed_buckets: Set[Any] = set()

        msg = ("Prompt bucket config (min, step, max_warmup) "
               f"bs:{self.bucketing_global_state.prompt_bs_bucket_cfg}, "
               f"seq:{self.bucketing_global_state.prompt_seq_bucket_cfg}")
        logger.info(msg)

        msg = ("Decode bucket config (min, step, max_warmup) "
               f"bs:{self.bucketing_global_state.decode_bs_bucket_cfg}, "
               f"block:{self.bucketing_global_state.decode_block_bucket_cfg}")
        logger.info(msg)

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

            mm_data = seq_group_metadata.multi_modal_data
            if mm_data:
                mm_kwargs = self.multi_modal_input_mapper(mm_data)
                multi_modal_kwargs_list.append(mm_kwargs)

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

                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping[-1].append(slot)

        max_query_len = max(query_lens)
        sum_query_len = sum(query_lens)
        real_num_seqs = len(query_lens)
        assert max_query_len > 0

        max_prompt_len = max(
            find_bucket(max(seq_lens),
                        self.bucketing_global_state.prompt_seq_bucket_cfg),
            self.block_size)

        lora_ids: List[int] = []
        for seq_group_metadata, context_len in zip(seq_group_metadata_list,
                                                   context_lens):
            lora_id = seq_group_metadata.lora_int_id
            lora_ids.append(lora_id)

            if lora_id > 0:
                lora_requests.add(seq_group_metadata.lora_request)

            lora_index_mapping += [lora_id] * (max_prompt_len - context_len)
            lora_prompt_mapping.extend(
                [lora_id] *
                (max_prompt_len - context_len
                 if seq_group_metadata.sampling_params.prompt_logprobs else 1))

        input_tokens = make_tensor_with_pad(input_tokens,
                                            max_len=max_prompt_len,
                                            pad=0,
                                            dtype=torch.long,
                                            device=self.device)

        input_positions = make_tensor_with_pad(input_positions,
                                               max_len=max_prompt_len,
                                               pad=0,
                                               dtype=torch.long,
                                               device=self.device)

        slot_mapping = make_tensor_with_pad(slot_mapping,
                                            max_len=max_prompt_len,
                                            pad=_PAD_SLOT_ID,
                                            dtype=torch.long,
                                            device=self.device)

        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.long,
                                       device=self.device)

        block_indices, block_offsets = precompute_indices_and_offsets(
            self.block_size, slot_mapping, True)
        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=True,
            block_list=None,
            block_mapping=None,
            block_usage=None,
            block_indices=block_indices,
            block_offsets=block_offsets,
            block_scales=None,
            block_groups=None,
            attn_bias=None,
            seq_lens_tensor=seq_lens_tensor,
            num_prefills=real_num_seqs,
            num_prefill_tokens=sum_query_len,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=
            None,  # FIXME(kzawora): mutli-modality will not work here
            enable_kv_scales_calculation=False,
        )
        multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_kwargs_list)

        return PreparePromptMetadata(input_tokens=input_tokens,
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
    ) -> PrepareDecodeMetadata:
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []
        seq_lens: List[int] = []
        block_tables: List[List[int]] = []
        lora_index_mapping: List[List[int]] = []
        lora_prompt_mapping: List[List[int]] = []
        lora_requests: Set[LoRARequest] = set()

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

            if lora_id > 0:
                lora_requests.add(seq_group_metadata.lora_request)

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append([generation_token])

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append([position])

                seq_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                seq_lens.append(seq_len)

                block_table = seq_group_metadata.block_tables[seq_id]
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

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device)
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device)

        num_decode_tokens = sum(seq_lens)

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

        padding_fn = None
        if self.use_contiguous_pa:
            block_bucket_size = max(max(block_list) + 1, len(block_list))
            block_bucket_size = find_bucket(
                block_bucket_size,
                self.bucketing_global_state.decode_block_bucket_cfg)
            indices: List[Any]
            indices = [None] * block_bucket_size
            for i, bid in enumerate(block_list):
                indices[bid] = i
            padding_fn = lambda tensor, pad_value: gather_list(
                tensor, indices, pad_value)
        else:
            block_bucket_size = find_bucket(
                len(block_list),
                self.bucketing_global_state.decode_block_bucket_cfg)
            padding_fn = lambda tensor, pad_value: pad_list(
                tensor, block_bucket_size, pad_value)

        block_list = padding_fn(block_list, _PAD_BLOCK_ID)
        block_groups = padding_fn(block_groups, -1)
        block_usage = padding_fn(block_usage, 1)

        block_list = torch.tensor(block_list,
                                  dtype=torch.int,
                                  device=self.device)
        block_groups = torch.tensor(block_groups,
                                    dtype=torch.int,
                                    device=self.device)
        block_usage = torch.tensor(block_usage,
                                   dtype=self.model_config.dtype,
                                   device=self.device)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=self.device)

        block_indices, block_offsets = precompute_indices_and_offsets(
            self.block_size, slot_mapping, False)

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=False,
            block_list=block_list,
            block_mapping=None,
            block_usage=block_usage,
            block_indices=block_indices,
            block_offsets=block_offsets,
            block_scales=None,
            block_groups=block_groups,
            attn_bias=None,
            seq_lens_tensor=None,
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

        real_batch_size = len(seq_group_metadata_list)
        bucket_cfg = self.bucketing_global_state.prompt_bs_bucket_cfg \
            if is_prompt else self.bucketing_global_state.decode_bs_bucket_cfg
        batch_size_padded = find_bucket(real_batch_size, bucket_cfg)
        batch_size_padding = batch_size_padded - real_batch_size
        seq_group_metadata_list = seq_group_metadata_list.copy()
        if batch_size_padding > 0:
            dummy_seq_group_metadata = self.create_dummy_seq_group_metadata(
                0, 0, is_prompt)
            seq_group_metadata_list.extend(dummy_seq_group_metadata
                                           for _ in range(batch_size_padding))

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
        sampling_metadata = SamplingMetadata.prepare(seq_group_metadata_list,
                                                     seq_lens, query_lens,
                                                     self.device,
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
        paddings = [max_len - s for s in seq_lens]
        paddings = [0] + paddings[:-1]
        paddings = list(itertools.accumulate(paddings))
        paddings_prompt_logprobs = []
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            if seq_group_metadata.sampling_params.prompt_logprobs is not None \
                              and seq_group_metadata.is_prompt:
                paddings_prompt_logprobs += ([paddings[i]] * seq_lens[i])
        paddings = torch.tensor(
            paddings_prompt_logprobs if paddings_prompt_logprobs else paddings,
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
            "selected_token_indices": sampling_metadata.selected_token_indices,
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
            'attn_bias', 'seq_lens_tensor', 'block_list', 'block_mapping',
            'block_usage', 'slot_mapping', 'is_prompt', 'block_indices',
            'block_offsets', 'block_scales', 'block_groups'
        ])
        return attention_metadata

    def create_dummy_seq_group_metadata(self,
                                        group_id,
                                        seq_len,
                                        is_prompt,
                                        lora_request=None):
        sampling_params = SamplingParams(temperature=0)
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
        max_seq_len = self.bucketing_global_state.prompt_seq_bucket_cfg[-1]
        max_batch_size = min(self.max_num_batched_tokens // max_seq_len,
                             self.scheduler_config.max_num_seqs)
        self.warmup_scenario(max_batch_size, max_seq_len, True, False, True)
        return

    def warmup_scenario(self,
                        batch_size,
                        seq_len,
                        is_prompt,
                        is_pt_profiler_run=False,
                        is_lora_profile_run=False) -> None:
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
                    if dummy_lora_requests_per_seq else None)
                for i in range(batch_size)
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
                    if dummy_lora_requests_per_seq else None)
                for i, b in enumerate(blocks)
            ]
        torch.hpu.synchronize()
        profiler = None
        if is_pt_profiler_run and self.is_driver_worker:
            profiler = setup_profiler()
            profiler.start()
        for _ in range(times):
            inputs = self.prepare_model_input(seqs)
            self.execute_model(inputs, None, warmup_mode=True)
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
        if phase == "Prompt":
            dim = "seq_len"
        msg = (f"[Warmup][{phase}][{i+1}/{max_i}] "
               f"batch_size:{batch_size} "
               f"{dim}:{seq_len} "
               f"free_mem:{free_mem}")
        logger.info(msg)

    def warmup_all_buckets(self, buckets, is_prompt):
        for i, (batch_size, seq_len) in enumerate(reversed(buckets)):
            self.log_warmup('Prompt' if is_prompt else 'Decode', i,
                            len(buckets), batch_size, seq_len)
            self.warmup_scenario(batch_size, seq_len, is_prompt)

    def warmup_graphs(self,
                      strategy,
                      buckets,
                      is_prompt,
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
                self.warmup_scenario(batch_size, seq_len, is_prompt)
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
            self.warmup_scenario(int(bs), int(seq_len), is_prompt, True)
            raise AssertionError("Finished profiling")
        if self.skip_warmup:
            logger.info("Skipping warmup...")
            return
        self.profiler.start('internal', 'warmup')
        max_blocks = kv_caches[0][0].size(0)

        self.bucketing_global_state.prompt_buckets, prompt_omitted_buckets = \
            generate_prompt_buckets(
            self.bucketing_global_state.prompt_bs_bucket_cfg,
            self.bucketing_global_state.prompt_seq_bucket_cfg,
            self.max_num_batched_tokens)

        msg = (f"Generated {len(self.bucketing_global_state.prompt_buckets)} "
               f"prompt buckets [bs, seq]: \
                {list(sorted(self.bucketing_global_state.prompt_buckets))}")
        logger.info(msg)

        msg = (f"Omitted {len(prompt_omitted_buckets)} "
               "prompt buckets due to exceeded token budget "
               f"(max_num_batched_tokens={self.max_num_batched_tokens})")
        logger.info(msg)

        msg = f"Omitted prompt buckets: {list(sorted(prompt_omitted_buckets))}"
        logger.debug(msg)

        self.bucketing_global_state.decode_buckets = generate_decode_buckets(
            self.bucketing_global_state.decode_bs_bucket_cfg,
            self.bucketing_global_state.decode_block_bucket_cfg, max_blocks)
        logger.info("Generated %d decode buckets [bs, total_blocks]: %s",
                    len(self.bucketing_global_state.decode_buckets),
                    list(sorted(self.bucketing_global_state.decode_buckets)))

        if not htorch.utils.internal.is_lazy() and not self.enforce_eager:
            cache_size_limit = 1 + 3 * (
                len(self.bucketing_global_state.prompt_buckets) +
                len(self.bucketing_global_state.decode_buckets))
            torch._dynamo.config.cache_size_limit = max(
                cache_size_limit, torch._dynamo.config.cache_size_limit)
            # Multiply by 8 to follow the original default ratio between
            # the cache_size_limit and accumulated_cache_size_limit
            torch._dynamo.config.accumulated_cache_size_limit = max(
                cache_size_limit * 8,
                torch._dynamo.config.accumulated_cache_size_limit)

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
            self.warmup_all_buckets(self.bucketing_global_state.prompt_buckets,
                                    True)
            self.warmup_all_buckets(self.bucketing_global_state.decode_buckets,
                                    False)

            if not self.enforce_eager and htorch.utils.internal.is_lazy():
                assert self.mem_margin is not None, \
                    ("HabanaWorker.determine_num_available_blocks needs "
                    "to be called before warming up the model.")
                free_mem = HabanaMemoryProfiler.current_free_device_memory()
                graph_free_mem = free_mem - self.mem_margin
                graph_free_mem = align_workers(graph_free_mem,
                                               torch.distributed.ReduceOp.MIN)
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
                    f"{format_bytes(prompt_available_memory)} for prompt and "
                    f"{format_bytes(decode_available_memory)} for decode "
                    f"(VLLM_GRAPH_PROMPT_RATIO={prompt_graph_mem_ratio})")
                logger.info(msg)
                prompt_strategy = os.environ.get('VLLM_GRAPH_PROMPT_STRATEGY',
                                                 'min_tokens')
                decode_strategy = os.environ.get('VLLM_GRAPH_DECODE_STRATEGY',
                                                 'max_bs')
                mem_post_prompt, prompt_batch_seq, prompt_captured_all = \
                    self.warmup_graphs(
                    prompt_strategy, self.bucketing_global_state.prompt_buckets,
                    True, prompt_available_memory)
                mem_post_decode, decode_batch_seq, decode_captured_all = \
                    self.warmup_graphs(
                    decode_strategy, self.bucketing_global_state.decode_buckets,
                    False, decode_available_memory)

                # Not all prompt buckets were captured, but all decode buckets
                # were captured and we have some free graph-allocated space
                # left. Let's try to use it for capturing more prompt buckets.
                if (mem_post_decode + mem_post_prompt < graph_free_mem
                        and not prompt_captured_all and decode_captured_all):
                    mem_post_prompt, _, prompt_captured_all = (
                        self.warmup_graphs(
                            prompt_strategy,
                            self.bucketing_global_state.prompt_buckets, True,
                            graph_free_mem - mem_post_prompt - mem_post_decode,
                            mem_post_prompt, prompt_batch_seq))

                # Not all decode buckets were captured, but all prompt buckets
                # were captured and we have some free graph-allocated space
                # left. Let's try to use it for capturing more decode buckets.
                if mem_post_decode + mem_post_prompt < graph_free_mem \
                    and not decode_captured_all \
                        and prompt_captured_all:
                    mem_post_decode, _, _ = self.warmup_graphs(
                        decode_strategy,
                        self.bucketing_global_state.decode_buckets, False,
                        graph_free_mem - mem_post_prompt - mem_post_decode,
                        mem_post_decode, decode_batch_seq)

                self.log_graph_warmup_summary(
                    self.bucketing_global_state.prompt_buckets, True,
                    mem_post_prompt)
                self.log_graph_warmup_summary(
                    self.bucketing_global_state.decode_buckets, False,
                    mem_post_decode)

        end_time = time.perf_counter()
        end_mem = HabanaMemoryProfiler.current_device_memory_usage()
        elapsed_time = end_time - start_time
        msg = (
            f"Warmup finished in {elapsed_time:.0f} secs, "
            f"allocated {format_bytes(end_mem - start_mem)} of device memory")
        logger.info(msg)
        self.profiler.end()

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()

    @property
    def mem_margin(self) -> Optional[int]:
        return self._mem_margin

    @mem_margin.setter
    def mem_margin(self, value):
        self._mem_margin = value


def _maybe_wrap_in_hpu_graph(*args, **kwargs):
    return htorch.hpu.wrap_in_hpu_graph(
        HpuModelAdapter(*args, **kwargs), disable_tensor_cache=True
    ) if htorch.utils.internal.is_lazy() else HpuModelAdapter(*args, **kwargs)


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


def unwrap_model(model):
    if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
        return unwrap_model(model._orig_mod)
    else:
        model = list(vars(model)['_modules'].values())[0]
        modules = list(vars(model)['_modules'].values())
        return modules


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

    def finish_measurements(self):
        from neural_compressor.torch.quantization import finalize_calibration
        finalize_calibration(self.model.model)

    def _check_config(self, batch_size, seq_len, is_prompt, warmup_mode):
        cfg = (batch_size, seq_len, is_prompt)
        seen = cfg in self.seen_configs
        self.seen_configs.add(cfg)
        if not seen and not warmup_mode:
            phase = 'prompt' if is_prompt else 'decode'
            logger.warning("Configuration: (%s, %s, %s) was not warmed-up!",
                           phase, batch_size, seq_len)

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

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForHPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
        warmup_mode=False,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        if num_steps > 1:
            raise ValueError(
                "num_steps > 1 is not supported in HPUModelRunner")

        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests,
                                  model_input.lora_mapping)
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
        self._check_config(batch_size, seq_len, is_prompt, warmup_mode)

        lora_mask: torch.Tensor = None
        lora_logits_mask: torch.Tensor = None
        if self.lora_config:
            assert model_input.lora_ids is not None
            lora_mask, lora_logits_mask = self.create_lora_mask(
                input_tokens, model_input.lora_ids, attn_metadata.is_prompt)

        execute_model_kwargs = {
            "input_ids": input_tokens,
            "positions": input_positions,
            "attn_metadata": self.trim_attn_metadata(attn_metadata),
            "intermediate_tensors": intermediate_tensors,
            "lora_mask": lora_mask,
            "virtual_engine": model_input.virtual_engine,
            **(model_input.multi_modal_kwargs or {}),
        }
        if htorch.utils.internal.is_lazy():
            execute_model_kwargs.update({"bypass_hpu_graphs": not use_graphs})

        htorch.core.mark_step()
        if self.is_driver_worker:
            model_event_name = ("model_"
                                f"{'prompt' if is_prompt else 'decode'}_"
                                f"bs{batch_size}_"
                                f"seq{seq_len}_"
                                f"graphs{'T' if use_graphs else 'F'}")
        else:
            model_event_name = 'model_executable'
        with self.profiler.record_event('internal', model_event_name):
            hidden_states = self.model.forward(
                **execute_model_kwargs,
                selected_token_indices=sampling_metadata.selected_token_indices
            )

        if self.lora_config:
            LoraMask.setLoraMask(
                lora_logits_mask.index_select(
                    0, sampling_metadata.selected_token_indices))

        # Compute the logits.
        with self.profiler.record_event(
                'internal', ('compute_logits_'
                             f'{"prompt" if is_prompt else "decode"}_bs'
                             f'{batch_size}_'
                             f'seq{seq_len}')):
            sampling_metadata.selected_token_indices = None
            logits = self.model.compute_logits(hidden_states,
                                               sampling_metadata)
        htorch.core.mark_step()
        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return []

        if model_input.async_callback is not None:
            model_input.async_callback()

        # Sample the next token.
        with self.profiler.record_event(
                'internal', ('sample_'
                             f'{"prompt" if is_prompt else "decode"}_'
                             f'bs{batch_size}_'
                             f'seq{seq_len}')):
            output = self.model.sample(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        output.outputs = output.outputs[:real_batch_size]
        htorch.core.mark_step()

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
        return [output]

    def shutdown_inc(self):
        can_finalize_inc = False
        from contextlib import suppress
        with suppress(AttributeError):
            can_finalize_inc = (self.model_config.quantization == 'inc') and \
                (self.model.model is not None) and \
                self.inc_initialized_successfully and \
                not getattr(self, "_is_inc_finalized", False)
        if can_finalize_inc:
            from neural_compressor.torch.quantization import (
                finalize_calibration)
            finalize_calibration(self.model.model)
            self._is_inc_finalized = True

    def __del__(self):
        self.shutdown_inc()
