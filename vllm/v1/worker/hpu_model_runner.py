# SPDX-License-Identifier: Apache-2.0
import collections
import contextlib
import functools
import itertools
import math
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import habana_frameworks.torch as htorch
import habana_frameworks.torch.internal.bridge_config as bc
import numpy as np
import torch
import torch.distributed
import vllm_hpu_extension.environment as environment
from vllm_hpu_extension.flags import enabled_flags
from vllm_hpu_extension.profiler import HabanaMemoryProfiler, format_bytes

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.attention.selector import get_attn_backend
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader import get_model
from vllm.sampling_params import SamplingType
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType, cdiv,
                        is_fake_hpu, is_pin_memory_available)
from vllm.v1.attention.backends.hpu_attn import HPUAttentionMetadataV1
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsLists,
                             LogprobsTensors, ModelRunnerOutput)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput
from vllm_hpu_extension.bucketing.common import get_bucketing_context

logger = init_logger(__name__)

_TYPE_CACHE = {}


class PhaseType(Enum):
    PREFILL = 'prefill'
    PREFIX_PREFILL = 'prefix_prefill'
    DECODE = 'decode'


@dataclass
class PromptDecodeInfo:
    prompt_req_ids: list[str]
    decode_req_ids: list[str]
    prompt_scheduled_tokens: list[int]


@dataclass
class PromptData:
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    attn_metadata: HPUAttentionMetadataV1


@dataclass
class DecodeData:
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    attn_metadata: Optional[HPUAttentionMetadataV1] = None


#TODO(kzawora): remove this
@dataclass
class PrefillInputData:
    request_ids: list
    prompt_lens: list
    token_ids: list
    position_ids: list
    attn_metadata: list
    logits_indices: list

    def zipped(self):
        return zip(self.request_ids, self.prompt_lens, self.token_ids,
                   self.position_ids, self.attn_metadata, self.logits_indices)


#TODO(kzawora): remove this
@dataclass
class DecodeInputData:
    num_decodes: int
    token_ids: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    attn_metadata: Optional[HPUAttentionMetadataV1] = None
    logits_indices: Optional[torch.Tensor] = None


def bool_helper(value):
    value = value.lower()
    return value in ("y", "yes", "t", "true", "on", "1")


@dataclass
class HpuEnvFlags:
    skip_warmup: bool
    enable_bucketing: bool
    use_contiguous_pa: bool
    __env_var_cfg_type = collections.namedtuple('__env_var_cfg_type',
                                                ['name', 'default', 'handler'])

    @classmethod
    def get_env_var_cfg_map(cls):
        return {
            "skip_warmup":
            cls.__env_var_cfg_type('VLLM_SKIP_WARMUP', 'true',
                                   cls.handle_boolean_env_var),
            "enable_bucketing":
            cls.__env_var_cfg_type('VLLM_ENABLE_BUCKETING', 'true',
                                   cls.handle_boolean_env_var),
            "use_contiguous_pa":
            cls.__env_var_cfg_type('VLLM_CONTIGUOUS_PA', 'false',
                                   cls.handle_boolean_env_var),
        }

    @classmethod
    def build(cls, vllm_config: VllmConfig, update_env=True):
        cfg_map = cls.get_env_var_cfg_map()
        env_vars = {
            key: handler(env_var, default, vllm_config, update_env)
            for key, (env_var, default, handler) in cfg_map.items()
        }
        return cls(**env_vars)

    @staticmethod
    def env_var_post_init(env_var, val, vllm_config):
        match env_var:
            case 'VLLM_SKIP_WARMUP':
                if not val:
                    logger.warning(
                        "HPU warmup is currently not supported in V1. "
                        "Forcing warmup off.")
                    val = True
            case 'VLLM_CONTIGUOUS_PA':
                can_use_contiguous_pa = not vllm_config.cache_config.\
                    enable_prefix_caching
                if val and not can_use_contiguous_pa:
                    logger.warning(
                        "Contiguous PA is not supported with prefix caching. "
                        "Forcing contiguous PA off.")
                    val = False
                if val:
                    logger.warning("Contiguous PA is not recommended in V1.")
            case _:
                pass
        return val

    @classmethod
    def handle_boolean_env_var(cls,
                               env_var,
                               default,
                               vllm_config,
                               update_env=True):
        x = bool_helper(os.environ.get(env_var, default))
        x = cls.env_var_post_init(env_var, x, vllm_config)
        if update_env:
            os.environ[env_var] = str(x).lower()
        logger.info('HpuEnvFlags %s: %s', env_var, x)
        return x


def flatten(in_list):
    return list(itertools.chain(*in_list))


def gather_list(input, indices, v):
    return [input[i] if i is not None else v for i in indices]


def _async_h2d_tensor_copy(source, device='hpu'):
    assert source.device.type == 'cpu', \
        "Source tensor is not present in host memory!"
    target = torch.empty(source.shape, dtype=source.dtype, device=device)
    target.copy_(source, non_blocking=True)
    return target


def ensure_decodes_first(b: InputBatch):
    num_reqs = b.num_reqs
    while True:
        # Find the first prompt index
        first_prompt_index = None
        for i in range(num_reqs):
            if b.num_computed_tokens_cpu[i] < b.num_prompt_tokens[i]:
                first_prompt_index = i
                break
        if first_prompt_index is None:
            break

        # Find the last decode index
        last_decode_index = None
        for i in reversed(range(num_reqs)):
            if b.num_computed_tokens_cpu[i] >= b.num_prompt_tokens[i]:
                last_decode_index = i
                break
        if last_decode_index is None:
            break

        # Sanity
        assert first_prompt_index != last_decode_index

        # Check if done
        if first_prompt_index > last_decode_index:
            break

        # Swap
        b.swap_states(first_prompt_index, last_decode_index)


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
        if (attn_metadata is None or
            (self.prefill_use_fusedsdpa and attn_metadata.block_list is None)
                or not attn_metadata.is_prompt):
            return attn_metadata

        prefill_metadata = attn_metadata

        seq_lens_t = prefill_metadata.seq_lens_tensor
        context_lens_t = prefill_metadata.context_lens_tensor

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
                                     seq_lens_t.unsqueeze(-1)).view(
                                         batch_size, 1, 1, seq_len))
        causal_mask = torch.triu(torch.ones((batch_size, 1, seq_len, seq_len),
                                            device=device,
                                            dtype=torch.bool),
                                 diagonal=1)
        mask = causal_mask.logical_or(len_mask)
        mask = torch.concat((past_mask, mask), dim=-1)
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

    def _set_indices_and_offsets(self, metadata, block_size, is_prompt):
        slot_mapping = metadata.slot_mapping.flatten()
        indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
        if is_prompt and metadata.block_list is None:
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
                            int):  # Indexed-based access (like Modulelist)
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
        # TODO(kzawora): something goes VERY WRONG when operating on
        # kwargs['attn_metadata'].slot_mapping, compared to untrimmed metadata
        kwargs = kwargs.copy()
        #        selected_token_indices = kwargs.pop('selected_token_indices')
        if 'warmup_mode' in kwargs:
            kwargs.pop('warmup_mode')
        input_ids = kwargs['input_ids']
        kwargs['attn_metadata'] = self._update_metadata(
            kwargs['attn_metadata'], input_ids.size(0), input_ids.size(1),
            input_ids.device, self.dtype)
        if self.layer_names is not None:
            self._prepare_cos_sin(kwargs['positions'])
        attn_meta = kwargs.pop('attn_metadata')
        if 'kv_caches' in kwargs:
            kwargs.pop('kv_caches')
        with set_forward_context(attn_meta, self.vllm_config):
            hidden_states = self.model(*args, **kwargs)
        return hidden_states

    def compute_logits(self, *args, **kwargs):
        return self.model.compute_logits(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)

    def generate_proposals(self, *args, **kwargs):
        return self.model.generate_proposals(*args, **kwargs)

    # sampler property will be used by spec_decode_worker
    # don't rename
    @property
    def sampler(self):
        return self.model.sampler


def _maybe_wrap_in_hpu_graph(*args, **kwargs):
    return htorch.hpu.wrap_in_hpu_graph(
        HpuModelAdapter(*args, **kwargs), disable_tensor_cache=True
    ) if htorch.utils.internal.is_lazy() else HpuModelAdapter(*args, **kwargs)


def subtuple(obj: object,
             typename: str,
             to_copy: list[str],
             to_override: Optional[dict[str, object]] = None):
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


def trim_attn_metadata(metadata: HPUAttentionMetadataV1) -> object:
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
        'attn_bias', 'seq_lens_tensor', 'context_lens_tensor', 'block_list',
        'block_mapping', 'block_usage', 'slot_mapping', 'is_prompt',
        'block_indices', 'block_offsets', 'block_groups'
    ])
    return attention_metadata


def next_pow2(value: int, base: int):
    res = base
    while value > 1:
        value = (value + 1) // 2
        res *= 2
    return res


def round_up(value: int, k: int):
    return (value + k - 1) // k * k


def pad_list(list, k, v):
    target_len = round_up(len(list), k)
    padding = target_len - len(list)
    return list + [v] * padding


class HPUModelRunner:

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device = 'hpu',
    ):
        # TODO: use ModelRunnerBase.__init__(self, vllm_config=vllm_config)
        environment.set_model_config(vllm_config.model_config)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        # NOTE(kzawora) update_env is a hack to work around VLLMKVCache in
        # hpu-extension which selects fetch_from_cache implementation based
        # on env vars... this should be fixed in the future
        self.env_flags = HpuEnvFlags.build(vllm_config, update_env=True)
        self.enable_bucketing = self.env_flags.enable_bucketing
        self.use_contiguous_pa = self.env_flags.use_contiguous_pa
        self.skip_warmup = self.env_flags.skip_warmup

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.max_num_tokens = scheduler_config.max_num_batched_tokens

        # Model-related.
        self.num_attn_layers = self.model_config.get_num_layers_by_block_type(
            self.parallel_config, LayerBlockType.attention)
        self.num_query_heads = self.model_config.get_num_attention_heads(
            self.parallel_config)
        self.num_kv_heads = self.model_config.get_num_kv_heads(
            self.parallel_config)
        self.head_size = self.model_config.get_head_size()
        self.hidden_size = self.model_config.get_hidden_size()

        self.attn_backend = get_attn_backend(
            self.head_size,
            self.dtype,
            self.kv_cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
            use_mla=self.model_config.use_mla,
        )

        # Lazy initialization
        # self.model: nn.Module  # set after load_model
        self.kv_caches: list[torch.Tensor] = []

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        # Persistent batch.
        self.input_batch = InputBatch(
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.max_model_len,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
        )
        self.mem_margin = None

        self.use_hpu_graph = not self.model_config.enforce_eager
        # TODO(woosuk): Provide an option to tune the max cudagraph batch size.
        self.max_batch_size = self.scheduler_config.max_num_seqs
        self.input_ids = torch.zeros(
            (self.max_batch_size, self.max_num_tokens),
            dtype=torch.int32,
            device=self.device)
        self.positions = torch.zeros(
            (self.max_batch_size, self.max_num_tokens),
            dtype=torch.int64,
            device=self.device)
        self.prefill_positions = torch.tensor(
            range(self.max_model_len),
            device="cpu",
        ).to(torch.int32).reshape(1, -1)
        self.max_num_seqs = self.scheduler_config.max_num_seqs
        self.max_prefill_batch_size = 1  # TODO(kzawora): add knob for that
        self.padding_aware_scheduling = True  # TODO(kzawora): add knob for that
        self.padding_ratio_threshold = 0.9  # TODO(kzawora): add knob for that
        self.seen_configs: set = set()
        if self.enable_bucketing:
            logger.info("Bucketing is ON.")
            HPUBucketingContext = get_bucketing_context()
            self.bucketing_ctx = HPUBucketingContext(
                self.max_num_seqs, self.max_prefill_batch_size,
                self.block_size, self.scheduler_config.max_num_batched_tokens,
                False)
            self.graphed_buckets: set[Any] = set()
        else:
            logger.info("Bucketing is OFF.")
        self._PAD_SLOT_ID = -1
        self._PAD_BLOCK_ID = -1
        self._tokenizer = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            parallel_config=vllm_config.parallel_config,
            lora_config=vllm_config.lora_config).tokenizer

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        forward_ctx = self.vllm_config.compilation_config.static_forward_context
        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in forward_ctx.items():
            if isinstance(attn_module, FusedMoE):
                continue

            # TODO: Support other attention modules, e.g., sliding window,
            # cross-attention
            assert isinstance(attn_module, Attention)
            if attn_module.attn_type == AttentionType.DECODER:
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                    use_mla=use_mla)
            elif attn_module.attn_type in (AttentionType.ENCODER,
                                           AttentionType.ENCODER_ONLY):
                # encoder-only attention does not need KV cache.
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown attention type: {attn_module.attn_type}")

        return kv_cache_spec

    def _update_states(self, scheduler_output: "SchedulerOutput") -> bool:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt=new_req_data.prompt,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            req_ids_to_add.append(req_id)
        # Update the states of the running/resumed requests.
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]
            # Update the cached states.
            num_computed_tokens = req_data.num_computed_tokens
            req_state.num_computed_tokens = num_computed_tokens
            # Add the sampled token(s) from the previous step (if any).
            # This doesn't include "unverified" tokens like spec decode tokens.
            num_new_tokens = (num_computed_tokens +
                              len(req_data.new_token_ids) -
                              req_state.num_tokens)
            if num_new_tokens == 1:
                # Avoid slicing list in most common case.
                req_state.output_token_ids.append(req_data.new_token_ids[-1])
            elif num_new_tokens > 0:
                req_state.output_token_ids.extend(
                    req_data.new_token_ids[-num_new_tokens:])
            # Update the block IDs.
            if not req_data.resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                req_state.block_ids.extend(req_data.new_block_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = req_data.new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)
            self.input_batch.block_table.append_row(req_data.new_block_ids,
                                                    req_index)
            # Add new_token_ids to token_ids_cpu.
            start_token_index = num_computed_tokens
            end_token_index = num_computed_tokens + len(req_data.new_token_ids)
            self.input_batch.token_ids_cpu[
                req_index,
                start_token_index:end_token_index] = req_data.new_token_ids
            self.input_batch.num_tokens_no_spec[req_index] = end_token_index
            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                req_id, ())
            if spec_token_ids:
                start_index = end_token_index
                end_token_index += len(spec_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
            # NOTE(woosuk): `num_tokens` here may include spec decode tokens.
            self.input_batch.num_tokens[req_index] = end_token_index

        # Check if the batch has changed. If not, we can skip copying the
        # sampling metadata from CPU to GPU.
        batch_changed = len(removed_req_indices) > 0 or len(req_ids_to_add) > 0

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        if batch_changed:
            self.input_batch.refresh_sampling_metadata()
        return batch_changed

    def get_model(self) -> torch.nn.Module:
        assert self.model is not None
        return self.model

    def _get_prompts_and_decodes(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> PromptDecodeInfo:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # Traverse decodes first
        decode_req_ids = []
        for i in range(num_reqs):
            req_id = self.input_batch.req_ids[i]
            assert req_id is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[i]
            num_prompt_tokens = self.input_batch.num_prompt_tokens[i]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]

            if num_computed_tokens < num_prompt_tokens:
                # This is prompt
                break

            # This is decode
            assert num_scheduled_tokens == 1
            decode_req_ids.append(req_id)

        # Traverse prompts
        prompt_req_ids = []
        prompt_scheduled_tokens = []
        for i in range(len(decode_req_ids), num_reqs):
            req_id = self.input_batch.req_ids[i]
            assert req_id is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[i]
            num_prompt_tokens = self.input_batch.num_prompt_tokens[i]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]

            # Must be prompt
            assert num_computed_tokens < num_prompt_tokens
            assert len(self.requests[req_id].output_token_ids) == 0

            prompt_req_ids.append(req_id)
            prompt_scheduled_tokens.append(num_scheduled_tokens)

        return PromptDecodeInfo(prompt_req_ids, decode_req_ids,
                                prompt_scheduled_tokens)

    def _prepare_sampling(self,
                          batch_changed: bool,
                          request_ids: Union[None, list[str]] = None,
                          pad_to: Optional[int] = None) -> SamplingMetadata:
        # Create the sampling metadata.
        req_id_output_token_ids: dict[str, list[int]] = \
            {req_id: req.output_token_ids \
                for req_id, req in self.requests.items()}
        if request_ids is not None:
            req_id_output_token_ids = {
                req_id: req_id_output_token_ids[req_id] \
                    for req_id in request_ids}
        req_id_output_token_ids_lst = list(req_id_output_token_ids.items())
        if pad_to is not None:
            while len(req_id_output_token_ids_lst) < pad_to:
                req_id_output_token_ids_lst.append(
                    req_id_output_token_ids_lst[0])
        sampling_metadata = self.input_batch.make_selective_sampling_metadata(
            req_id_output_token_ids_lst, skip_copy=not batch_changed)
        return sampling_metadata

    def get_habana_paged_attn_buffers(self,
                                      block_tables,
                                      slot_mapping,
                                      bucketing=True):

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
        block_bucket_size: int
        if self.use_contiguous_pa:
            block_bucket_size = max(max(block_list) + 1, len(block_list))
            if bucketing:
                block_bucket_size = \
                    self.bucketing_ctx.get_padded_decode_num_blocks(
                    block_bucket_size)
            indices: list[Any]
            indices = [None] * block_bucket_size
            for i, bid in enumerate(block_list):
                indices[bid] = i
            padding_fn = lambda tensor, pad_value: gather_list(
                tensor, indices, pad_value)
        else:
            if bucketing:
                block_bucket_size = \
                    self.bucketing_ctx.get_padded_decode_num_blocks(
                    len(block_list))
            else:
                block_bucket_size = len(block_list)
            padding_fn = lambda tensor, pad_value: pad_list(
                tensor, block_bucket_size, pad_value)

        block_list = padding_fn(block_list, self._PAD_BLOCK_ID)
        block_groups = padding_fn(block_groups, -1)
        block_usage = padding_fn(block_usage, 1)

        block_list = torch.tensor(block_list, dtype=torch.long, device='cpu')
        block_groups = torch.tensor(block_groups,
                                    dtype=torch.long,
                                    device='cpu')
        block_usage = torch.tensor(block_usage,
                                   dtype=self.model_config.dtype,
                                   device='cpu')
        return block_list, block_groups, block_usage

    def _get_padded_prefill_dims(self, num_prefills, max_prompt_len,
                                 bucketing):
        if bucketing:
            padded_batch_size = self.bucketing_ctx.get_padded_batch_size(
                num_prefills, True)
            padded_prompt_len = self.bucketing_ctx.get_padded_prompt_seq_len(
                max_prompt_len)
        else:
            #NOTE(kzawora): On HPU prompt length needs to be block_size
            # aligned, so we're padding to that, even if bucketing
            # is disabled.
            padded_batch_size = num_prefills
            padded_prompt_len = math.ceil(
                max_prompt_len / self.block_size) * self.block_size
        assert padded_prompt_len <= self.max_model_len
        return padded_batch_size, padded_prompt_len

    def _prefill_find_batch_size(self, num_scheduled_tokens, batch_idx,
                                 num_reqs, fake_prefix_prefill, bucketing):
        num_prefills: int
        padded_batch_size: int
        padded_prompt_len: int
        padded_num_tokens: int
        padding_ratio: float
        for possible_batch_size in reversed(
                range(1, self.max_prefill_batch_size + 1)):
            if batch_idx + possible_batch_size > num_reqs:
                continue
            num_prefills = possible_batch_size
            batch_req_ids = self.input_batch.req_ids[batch_idx:batch_idx +
                                                     num_prefills]
            batch_context_lens = self.input_batch.num_computed_tokens_cpu[
                batch_idx:batch_idx + num_prefills]
            batch_num_prompt_tokens = self.input_batch.num_prompt_tokens[
                batch_idx:batch_idx + num_prefills]
            batch_num_scheduled_tokens = num_scheduled_tokens[
                batch_idx:batch_idx + num_prefills]

            prompt_lens = num_scheduled_tokens[batch_idx:batch_idx +
                                               num_prefills]

            if fake_prefix_prefill:
                for i in range(num_prefills):
                    if batch_context_lens[i] > 0 and batch_num_scheduled_tokens[
                            i] != batch_num_prompt_tokens[i]:
                        prompt_lens[i] = batch_num_prompt_tokens[i]

            max_prompt_len = max(prompt_lens)
            num_tokens = sum(prompt_lens)
            padded_batch_size, padded_prompt_len = \
                self._get_padded_prefill_dims(num_prefills,
                    max_prompt_len, bucketing)
            padded_num_tokens = padded_batch_size * padded_prompt_len
            padding_ratio = 1 - (num_tokens / padded_num_tokens)
            is_within_token_budget = padded_batch_size * padded_prompt_len \
                < self.scheduler_config.max_num_batched_tokens
            is_within_padding_ratio_threshold = padding_ratio < \
                self.padding_ratio_threshold
            can_schedule = is_within_token_budget and \
                is_within_padding_ratio_threshold
            # If padding aware scheduling is off, we'll break on the first
            # loop iteration (==max_prefill_batch_size).
            # Else, we'll break on first batch size that fits token budget.
            if not self.padding_aware_scheduling or can_schedule:
                break
        return batch_req_ids, padded_batch_size, padded_prompt_len

    def _prepare_prefill_inputs(self,
                                total_num_prefills,
                                num_decodes,
                                num_scheduled_tokens: list[int],
                                bucketing=True) -> PrefillInputData:
        # Each prefill run separately with shape [1, padded_prompt_len].
        # So we create lists that will be used in execute_model().

        prefill_request_ids = []
        prefill_prompt_lens = []
        prefill_token_ids = []
        prefill_position_ids = []
        prefill_attn_metadata = []
        prefill_logits_indices = []
        block_table_cpu_tensor = self.input_batch.block_table.get_cpu_tensor()
        fake_prefix_prefill = False

        # DECODES are the first num_decodes REQUESTS.
        # PREFILLS are the next num_reqs - num_decodes REQUESTS.
        num_reqs = total_num_prefills + num_decodes
        # NOTE(kzawora): This loop was initially implemented as
        # for batch_idx in range(num_decodes, num_reqs, max_prefill_batch_size)
        # but was changed to accommodate variable loop step size for
        # padding-aware scheduling
        batch_idx = num_decodes
        while batch_idx < num_reqs:
            # Find the largest batch size in range [1, max_prefill_batch_size]
            # that can fit within specified token budget

            batch_req_ids, padded_batch_size, padded_prompt_len = (
                self._prefill_find_batch_size(num_scheduled_tokens, batch_idx,
                                              num_reqs, fake_prefix_prefill,
                                              bucketing))
            num_prefills = len(batch_req_ids)
            context_lens = self.input_batch.num_computed_tokens_cpu[
                batch_idx:batch_idx + num_prefills]
            batch_num_scheduled_tokens = num_scheduled_tokens[
                batch_idx:batch_idx + num_prefills]

            use_prefix_prefill = any(context_lens) and not fake_prefix_prefill
            # TODO(kzawora): this is an ugly hack for prefix caching, remove
            # padded_batch_size = num_prefills
            if use_prefix_prefill:
                padded_batch_size = num_prefills
                #padded_prompt_len = max(batch_num_scheduled_tokens)

            padded_prompt_lens = [
                padded_prompt_len for _ in range(padded_batch_size)
            ]

            # TOKEN_IDS.
            token_ids = torch.zeros((padded_batch_size, padded_prompt_len),
                                    dtype=torch.int32,
                                    device='cpu')
            # POSITIONS.
            positions = torch.zeros((padded_batch_size, padded_prompt_len),
                                    dtype=torch.int32,
                                    device='cpu')
            # SLOT_MAPPING.
            # The "slot" is the "physical index" of a token in the KV cache.
            # Look up the block_idx in the block table (logical<>physical map)
            # to compute this.
            slot_mapping = torch.ones((padded_batch_size, padded_prompt_len),
                                      dtype=torch.int32,
                                      device='cpu') * self._PAD_SLOT_ID
            dummy_slots = itertools.cycle(
                range(self._PAD_SLOT_ID, self._PAD_SLOT_ID + self.block_size))
            slot_mapping.apply_(lambda _, ds=dummy_slots: next(ds))
            # NOTE(kzawora): this has no right to work on prefix prefills
            iterable = zip(batch_num_scheduled_tokens, [0] *
                           len(batch_num_scheduled_tokens)
                           ) if not use_prefix_prefill else zip(
                               batch_num_scheduled_tokens, context_lens)
            for i, (prompt_scheduled_tokens,
                    prompt_start_idx) in enumerate(iterable):
                # Prepare and sanitize token ids (cpu)
                batch_offset = batch_idx + i
                token_ids[i, :prompt_scheduled_tokens] = torch.from_numpy(
                    self.input_batch.token_ids_cpu[
                        batch_offset, prompt_start_idx:prompt_start_idx +
                        prompt_scheduled_tokens])
                #token_ids[i, prompt_len:] = 0 # no need to sanitize - buffer
                # is pre-filled with 0s

                # Prepare and sanitize positions ids (cpu)
                positions[
                    i, :
                    prompt_scheduled_tokens] = self.prefill_positions[:,
                                                                      prompt_start_idx:
                                                                      prompt_start_idx
                                                                      +
                                                                      prompt_scheduled_tokens]
                #positions[i, prompt_len:] = 0 # no need to sanitize - buffer
                # is pre-filled with 0s

                # Prepare and sanitize slot_mapping (cpu)
                flat_prefill_positions = positions[
                    i, :prompt_scheduled_tokens].flatten()
                block_numbers = block_table_cpu_tensor[
                    batch_offset, flat_prefill_positions // self.block_size]
                block_offsets = flat_prefill_positions % self.block_size
                slot_mapping[
                    i, :
                    prompt_scheduled_tokens] = block_numbers * self.block_size \
                        + block_offsets
                #slot_mapping[i, prompt_len:] = _PAD_SLOT_ID # no need to
                # sanitize - buffer is pre-filled with _PAD_SLOT_IDs
            slot_mapping = slot_mapping.long()

            logits_indices = torch.zeros(padded_batch_size,
                                         dtype=torch.int32,
                                         device='cpu')
            query_start_loc = torch.empty((num_prefills + 1, ),
                                          dtype=torch.int32,
                                          device="cpu")
            query_start_loc_np = query_start_loc.numpy()
            query_start_loc_np[0] = 0

            # logits indices in prefill must account for padding: last
            # token logits will be emitted at index
            # (idx - 1) * padded_seq_len + seq_len[idx] - 1
            np.cumsum(padded_prompt_lens[:num_prefills],
                      out=query_start_loc_np[1:])
            query_start_loc_np[:num_prefills] += num_scheduled_tokens[
                batch_idx:batch_idx + num_prefills]
            logits_indices[:num_prefills] = query_start_loc[:num_prefills] - 1

            # HPU should *not* sync here with CPU
            seq_lens_tensor = torch.zeros((padded_batch_size),
                                          dtype=torch.int32,
                                          device='cpu')
            seq_lens_tensor[:num_prefills] = torch.tensor(
                batch_num_scheduled_tokens, device='cpu')
            token_ids_device = _async_h2d_tensor_copy(token_ids, self.device)
            positions_device = _async_h2d_tensor_copy(positions, self.device)
            seq_lens_tensor_device = _async_h2d_tensor_copy(
                seq_lens_tensor, self.device)
            slot_mapping_device = _async_h2d_tensor_copy(
                slot_mapping, self.device)
            logits_indices_device = _async_h2d_tensor_copy(
                logits_indices, self.device)

            prefill_request_ids.append(batch_req_ids)
            prefill_prompt_lens.append(batch_num_scheduled_tokens)
            prefill_token_ids.append(token_ids_device)
            prefill_position_ids.append(positions_device)
            prefill_logits_indices.append(logits_indices_device)
            attn_metadata = None
            if use_prefix_prefill:
                # Prefix caching
                num_blocks = np.ceil(context_lens / self.block_size).astype(
                    np.int32).tolist()
                max_num_blocks = max(num_blocks)
                prefix_block_tables = torch.ones(
                    (padded_batch_size, max_num_blocks),
                    dtype=torch.int32,
                    device='cpu') * self._PAD_BLOCK_ID
                for i, n in enumerate(num_blocks):
                    prefix_block_tables[i, :n] = block_table_cpu_tensor[
                        batch_idx + i, :n]
                context_lens_tensor = torch.zeros((padded_batch_size),
                                                  dtype=torch.int32,
                                                  device='cpu')
                context_lens_tensor[:num_prefills] = torch.tensor(context_lens,
                                                                  device='cpu')

                block_list_device = _async_h2d_tensor_copy(
                    prefix_block_tables.flatten(), self.device)
                context_lens_tensor_device = _async_h2d_tensor_copy(
                    context_lens_tensor, self.device)
                attn_metadata = \
                    HPUAttentionMetadataV1.make_cached_prefill_metadata(
                    seq_lens_tensor=seq_lens_tensor_device,
                    context_lens_tensor=context_lens_tensor_device,
                    num_prefills=num_prefills,
                    num_prefill_tokens=sum(batch_num_scheduled_tokens),
                    slot_mapping=slot_mapping_device,
                    block_list=block_list_device)
            else:
                attn_metadata = HPUAttentionMetadataV1.make_prefill_metadata(
                    seq_lens_tensor=seq_lens_tensor_device,
                    num_prefills=num_prefills,
                    num_prefill_tokens=sum(batch_num_scheduled_tokens),
                    slot_mapping=slot_mapping_device,
                )
            # ATTN_METADATA.
            prefill_attn_metadata.append(attn_metadata)
            batch_idx += num_prefills
        return PrefillInputData(request_ids=prefill_request_ids,
                                prompt_lens=prefill_prompt_lens,
                                token_ids=prefill_token_ids,
                                position_ids=prefill_position_ids,
                                attn_metadata=prefill_attn_metadata,
                                logits_indices=prefill_logits_indices)

    def _prepare_decode_inputs(self,
                               num_decodes,
                               num_scheduled_tokens,
                               bucketing=True) -> DecodeInputData:
        # Decodes run as one single padded batch with shape [batch, 1]
        #
        # We need to set _PAD_SLOT_ID for the padding tokens in the
        # slot_mapping, such that the attention KV cache insertion
        # logic knows to ignore those indicies. Otherwise, the
        # padding data can be dummy since we have a causal mask.

        block_table_cpu_tensor = self.input_batch.block_table.get_cpu_tensor()
        if num_decodes == 0:
            return DecodeInputData(num_decodes=0)

        # PAD FOR STATIC SHAPES.
        padded_batch_size: int
        if bucketing:
            padded_batch_size = self.bucketing_ctx.get_padded_batch_size(
                num_decodes, False)
        else:
            padded_batch_size = num_decodes

        # POSITIONS. [batch, 1]
        # We slice at the end, since we use the positions for gathering.
        positions = torch.zeros((padded_batch_size, 1), dtype=torch.int32)
        positions[:num_decodes] = torch.from_numpy(
            self.input_batch.num_computed_tokens_cpu.reshape(-1,
                                                             1)[:num_decodes])
        positions = positions[:padded_batch_size]

        padded_index = torch.zeros((padded_batch_size, 1), dtype=torch.int64)
        index = positions.to(torch.int64)[:num_decodes]
        padded_index[:num_decodes] = index

        # TOKEN_IDS. [batch, 1]
        token_ids = torch.zeros((padded_batch_size, 1), dtype=torch.int32)
        token_ids[:num_decodes] = torch.gather(input=torch.from_numpy(
            self.input_batch.token_ids_cpu),
                                               dim=1,
                                               index=index)

        # SLOT_MAPPING [batch, 1]
        # The "slot" is the "physical index" of a token in the KV cache.
        # Look up the block_idx in the block table (logical<>physical map)
        # to compute this.
        block_number = torch.ones(
            (padded_batch_size, 1), dtype=torch.int32) * self._PAD_BLOCK_ID
        block_number[:num_decodes] = torch.gather(input=block_table_cpu_tensor,
                                                  dim=1,
                                                  index=(index //
                                                         self.block_size))
        block_offsets = padded_index % self.block_size
        slot_mapping = block_number * self.block_size + block_offsets
        # set an out of range value for the padding tokens so that they
        # are ignored when inserting into the KV cache.
        slot_mapping = slot_mapping[:padded_batch_size]
        dummy_slots = itertools.cycle(
            range(self._PAD_SLOT_ID, self._PAD_SLOT_ID + self.block_size))
        slot_mapping[num_decodes:].apply_(lambda _, ds=dummy_slots: next(ds))
        # BLOCK_TABLE [batch, max_num_blocks_per_req]
        context_lens = self.input_batch.num_computed_tokens_cpu[:num_decodes]

        # NOTE(kzawora): the +1 is what causes this entire thing to work,
        # as in the paged attention, we don't fetch just the context from cache,
        # but also kvs for the current token
        num_blocks = np.ceil(
            (context_lens + 1) / self.block_size).astype(np.int32).tolist()
        block_tables_list = []
        for i, n in enumerate(num_blocks):
            seq_block_table = block_table_cpu_tensor[i, :n].tolist()
            assert len(seq_block_table) == n
            block_tables_list.append(seq_block_table)

        # CONTEXT_LENS [batch_size]
        block_list, block_groups, block_usage = \
            self.get_habana_paged_attn_buffers(
            block_tables_list, slot_mapping.tolist(), bucketing)

        logits_indices = torch.zeros(padded_batch_size,
                                     dtype=torch.int32,
                                     device='cpu')
        query_start_loc = torch.empty((num_decodes + 1, ),
                                      dtype=torch.int32,
                                      device="cpu",
                                      pin_memory=self.pin_memory)
        query_start_loc_np = query_start_loc.numpy()
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens[:num_decodes],
                  out=query_start_loc_np[1:])
        logits_indices[:num_decodes] = query_start_loc[1:] - 1
        num_decode_tokens = torch.tensor(np.sum(context_lens), device='cpu')

        # CPU<>HPU sync *should not* happen here.
        token_ids_device = _async_h2d_tensor_copy(token_ids, self.device)
        positions_device = _async_h2d_tensor_copy(positions, self.device)
        logits_indices_device = _async_h2d_tensor_copy(logits_indices,
                                                       self.device)
        block_list_device = _async_h2d_tensor_copy(block_list, self.device)
        block_usage_device = _async_h2d_tensor_copy(block_usage, self.device)
        block_groups_device = _async_h2d_tensor_copy(block_groups, self.device)
        num_decode_tokens_device = _async_h2d_tensor_copy(
            num_decode_tokens, self.device)
        slot_mapping_device = _async_h2d_tensor_copy(slot_mapping, self.device)
        return DecodeInputData(
            num_decodes=num_decodes,
            token_ids=token_ids_device,
            position_ids=positions_device,
            logits_indices=logits_indices_device,
            attn_metadata=HPUAttentionMetadataV1.make_decode_metadata(
                block_list=block_list_device,
                block_usage=block_usage_device,
                block_groups=block_groups_device,
                num_decode_tokens=num_decode_tokens_device,
                slot_mapping=slot_mapping_device,
            ))

    def _prepare_inputs(
            self,
            scheduler_output: "SchedulerOutput",
            num_prefills,
            num_decodes,
            bucketing=True
    ) -> tuple[PrefillInputData, Optional[DecodeInputData]]:

        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0

        num_reqs = num_prefills + num_decodes

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = []
        num_prompt_tokens = []
        for idx, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            assert req_id is not None
            seq_num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            seq_num_prompt_tokens = self.input_batch.num_prompt_tokens[idx]
            num_scheduled_tokens.append(seq_num_scheduled_tokens)
            num_prompt_tokens.append(seq_num_prompt_tokens)
            # NOTE: assert that all the decodes are "decodes".
            if idx < num_decodes:
                assert seq_num_scheduled_tokens == 1
        return (
            self._prepare_prefill_inputs(num_prefills, num_decodes,
                                         num_scheduled_tokens, bucketing),
            self._prepare_decode_inputs(num_decodes, num_scheduled_tokens,
                                        bucketing),
        )

    def _seq_len(self, attn_metadata):
        return attn_metadata.slot_mapping.size(1)

    def _num_blocks(self, attn_metadata):
        if attn_metadata.block_list is None:
            return 0
        return attn_metadata.block_list.numel()

    def _phase(self, attn_metadata):
        phase_type: PhaseType
        is_prompt = attn_metadata.is_prompt
        is_prefix_cached = is_prompt and attn_metadata.block_list is not None
        if is_prompt and is_prefix_cached:
            phase_type = PhaseType.PREFIX_PREFILL
        elif is_prompt and not is_prefix_cached:
            phase_type = PhaseType.PREFILL
        elif not is_prompt:
            phase_type = PhaseType.DECODE
        else:
            raise ValueError("Unrecognized pass type, likely due to malformed "
                             "attention metadata")
        return phase_type

    def _check_config(self, batch_size, seq_len, num_blocks, attn_metadata,
                      warmup_mode):
        phase = self._phase(attn_metadata)
        cfg = (batch_size, seq_len, num_blocks, phase)
        seen = cfg in self.seen_configs
        self.seen_configs.add(cfg)
        if not seen and not warmup_mode:
            phase = phase.value
            logger.warning(
                "Configuration: (%s, %s, %s, %s) was not warmed-up!", phase,
                batch_size, seq_len, num_blocks)

    def _execute_model_generic(self,
                               token_ids,
                               position_ids,
                               attn_metadata,
                               logits_indices,
                               kv_caches,
                               warmup_mode=False):
        # FORWARD.
        batch_size = token_ids.size(0)
        seq_len = self._seq_len(attn_metadata)
        num_blocks = self._num_blocks(attn_metadata)

        is_prompt = attn_metadata.is_prompt
        self._check_config(batch_size, seq_len, num_blocks, attn_metadata,
                           warmup_mode)
        additional_kwargs = {}
        if htorch.utils.internal.is_lazy(
        ) and not self.model_config.enforce_eager:
            use_graphs = self._use_graphs(batch_size, seq_len, num_blocks,
                                          is_prompt)
            additional_kwargs.update({"bypass_hpu_graphs": not use_graphs})
        trimmed_attn_metadata = trim_attn_metadata(attn_metadata)
        hidden_states = self.model.forward(input_ids=token_ids,
                                           positions=position_ids,
                                           attn_metadata=trimmed_attn_metadata,
                                           kv_caches=kv_caches)
        #hidden_states = hidden_states[:num_scheduled_tokens]
        # NOTE(kzawora): returning hidden_states is required in prompt logprobs
        # scenarios, as they will do logit processing on their own
        non_flattened_hidden_states = hidden_states

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(hidden_states, None)
        return non_flattened_hidden_states, logits

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        scheduler_output: "SchedulerOutput",
    ) -> dict[str, Optional[LogprobsTensors]]:
        num_prompt_logprobs_dict = self.input_batch.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for i, (req_id, num_prompt_logprobs) in enumerate(
                num_prompt_logprobs_dict.items()):

            num_tokens = scheduler_output.num_scheduled_tokens[req_id]

            # Get metadata for this request.
            request = self.requests[req_id]
            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True)

            # Determine number of logits to retrieve.
            start_tok = request.num_computed_tokens + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens < num_remaining_tokens:
                # This is a chunk, more tokens remain.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            prompt_hidden_states = hidden_states[i, :num_logits]
            logits = self.model.compute_logits(prompt_hidden_states, None)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok:start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.model.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks = self.model.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids)

            # Transfer GPU->CPU async.
            prompt_logprobs_dict[req_id] = LogprobsTensors(
                token_ids.to("cpu", non_blocking=True),
                logprobs.to("cpu", non_blocking=True),
                ranks.to("cpu", non_blocking=True),
            )

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        torch.hpu.synchronize()

        return prompt_logprobs_dict

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        # NOTE(kzawora): Since scheduler doesn't differentiate between prefills
        # and decodes, we must handle mixed batches. In _update_states we make
        # sure that first self.input_batch.num_decodes requests are decodes,
        # and remaining ones until the end are prefills. _update_states also
        # handles changes in request cache based on scheduler outputs and
        # previous iterations (e.g. keeping block tables and context lengths up
        # to date, creating, pruning and updating request caches,
        # and some more stuff)

        # If num_decodes == self.input_batch.num_reqs, then batch is all decode, and only a single decode forward pass will be executed in this method. # noqa
        # If num_decodes == 0, then batch is all prefill, and only prefill forward passes will be executed  in this method. # noqa
        # If neither apply, then batch is mixed, and both prefill and decode forward passes will be executed in this method. # noqa

        # First, we will execute all decodes (if any) in a single batch,
        # then we'll execute prefills in batches of up to max_prefill_batch_size elements. # noqa
        # All shapes used in forward passes are bucketed appropriately to mitigate risk of graph recompilations. # noqa

        # We perform sampling directly after executing each forward pass
        # Everything is done asynchronously - the only sync point is the place
        # where we copy the generated tokens back to the host.

        # Example: If a batch has 6 requests, 3 prefills and 3 decodes, the unprocessed sequences in batch will be laid as follows: # noqa
        # [D0, D1, D2, P0, P1, P2]
        # If we assume max_prefill_batch_size=2, the flow of this method will look as follows: # noqa
        # prepare_inputs: bucket [D0, D1, D2] -> [D0, D1, D2, 0] (BS=4 bucket, 1 seq padding) # noqa
        # prepare_inputs: bucket [P0, P1, P2] -> [P0, P1], [P2] (BS=2 + BS=1 bucket, no seqs padding) # noqa
        # decode forward pass BS4 [D0, D1, D2, 0]
        # decode compute_logits BS4 [D0, D1, D2, 0]
        # decode sampler BS4 [D0, D1, D2, 0] -> [tokD0, tokD1, tokD2, 0]
        # prefill[iter 0] forward pass BS2 [P0, P1]
        # prefill[iter 0] compute_logits BS2 [P0, P1]
        # prefill[iter 0] sampler BS2 [P0, P1] -> [tokP0, tokP1]
        # prefill[iter 1] forward pass BS1 [P0, P1]
        # prefill[iter 1] compute_logits BS1 [P0, P1]
        # prefill[iter 1] sampler BS1 [P0, P1] -> [tokP2]
        # prefill concat sampler results [tokP0, tokP1], [tokP2] -> [tokP0, tokP1, tokP2] # noqa
        # Join the prefill and decode on device into [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2] # noqa
        # Transfer [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2] to CPU
        # On CPU, sanitize [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2] -> [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2] # noqa
        # Return [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2]

        # Example2: Same thing, but with max_prefill_batch_size=4:
        # prepare_inputs: bucket [D0, D1, D2] -> [D0, D1, D2, 0] (BS=4 bucket, 1 seq padding) # noqa
        # prepare_inputs: bucket [P0, P1, P2] -> [P0, P1, P2, 0] (BS=4 bucket, 1 seq padding) # noqa
        # decode forward pass BS4 [D0, D1, D2, 0]
        # decode compute_logits BS4 [D0, D1, D2, 0]
        # decode sampler BS4 [D0, D1, D2, 0] -> [tokD0, tokD1, tokD2, 0]
        # prefill[iter 0] forward pass BS4 [P0, P1, P2, 0]
        # prefill[iter 0] compute_logits BS4 [P0, P1, P2, 0]
        # prefill[iter 0] sampler BS4 [P0, P1, P2, 0] -> [tokP0, tokP1, tokP2, 0] # noqa
        # Join the prefill and decode on device into [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2, 0] # noqa
        # Transfer [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2, 0] to CPU
        # On CPU, sanitize [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2, 0] -> [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2] # noqa
        # Return [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2]

        batch_changed = self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOuptut if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT
        # If necessary, swap decodes/prompts to have all decodes on the start
        ensure_decodes_first(self.input_batch)
        # Prepare prompts/decodes info
        pd_info = self._get_prompts_and_decodes(scheduler_output)
        num_decodes = len(pd_info.decode_req_ids)
        num_prefills = len(pd_info.prompt_req_ids)
        num_reqs = num_decodes + num_prefills
        prefill_data, decode_data = self._prepare_inputs(
            scheduler_output,
            num_prefills,
            num_decodes,
            bucketing=self.enable_bucketing)

        #num_padded_decodes = decode_data.token_ids.shape[
        #    0] if num_decodes > 0 else 0

        #FIXME(kzawora): Currently there's no handling of logprobs. Fix that
        # later.
        prefill_sampler_outputs = []
        prefill_hidden_states = []
        decode_sampler_outputs = []
        prefill_output_tokens_device = None
        prefill_hidden_states_device = None
        decode_output_tokens_device = None
        ######################### PREFILLS #########################
        # Prefills run with shape [padded_prefill_bs, padded_prefill_len]
        if num_prefills > 0:
            htorch.core.mark_step()
            for idx, (req_id, prompt_len, token_ids, position_ids,
                      attn_metadata,
                      logits_indices) in enumerate(prefill_data.zipped()):
                htorch.core.mark_step()
                prefill_hidden_states_ts, logits_device = \
                    self._execute_model_generic(
                    token_ids, position_ids, attn_metadata, logits_indices,
                    self.kv_caches)
                htorch.core.mark_step()
                sampling_metadata = self._prepare_sampling(
                    batch_changed, req_id, pad_to=logits_device.shape[0])
                sampler_output = self.model.sample(
                    logits=logits_device, sampling_metadata=sampling_metadata)
                htorch.core.mark_step()
                prefill_sampler_outputs.append(sampler_output)
                if self.input_batch.num_prompt_logprobs:
                    prefill_hidden_states.append(prefill_hidden_states_ts)
            # sampler returns device tensors, concat will happen on device
            if len(prefill_sampler_outputs) > 0:
                prefill_output_tokens_device = torch.tensor(
                    [o.sampled_token_ids for o in prefill_sampler_outputs],
                    device=prefill_sampler_outputs[0].sampled_token_ids.device,
                    dtype=prefill_sampler_outputs[0].sampled_token_ids.dtype)
                #torch.cat(prefill_output_tokens, dim=0)
            if len(prefill_hidden_states) > 0:
                prefill_hidden_states_device = torch.cat(prefill_hidden_states)
            htorch.core.mark_step()

        ######################### DECODES #########################
        # Decodes run as one single batch with [padded_decode_bs, 1]
        if num_decodes > 0:
            assert decode_data is not None
            htorch.core.mark_step()
            _, logits_device = self._execute_model_generic(
                decode_data.token_ids, decode_data.position_ids,
                decode_data.attn_metadata, decode_data.logits_indices,
                self.kv_caches)
            htorch.core.mark_step()
            sampling_metadata = self._prepare_sampling(
                batch_changed,
                pd_info.decode_req_ids,
                pad_to=logits_device.shape[0])
            sampler_output = self.model.sample(
                logits=logits_device, sampling_metadata=sampling_metadata)
            decode_sampler_outputs.append(sampler_output)
            decode_output_tokens_device = sampler_output.sampled_token_ids
            htorch.core.mark_step()

        # From this point onward, all operations are done on CPU.
        # We already have tokens. Let's copy the data to
        # CPU as is, and then discard padded tokens.
        prefill_output_tokens_cpu = prefill_output_tokens_device.cpu(
        ) if prefill_output_tokens_device is not None else None
        decode_output_tokens_cpu = decode_output_tokens_device.cpu(
        ) if decode_output_tokens_device is not None else None
        # From this point onward, all operations are done on CPU.

        # Discard garbage tokens from prefills and/or decodes
        if prefill_output_tokens_cpu is not None \
            and decode_output_tokens_cpu is not None:
            sampled_token_ids_cpu = torch.cat(
                (decode_output_tokens_cpu[:num_decodes].flatten(),
                 prefill_output_tokens_cpu[:num_prefills].flatten()),
                dim=0)
        else:
            sampled_token_ids_cpu = (
                decode_output_tokens_cpu[:num_decodes].flatten()
                if decode_output_tokens_cpu is not None else
                prefill_output_tokens_cpu[:num_prefills].flatten())

        sampled_token_ids_list = sampled_token_ids_cpu.tolist()
        logprobs = None
        all_outputs = [*prefill_sampler_outputs, *decode_sampler_outputs]
        # NOTE(kzawora): idk what happens if part of batch doesn't have logprobs
        has_logprobs = all(
            [o.logprobs_tensors is not None for o in all_outputs])
        if has_logprobs:
            logprob_token_ids = []
            logprob_values = []
            selected_token_ranks = []
            for out in all_outputs:
                # NOTE(kzawora): this is likely wrong - we're including
                # padded sequence data here
                logprob_token_ids.extend(
                    out.logprobs_tensors.logprob_token_ids.tolist())
                logprob_values.extend(out.logprobs_tensors.logprobs.tolist())
                selected_token_ranks.extend(
                    out.logprobs_tensors.selected_token_ranks.tolist())
            logprobs = LogprobsLists(
                logprob_token_ids,
                logprob_values,
                selected_token_ranks,
            )

        ######### UPDATE REQUEST STATE WITH GENERATED TOKENS #########
        seqs_to_discard = []
        for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            req_state = self.requests[req_id]

            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            # NOTE(kzawora): this is crucial!!! scheduler can send us partial
            # prefills to do, e.g. if we have token budget of 2048 tokens and 3
            # prefills with 768 tokens, we'd process 2 full prefills and first
            # 512 tokens of the last one - but the token that's emitted is
            # obviously garbage and we should not include it in the state
            if seq_len >= len(req_state.prompt_token_ids):
                token_id = sampled_token_ids_list[i]
                self.input_batch.token_ids_cpu[i, seq_len] = token_id
                self.input_batch.num_tokens[i] += 1
                req_state.output_token_ids.append(token_id)
            else:
                seqs_to_discard.append(i)
        ################## RETURN ##################
        # Create output.
        all_req_ids = pd_info.decode_req_ids + pd_info.prompt_req_ids
        prompt_logprobs_dict: dict[
            str, Optional[LogprobsTensors]] = self._get_prompt_logprobs_dict(
                prefill_hidden_states_device, scheduler_output)
        #prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}
        #for req_id in all_req_ids:
        #    prompt_logprobs_dict[req_id] = None
        all_req_ids = pd_info.decode_req_ids + pd_info.prompt_req_ids

        # in spec decode, multiple tokens can be returned, so
        # scheduler expects a list of tokens per seq here
        postprocessed_sampled_token_ids = [
            ([tok] if i not in seqs_to_discard else [])
            for i, tok in enumerate(sampled_token_ids_list)
        ]
        model_runner_output = ModelRunnerOutput(
            req_ids=all_req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=postprocessed_sampled_token_ids,
            logprobs=logprobs,
            spec_token_ids=None,
            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
        )

        return model_runner_output

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        with HabanaMemoryProfiler() as m:  # noqa: SIM117
            self.model = get_model(vllm_config=self.vllm_config)
        self.model_memory_usage = m.consumed_device_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))
        hidden_layer_markstep_interval = int(
            os.getenv('VLLM_CONFIG_HIDDEN_LAYERS', '1'))
        model_config = getattr(self.model, "config", None)
        modify_model_layers(
            self.model,
            get_target_layer_suffix_list(
                model_config.model_type if model_config is not None else None),
            hidden_layer_markstep_interval)
        path_to_rope = get_path_to_rope(self.model)
        torch.hpu.synchronize()
        with HabanaMemoryProfiler() as m:  # noqa: SIM117
            self.model = _maybe_wrap_in_hpu_graph(self.model,
                                                  vllm_config=self.vllm_config,
                                                  layer_names=path_to_rope)
        self.model_memory_usage = m.consumed_device_memory
        logger.info("Wrapping in HPUGraph took %.4f GB",
                    self.model_memory_usage / float(2**30))

    def _use_graphs(self, batch_size, seq_len, num_blocks, phase):
        if self.model_config.enforce_eager:
            return False
        if self.skip_warmup:
            return True
        return (batch_size, seq_len, num_blocks, phase) in self.graphed_buckets

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

    def warmup_scenario(self, batch_size, seq_or_block, is_prompt,
                        kv_caches) -> None:
        """Dummy warmup run for memory usage and graph compilation."""

        query_seq_len = seq_or_block if is_prompt else 1
        input_ids = torch.zeros((batch_size, query_seq_len),
                                dtype=torch.int32,
                                device='cpu')
        position_ids = torch.zeros((batch_size, query_seq_len),
                                   dtype=torch.int32,
                                   device='cpu')
        slot_mapping = torch.zeros((batch_size, query_seq_len),
                                   dtype=torch.int64,
                                   device='cpu')

        input_ids_device = _async_h2d_tensor_copy(input_ids, self.device)
        position_ids_device = _async_h2d_tensor_copy(position_ids, self.device)
        slot_mapping_device = _async_h2d_tensor_copy(slot_mapping, self.device)

        if is_prompt:
            seq_lens = torch.zeros((batch_size),
                                   dtype=torch.int32,
                                   device='cpu')
            seq_lens.fill_(seq_or_block)
            seq_lens_device = _async_h2d_tensor_copy(seq_lens, self.device)
            attn_metadata = HPUAttentionMetadataV1.make_prefill_metadata(
                seq_lens_tensor=seq_lens_device,
                num_prefills=batch_size,
                num_prefill_tokens=batch_size * seq_or_block,
                slot_mapping=slot_mapping_device)
        else:
            block_tables = [
                x.tolist()
                for x in np.array_split(np.arange(seq_or_block), batch_size)
            ]
            block_list, block_groups, block_usage = \
                self.get_habana_paged_attn_buffers(
                block_tables=block_tables,
                slot_mapping=slot_mapping,
                bucketing=True)
            block_list_device = _async_h2d_tensor_copy(block_list, self.device)
            block_usage_device = _async_h2d_tensor_copy(
                block_usage, self.device)
            block_groups_device = _async_h2d_tensor_copy(
                block_groups, self.device)
            attn_metadata = HPUAttentionMetadataV1.make_decode_metadata(
                block_list=block_list_device,
                block_usage=block_usage_device,
                block_groups=block_groups_device,
                num_decode_tokens=batch_size,
                slot_mapping=slot_mapping_device)

        logits_indices = torch.arange(0, batch_size, device='cpu')
        logits_indices_device = _async_h2d_tensor_copy(logits_indices,
                                                       self.device)
        # Dummy run.
        htorch.core.mark_step()
        logits = self._execute_model_generic(input_ids_device,
                                             position_ids_device,
                                             attn_metadata,
                                             logits_indices_device, kv_caches,
                                             True)
        # TODO: do sampling on logits, warmup sampler and prefill joiner
        htorch.core.mark_step()
        temperature = torch.ones(batch_size, dtype=torch.float32, device='cpu')
        top_p = torch.ones(batch_size, dtype=torch.float32, device='cpu')
        top_k = torch.ones(batch_size, dtype=torch.float32, device='cpu')
        temperature_device = _async_h2d_tensor_copy(temperature, self.device)
        top_p_device = _async_h2d_tensor_copy(top_p, self.device)
        top_k_device = _async_h2d_tensor_copy(top_k, self.device)
        generators = {
            i: None
            for i in range(batch_size)
        }  # NOTE(kzawora): idk what to set here
        max_num_logprobs = 0  # NOTE(kzawora): idk what to set here
        # NOTE(kzawora: do this in a smarter way)
        return None
        htorch.core.mark_step()
        sampling_metadata = SamplingMetadata(
            temperature=temperature_device,
            all_greedy=False,  # hacky
            all_random=True,  # hacky
            top_p=top_p_device,
            top_k=top_k_device,
            no_top_p=True,
            no_top_k=True,
            generators=generators,
            max_num_logprobs=max_num_logprobs,
        )
        tokens_all_random = self.model.sample(logits, sampling_metadata)
        htorch.core.mark_step()
        sampling_metadata = SamplingMetadata(
            temperature=temperature_device,
            all_greedy=True,  # hacky
            all_random=False,  # hacky
            top_p=top_p_device,
            top_k=top_k_device,
            no_top_p=True,
            no_top_k=True,
            generators=generators,
            max_num_logprobs=max_num_logprobs,
        )
        tokens_all_greedy = self.model.sample(logits, sampling_metadata)
        htorch.core.mark_step()
        sampling_metadata = SamplingMetadata(
            temperature=temperature_device,
            all_greedy=False,  # hacky
            all_random=False,  # hacky
            top_p=top_p_device,
            top_k=top_k_device,
            no_top_p=True,
            no_top_k=True,
            generators=generators,
            max_num_logprobs=max_num_logprobs,
        )
        tokens_mixed = self.model.sample(logits, sampling_metadata)
        htorch.core.mark_step()
        return tokens_all_random, tokens_all_greedy, tokens_mixed

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

    def warmup_all_buckets(self, buckets, is_prompt, kv_caches):
        for i, (batch_size, seq_len) in enumerate(reversed(buckets)):
            self.log_warmup('Prompt' if is_prompt else 'Decode', i,
                            len(buckets), batch_size, seq_len)
            self.warmup_scenario(batch_size, seq_len, is_prompt, kv_caches)
            torch.hpu.synchronize()

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
        ordering : Union[Callable[[Any], tuple[Any, Any]], \
            Callable[[Any], tuple[Any, Any, Any]]]
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
                self.warmup_scenario(batch_size, seq_len, is_prompt, kv_caches)
            #TODO(kzawora): align_workers
            used_mem = mem_prof.consumed_device_memory
            available_mem -= used_mem
            total_mem += used_mem
            total_batch_seq += batch_seq

        return total_mem, total_batch_seq, captured_all

    @torch.inference_mode()
    def warmup_model(self) -> None:
        kv_caches = self.kv_caches
        if profile := os.environ.get('VLLM_PT_PROFILE', None):
            phase, bs, seq_len, graph = profile.split('_')
            is_prompt = phase == 'prompt'
            graphs = graph == 't'
            if graphs:
                self.graphed_buckets.add((int(bs), int(seq_len), is_prompt))
            #self.warmup_scenario(int(bs), int(seq_len), is_prompt, kv_caches,
            #                     True)
            raise AssertionError("Finished profiling")
        if self.skip_warmup:
            logger.info("Skipping warmup...")
            return
        max_blocks = kv_caches[0][0].size(0)
        self.bucketing_ctx.generate_decode_buckets(max_blocks)

        if not htorch.utils.internal.is_lazy(
        ) and not self.model_config.enforce_eager:
            cache_size_limit = len(self.bucketing_ctx.prompt_buckets) + len(
                self.bucketing_ctx.decode_buckets) + 1
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
            self.warmup_all_buckets(self.bucketing_ctx.prompt_buckets, True,
                                    kv_caches)
            self.warmup_all_buckets(self.bucketing_ctx.decode_buckets, False,
                                    kv_caches)

            if (not self.model_config.enforce_eager
                    and htorch.utils.internal.is_lazy()):
                assert self.mem_margin is not None, \
                    ("HabanaWorker.determine_num_available_blocks needs "
                    "to be called before warming up the model.")
                free_mem = HabanaMemoryProfiler.current_free_device_memory()
                graph_free_mem = free_mem - self.mem_margin
                #TODO(kzawora): align_workers
                graph_free_mem = graph_free_mem
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
                    prompt_strategy, self.bucketing_ctx.prompt_buckets,
                    True, kv_caches, prompt_available_memory)
                mem_post_decode, decode_batch_seq, decode_captured_all = \
                    self.warmup_graphs(
                    decode_strategy, self.bucketing_ctx.decode_buckets,
                    False, kv_caches, decode_available_memory)

                # Not all prompt buckets were captured, but all decode buckets
                # were captured and we have some free graph-allocated space
                # left. Let's try to use it for capturing more prompt buckets.
                if (mem_post_decode + mem_post_prompt < graph_free_mem
                        and not prompt_captured_all and decode_captured_all):
                    mem_post_prompt, _, prompt_captured_all = (
                        self.warmup_graphs(
                            prompt_strategy, self.bucketing_ctx.prompt_buckets,
                            True, kv_caches,
                            graph_free_mem - mem_post_prompt - mem_post_decode,
                            mem_post_prompt, prompt_batch_seq))

                # Not all decode buckets were captured, but all prompt buckets
                # were captured and we have some free graph-allocated space
                # left. Let's try to use it for capturing more decode buckets.
                if mem_post_decode + mem_post_prompt < graph_free_mem \
                    and not decode_captured_all \
                        and prompt_captured_all:
                    mem_post_decode, _, _ = self.warmup_graphs(
                        decode_strategy, self.bucketing_ctx.decode_buckets,
                        False, kv_caches,
                        graph_free_mem - mem_post_prompt - mem_post_decode,
                        mem_post_decode, decode_batch_seq)

                self.log_graph_warmup_summary(
                    self.bucketing_ctx.prompt_buckets, True, mem_post_prompt)
                self.log_graph_warmup_summary(
                    self.bucketing_ctx.decode_buckets, False, mem_post_decode)

        end_time = time.perf_counter()
        end_mem = HabanaMemoryProfiler.current_device_memory_usage()
        elapsed_time = end_time - start_time
        msg = (
            f"Warmup finished in {elapsed_time:.0f} secs, "
            f"allocated {format_bytes(end_mem - start_mem)} of device memory")
        logger.info(msg)

    @torch.inference_mode()
    def profile_run(self) -> None:
        return
        """Profile to measure peak memory during forward pass."""

        # use an empty tensor instead of `None`` to force Dynamo to pass
        # it by reference, rather by specializing on the value `None`.
        # the `dtype` argument does not matter, and we use `float32` as
        # a placeholder (it has wide hardware support).
        # it is important to create tensors inside the loop, rather than
        # multiplying the list, to avoid Dynamo from treating them as
        # tensor aliasing.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers

        # Run empty prefill forwards - prefill max batch and prefill max seq
        self.warmup_scenario(batch_size=1,
                             seq_or_block=self.max_model_len,
                             is_prompt=True,
                             kv_caches=kv_caches)
        max_seq_len = math.ceil(
            (self.max_num_tokens // self.max_prefill_batch_size) /
            self.block_size) * self.block_size
        self.warmup_scenario(batch_size=self.max_prefill_batch_size,
                             seq_or_block=max_seq_len,
                             is_prompt=True,
                             kv_caches=kv_caches)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        if len(kv_cache_config.kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")

        kv_caches: dict[str, torch.Tensor] = {}

        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_config = kv_cache_config.tensors[layer_name]
                assert tensor_config.size % kv_cache_spec.page_size_bytes == 0
                num_blocks = tensor_config.size // kv_cache_spec.page_size_bytes
                # `num_blocks` is the number of blocks the model runner can use.
                # `kv_cache_config.num_blocks` is the number of blocks that
                # KVCacheManager may allocate.
                # Since different GPUs may have different number of layers and
                # different memory capacities, `num_blocks` can be different on
                # different GPUs, and `kv_cache_config.num_blocks` is set to
                # the min of all `num_blocks`. Verify it here.
                assert num_blocks >= kv_cache_config.num_blocks
                if isinstance(kv_cache_spec, FullAttentionSpec):
                    kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                        num_blocks + 1, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    key_cache = torch.zeros(kv_cache_shape,
                                            dtype=dtype,
                                            device=self.device)
                    value_cache = torch.zeros_like(key_cache)
                    kv_caches[layer_name] = (key_cache, value_cache)
                else:
                    # TODO: add new branches when introducing more types of
                    # KV cache specs.
                    raise ValueError("Unknown KV cache spec type.")

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)

        if self.enable_bucketing:
            self.bucketing_ctx.num_hpu_blocks = num_blocks
        self._PAD_BLOCK_ID = num_blocks
        self._PAD_SLOT_ID = num_blocks * self.block_size

        htorch.hpu.synchronize()
