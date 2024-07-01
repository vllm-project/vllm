###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

import time
from enum import IntEnum
from typing import List, NamedTuple, Optional, Set, Tuple, Dict

import collections
import gc
import os
import math
import itertools
import operator
import torch
import habana_frameworks.torch as htorch
import contextlib
from vllm.attention import (AttentionMetadata, get_attn_backend)
from vllm.config import (DeviceConfig, LoadConfig, CacheConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VisionLanguageConfig)
from vllm.distributed import broadcast_tensor_dict
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.model_loader import get_model
from vllm.sampling_params import SamplingParams
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.utils import (HabanaMemoryProfiler, is_pin_memory_available,
                        make_tensor_with_pad, format_bytes)

from .profiler import Profiler

logger = init_logger(__name__)

_PAD_SLOT_ID = 0
LORA_WARMUP_RANK = 8
_TYPE_CACHE = {}


# Read bucketing configuration from env variables
# phase is either 'prompt' or 'decode'
# dim is either 'bs' or 'seq'
# param is either 'min', 'step' or 'max'
# example env variable: VLLM_DECODE_BS_BUCKET_STEP=128
def read_bucket_settings(phase: str, dim: str, **defaults: Dict):
    params = ['min', 'step', 'max']
    values = [int(os.environ.get(f'VLLM_{phase}_{dim}_BUCKET_{p}'.upper(), defaults[p])) for p in params]
    return values


def warmup_range(config: Tuple[int, int, int]):
    bmin, bstep, bmax = config
    base = itertools.repeat(2)
    ramp_up = itertools.accumulate(base, func=operator.mul, initial=bmin)
    ramp_up = itertools.takewhile(lambda x: x < bstep and x <= bmax, ramp_up)
    stable = range(bstep, bmax + 1, bstep)
    return list(ramp_up) + list(stable)


def warmup_buckets(bs_bucket_config, seq_bucket_config):
    buckets = itertools.product(warmup_range(bs_bucket_config), warmup_range(seq_bucket_config))
    return list(sorted(buckets, key=lambda b: (b[0] * b[1], b[1], b[0])))


def next_pow2(value: int):
    res = 1
    while value > 1:
        value = (value + 1) // 2
        res *= 2
    return res


def round_up(value: int, k: int):
    return (value + k - 1) // k * k


def find_bucket(value: int, config: Tuple[int, int, int]):
    bmin, bstep, bmax = config
    if value < bstep:
        result = min(next_pow2(value), bstep)
    else:
        result = round_up(value, bstep)
    return result


def subtuple(obj: object, typename: str, to_copy: List[str], to_override: Dict[str, object] = {}):
    if obj is None:
        return None
    fields = set(to_copy) | set(to_override.keys())
    values = {f: to_override.get(f, getattr(obj, f)) for f in fields}
    if typename not in _TYPE_CACHE:
        _TYPE_CACHE[typename] = collections.namedtuple(typename, ' '.join(fields))
    return _TYPE_CACHE[typename](**values)


def align_workers(value, op):
    group = get_world_group().cpu_group
    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        return value
    value_t = torch.tensor(value, device='cpu')
    torch.distributed.all_reduce(value_t, op=op, group=group)
    return value_t.item()


class HpuModelAdapter():
    def __init__(self, model):
        self.model = model

    def _set_attn_bias(self, attn_metadata, batch_size, seq_len, device, dtype):
        prefill_metadata = attn_metadata
        if prefill_metadata is None:
            return attn_metadata

        seq_lens_t = prefill_metadata.seq_lens_tensor
        len_mask = (torch.arange(0, seq_len, device=device, dtype=torch.int32)
                    .view(1, seq_len)
                    .ge(seq_lens_t.unsqueeze(-1))
                    .view(batch_size, 1, 1, seq_len))
        causal_mask = torch.triu(
            torch.ones((batch_size, 1, seq_len, seq_len), device=device, dtype=torch.bool),
            diagonal=1
        )
        mask = causal_mask.logical_or(len_mask)
        attn_bias = (torch.zeros_like(mask, dtype=dtype)
                      .masked_fill_(mask, -math.inf))
        #FIXME: Restore sliding window support
        #if self.sliding_window is not None:
        attn_metadata = prefill_metadata._replace(attn_bias=attn_bias)
        return attn_metadata


    def forward(self, *args, **kwargs):
        kwargs = kwargs.copy()
        selected_token_indices = kwargs.pop('selected_token_indices')
        if 'bypass_hpu_graphs' in kwargs:
            kwargs.pop('bypass_hpu_graphs') # required for PT eager
        input_ids = kwargs['input_ids']
        kwargs['attn_metadata'] = self._set_attn_bias(kwargs['attn_metadata'], input_ids.size(0), input_ids.size(1), input_ids.device, torch.bfloat16)
        hidden_states = self.model(*args, **kwargs)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = hidden_states.index_select(0, selected_token_indices)
        return hidden_states

    def compute_logits(self, *args, **kwargs):
        return self.model.compute_logits(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)


class PreparePromptMetadata(NamedTuple):
    input_tokens: List[int]
    input_positions: List[int]
    attn_metadata: Optional[AttentionMetadata]
    seq_lens: List[int]
    query_lens: List[int]
    lora_index_mapping: List[int]
    lora_prompt_mapping: List[int]
    lora_requests: Set[LoRARequest]
    multi_modal_input: Optional[torch.Tensor]
    slot_mapping: List[int]

    @classmethod
    def empty(cls):
        return PreparePromptMetadata(
            input_tokens=[],
            input_positions=[],
            attn_metadata=None,
            seq_lens=[],
            query_lens=[],
            lora_index_mapping=[],
            lora_prompt_mapping=[],
            lora_requests=set(),
            multi_modal_input=None,
            slot_mapping=[],
        )


class PrepareDecodeMetadata(NamedTuple):
    input_tokens: List[int]
    input_positions: List[int]
    attn_metadata: Optional[AttentionMetadata]
    lora_index_mapping: List[int]
    lora_prompt_mapping: List[int]
    lora_requests: Set[LoRARequest]
    slot_mapping: List[int]

    @classmethod
    def empty(cls):
        return PrepareDecodeMetadata(
            input_tokens=[],
            input_positions=[],
            attn_metadata=None,
            lora_index_mapping=[],
            lora_prompt_mapping=[],
            lora_requests=set(),
            slot_mapping=[],
        )


# How batches are constructed.
class BatchType(IntEnum):
    # Every batch is prefill.
    PREFILL = 0
    # Every batch is decode.
    DECODE = 1
    # Batch is a mixture of prefill and decode.
    MIXED = 2


class HabanaModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        vision_language_config: Optional[VisionLanguageConfig] = None,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.lora_config = lora_config
        self.load_config = load_config
        self.cache_config = cache_config
        self.is_driver_worker = is_driver_worker
        self.profiler = Profiler()

        self.sliding_window = (model_config.get_sliding_window()
                               if model_config is not None else None)
        self.device_config = (device_config
                              if device_config is not None else DeviceConfig())

        self.device = self.device_config.device
        self.enforce_eager = self.model_config.enforce_eager
        self.max_num_seqs = self.scheduler_config.max_num_seqs
        self.max_model_len = self.scheduler_config.max_model_len
        self.max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        self.block_size = cache_config.block_size

        self.pin_memory = is_pin_memory_available()
        self.kv_cache_dtype = kv_cache_dtype
        self.vision_language_config = vision_language_config

        self.attn_backend = get_attn_backend(
            self.model_config.get_num_attention_heads(self.parallel_config),
            self.model_config.get_head_size(),
            self.model_config.get_num_kv_heads(self.parallel_config),
            self.model_config.get_sliding_window(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
        )

        # Lazy initialization
        self.lora_manager: LRUCacheWorkerLoRAManager = None
        self.model: torch.nn.Module = None

        # Profiler stats
        self.profiler_counter_helper = HabanaProfilerCounterHelper()

        self._setup_buckets()

    def load_model(self) -> None:
        with HabanaMemoryProfiler() as m:
            with HabanaMemoryProfiler() as m_getmodel:
                self.model = get_model(
                    model_config=self.model_config,
                    device_config=self.device_config,
                    load_config=self.load_config,
                    lora_config=self.lora_config,
                    vision_language_config=self.vision_language_config,
                    parallel_config=self.parallel_config,
                    scheduler_config=self.scheduler_config,
                    cache_config=self.cache_config
                )
            logger.info(f"Pre-loading model weights on {next(self.model.parameters()).device} took {m_getmodel.get_summary_string()}")

            # FIXME: Running with disable_tensor_cache=True causes RuntimeErrors. This needs to be debugged
            with HabanaMemoryProfiler() as m_wrap:
                self.model = _maybe_wrap_in_hpu_graph(self.model)
            logger.info(f"Wrapping in HPU Graph took {m_wrap.get_summary_string()}")
            
        self.model_memory_usage = m.consumed_device_memory
        logger.info(f"Loading model weights took in total {m.get_summary_string()}")

        if self.lora_config:
            assert hasattr(self.model, "supported_lora_modules"
                           ) and self.model.supported_lora_modules, (
                               "Model does not support LoRA")
            assert hasattr(
                self.model,
                "embedding_modules"), "Model does not have embedding_modules"
            assert hasattr(self.model, "embedding_padding_modules"
                           ), "Model does not have embedding_padding_modules"
            self.lora_manager = LRUCacheWorkerLoRAManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens, self.vocab_size,
                self.lora_config, self.device, self.model.embedding_modules,
                self.model.embedding_padding_modules)
            self.model = self.lora_manager.create_lora_manager(self.model)

    def _use_graphs(self, batch_size, seq_len, is_prompt):
        if self.enforce_eager:
            return False
        return (batch_size, seq_len, is_prompt) in self.graphed_buckets

    def _setup_buckets(self) -> None:
        self.prompt_bs_bucket_cfg = read_bucket_settings('prompt', 'bs', min=1, step=32, max=min(self.max_num_seqs, 64))
        self.decode_bs_bucket_cfg = read_bucket_settings('decode', 'bs', min=1, step=128, max=self.max_num_seqs)
        self.prompt_seq_bucket_cfg = read_bucket_settings('prompt', 'seq', min=self.block_size, step=self.block_size, max=1024)
        self.decode_seq_bucket_cfg = read_bucket_settings('decode', 'seq', min=self.block_size, step=self.block_size, max=2048)
        self.graphed_buckets = set()

        logger.info(f"Prompt bucket config (min, step, max_warmup) bs:{self.prompt_bs_bucket_cfg}, seq:{self.prompt_seq_bucket_cfg}")
        self.prompt_buckets = warmup_buckets(self.prompt_bs_bucket_cfg, self.prompt_seq_bucket_cfg)
        logger.info(f"Generated {len(self.prompt_buckets)} prompt buckets: {list(sorted(self.prompt_buckets))}")

        logger.info(f"Decode bucket config (min, step, max_warmup) bs:{self.decode_bs_bucket_cfg}, seq:{self.decode_seq_bucket_cfg}")
        self.decode_buckets = warmup_buckets(self.decode_bs_bucket_cfg, self.decode_seq_bucket_cfg)
        logger.info(f"Generated {len(self.decode_buckets)} decode buckets: {list(sorted(self.decode_buckets))}")

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
        multi_modal_input_list: List[torch.Tensor] = []

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
            lora_id = seq_group_metadata.lora_int_id

            if lora_id > 0:
                lora_requests.add(seq_group_metadata.lora_request)

            lora_index_mapping += [lora_id] * (seq_len - context_len)
            lora_prompt_mapping.append(
                [lora_id] *
                (seq_len - context_len
                 if seq_group_metadata.sampling_params.prompt_logprobs else 1))

            if seq_group_metadata.multi_modal_data:
                multi_modal_input_list.append(
                    seq_group_metadata.multi_modal_data.data)

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

        context_lens_tensor = torch.tensor(context_lens,
                                           dtype=torch.int,
                                           device=self.device)

        if multi_modal_input_list:
            assert self.vision_language_config, (
                "Multi-modal inputs are only supported by "
                "vision language models.")
            multi_modal_input = torch.cat(multi_modal_input_list,
                                          dim=0).to(self.device)
        else:
            multi_modal_input = None

        max_prompt_block_table_len = max(len(t) for t in prefix_block_tables)
        max_prompt_len = max(find_bucket(max(seq_lens), self.prompt_seq_bucket_cfg), self.block_size)

        input_tokens = make_tensor_with_pad(input_tokens,
                                            max_prompt_len,
                                            pad=0,
                                            dtype=torch.long,
                                            device=self.device)

        input_positions = make_tensor_with_pad(input_positions,
                                               max_prompt_len,
                                               pad=0,
                                               dtype=torch.long,
                                               device=self.device)

        slot_mapping = make_tensor_with_pad(slot_mapping,
                                            max_prompt_len,
                                            pad=_PAD_SLOT_ID,
                                            dtype=torch.long,
                                            device=self.device)

        block_tables = make_tensor_with_pad(prefix_block_tables,
                                            max_len=max_prompt_block_table_len,
                                            pad=0,
                                            dtype=torch.int,
                                            device=self.device)

        # Query length can be shorter than key (i.e., prompt) when prefill
        # is chunked or prefix cached.
        query_lens_tensor = torch.tensor(query_lens,
                                         dtype=torch.long,
                                         device=self.device)
        subquery_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                         dtype=torch.int32,
                                         device=self.device)
        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.long,
                                       device=self.device)
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=self.device)

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=True,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            subquery_start_loc=subquery_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            num_prefills=real_num_seqs,
            num_prefill_tokens=sum_query_len,
            num_decode_tokens=0, 
            slot_mapping=slot_mapping,
        )
        return PreparePromptMetadata(
            input_tokens=input_tokens,
            input_positions=input_positions,
            attn_metadata=attn_metadata,
            seq_lens=seq_lens,
            query_lens=query_lens,
            lora_index_mapping=lora_index_mapping,
            lora_prompt_mapping=lora_prompt_mapping,
            lora_requests=lora_requests,
            multi_modal_input=multi_modal_input,
            slot_mapping=slot_mapping,
        )

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> PrepareDecodeMetadata:
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []
        seq_lens: List[int] = []
        block_tables: List[List[int]] = []
        lora_index_mapping: List[int] = []
        lora_prompt_mapping: List[int] = []
        lora_requests: Set[LoRARequest] = set()

        if len(seq_group_metadata_list) == 0:
            return PrepareDecodeMetadata.empty()

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            assert seq_group_metadata.token_chunk_size == 1

            seq_ids = list(seq_group_metadata.seq_data.keys())
            lora_id = seq_group_metadata.lora_int_id

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
                block_number = block_table[position // self.block_size]
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
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=self.device)
        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.int,
                                       device=self.device)
        num_decode_tokens = sum(seq_lens)
        max_block_table_len = max(
            len(block_table) for block_table in block_tables)
        block_tables = make_tensor_with_pad(
            block_tables,
            max_len=max_block_table_len,
            pad=0,
            dtype=torch.int,
            device=self.device,
        )
        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=False,
            seq_lens=None,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=None,
            subquery_start_loc=None,
            seq_start_loc=None,
            context_lens_tensor=None,
            block_tables=block_tables,
            use_cuda_graph=False,
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=num_decode_tokens,
            slot_mapping=slot_mapping,
        )
        return PrepareDecodeMetadata(
            input_tokens=input_tokens,
            input_positions=input_positions,
            attn_metadata=attn_metadata,
            lora_index_mapping=lora_index_mapping,
            lora_prompt_mapping=lora_prompt_mapping,
            lora_requests=lora_requests,
            slot_mapping=slot_mapping,
        )

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, SamplingMetadata,
               Set[LoRARequest], LoRAMapping, torch.Tensor]:
        if self.is_driver_worker:
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
                multi_modal_input,
                slot_mapping,
            ) = self._prepare_prompt(prefill_reqs)
            (
                decode_input_tokens,
                decode_input_positions,
                decode_attn_metadata,
                decode_lora_index_mapping,
                decode_lora_prompt_mapping,
                decode_lora_requests,
                decode_slot_mapping,
            ) = self._prepare_decode(decode_reqs)
            sampling_metadata = SamplingMetadata.prepare(
                seq_group_metadata_list, seq_lens, query_lens, self.device,
                self.pin_memory)

            if not self.scheduler_config.chunked_prefill_enabled:
                assert (len(prefill_reqs) and len(decode_reqs)) == 0

            num_prefills = len(seq_lens)
            num_prefill_tokens = len(input_tokens)
            num_decode_tokens = len(decode_input_tokens)

            # NOTE(kzawora): Here we diverge from GPU code - we don't support mixed batches, so we either use decode or prefill inputs, without coalescing.
            assert (num_prefills == 0 and num_decode_tokens > 0) or (num_prefills > 0 and num_decode_tokens == 0), "HPU does not support mixed batches!"
            if num_decode_tokens > 0:
                input_tokens = decode_input_tokens
                input_positions = decode_input_positions
                slot_mapping = decode_slot_mapping
                lora_index_mapping = decode_lora_index_mapping
                lora_prompt_mapping = decode_lora_prompt_mapping
                lora_requests = decode_lora_requests

            # FIXME: We need to adjust selected_token_indices to accomodate for padding
            max_len = input_tokens.size(1)
            paddings = [max_len - s for s in seq_lens]
            paddings = [0] + paddings[:-1]
            paddings = list(itertools.accumulate(paddings))
            paddings = torch.tensor(paddings, dtype=sampling_metadata.selected_token_indices.dtype, device=sampling_metadata.selected_token_indices.device)
            sampling_metadata.selected_token_indices.add_(paddings)

            if self.lora_config:
                lora_mapping = LoRAMapping(
                    lora_index_mapping,
                    lora_prompt_mapping,
                )
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
                "selected_token_indices":
                sampling_metadata.selected_token_indices,
                "lora_requests": lora_requests,
                "lora_mapping": lora_mapping,
                "multi_modal_input": multi_modal_input,
                "num_prefill_tokens": num_prefill_tokens,
                "num_decode_tokens": num_decode_tokens,
                "slot_mapping": slot_mapping,
                "num_prefills": num_prefills,
                "batch_type": batch_type,
            }
            if prefill_attn_metadata is not None:
                metadata_dict.update(prefill_attn_metadata.asdict_zerocopy())
            else:
                assert decode_attn_metadata is not None
                metadata_dict.update(decode_attn_metadata.asdict_zerocopy())
            broadcast_tensor_dict(metadata_dict, src=0)

            # Broadcast decode attn metadata for mixed batch type.
            # The additional broadcast costs 300us overhead on 4 A10 GPUs.
            # We can potentially reduce the overhead by coelescing tensors.
            if batch_type == BatchType.MIXED:
                assert decode_attn_metadata is not None
                metadata_dict = decode_attn_metadata.asdict_zerocopy()
                broadcast_tensor_dict(metadata_dict, src=0)
        else:
            metadata_dict = broadcast_tensor_dict(src=0)
            input_tokens = metadata_dict.pop("input_tokens")
            input_positions = metadata_dict.pop("input_positions")
            selected_token_indices = metadata_dict.pop(
                "selected_token_indices")
            lora_mapping = metadata_dict.pop("lora_mapping")
            lora_requests = metadata_dict.pop("lora_requests")
            multi_modal_input = metadata_dict.pop("multi_modal_input")
            batch_type = metadata_dict.pop("batch_type")

            # Create an attention metadata.
            prefill_attn_metadata = None
            decode_attn_metadata = None
            if batch_type == BatchType.PREFILL or batch_type == BatchType.MIXED:
                prefill_attn_metadata = self.attn_backend.make_metadata(
                    **metadata_dict)
            else:
                decode_attn_metadata = self.attn_backend.make_metadata(
                    **metadata_dict)
            sampling_metadata = SamplingMetadata(
                seq_groups=None,
                selected_token_indices=selected_token_indices,
                categorized_sample_indices=None,
                num_prompts=0,
            )

            # if it is a mixed batch, decode attn_metadata is broadcasted
            # separately.
            if batch_type == BatchType.MIXED:
                metadata_dict = broadcast_tensor_dict(src=0)
                decode_attn_metadata = self.attn_backend.make_metadata(
                    **metadata_dict)

        attn_metadata = prefill_attn_metadata if prefill_attn_metadata is not None else decode_attn_metadata
#        attn_metadata = AttentionMetadata(
#            num_prefills=num_prefills,
#            slot_mapping=slot_mapping,
#            num_prefill_tokens=num_prefill_tokens,
#            num_decode_tokens=num_decode_tokens,
#            prefill_metadata=prefill_attn_metadata,
#            decode_metadata=decode_attn_metadata,
#            kv_cache_dtype=self.kv_cache_dtype,
#        )

        return (input_tokens, input_positions, attn_metadata,
                sampling_metadata, lora_requests, lora_mapping,
                multi_modal_input)

    def _seq_len(self, attn_metadata):
        if attn_metadata.num_prefills != 0:
            return attn_metadata.slot_mapping.size(1)
        else:
            return attn_metadata.block_tables.size(1) * self.block_size

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
        attention_metadata = subtuple(metadata,
                                      'TrimmedAttentionMetadata',
                                      ['block_tables',
                                       'seq_lens_tensor',
                                       'attn_bias',
                                       'slot_mapping',
                                       'is_prompt'])
        return attention_metadata

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[torch.Tensor],
    ) -> Optional[SamplerOutput]:
        if self.is_driver_worker:
            event_start = self.profiler.get_timestamp_us()
            is_prompt = seq_group_metadata_list[0].is_prompt
            base_event_name = 'prompt' if is_prompt else 'decode'
            self.profiler.start('internal', base_event_name)

            real_batch_size = len(seq_group_metadata_list)
            bucket_cfg = self.prompt_bs_bucket_cfg if is_prompt else self.decode_bs_bucket_cfg
            batch_size_padded = find_bucket(real_batch_size, bucket_cfg)
            batch_size_padding = batch_size_padded - real_batch_size
            seq_group_metadata_list = seq_group_metadata_list.copy()
            seq_group_metadata_list.extend(seq_group_metadata_list[0] for _ in range(batch_size_padding))
        with self.profiler.record_event('internal', 'prepare_input_tensors'):
            (input_tokens, input_positions, attn_metadata, sampling_metadata,
            lora_requests, lora_mapping, multi_modal_input
            ) = self.prepare_input_tensors(seq_group_metadata_list)
            is_prompt = attn_metadata.prefill_metadata is not None

        if self.lora_config:
            self.set_active_loras(lora_requests, lora_mapping)

        batch_size = input_tokens.size(0)
        seq_len = self._seq_len(attn_metadata)
        use_graphs = self._use_graphs(batch_size, seq_len, is_prompt)
        execute_model_kwargs = {
            "input_ids": input_tokens,
            "positions": input_positions,
            "kv_caches": kv_caches,
            "attn_metadata": self.trim_attn_metadata(attn_metadata),
        }
        if self.vision_language_config:
            execute_model_kwargs.update({"image_input": multi_modal_input})

        htorch.core.mark_step()
        if self.is_driver_worker:
            model_event_name = f"model_{'prompt' if is_prompt else 'decode'}_bs{batch_size}_seq{seq_len}_graphs{'T' if use_graphs else 'F'}"
        else:
            model_event_name = 'model_executable'
        with self.profiler.record_event('internal', model_event_name):
            hidden_states = self.model.forward(**execute_model_kwargs, selected_token_indices=sampling_metadata.selected_token_indices, bypass_hpu_graphs=not use_graphs)

        # Compute the logits.
        with self.profiler.record_event('internal', f'compute_logits_{"prompt" if is_prompt else "decode"}_bs{batch_size}_seq{seq_len}'):
            sampling_metadata.selected_token_indices = None
            logits = self.model.compute_logits(hidden_states, sampling_metadata)
        htorch.core.mark_step()

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return None

        # Sample the next token.
        with self.profiler.record_event('internal', f'sample_{"prompt" if is_prompt else "decode"}_bs{batch_size}_seq{seq_len}'):
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
                duration=event_end-event_start, 
                seq_len=seq_len, 
                batch_size_padded=batch_size_padded, 
                real_batch_size=real_batch_size, 
                seq_group_metadata_list=seq_group_metadata_list, 
                is_prompt=is_prompt)
            self.profiler.record_counter(event_start, counters)

        return output

    def create_dummy_seq_group_metadata(self, group_id, seq_len, is_prompt):
        sampling_params = SamplingParams(temperature=0)
        num_blocks = math.ceil(seq_len / self.block_size)
        if is_prompt:
            input_len = seq_len
            output_len = 0
            block_tables = None
        else:
            input_len = seq_len - 1
            output_len = 1
            block_tables = {group_id: [0] * num_blocks}
        prompt_token_ids = [0] * input_len
        output_token_ids = [1] * output_len
        seq_data = SequenceData(prompt_token_ids)
        seq_data.output_token_ids = output_token_ids
        return SequenceGroupMetadata(
            request_id=str(group_id),
            is_prompt=(output_len == 0),
            seq_data={group_id: seq_data},
            sampling_params=sampling_params,
            block_tables=block_tables,
        )

    def profile_run(self) -> None:
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers
        max_batch_size = self.prompt_bs_bucket_cfg[-1]
        max_seq_len = self.prompt_seq_bucket_cfg[-1]

        self.warmup_scenario(max_batch_size, max_seq_len, True, kv_caches)

    def warmup_scenario(self, batch_size, seq_len, is_prompt, kv_caches) -> None:
        use_graphs = self._use_graphs(batch_size, seq_len, is_prompt)
        scenario_name = f"warmup_{'prompt' if is_prompt else 'decode'}_bs{batch_size}_seq{seq_len}_graphs{'T' if use_graphs else 'F'}"
        self.profiler.start('internal', scenario_name)
        times = 3 if use_graphs else 1
        seqs = [self.create_dummy_seq_group_metadata(i, seq_len, is_prompt) for i in range(batch_size)]
        torch.hpu.synchronize()
        for _ in range(times):
            self.execute_model(seqs, kv_caches)
            torch.hpu.synchronize()
        self.profiler.end()
        gc.collect()

    def log_warmup(self, phase, i, max_i, batch_size, seq_len):
        free_mem = format_bytes(HabanaMemoryProfiler.current_free_device_memory())
        logger.info(f"[Warmup][{phase}][{i+1}/{max_i}] batch_size:{batch_size} seq_len:{seq_len} free_mem:{free_mem}")

    def warmup_all_buckets(self, buckets, is_prompt, kv_caches):
        for i, (batch_size, seq_len) in enumerate(reversed(buckets)):
            mem_usage = 100.0 * HabanaMemoryProfiler.current_device_memory_usage() / HabanaMemoryProfiler.total_device_memory()
            self.log_warmup('Prompt' if is_prompt else 'Decode', i, len(buckets), batch_size, seq_len)
            self.warmup_scenario(batch_size, seq_len, is_prompt, kv_caches)

    def warmup_graphs(self, strategy, buckets, is_prompt, kv_caches, available_mem):
        total_batch_seq = 0.001
        total_mem = 0
        idx = 0
        phase = f'Graph/{"Prompt" if is_prompt else "Decode"}'
        num_candidates = len(buckets)

        if strategy == 'min_tokens':
            ordering = lambda b: (b[0] * b[1], b[1], b[0])
        elif strategy == 'max_bs':
            ordering = lambda b: (-b[0], b[1])
        else:
            raise NotImplementedError(f'Unsupported graph allocation strategy: {strategy}')
        buckets = list(sorted(buckets, key=ordering))

        for idx, (batch_size, seq_len) in enumerate(buckets):
            # Graph memory usage is proportional to seq dimension in a batch
            batch_seq = batch_size * seq_len if is_prompt else batch_size
            mem_estimate = batch_seq / total_batch_seq * total_mem
            if mem_estimate >= available_mem:
                continue
            self.graphed_buckets.add((batch_size, seq_len, is_prompt))
            self.log_warmup(phase, idx, num_candidates, batch_size, seq_len)
            with HabanaMemoryProfiler() as mem_prof:
                self.warmup_scenario(batch_size, seq_len, is_prompt, kv_caches)
            used_mem = align_workers(mem_prof.consumed_device_memory, torch.distributed.ReduceOp.MAX)
            available_mem -= used_mem
            total_mem += used_mem
            total_batch_seq += batch_seq
        graphed = list(c[:2] for c in self.graphed_buckets if c[2] == is_prompt)
        logger.info(f'{phase} captured:{len(graphed)} ({100 * len(graphed) / num_candidates:.1f}%) used_mem:{format_bytes(total_mem)} buckets:{sorted(list(graphed))}')

    @torch.inference_mode()
    def warmup_model(self, kv_caches: List[torch.Tensor]) -> None:
        if os.environ.get('VLLM_SKIP_WARMUP', 'false').lower() == 'true':
            logger.info("Skipping warmup...")
            return
        self.profiler.start('internal', 'warmup')
        start_mem = HabanaMemoryProfiler.current_device_memory_usage()
        start_time = time.perf_counter()
        self.warmup_all_buckets(self.prompt_buckets, True, kv_caches)
        self.warmup_all_buckets(self.decode_buckets, False, kv_caches)

        if not self.enforce_eager:
            mem_margin = 1.0 - float(os.environ.get('VLLM_GRAPH_MEM_MARGIN', '0.02'))
            free_mem = mem_margin * HabanaMemoryProfiler.current_free_device_memory()
            free_mem = align_workers(free_mem, torch.distributed.ReduceOp.MIN)
            prompt_graph_mem_ratio = float(os.environ.get('VLLM_GRAPH_PROMPT_RATIO', '0.5'))
            prompt_available_memory = prompt_graph_mem_ratio * free_mem
            decode_available_memory = free_mem - prompt_available_memory
            prompt_strategy = 'min_tokens'
            decode_strategy = os.environ.get('VLLM_GRAPH_DECODE_STRATEGY', 'max_bs')
            self.warmup_graphs(prompt_strategy, self.prompt_buckets, True, kv_caches, prompt_available_memory)
            self.warmup_graphs(decode_strategy, self.decode_buckets, False, kv_caches, decode_available_memory)

        end_time = time.perf_counter()
        end_mem = HabanaMemoryProfiler.current_device_memory_usage()
        elapsed_time = end_time - start_time
        logger.info(f"Warmup finished in {elapsed_time:.0f} secs, allocated {format_bytes(end_mem - start_mem)} of device memory")
        self.profiler.end()

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()

def _maybe_wrap_in_hpu_graph(model):
    return htorch.hpu.wrap_in_hpu_graph(HpuModelAdapter(model)) if htorch.utils.internal.is_lazy() else HpuModelAdapter(model)


class HabanaProfilerCounterHelper():
    def __init__(self):
        self.niter = 0
        self.average_real_throughput = None
        self.logged_once = False
    
    def get_counter_dict(self, cache_config, duration, seq_len, batch_size_padded, real_batch_size, seq_group_metadata_list, is_prompt):
        throughput = batch_size_padded / (duration / 1e6)
        throughput_effective = real_batch_size / (duration / 1e6)
        real_seq_lens = [len(seq_data.prompt_token_ids) + len(seq_data.output_token_ids) for seq_group_metadata in seq_group_metadata_list for seq_data in seq_group_metadata.seq_data.values()]
        real_max_seq_len = max(real_seq_lens)
        real_num_tokens = sum(real_seq_lens)
        padded_num_tokens = batch_size_padded * seq_len
        batch_token_utilization = real_num_tokens / padded_num_tokens
        if self.average_real_throughput is None:
            self.average_real_throughput = throughput_effective
        else: # https://www.heikohoffmann.de/htmlthesis/node134.html
            self.average_real_throughput = self.average_real_throughput + 1/(self.niter+1) * (throughput_effective-self.average_real_throughput)
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
            prompt_seq_lens = [len(seq_data.prompt_token_ids) for seq_group_metadata in seq_group_metadata_list for seq_data in seq_group_metadata.seq_data.values()]
            prompt_bucket_in_throughput = (seq_len*batch_size_padded) / (duration / 1e6) 
            prompt_real_in_throughput = sum(prompt_seq_lens) / (duration / 1e6) 
            counters[f'{phase}_bucket_in_throughput'] = prompt_bucket_in_throughput
            counters[f'{phase}_real_in_throughput'] = prompt_real_in_throughput

        # KV cache might not be created yet (e.g. for profiling run)
        if cache_config.num_gpu_blocks is not None and cache_config.num_gpu_blocks != 0:
            cache_num_blocks_used = [math.ceil(sl/cache_config.block_size) for sl in real_seq_lens]
            cache_total_num_blocks_used = sum(cache_num_blocks_used)
            num_cache_blocks = cache_config.num_gpu_blocks 
            cache_total_num_free_blocks = num_cache_blocks - cache_total_num_blocks_used
            cache_computed_utilization = cache_total_num_blocks_used / num_cache_blocks
            max_blocks_per_seq = math.ceil(seq_len/cache_config.block_size)
            batch_block_utilization = cache_total_num_blocks_used / (batch_size_padded * max_blocks_per_seq)
            counters['cache_num_blocks_used'] = cache_total_num_blocks_used
            counters['cache_num_free_blocks'] = cache_total_num_free_blocks
            counters['cache_computed_utilization'] = cache_computed_utilization
            counters[f'{phase}_batch_block_utilization'] = batch_block_utilization
        if not self.logged_once:
            counters['const_cache_num_blocks'] = cache_config.num_gpu_blocks
            counters['const_gpu_memory_utilization'] = cache_config.gpu_memory_utilization
            counters['const_block_size'] = cache_config.block_size
            self.logged_once = True
        return counters
