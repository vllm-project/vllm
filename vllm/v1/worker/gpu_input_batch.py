# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Datastructures defining a GPU input batch

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl
from typing_extensions import deprecated

from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import (MultiModalKwargsItem,
                                    MultiModalKwargsItems, PlaceholderRange)
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils import cdiv, get_cuda_view_from_cpu_tensor, is_uva_available
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata

PAD_SLOT_ID = -1


@dataclass
class CachedRequestState:

    req_id: str
    mm_kwargs: list[MultiModalKwargsItem]
    mm_positions: list[PlaceholderRange]
    sampling_params: Optional[SamplingParams]
    pooling_params: Optional[PoolingParams]

    # M-RoPE (only for Qwen2/2.5-VL)
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[int] = None

    lora_request: Optional[LoRARequest] = None

    # Temporary back-compatibility for plugins that define model runner
    @property
    @deprecated("`mm_inputs` is superseded by `mm_kwargs` and will be "
                "removed in v0.13. Please use `mm_kwargs` instead.")
    def mm_inputs(self) -> list[MultiModalKwargsItems]:
        return [
            MultiModalKwargsItems.from_seq([item]) for item in self.mm_kwargs
        ]


class PerRequestAttribute:

    def __init__(
        self,
        num_rows_cpu: int,
        num_cols: int,
        num_rows_gpu: int,
        dtype: torch.dtype,
        device: torch.device,
        is_scalar: bool = False,
    ):
        assert is_uva_available(), "UVA is not available."
        self.cpu = torch.zeros(num_rows_cpu,
                               num_cols,
                               dtype=dtype,
                               device="cpu",
                               pin_memory=True)
        self.np = self.cpu.numpy()
        self.uva = get_cuda_view_from_cpu_tensor(self.cpu)
        self.gpu = torch.zeros(num_rows_gpu,
                               num_cols,
                               dtype=dtype,
                               device=device)
        if is_scalar:
            assert num_cols == 1
            self.cpu.squeeze_(1)
            self.np = self.cpu.numpy()
            self.uva.squeeze_(1)
            self.gpu.squeeze_(1)


class InputBatch:

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        max_num_cached_reqs: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
        block_sizes: list[int],  # The block_size of each kv cache group
        logitsprocs: Optional[LogitsProcessors] = None,
        is_spec_decode: bool = False,
        is_pooling_model: bool = False,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_cached_reqs = max_num_cached_reqs
        self.device = device
        self.pin_memory = pin_memory
        self.vocab_size = vocab_size
        self.is_spec_decode = is_spec_decode
        self.pooling_params = None
        self.block_sizes = block_sizes
        self.num_prompt_logprobs = {}

        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        self.free_indices = list(range(max_num_cached_reqs))
        self._add_scalar_attr("idx_mapping", torch.int32)

        # Request states.
        # TODO(woosuk): Because the token_ids tensor can be very big, we only
        # initialize it on CPU memory.
        self._add_vector_attr_cpu("token_ids", self.max_model_len, torch.int32)
        self._add_scalar_attr("num_prompt_tokens", torch.int32)
        self._add_scalar_attr("num_tokens", torch.int32)
        self._add_scalar_attr("num_computed_tokens", torch.int32)

        # Sampling-related.
        self._add_scalar_attr("temperature", torch.float32)
        self.greedy_reqs: set[str] = set()
        self.random_reqs: set[str] = set()
        self._add_scalar_attr("top_p", torch.float32)
        self.top_p_reqs: set[str] = set()
        self._add_scalar_attr("top_k", torch.int32)
        self.top_k_reqs: set[str] = set()
        self._add_scalar_attr("frequency_penalties", torch.float32)
        self.frequency_penalties_reqs: set[str] = set()
        self._add_scalar_attr("presence_penalties", torch.float32)
        self.presence_penalties_reqs: set[str] = set()
        self._add_scalar_attr("repetition_penalties", torch.float32)
        self.repetition_penalties_reqs: set[str] = set()

        # req_idx -> generator
        self.generators: dict[int, torch.Generator] = {}

        # Block table(s).
        self._init_block_tables()

    def add_request(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        num_computed_tokens: int,
        block_ids: tuple[list[int], ...],
        sampling_params: SamplingParams,
    ) -> None:
        req_idx = self.free_indices.pop()
        self.req_id_to_index[req_id] = req_idx
        self.index_to_req_id[req_idx] = req_id

        self.num_prompt_tokens.np[req_idx] = len(prompt_token_ids)
        self.num_computed_tokens.np[req_idx] = num_computed_tokens
        self.append_token_ids(req_idx, prompt_token_ids)
        self.append_block_ids(req_idx, block_ids, overwrite=True)

        self.temperature.np[req_idx] = sampling_params.temperature
        if sampling_params.sampling_type == SamplingType.GREEDY:
            # NOTE: Be careful about division by zero.
            self.greedy_reqs.add(req_id)
        elif sampling_params.sampling_type == SamplingType.RANDOM:
            self.random_reqs.add(req_id)

        self.top_p.np[req_idx] = sampling_params.top_p
        if sampling_params.top_p < 1.0:
            self.top_p_reqs.add(req_id)

        top_k = sampling_params.top_k
        if 0 < top_k < self.vocab_size:
            self.top_k_reqs.add(req_id)
        else:
            top_k = self.vocab_size
        self.top_k.np[req_idx] = top_k

        self.frequency_penalties.np[
            req_idx] = sampling_params.frequency_penalty
        if sampling_params.frequency_penalty != 0.0:
            self.frequency_penalties_reqs.add(req_id)
        self.presence_penalties.np[req_idx] = sampling_params.presence_penalty
        if sampling_params.presence_penalty != 0.0:
            self.presence_penalties_reqs.add(req_id)
        self.repetition_penalties.np[
            req_idx] = sampling_params.repetition_penalty
        if sampling_params.repetition_penalty != 1.0:
            self.repetition_penalties_reqs.add(req_id)

        if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(sampling_params.seed)
            self.generators[req_idx] = generator

    def append_token_ids(self, req_idx: int, token_ids: list[int]) -> None:
        start_idx = self.num_tokens.np[req_idx]
        end_idx = start_idx + len(token_ids)
        self.token_ids.np[req_idx, start_idx:end_idx] = token_ids
        self.num_tokens.np[req_idx] = end_idx

    # TODO(woosuk): Further vectorize this to minimize overheads.
    def append_block_ids(
        self,
        req_idx: int,
        new_block_ids: tuple[list[int], ...],
        overwrite: bool,
    ) -> None:
        for i in range(self.num_block_tables):
            block_table = self.block_tables[i]
            num_blocks = self.num_blocks[i]
            num_new_blocks = len(new_block_ids[i])
            if overwrite:
                # Replace the existing block IDs with the new ones.
                # This happens when the request is resumed from preemption.
                block_table.np[req_idx, :num_new_blocks] = new_block_ids[i]
                num_blocks.np[req_idx] = num_new_blocks
            else:
                # Append the new block IDs to the existing ones (common case).
                start_idx = num_blocks.np[req_idx]
                end_idx = start_idx + num_new_blocks
                block_table.np[req_idx, start_idx:end_idx] = new_block_ids[i]
                num_blocks.np[req_idx] = end_idx

    def remove_request(self, req_id: str) -> None:
        req_idx = self.req_id_to_index.pop(req_id, None)
        if req_idx is None:
            # Request not found.
            return
        self.index_to_req_id.pop(req_idx, None)
        self.free_indices.append(req_idx)

        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)
        self.frequency_penalties_reqs.discard(req_id)
        self.presence_penalties_reqs.discard(req_id)
        self.repetition_penalties_reqs.discard(req_id)
        self.generators.pop(req_idx, None)

    def get_index_mapping(self, idx_mapping: list[int]) -> torch.Tensor:
        num_reqs = len(idx_mapping)
        self.idx_mapping.np[:num_reqs] = idx_mapping
        return self.idx_mapping.gpu[:num_reqs].copy_(
            self.idx_mapping.uva[:num_reqs], non_blocking=True)

    def make_sampling_metadata(
        self,
        batch_idx_to_req_idx: torch.Tensor,
    ) -> SamplingMetadata:
        batch_size = batch_idx_to_req_idx.shape[0]
        _make_sampling_metadata_kernel[(batch_size, )](
            batch_idx_to_req_idx,
            self.temperature.uva,
            self.temperature.gpu,
            self.top_p.uva,
            self.top_p.gpu,
            self.top_k.uva,
            self.top_k.gpu,
            self.frequency_penalties.uva,
            self.frequency_penalties.gpu,
            self.presence_penalties.uva,
            self.presence_penalties.gpu,
            self.repetition_penalties.uva,
            self.repetition_penalties.gpu,
            num_warps=1,
            num_stages=1,
        )
        no_penalties = not (self.frequency_penalties_reqs
                            or self.presence_penalties_reqs
                            or self.repetition_penalties_reqs)
        return SamplingMetadata(
            temperature=self.temperature.gpu[:batch_size],
            all_greedy=not self.random_reqs,
            all_random=not self.greedy_reqs,
            top_p=self.top_p.gpu[:batch_size],
            top_k=self.top_k.gpu[:batch_size],
            frequency_penalties=self.frequency_penalties.gpu[:batch_size],
            presence_penalties=self.presence_penalties.gpu[:batch_size],
            repetition_penalties=self.repetition_penalties.gpu[:batch_size],
            no_penalties=no_penalties,
            # TODO
            generators={},
            token_ids=self.token_ids.gpu[:batch_size],
            max_num_logprobs=None,
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=None,
        )

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    def _add_scalar_attr(self, name: str, dtype: torch.dtype):
        attr = PerRequestAttribute(self.max_num_cached_reqs,
                                   1,
                                   self.max_num_reqs,
                                   dtype,
                                   self.device,
                                   is_scalar=True)
        setattr(self, name, attr)

    def _add_vector_attr(self, name: str, max_len: int, dtype: torch.dtype):
        attr = PerRequestAttribute(self.max_num_cached_reqs, max_len,
                                   self.max_num_reqs, dtype, self.device)
        setattr(self, name, attr)

    def _add_vector_attr_cpu(self, name: str, max_len: int,
                             dtype: torch.dtype):
        attr = PerRequestAttribute(self.max_num_cached_reqs, max_len, 0, dtype,
                                   self.device)
        setattr(self, name, attr)

    def _init_block_tables(self):
        self.num_block_tables = len(self.block_sizes)
        self.block_tables = []
        self.num_blocks = []
        self.slot_mappings: list[torch.Tensor] = []
        for i in range(self.num_block_tables):
            max_num_blocks = cdiv(self.max_model_len, self.block_sizes[i])
            block_table = PerRequestAttribute(self.max_num_cached_reqs,
                                              max_num_blocks,
                                              self.max_num_reqs, torch.int32,
                                              self.device)
            self.block_tables.append(block_table)
            num_blocks = PerRequestAttribute(self.max_num_cached_reqs,
                                             1,
                                             self.max_num_reqs,
                                             torch.int32,
                                             self.device,
                                             is_scalar=True)
            self.num_blocks.append(num_blocks)
            slot_mapping = torch.zeros(self.max_num_batched_tokens,
                                       dtype=torch.int64,
                                       device=self.device)
            self.slot_mappings.append(slot_mapping)

        def make_ptr_tensor(x: list[torch.Tensor]) -> torch.Tensor:
            return torch.tensor([t.data_ptr() for t in x],
                                dtype=torch.int64,
                                device=self.device)

        self.uva_block_table_ptrs = make_ptr_tensor(
            [b.uva for b in self.block_tables])
        self.gpu_block_table_ptrs = make_ptr_tensor(
            [b.gpu for b in self.block_tables])
        self.uva_num_blocks_ptrs = make_ptr_tensor(
            [n.uva for n in self.num_blocks])
        self.gpu_num_blocks_ptrs = make_ptr_tensor(
            [n.gpu for n in self.num_blocks])
        self.block_table_strides = torch.tensor(
            [b.gpu.shape[1] for b in self.block_tables],
            dtype=torch.int64,
            device=self.device)
        self.block_sizes_tensor = torch.tensor(self.block_sizes,
                                               dtype=torch.int32,
                                               device=self.device)
        self.slot_mapping_ptrs = make_ptr_tensor(self.slot_mappings)

    def make_block_tables(
        self,
        idx_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        batch_size = idx_mapping.shape[0]
        _make_block_tables_kernel[(batch_size, self.num_block_tables)](
            idx_mapping,
            self.uva_block_table_ptrs,
            self.gpu_block_table_ptrs,
            self.block_table_strides,
            self.uva_num_blocks_ptrs,
            self.gpu_num_blocks_ptrs,
            BLOCK_SIZE=1024,
        )
        return tuple(b.gpu[:batch_size] for b in self.block_tables)

    def make_slot_mappings(
        self,
        cu_num_tokens: torch.Tensor,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        num_tokens = pos.shape[0]
        num_reqs = cu_num_tokens.shape[0] - 1
        _make_slot_mappings_kernel[(num_reqs + 1, self.num_block_tables)](
            num_tokens,
            self.max_num_batched_tokens,
            cu_num_tokens,
            pos,
            self.gpu_block_table_ptrs,
            self.block_table_strides,
            self.block_sizes_tensor,
            self.slot_mapping_ptrs,
            PAD_ID=PAD_SLOT_ID,
            BLOCK_SIZE=1024,
        )
        return tuple(x[:num_tokens] for x in self.slot_mappings)


@triton.jit
def _make_sampling_metadata_kernel(
    batch_idx_to_req_idx,  # [batch_size]
    src_temperature,
    dst_temperature,
    src_top_p,
    dst_top_p,
    src_top_k,
    dst_top_k,
    src_frequency_penalties,
    dst_frequency_penalties,
    src_presence_penalties,
    dst_presence_penalties,
    src_repetition_penalties,
    dst_repetition_penalties,
):
    batch_idx = tl.program_id(0)
    req_idx = tl.load(batch_idx_to_req_idx + batch_idx)

    temperature = tl.load(src_temperature + req_idx)
    tl.store(dst_temperature + batch_idx, temperature)

    top_p = tl.load(src_top_p + req_idx)
    tl.store(dst_top_p + batch_idx, top_p)

    top_k = tl.load(src_top_k + req_idx)
    tl.store(dst_top_k + batch_idx, top_k)

    frequency_penalties = tl.load(src_frequency_penalties + req_idx)
    tl.store(dst_frequency_penalties + batch_idx, frequency_penalties)

    presence_penalties = tl.load(src_presence_penalties + req_idx)
    tl.store(dst_presence_penalties + batch_idx, presence_penalties)

    repetition_penalties = tl.load(src_repetition_penalties + req_idx)
    tl.store(dst_repetition_penalties + batch_idx, repetition_penalties)


@triton.jit
def _make_block_tables_kernel(
    batch_idx_to_req_idx,  # [batch_size]
    src_block_table_ptrs,  # [num_block_tables]
    dst_block_table_ptrs,  # [num_block_tables]
    block_table_strides,  # [num_block_tables]
    src_num_blocks_ptrs,  # [num_block_tables]
    dst_num_blocks_ptrs,  # [num_block_tables]
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    # kv cache group id
    group_id = tl.program_id(1)
    req_idx = tl.load(batch_idx_to_req_idx + batch_idx)

    src_num_blocks_ptr = _load_ptr(src_num_blocks_ptrs, group_id, tl.int32)
    dst_num_blocks_ptr = _load_ptr(dst_num_blocks_ptrs, group_id, tl.int32)
    num_blocks = tl.load(src_num_blocks_ptr + req_idx)
    tl.store(dst_num_blocks_ptr + batch_idx, num_blocks)

    stride = tl.load(block_table_strides + group_id)
    src_block_table_ptr = _load_ptr(src_block_table_ptrs, group_id, tl.int32)
    src_row_ptr = src_block_table_ptr + req_idx * stride
    dst_block_table_ptr = _load_ptr(dst_block_table_ptrs, group_id, tl.int32)
    dst_row_ptr = dst_block_table_ptr + batch_idx * stride

    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        block_ids = tl.load(src_row_ptr + offset, mask=offset < num_blocks)
        tl.store(dst_row_ptr + offset, block_ids, mask=offset < num_blocks)


@triton.jit
def _make_slot_mappings_kernel(
    num_tokens,
    max_num_tokens,
    cu_num_tokens,  # [num_reqs + 1]
    pos,  # [num_tokens]
    block_table_ptrs,  # [num_block_tables]
    block_table_strides,  # [num_block_tables]
    page_sizes,  # [num_block_tables]
    slot_mapping_ptrs,  # [num_block_tables]
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0)
    # kv cache group id
    group_id = tl.program_id(1)
    slot_mapping_ptr = _load_ptr(slot_mapping_ptrs, group_id, tl.int64)

    if req_idx == num_reqs - 1:
        # Pad remaining slots to -1. This is needed for CUDA graphs.
        for i in tl.range(num_tokens, max_num_tokens, BLOCK_SIZE):
            offset = num_tokens + i + tl.arange(0, BLOCK_SIZE)
            tl.store(slot_mapping_ptr + offset,
                     PAD_ID,
                     mask=offset < max_num_tokens)
        return

    block_table_ptr = _load_ptr(block_table_ptrs, group_id, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    page_size = tl.load(page_sizes + group_id)

    start_idx = tl.load(cu_num_tokens + req_idx)
    end_idx = tl.load(cu_num_tokens + req_idx + 1)
    for i in tl.range(start_idx, end_idx, BLOCK_SIZE):
        offset = start_idx + i + tl.arange(0, BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)
        block_indices = positions // page_size
        block_numbers = tl.load(block_table_ptr +
                                req_idx * block_table_stride + block_indices)
        slot_ids = block_numbers * page_size + positions % page_size
        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)


@triton.jit
def _load_ptr(base, offset, elem_dtype):
    ptr = tl.load(base + offset)
    return tl.cast(ptr, tl.pointer_type(elem_dtype))
