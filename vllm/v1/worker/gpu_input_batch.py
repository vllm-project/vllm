# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Datastructures defining a GPU input batch

from dataclasses import dataclass
from typing import Optional

import torch
from typing_extensions import deprecated

from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import (MultiModalKwargs, MultiModalKwargsItem,
                                    PlaceholderRange)
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils import cdiv, get_cuda_view_from_cpu_tensor, is_uva_available
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata


@dataclass
class CachedRequestState:

    req_id: str
    mm_kwargs: list[MultiModalKwargsItem]
    mm_positions: list[PlaceholderRange]
    sampling_params: Optional[SamplingParams]
    pooling_params: Optional[PoolingParams]

    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[int] = None

    lora_request: Optional[LoRARequest] = None

    # Temporary back-compatibility for plugins that define model runner
    @property
    @deprecated("`mm_inputs` is superseded by `mm_kwargs` and will be "
                "removed in v0.13. Please use `mm_kwargs` instead.")
    def mm_inputs(self) -> list[MultiModalKwargs]:
        return [MultiModalKwargs([item]) for item in self.mm_kwargs]


class PerRequestAttribute:

    def __init__(
        self,
        N: int,
        M: int,
        K: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        assert is_uva_available(), "UVA is not available."
        self.cpu_tensor = torch.zeros(N,
                                      M,
                                      dtype=dtype,
                                      device="cpu",
                                      pin_memory=True)
        self.np = self.cpu_tensor.numpy()
        self.uva_tensor = get_cuda_view_from_cpu_tensor(self.cpu_tensor)
        self.gpu_tensor = torch.zeros(K, M, dtype=dtype, device=device)


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

        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        self.free_indices = list(range(max_num_cached_reqs))

        # Request states.
        # TODO(woosuk): This buffer could be too large if max_model_len is big.
        # Find a way to reduce the memory usage.
        self._add_vector_attr("token_ids", self.max_model_len, torch.int32)
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
        self.block_tables = []
        self.num_blocks = []
        for block_size in block_sizes:
            max_num_blocks = cdiv(max_model_len, block_size)
            block_table = PerRequestAttribute(self.max_num_cached_reqs,
                                              max_num_blocks,
                                              self.max_num_reqs, torch.int32,
                                              self.device)
            self.block_tables.append(block_table)
            num_blocks = PerRequestAttribute(self.max_num_cached_reqs, 1,
                                             self.max_num_reqs, torch.int32,
                                             self.device)
            self.num_blocks.append(num_blocks)
        self.num_block_tables = len(block_sizes)

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

        num_prompt_tokens = len(prompt_token_ids)
        self.token_ids.np[req_idx, :num_prompt_tokens] = prompt_token_ids
        self.num_prompt_tokens.np[req_idx] = num_prompt_tokens
        self.num_tokens.np[req_idx] = num_prompt_tokens
        self.num_computed_tokens.np[req_idx] = num_computed_tokens

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
            req_idx] = sampling_params.frequency_penalties
        if sampling_params.frequency_penalties != 0.0:
            self.frequency_penalties_reqs.add(req_id)
        self.presence_penalties.np[
            req_idx] = sampling_params.presence_penalties
        if sampling_params.presence_penalties != 0.0:
            self.presence_penalties_reqs.add(req_id)
        self.repetition_penalties.np[
            req_idx] = sampling_params.repetition_penalties
        if sampling_params.repetition_penalties != 1.0:
            self.repetition_penalties_reqs.add(req_id)

        if sampling_params.seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(sampling_params.seed)
            self.generators[req_idx] = generator

        for i in range(self.num_block_tables):
            self.block_tables[i].np[req_idx, :len(block_ids[i])] = block_ids[i]
            self.num_blocks[i].np[req_idx] = len(block_ids[i])

    def append_token_ids(self, req_id: str, token_ids: list[int]) -> None:
        req_idx = self.req_id_to_index.get(req_id)
        assert req_idx is not None
        start_idx = self.num_tokens.np[req_idx]
        end_idx = start_idx + len(token_ids)
        self.token_ids.np[req_idx, start_idx:end_idx] = token_ids
        self.num_tokens.np[req_idx] = end_idx

    def append_block_ids(
        self,
        req_id: str,
        new_block_ids: tuple[list[int], ...],
        overwrite: bool,
    ) -> None:
        req_idx = self.req_id_to_index.get(req_id)
        assert req_idx is not None
        for i in range(self.num_block_tables):
            block_table = self.block_tables[i]
            num_blocks = self.num_blocks[i]
            if overwrite:
                # Replace the existing block IDs with the new ones.
                # This happens when the request is resumed from preemption.
                block_table.np[
                    req_idx, :len(new_block_ids[i])] = new_block_ids[i]
                num_blocks.np[req_idx] = len(new_block_ids[i])
            else:
                # Append the new block IDs to the existing ones (common case).
                start_idx = num_blocks.np[req_idx]
                end_idx = start_idx + len(new_block_ids[i])
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

    def make_block_table(self, req_idx: int) -> tuple[torch.Tensor, ...]:
        pass

    def make_sampling_metadata(self,
                               req_indices: list[int]) -> SamplingMetadata:
        batch_size = len(req_indices)
        _make_sampling_metadata_kernel[(batch_size, )](
            req_indices,
            self.temperature.uva_tensor,
            self.temperature.gpu_tensor,
            self.top_p.uva_tensor,
            self.top_p.gpu_tensor,
            self.top_k.uva_tensor,
            self.top_k.gpu_tensor,
            self.frequency_penalties.uva_tensor,
            self.frequency_penalties.gpu_tensor,
            self.presence_penalties.uva_tensor,
            self.presence_penalties.gpu_tensor,
            self.repetition_penalties.uva_tensor,
            self.repetition_penalties.gpu_tensor,
            num_warps=1,
            num_stages=1,
        )
        generators = {}
        if self.generators:
            for i, req_idx in enumerate(req_indices):
                generator = self.generators.get(req_idx)
                if generator is not None:
                    generators[i] = generator
        no_penalties = not (self.frequency_penalties_reqs
                            or self.presence_penalties_reqs
                            or self.repetition_penalties_reqs)
        return SamplingMetadata(
            temperature=self.temperature.gpu_tensor[:batch_size],
            all_greedy=not self.random_reqs,
            all_random=not self.greedy_reqs,
            top_p=self.top_p.gpu_tensor[:batch_size],
            top_k=self.top_k.gpu_tensor[:batch_size],
            frequency_penalties=self.frequency_penalties.
            gpu_tensor[:batch_size],
            presence_penalties=self.presence_penalties.gpu_tensor[:batch_size],
            repetition_penalties=self.repetition_penalties.
            gpu_tensor[:batch_size],
            no_penalties=no_penalties,
            generators=generators,
            token_ids=self.token_ids.gpu_tensor[:batch_size],
            max_num_logprobs=None,
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=None,
        )

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    def _add_vector_attr(self, name: str, max_len: int, dtype: torch.dtype):
        attr = PerRequestAttribute(self.max_num_cached_reqs, max_len,
                                   self.max_num_reqs, dtype, self.device)
        setattr(self, name, attr)

    def _add_scalar_attr(self, name: str, dtype: torch.dtype):
        self._add_vector_attr(name, max_len=1, dtype=dtype)


import triton
import triton.language as tl


@triton.jit
def _make_sampling_metadata_kernel(
    req_indices,  # [batch_size]
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
    req_index = tl.load(req_indices + batch_idx)

    temperature = tl.load(src_temperature + req_index)
    tl.store(dst_temperature + req_index, temperature)

    top_p = tl.load(src_top_p + req_index)
    tl.store(dst_top_p + req_index, top_p)

    top_k = tl.load(src_top_k + req_index)
    tl.store(dst_top_k + req_index, top_k)

    frequency_penalties = tl.load(src_frequency_penalties + req_index)
    tl.store(dst_frequency_penalties + req_index, frequency_penalties)

    presence_penalties = tl.load(src_presence_penalties + req_index)
    tl.store(dst_presence_penalties + req_index, presence_penalties)

    repetition_penalties = tl.load(src_repetition_penalties + req_index)
    tl.store(dst_repetition_penalties + req_index, repetition_penalties)


@triton.jit
def _make_block_table_kernel(
    req_indices,  # [batch_size]
    src_block_table_ptrs,
    dst_block_table_ptrs,
    src_num_blocks_ptrs,
    dst_num_blocks_ptrs,
    num_block_tables: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_index = tl.load(req_indices + batch_idx)

    for i in tl.range(num_block_tables):
        src_num_blocks_ptr = tl.load(src_num_blocks_ptrs + i)
        dst_num_blocks_ptr = tl.load(dst_num_blocks_ptrs + i)
        num_blocks = tl.load(src_num_blocks_ptr + req_index)
        tl.store(dst_num_blocks_ptr + req_index, num_blocks)

        src_block_table_ptr = tl.load(src_block_table_ptrs + i)
        dst_block_table_ptr = tl.load(dst_block_table_ptrs + i)
        for j in tl.range(num_blocks, BLOCK_SIZE):
            offset = tl.arange(0, BLOCK_SIZE)
            block_ids = tl.load(src_block_table_ptr + j * BLOCK_SIZE + offset,
                                mask=offset < num_blocks)
            tl.store(dst_block_table_ptr + j * BLOCK_SIZE + offset,
                     block_ids,
                     mask=offset < num_blocks)
