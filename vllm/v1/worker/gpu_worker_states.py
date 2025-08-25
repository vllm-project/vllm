# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Datastructures defining a GPU input batch

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import triton
import triton.language as tl
from typing_extensions import deprecated

from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import (MultiModalKwargsItem,
                                    MultiModalKwargsItems, PlaceholderRange)
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata


@dataclass
class RequestData:

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


class RequestAttribute:

    def __init__(
        self,
        num_rows_cpu: int,
        num_cols: int,
        num_rows_gpu: int,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
        is_scalar: bool = False,
    ):
        self.cpu = torch.zeros(num_rows_cpu,
                               num_cols,
                               dtype=dtype,
                               device="cpu",
                               pin_memory=pin_memory)
        self.np = self.cpu.numpy()
        self.gpu = torch.zeros(num_rows_gpu,
                               num_cols,
                               dtype=dtype,
                               device=device)
        if is_scalar:
            assert num_cols == 1
            self.cpu.squeeze_(1)
            self.np = self.cpu.numpy()
            self.gpu.squeeze_(1)

        self.gpu_buffer = self.cpu.to(device)

    def mirror_to_gpu(self) -> torch.Tensor:
        return self.gpu_buffer.copy_(self.cpu, non_blocking=True)


class RequestState:

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
        # Used to construct the input batch.
        self._add_scalar_attr("idx_mapping", torch.int32)

        # Request states.
        self.req_data: dict[int, RequestData] = {}
        # TODO(woosuk): Because the token_ids tensor can be very big, we only
        # initialize it on CPU memory.
        self._add_vector_attr("token_ids",
                              self.max_model_len,
                              torch.int32,
                              cpu_only=True)
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

    def add_request(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        num_computed_tokens: int,
        sampling_params: SamplingParams,
    ) -> None:
        req_idx = self.free_indices.pop()
        self.req_id_to_index[req_id] = req_idx
        self.index_to_req_id[req_idx] = req_id

        self.num_prompt_tokens.np[req_idx] = len(prompt_token_ids)
        self.num_computed_tokens.np[req_idx] = num_computed_tokens
        self.append_token_ids(req_idx, prompt_token_ids)

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

    def append_token_ids(
        self,
        req_idx: int,
        token_ids: Union[list[int], np.ndarray],
    ) -> None:
        start_idx = self.num_tokens.np[req_idx]
        end_idx = start_idx + len(token_ids)
        self.token_ids.np[req_idx, start_idx:end_idx] = token_ids
        self.num_tokens.np[req_idx] = end_idx

    def append_sampled_token_ids(
        self,
        idx_mapping: np.ndarray,
        sampled_token_ids: np.ndarray,
    ) -> None:
        num_reqs = idx_mapping.shape[0]
        for i in range(num_reqs):
            req_idx = idx_mapping[i]
            self.append_token_ids(req_idx, sampled_token_ids[i])

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

    def make_sampling_metadata(
        self,
        batch_idx_to_req_idx: torch.Tensor,
    ) -> SamplingMetadata:
        batch_size = batch_idx_to_req_idx.shape[0]
        _make_sampling_metadata_kernel[(batch_size, )](
            batch_idx_to_req_idx,
            self.temperature.mirror_to_gpu(),
            self.temperature.gpu,
            self.top_p.mirror_to_gpu(),
            self.top_p.gpu,
            self.top_k.mirror_to_gpu(),
            self.top_k.gpu,
            self.frequency_penalties.mirror_to_gpu(),
            self.frequency_penalties.gpu,
            self.presence_penalties.mirror_to_gpu(),
            self.presence_penalties.gpu,
            self.repetition_penalties.mirror_to_gpu(),
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
            token_ids=self.token_ids.cpu[:batch_size],
            max_num_logprobs=None,
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=None,
        )

    @property
    def num_cached_reqs(self) -> int:
        return len(self.req_id_to_index)

    def _add_scalar_attr(self, name: str, dtype: torch.dtype):
        attr = RequestAttribute(self.max_num_cached_reqs,
                                1,
                                self.max_num_reqs,
                                dtype,
                                self.device,
                                self.pin_memory,
                                is_scalar=True)
        setattr(self, name, attr)

    def _add_vector_attr(
        self,
        name: str,
        max_len: int,
        dtype: torch.dtype,
        cpu_only: bool = False,
    ):
        if cpu_only:
            num_rows_gpu = 0
        else:
            num_rows_gpu = self.max_num_reqs
        attr = RequestAttribute(self.max_num_cached_reqs, max_len,
                                num_rows_gpu, dtype, self.device,
                                self.pin_memory)
        setattr(self, name, attr)


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
