# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

_MAX_SPEC_LEN = 32


@dataclass
class RequestData:

    mm_kwargs: list[MultiModalKwargsItem]
    mm_positions: list[PlaceholderRange]
    sampling_params: Optional[SamplingParams]
    pooling_params: Optional[PoolingParams]

    mm_hashes: list[str]
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


class SamplingStates:

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_cached_reqs: int,
        vocab_size: int,
        device: torch.device,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_cached_reqs = max_num_cached_reqs
        self.vocab_size = vocab_size
        self.device = device

        self.temperature = self._make_param(torch.float32)
        self.greedy_req_indices: set[int] = set()
        self.top_p = self._make_param(torch.float32)
        self.top_p_req_indices: set[int] = set()
        self.top_k = self._make_param(torch.int32)
        self.top_k_req_indices: set[int] = set()

        self.frequency_penalties = self._make_param(torch.float32)
        self.presence_penalties = self._make_param(torch.float32)
        self.repetition_penalties = self._make_param(torch.float32)
        self.penalty_req_indices: set[int] = set()

        self.generators: dict[int, torch.Generator] = {}

    def _make_param(self, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(self.max_num_reqs, dtype=dtype, device=self.device)

    def add_requests(
        self,
        req_indices: list[int],
        sampling_params: list[SamplingParams],
    ) -> None:
        num_reqs = len(req_indices)
        for i in range(num_reqs):
            req_idx = req_indices[i]
            sampling_param = sampling_params[i]

            temp = sampling_param.temperature
            if temp == 0.0:
                self.greedy_req_indices.add(req_idx)

            top_p = sampling_param.top_p
            if top_p < 1.0:
                self.top_p_req_indices.add(req_idx)
            top_k = sampling_param.top_k
            if 0 < top_k < self.vocab_size:
                self.top_k_req_indices.add(req_idx)
            else:
                top_k = self.vocab_size

            if sampling_param.frequency_penalty != 0.0 or sampling_param.presence_penalty != 0.0 or sampling_param.repetition_penalty != 1.0:
                self.penalty_req_indices.add(req_idx)

            if sampling_param.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_param.seed)
                self.generators[req_idx] = generator

    def remove_request(self, req_idx: int) -> None:
        self.greedy_req_indices.discard(req_idx)
        self.top_p_req_indices.discard(req_idx)
        self.top_k_req_indices.discard(req_idx)
        self.penalty_req_indices.discard(req_idx)
        self.generators.pop(req_idx, None)


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
        self.num_prompt_logprobs: dict[int, int] = {}

        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        self.free_indices = list(range(max_num_cached_reqs))

        # Request states.
        self.req_data: dict[int, RequestData] = {}
        # TODO(woosuk): Because the token_ids tensor can be very big, we only
        # initialize it on CPU memory.
        self.token_ids = self._make_param(
            num_cols=self.max_model_len,
            dtype=torch.int32,
            cpu_only=True,
        )
        self.num_prompt_tokens = self._make_param(torch.int32)
        self.num_tokens = self._make_param(torch.int32)
        self.num_computed_tokens = self._make_param(torch.int32)

        self.sampling_states = SamplingStates(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_cached_reqs=max_num_cached_reqs,
            device=device,
        )

    def _make_param(
        self,
        dtype: torch.dtype,
        num_cols: int = 1,
        cpu_only: bool = False,
    ) -> Param:
        return Param(
            self.max_num_cached_reqs,
            num_cols,
            self.max_num_reqs if not cpu_only else 0,
            dtype,
            self.device,
            self.pin_memory,
            is_scalar=num_cols == 1,
        )

    @property
    def num_cached_reqs(self) -> int:
        return len(self.req_id_to_index)

    def add_request(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        num_computed_tokens: int,
        sampling_params: SamplingParams,
    ) -> None:
        assert len(self.free_indices) > 0, "No free space in GPU worker states"
        req_idx = self.free_indices.pop()
        self.req_id_to_index[req_id] = req_idx
        self.index_to_req_id[req_idx] = req_id

        prompt_len = len(prompt_token_ids)
        self.num_prompt_tokens.np[req_idx] = prompt_len
        self.num_tokens.np[req_idx] = prompt_len
        self.token_ids.np[req_idx, :prompt_len] = prompt_token_ids
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
        if self.top_p_reqs:
            top_p_buffer = self.top_p.mirror_to_gpu()
            top_p = self.top_p.gpu
        else:
            top_p_buffer = self.top_p.gpu_buffer
            top_p = None
        if self.top_k_reqs:
            top_k_buffer = self.top_k.mirror_to_gpu()
            top_k = self.top_k.gpu
        else:
            top_k_buffer = self.top_k.gpu_buffer
            top_k = None
        # TODO(woosuk): Use UVA to optimize CPU -> GPU copy.
        _make_sampling_metadata_kernel[(batch_size, )](
            batch_idx_to_req_idx,
            self.temperature.mirror_to_gpu(),
            self.temperature.gpu,
            top_p_buffer,
            self.top_p.gpu,
            top_k_buffer,
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
            top_p=top_p,
            top_k=top_k,
            frequency_penalties=self.frequency_penalties.gpu[:batch_size],
            presence_penalties=self.presence_penalties.gpu[:batch_size],
            repetition_penalties=self.repetition_penalties.gpu[:batch_size],
            no_penalties=no_penalties,
            # TODO
            generators={},
            token_ids=None,
            num_tokens=None,
            num_prompt_tokens=None,
            max_num_logprobs=None,
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=None,
        )

    def make_spec_decode_metadata(
        self,
        query_start_loc: torch.Tensor,
        cu_num_draft_tokens: torch.Tensor,
        cu_num_draft_tokens_np: np.ndarray,
        input_ids: torch.Tensor,
    ) -> SpecDecodeMetadata:
        batch_size = query_start_loc.shape[0] - 1
        total_num_draft_tokens = cu_num_draft_tokens_np[batch_size - 1]
        logits_indices = torch.empty(total_num_draft_tokens + batch_size,
                                     dtype=torch.int32,
                                     device=self.device)
        target_logits_indices = torch.empty(total_num_draft_tokens,
                                            dtype=torch.int32,
                                            device=self.device)
        bonus_logits_indices = torch.empty(batch_size,
                                           dtype=torch.int32,
                                           device=self.device)
        _prepare_spec_decode_kernel[(batch_size, )](
            query_start_loc,
            cu_num_draft_tokens,
            logits_indices,
            target_logits_indices,
            bonus_logits_indices,
            BLOCK_SIZE=triton.next_power_of_2(_MAX_SPEC_LEN + 1),
        )

        draft_token_ids = input_ids[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]
        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=cu_num_draft_tokens_np.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )


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
def _prepare_spec_decode_kernel(
    query_start_loc,  # [B + 1]
    cu_num_draft_tokens,  # [B]
    logits_indices,  # [N + B]
    target_logits_indices,  # [N]
    bonus_logits_indices,  # [B]
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    if batch_idx == 0:
        draft_start_idx = 0
    else:
        draft_start_idx = tl.load(cu_num_draft_tokens + batch_idx - 1)
    draft_end_idx = tl.load(cu_num_draft_tokens + batch_idx)
    draft_len = draft_end_idx - draft_start_idx
    sample_len = draft_len + 1

    q_end_idx = tl.load(query_start_loc + batch_idx + 1)

    sample_start_idx = draft_start_idx + batch_idx
    sample_end_idx = sample_start_idx + sample_len
    offset = tl.arange(0, BLOCK_SIZE)
    tl.store(logits_indices + sample_start_idx + offset,
             q_end_idx - sample_len + offset,
             mask=offset < sample_len)
    tl.store(target_logits_indices + draft_start_idx + offset,
             sample_start_idx + offset,
             mask=offset < draft_len)
    tl.store(bonus_logits_indices + batch_idx, sample_end_idx - 1)
