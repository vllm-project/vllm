# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import triton
import triton.language as tl

from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalKwargsItem, PlaceholderRange
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
        self.num_prompt_logprobs: dict[int, int] = {}

        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        self.free_indices = list(range(max_num_cached_reqs))

        # Request states.
        self.req_data: dict[int, RequestData] = {}
        # TODO(woosuk): Because the token_ids tensor can be very big, we only
        # initialize it on CPU memory.
        self.token_ids = np.zeros(
            (self.max_num_cached_reqs, self.max_model_len),
            dtype=np.int32,
        )
        self.num_prompt_tokens = np.zeros(self.max_num_cached_reqs,
                                          dtype=np.int32)
        self.num_tokens = np.zeros(self.max_num_cached_reqs, dtype=np.int32)
        self.num_computed_tokens = np.zeros(self.max_num_cached_reqs,
                                            dtype=np.int32)

        # Last sampled token ids.
        self.last_sampled_token = torch.zeros(
            self.max_num_cached_reqs,
            dtype=torch.int32,
            device=self.device,
        )

        self.temperature = np.zeros(self.max_num_cached_reqs, dtype=np.float32)
        self.top_p = np.zeros(self.max_num_cached_reqs, dtype=np.float32)
        self.top_k = np.zeros(self.max_num_cached_reqs, dtype=np.int32)

        self.frequency_penalties = np.zeros(self.max_num_cached_reqs,
                                            dtype=np.float32)
        self.presence_penalties = np.zeros(self.max_num_cached_reqs,
                                           dtype=np.float32)
        self.repetition_penalties = np.zeros(self.max_num_cached_reqs,
                                             dtype=np.float32)

        self.generators: dict[int, torch.Generator] = {}

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
        self.num_prompt_tokens[req_idx] = prompt_len
        self.num_tokens[req_idx] = prompt_len
        self.token_ids[req_idx, :prompt_len] = prompt_token_ids
        self.num_computed_tokens[req_idx] = num_computed_tokens

        self.temperature[req_idx] = sampling_params.temperature
        self.top_p[req_idx] = sampling_params.top_p
        if 0 < sampling_params.top_k < self.vocab_size:
            top_k = sampling_params.top_k
        else:
            top_k = self.vocab_size
        self.top_k[req_idx] = top_k
        self.frequency_penalties[req_idx] = sampling_params.frequency_penalty
        self.presence_penalties[req_idx] = sampling_params.presence_penalty
        self.repetition_penalties[req_idx] = sampling_params.repetition_penalty

        if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(sampling_params.seed)
            self.generators[req_idx] = generator

    @property
    def num_cached_reqs(self) -> int:
        return len(self.req_id_to_index)

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

    def make_sampling_metadata(
        self,
        idx_mapping: np.ndarray,
    ) -> SamplingMetadata:
        temperature = self.temperature[idx_mapping]
        all_greedy = np.all(temperature == 0.0)
        all_random = np.all(temperature != 0.0)
        temperature = self._copy_np_to_gpu(temperature)

        top_p = self.top_p[idx_mapping]
        no_top_p = np.all(top_p == 1.0)
        top_p = self._copy_np_to_gpu(top_p) if not no_top_p else None
        top_k = self.top_k[idx_mapping]
        no_top_k = np.all(top_k == self.vocab_size)
        top_k = self._copy_np_to_gpu(top_k) if not no_top_k else None

        frequency_penalties = self.frequency_penalties[idx_mapping]
        presence_penalties = self.presence_penalties[idx_mapping]
        repetition_penalties = self.repetition_penalties[idx_mapping]
        no_penalties = (np.all(frequency_penalties == 0.0)
                        and np.all(presence_penalties == 0.0)
                        and np.all(repetition_penalties == 1.0))
        if no_penalties:
            frequency_penalties = None
            presence_penalties = None
            repetition_penalties = None
        else:
            frequency_penalties = self._copy_np_to_gpu(frequency_penalties)
            presence_penalties = self._copy_np_to_gpu(presence_penalties)
            repetition_penalties = self._copy_np_to_gpu(repetition_penalties)

        if self.generators:
            generators = {
                req_idx: self.generators[req_idx]
                for req_idx in idx_mapping if req_idx in self.generators
            }
        else:
            generators = {}

        return SamplingMetadata(
            temperature=temperature,
            all_greedy=all_greedy,
            all_random=all_random,
            top_p=top_p,
            top_k=top_k,
            frequency_penalties=frequency_penalties,
            presence_penalties=presence_penalties,
            repetition_penalties=repetition_penalties,
            no_penalties=no_penalties,
            generators=generators,
            token_ids=None,
            num_tokens=None,
            num_prompt_tokens=None,
            max_num_logprobs=None,
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=None,
        )

    def _copy_np_to_gpu(self, src: np.ndarray) -> torch.Tensor:
        cpu_tensor = torch.from_numpy(src)
        return cpu_tensor.to(device=self.device, non_blocking=True)

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
