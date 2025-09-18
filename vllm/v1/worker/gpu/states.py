# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import numba
import numba.types as types
import numpy as np
import torch

from vllm.sampling_params import SamplingParams

_NP_INT64_MIN = np.iinfo(np.int64).min
_NP_INT64_MAX = np.iinfo(np.int64).max


@dataclass
class SamplingMetadata:

    temperature: torch.Tensor

    top_p: Optional[torch.Tensor]
    top_k: Optional[torch.Tensor]

    seeds: torch.Tensor
    pos: torch.Tensor

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: Optional[int]


class RequestState:

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        device: torch.device,
        pin_memory: bool,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.vocab_size = vocab_size
        self.device = device
        self.pin_memory = pin_memory

        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        self.free_indices = list(range(max_num_reqs))

        # TODO(woosuk): Because the token_ids tensor can be very big, we only
        # initialize it on CPU memory.
        self.token_ids = np.zeros(
            (self.max_num_reqs, self.max_model_len),
            dtype=np.int32,
        )
        self.num_tokens = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.num_computed_tokens = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.num_prompt_tokens = np.zeros(self.max_num_reqs, dtype=np.int32)

        # Last sampled token ids.
        self.last_token = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )

        # Sampling parameters.
        self.temperature = self._make_buffer(self.max_num_reqs, torch.float32)
        self.top_p = self._make_buffer(self.max_num_reqs, torch.float32)
        self.top_k = self._make_buffer(self.max_num_reqs, torch.int32)
        self.seeds = self._make_buffer(self.max_num_reqs, torch.int64)

        self.num_logprobs = np.empty(self.max_num_reqs, dtype=np.int32)
        # -1 means no logprobs are requested.
        self.num_logprobs.fill(-1)
        self.needs_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=bool)

    def _make_buffer(self, size, dtype: torch.dtype) -> "Buffer":
        return Buffer(size,
                      dtype=dtype,
                      pin_memory=self.pin_memory,
                      device=self.device)

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    def add_request(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        num_computed_tokens: int,
        sampling_params: SamplingParams,
    ) -> None:
        assert len(self.free_indices) > 0
        req_idx = self.free_indices.pop()
        self.req_id_to_index[req_id] = req_idx
        self.index_to_req_id[req_idx] = req_id

        prompt_len = len(prompt_token_ids)
        self.num_tokens[req_idx] = prompt_len
        self.num_prompt_tokens[req_idx] = prompt_len
        self.token_ids[req_idx, :prompt_len] = prompt_token_ids
        self.num_computed_tokens[req_idx] = num_computed_tokens

        self.temperature.np[req_idx] = sampling_params.temperature
        self.top_p.np[req_idx] = sampling_params.top_p
        if 0 < sampling_params.top_k < self.vocab_size:
            top_k = sampling_params.top_k
        else:
            top_k = self.vocab_size
        self.top_k.np[req_idx] = top_k

        if sampling_params.seed is not None:
            seed = sampling_params.seed
        else:
            seed = np.random.randint(_NP_INT64_MIN, _NP_INT64_MAX)
        self.seeds.np[req_idx] = seed

        if sampling_params.logprobs is not None:
            num_logprobs = sampling_params.logprobs
        else:
            num_logprobs = -1
        self.num_logprobs[req_idx] = num_logprobs

        # For now, only support prompt logprobs for the prompt tokens.
        needs_prompt_logprobs = sampling_params.prompt_logprobs is not None
        self.needs_prompt_logprobs[req_idx] = needs_prompt_logprobs

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
        pos: torch.Tensor,
    ) -> SamplingMetadata:
        temperature = self.temperature.np[idx_mapping]
        temperature = self.temperature.copy_np_to_gpu(temperature)

        top_p = self.top_p.np[idx_mapping]
        no_top_p = np.all(top_p == 1.0)
        top_p = self.top_p.copy_np_to_gpu(top_p) if not no_top_p else None

        top_k = self.top_k.np[idx_mapping]
        no_top_k = np.all(top_k == self.vocab_size)
        top_k = self.top_k.copy_np_to_gpu(top_k) if not no_top_k else None

        seeds = self.seeds.np[idx_mapping]
        seeds = self.seeds.copy_np_to_gpu(seeds)

        num_logprobs = self.num_logprobs[idx_mapping]
        max_num_logprobs = np.max(num_logprobs)
        if max_num_logprobs == -1:
            max_num_logprobs = None

        return SamplingMetadata(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seeds=seeds,
            pos=pos,
            max_num_logprobs=max_num_logprobs,
        )

    def append_token_ids(
        self,
        req_indices: np.ndarray,
        sampled_ids: np.ndarray,
        num_sampled_tokens: np.ndarray,
    ) -> None:
        _append_token_ids(
            req_indices,
            sampled_ids,
            num_sampled_tokens,
            self.token_ids,
            self.num_tokens,
        )


@numba.jit(
    [
        types.none(
            types.int32[:],
            types.int64[:, :],
            types.int32[:],
            types.int32[:, :],
            types.int32[:],
        )
    ],
    nopython=True,
    cache=True,
)
def _append_token_ids(
    req_indices: np.ndarray,
    sampled_ids: np.ndarray,
    num_sampled_tokens: np.ndarray,
    token_ids: np.ndarray,
    num_tokens: np.ndarray,
) -> None:
    num_reqs = num_sampled_tokens.shape[0]
    for i in range(num_reqs):
        req_idx = req_indices[i]
        n = num_sampled_tokens[i]
        start_idx = num_tokens[req_idx]
        end_idx = start_idx + n
        token_ids[req_idx, start_idx:end_idx] = sampled_ids[i, :n]
        num_tokens[req_idx] = end_idx


class Buffer:

    def __init__(
        self,
        *args,
        dtype: torch.dtype,
        pin_memory: bool,
        device: torch.device,
    ):
        # NOTE(woosuk): Unlike CpuGpuBuffer, the Numpy array and CPU tensor
        # in this class do not share the same storage.
        self.np = np.zeros(*args, dtype=dtype)
        self.cpu = torch.zeros(
            *args,
            dtype=dtype,
            pin_memory=pin_memory,
            device=device,
        )
        self.gpu = self.cpu.to(device)

    def copy_np_to_gpu(self, x: np.ndarray) -> torch.Tensor:
        n = x.shape[0]
        self.cpu[:n] = x
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)
