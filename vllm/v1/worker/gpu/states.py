# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field

import numpy as np
import torch

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.utils import CpuGpuBuffer

_NP_INT64_MIN = np.iinfo(np.int64).min
_NP_INT64_MAX = np.iinfo(np.int64).max
NO_LORA_ID = 0


@dataclass
class SamplingMetadata:
    temperature: torch.Tensor

    top_p: torch.Tensor | None
    top_k: torch.Tensor | None

    seeds: torch.Tensor
    pos: torch.Tensor

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: int | None

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        device: torch.device,
    ) -> "SamplingMetadata":
        assert num_reqs > 0
        temperature = torch.zeros(num_reqs, dtype=torch.float32, device=device)
        temperature[0] = 0.5
        # TODO(woosuk): Use top-p and top-k for dummy sampler.
        # Currently, they are disabled because of memory usage.
        top_p = None
        top_k = None
        seeds = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        pos = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        max_num_logprobs = 20

        return cls(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seeds=seeds,
            pos=pos,
            max_num_logprobs=max_num_logprobs,
        )


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
        self.extra_data: dict[str, ExtraData] = {}

        self.prompt_len = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.prefill_token_ids = np.zeros(
            (self.max_num_reqs, self.max_model_len),
            dtype=np.int32,
        )
        self.prefill_len = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        self.num_tokens = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.num_computed_tokens = np.zeros(self.max_num_reqs, dtype=np.int32)

        # Last sampled tokens.
        self.last_sampled_tokens = torch.zeros(
            self.max_num_reqs,
            1,
            dtype=torch.int64,
            device=device,
        )

        # LoRA.
        self.lora_ids = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.lora_ids.fill(NO_LORA_ID)

        # Sampling parameters.
        self.temperature = self._make_param(self.max_num_reqs, torch.float32)
        self.top_p = self._make_param(self.max_num_reqs, torch.float32)
        self.top_k = self._make_param(self.max_num_reqs, torch.int32)
        self.seeds = self._make_param(self.max_num_reqs, torch.int64)

        self.num_logprobs = np.empty(self.max_num_reqs, dtype=np.int32)
        # -1 means no logprobs are requested.
        self.num_logprobs.fill(-1)
        self.needs_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=bool)

    def _make_param(self, size: int, dtype: torch.dtype) -> "Param":
        return Param(size, dtype=dtype, device=self.device, pin_memory=self.pin_memory)

    def _make_buffer(self, size: int, dtype: torch.dtype) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            size, dtype=dtype, device=self.device, pin_memory=self.pin_memory
        )

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    def add_request(
        self,
        req_id: str,
        prompt_len: int,
        prefill_token_ids: list[int],
        num_computed_tokens: int,
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None,
    ) -> None:
        assert len(self.free_indices) > 0, "No free indices"
        req_idx = self.free_indices.pop()
        self.req_id_to_index[req_id] = req_idx
        self.index_to_req_id[req_idx] = req_id
        self.extra_data[req_id] = ExtraData(lora_request)

        self.prompt_len[req_idx] = prompt_len
        prefill_len = len(prefill_token_ids)
        assert prefill_len >= prompt_len, (
            f"prefill_len {prefill_len} < prompt_len {prompt_len}"
        )
        self.prefill_len.np[req_idx] = prefill_len
        self.prefill_token_ids[req_idx, :prefill_len] = prefill_token_ids
        self.num_tokens[req_idx] = prefill_len
        self.num_computed_tokens[req_idx] = num_computed_tokens

        if lora_request is not None:
            self.lora_ids[req_idx] = lora_request.lora_int_id
        else:
            self.lora_ids[req_idx] = NO_LORA_ID

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
        self.extra_data.pop(req_id, None)
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
        max_num_logprobs = int(np.max(num_logprobs))
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

    def make_lora_inputs(
        self,
        req_ids: list[str],
        idx_mapping: np.ndarray,
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]:
        lora_ids = self.lora_ids[idx_mapping]
        prompt_lora_mapping = tuple(lora_ids)
        token_lora_mapping = tuple(lora_ids.repeat(num_scheduled_tokens))

        active_lora_requests: set[LoRARequest] = set()
        for req_id in req_ids:
            lora_request = self.extra_data[req_id].lora_request
            if lora_request is not None:
                active_lora_requests.add(lora_request)
        return prompt_lora_mapping, token_lora_mapping, active_lora_requests


class Param:
    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
    ):
        self.buffer = CpuGpuBuffer(
            size,
            dtype=dtype,
            device=device,
            pin_memory=pin_memory,
        )
        self.np = np.zeros_like(self.buffer.np)

    def copy_np_to_gpu(self, x: np.ndarray) -> torch.Tensor:
        n = x.shape[0]
        self.buffer.np[:n] = x
        return self.buffer.copy_to_gpu(n)


@dataclass
class ExtraData:
    lora_request: LoRARequest | None
    in_progress_prompt_logprobs: list[LogprobsTensors] = field(default_factory=list)
