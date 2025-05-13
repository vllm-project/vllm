# SPDX-License-Identifier: Apache-2.0
# Datastructures defining an input batch

from dataclasses import dataclass
from typing import Any, Optional, cast

import jax
import jax.numpy as jnp

from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils import swap_dict_values
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.block_table import BlockTable

_SAMPLING_EPS = 1e-5


@dataclass
class CachedRequestState:

    req_id: str
    prompt_token_ids: list[int]
    mm_inputs: list[MultiModalKwargs]
    mm_positions: list[PlaceholderRange]
    sampling_params: SamplingParams
    generator: Any

    block_ids: list[int]
    num_computed_tokens: int
    output_token_ids: list[int]

    mrope_positions: Optional[jax.Array] = None
    mrope_position_delta: Optional[int] = None

    lora_request: Optional[LoRARequest] = None

    def __post_init__(self):
        self.num_prompt_tokens = len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self.output_token_ids)

    def get_token_id(self, idx: int) -> int:
        if idx < self.num_prompt_tokens:
            return self.prompt_token_ids[idx]
        else:
            return self.output_token_ids[idx - self.num_prompt_tokens]


class InputBatch:

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        device: jax.Device,
        pin_memory: bool,
        vocab_size: int,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device
        self.pin_memory = pin_memory
        self.vocab_size = vocab_size

        self._req_ids: list[Optional[str]] = []
        self.req_id_to_index: dict[str, int] = {}

        self.token_ids_cpu = jnp.zeros(
            (max_num_reqs, max_model_len),
            dtype=jnp.int32,
        )
        self.num_tokens = jnp.zeros(max_num_reqs, dtype=jnp.int32)
        self.num_tokens_no_spec = jnp.zeros(max_num_reqs, dtype=jnp.int32)
        self.num_prompt_tokens = jnp.zeros(max_num_reqs, dtype=jnp.int32)
        self.num_computed_tokens_cpu = jnp.zeros(
            (max_num_reqs, ),
            dtype=jnp.int32,
        )

        self.block_table = BlockTable(
            max_num_reqs=max_num_reqs,
            max_num_blocks_per_req=max_num_blocks_per_req,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=pin_memory,
            device=device,
        )

        self.temperature = jax.device_put(
            jnp.empty((max_num_reqs, ), dtype=jnp.float32), self.device)
        self.temperature_cpu = jnp.empty((max_num_reqs, ), dtype=jnp.float32)
        self.greedy_reqs: set[str] = set()
        self.random_reqs: set[str] = set()

        self.top_p = jax.device_put(
            jnp.empty((max_num_reqs, ), dtype=jnp.float32), self.device)
        self.top_p_cpu = jnp.empty((max_num_reqs, ), dtype=jnp.float32)
        self.top_p_reqs: set[str] = set()

        self.top_k = jax.device_put(
            jnp.empty((max_num_reqs, ), dtype=jnp.int32), self.device)
        self.top_k_cpu = jnp.empty((max_num_reqs, ), dtype=jnp.int32)
        self.top_k_reqs: set[str] = set()

        self.min_p = jax.device_put(
            jnp.empty((max_num_reqs, ), dtype=jnp.float32), self.device)
        self.min_p_cpu = jnp.empty((max_num_reqs, ), dtype=jnp.float32)
        self.min_p_reqs: set[str] = set()

        self.frequency_penalties = jax.device_put(
            jnp.empty((max_num_reqs, ), dtype=jnp.float32), self.device)
        self.frequency_penalties_cpu = jnp.empty((max_num_reqs, ),
                                                 dtype=jnp.float32)
        self.frequency_penalties_reqs: set[str] = set()

        self.presence_penalties = jax.device_put(
            jnp.empty((max_num_reqs, ), dtype=jnp.float32), self.device)
        self.presence_penalties_cpu = jnp.empty((max_num_reqs, ),
                                                dtype=jnp.float32)
        self.presence_penalties_reqs: set[str] = set()

        self.repetition_penalties = jax.device_put(
            jnp.empty((max_num_reqs, ), dtype=jnp.float32), self.device)
        self.repetition_penalties_cpu = jnp.empty((max_num_reqs, ),
                                                  dtype=jnp.float32)
        self.repetition_penalties_reqs: set[str] = set()

        self.min_tokens: dict[int, tuple[int, set[int]]] = {}

        self.request_lora_mapping = jnp.zeros((self.max_num_reqs, ),
                                              dtype=jnp.int32)
        self.lora_id_to_request_ids: dict[int, set[str]] = {}
        self.lora_id_to_lora_request: dict[int, LoRARequest] = {}

        self.generators: dict[int, jax.random.PRNGKeyArray] = {}

        self.num_logprobs: dict[str, int] = {}
        self.num_prompt_logprobs: dict[str, int] = {}

        self.in_progress_prompt_logprobs_cpu: dict[str, LogprobsTensors] = {}

        self.logit_bias: list[Optional[dict[int,
                                            float]]] = [None] * max_num_reqs
        self.has_allowed_token_ids: set[str] = set()
        self.allowed_token_ids_mask: Optional[jax.Array] = None
        self.allowed_token_ids_mask_cpu: Optional[jax.Array] = None

        self.bad_words_token_ids: dict[int, list[list[int]]] = {}
        self.req_output_token_ids: list[Optional[list[int]]] = []
        self.sampling_metadata = self._make_sampling_metadata()

    @property
    def req_ids(self) -> list[str]:
        return cast(list[str], self._req_ids)

    def add_request(
        self,
        request: "CachedRequestState",
        req_index: Optional[int] = None,
    ) -> None:
        if req_index is None:
            req_index = self.num_reqs
        assert req_index < self.max_num_reqs

        req_id = request.req_id
        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
        else:
            self._req_ids[req_index] = req_id
            self.req_output_token_ids[req_index] = request.output_token_ids

        self.req_id_to_index[req_id] = req_index

        num_prompt_tokens = len(request.prompt_token_ids)
        self.num_prompt_tokens = self.num_prompt_tokens.at[req_index].set(
            num_prompt_tokens)
        self.token_ids_cpu = self.token_ids_cpu.at[
            req_index, :num_prompt_tokens].set(
                jnp.array(request.prompt_token_ids, dtype=jnp.int32))
        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        if request.output_token_ids:
            self.token_ids_cpu = self.token_ids_cpu.at[
                req_index, start_idx:end_idx].set(
                    jnp.array(request.output_token_ids, dtype=jnp.int32))

        self.num_tokens = self.num_tokens.at[req_index].set(request.num_tokens)
        self.num_tokens_no_spec = self.num_tokens_no_spec.at[req_index].set(
            request.num_tokens)

        self.num_computed_tokens_cpu = self.num_computed_tokens_cpu.at[
            req_index].set(request.num_computed_tokens)
        self.block_table.add_row(request.block_ids, req_index)

        sampling_params = request.sampling_params
        if sampling_params.sampling_type == SamplingType.GREEDY:
            self.temperature_cpu = self.temperature_cpu.at[req_index].set(-1.0)
            self.greedy_reqs.add(req_id)
        else:
            self.temperature_cpu = self.temperature_cpu.at[req_index].set(
                sampling_params.temperature)
            self.random_reqs.add(req_id)

        self.top_p_cpu = self.top_p_cpu.at[req_index].set(
            sampling_params.top_p)
        if sampling_params.top_p < 1:
            self.top_p_reqs.add(req_id)
        top_k = sampling_params.top_k
        if 0 < top_k < self.vocab_size:
            self.top_k_reqs.add(req_id)
        else:
            top_k = self.vocab_size
        self.top_k_cpu = self.top_k_cpu.at[req_index].set(top_k)
        self.min_p_cpu = self.min_p_cpu.at[req_index].set(
            sampling_params.min_p)
        if sampling_params.min_p > _SAMPLING_EPS:
            self.min_p_reqs.add(req_id)

        self.frequency_penalties_cpu = self.frequency_penalties_cpu.at[
            req_index].set(sampling_params.frequency_penalty)
        if sampling_params.frequency_penalty != 0.0:
            self.frequency_penalties_reqs.add(req_id)

        self.presence_penalties_cpu = self.presence_penalties_cpu.at[
            req_index].set(sampling_params.presence_penalty)
        if sampling_params.presence_penalty != 0.0:
            self.presence_penalties_reqs.add(req_id)

        self.repetition_penalties_cpu = self.repetition_penalties_cpu.at[
            req_index].set(sampling_params.repetition_penalty)
        if sampling_params.repetition_penalty != 1.0:
            self.repetition_penalties_reqs.add(req_id)

        if sampling_params.min_tokens:
            self.min_tokens[req_index] = (sampling_params.min_tokens,
                                          sampling_params.all_stop_token_ids)

        if request.generator is not None:
            self.generators[req_index] = request.generator

        if sampling_params.logprobs is not None:
            self.num_logprobs[req_id] = sampling_params.logprobs
        if sampling_params.prompt_logprobs is not None:
            self.num_prompt_logprobs[req_id] = sampling_params.prompt_logprobs
        if sampling_params.logit_bias is not None:
            self.logit_bias[req_index] = sampling_params.logit_bias

        if sampling_params.allowed_token_ids:
            self.has_allowed_token_ids.add(req_id)
            if self.allowed_token_ids_mask_cpu is None:
                self.allowed_token_ids_mask_cpu = jnp.zeros(
                    (self.max_num_reqs, self.vocab_size), dtype=jnp.bool_)
                self.allowed_token_ids_mask = jax.device_put(
                    jnp.zeros((self.max_num_reqs, self.vocab_size),
                              dtype=jnp.bool_), self.device)

            current_mask_row = jnp.ones(self.vocab_size, dtype=jnp.bool_)
            allowed_indices = jnp.array(list(
                sampling_params.allowed_token_ids),
                                        dtype=jnp.int32)
            current_mask_row = current_mask_row.at[allowed_indices].set(False)
            self.allowed_token_ids_mask_cpu = self.allowed_token_ids_mask_cpu.at[
                req_index].set(current_mask_row)

        if sampling_params.bad_words_token_ids:
            self.bad_words_token_ids[
                req_index] = sampling_params.bad_words_token_ids

        if request.lora_request:
            lora_id = request.lora_request.lora_int_id
            if lora_id not in self.lora_id_to_request_ids:
                self.lora_id_to_request_ids[lora_id] = set()

            self.request_lora_mapping = self.request_lora_mapping.at[
                req_index].set(lora_id)
            self.lora_id_to_request_ids[lora_id].add(request.req_id)
            self.lora_id_to_lora_request[lora_id] = request.lora_request
        else:
            self.request_lora_mapping = self.request_lora_mapping.at[
                req_index].set(0)

    def remove_request(self, req_id: str) -> Optional[int]:
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        self._req_ids[req_index] = None
        self.req_output_token_ids[req_index] = None

        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)
        self.min_p_reqs.discard(req_id)
        self.min_tokens.pop(req_index, None)
        self.frequency_penalties_reqs.discard(req_id)
        self.presence_penalties_reqs.discard(req_id)
        self.repetition_penalties_reqs.discard(req_id)
        self.generators.pop(req_index, None)
        self.num_logprobs.pop(req_id, None)
        self.num_prompt_logprobs.pop(req_id, None)
        self.in_progress_prompt_logprobs_cpu.pop(req_id, None)

        lora_id = self.request_lora_mapping[req_index].item(
        )  # .item() to get Python int
        if lora_id != 0:
            self.lora_id_to_request_ids[lora_id].discard(req_id)
            if len(self.lora_id_to_request_ids[lora_id]) == 0:
                self.lora_id_to_request_ids.pop(lora_id)
                self.lora_id_to_lora_request.pop(lora_id)
            self.request_lora_mapping = self.request_lora_mapping.at[
                req_index].set(0)

        self.logit_bias[req_index] = None
        self.has_allowed_token_ids.discard(req_id)
        if self.allowed_token_ids_mask_cpu is not None:
            self.allowed_token_ids_mask_cpu = self.allowed_token_ids_mask_cpu.at[
                req_index].set(False)
        self.bad_words_token_ids.pop(req_index, None)
        return req_index

    def swap_states(self, i1: int, i2: int) -> None:
        old_id_i1 = self._req_ids[i1]
        old_id_i2 = self._req_ids[i2]
        self._req_ids[i1], self._req_ids[i2] = self._req_ids[
            i2], self._req_ids[i1]  # noqa
        self.req_output_token_ids[i1], self.req_output_token_ids[
            i2] = self.req_output_token_ids[i2], self.req_output_token_ids[i1]
        assert old_id_i1 is not None and old_id_i2 is not None
        self.req_id_to_index[old_id_i1], self.req_id_to_index[
            old_id_i2] = self.req_id_to_index[old_id_i2], self.req_id_to_index[
                old_id_i1]

        self.num_tokens = self.num_tokens.at[i1].set(
            self.num_tokens[i2]).at[i2].set(self.num_tokens[i1])
        self.num_tokens_no_spec = self.num_tokens_no_spec.at[i1].set(
            self.num_tokens_no_spec[i2]).at[i2].set(
                self.num_tokens_no_spec[i1])
        self.num_prompt_tokens = self.num_prompt_tokens.at[i1].set(
            self.num_prompt_tokens[i2]).at[i2].set(self.num_prompt_tokens[i1])
        self.num_computed_tokens_cpu = self.num_computed_tokens_cpu.at[i1].set(
            self.num_computed_tokens_cpu[i2]).at[i2].set(
                self.num_computed_tokens_cpu[i1])
        self.temperature_cpu = self.temperature_cpu.at[i1].set(
            self.temperature_cpu[i2]).at[i2].set(self.temperature_cpu[i1])
        self.top_p_cpu = self.top_p_cpu.at[i1].set(
            self.top_p_cpu[i2]).at[i2].set(self.top_p_cpu[i1])
        self.top_k_cpu = self.top_k_cpu.at[i1].set(
            self.top_k_cpu[i2]).at[i2].set(self.top_k_cpu[i1])
        self.frequency_penalties_cpu = self.frequency_penalties_cpu.at[i1].set(
            self.frequency_penalties_cpu[i2]).at[i2].set(
                self.frequency_penalties_cpu[i1])
        self.presence_penalties_cpu = self.presence_penalties_cpu.at[i1].set(
            self.presence_penalties_cpu[i2]).at[i2].set(
                self.presence_penalties_cpu[i1])
        self.repetition_penalties_cpu = self.repetition_penalties_cpu.at[
            i1].set(self.repetition_penalties_cpu[i2]).at[i2].set(
                self.repetition_penalties_cpu[i1])
        self.min_p_cpu = self.min_p_cpu.at[i1].set(
            self.min_p_cpu[i2]).at[i2].set(self.min_p_cpu[i1])

        tmp_row = jnp.copy(self.token_ids_cpu[i1, :])
        self.token_ids_cpu = self.token_ids_cpu.at[i1, :].set(
            self.token_ids_cpu[i2, :])
        self.token_ids_cpu = self.token_ids_cpu.at[i2, :].set(tmp_row)

        swap_dict_values(self.generators, i1, i2)
        swap_dict_values(self.min_tokens, i1, i2)
        swap_dict_values(self.bad_words_token_ids, i1, i2)

        self.request_lora_mapping = self.request_lora_mapping.at[i1].set(
            self.request_lora_mapping[i2]).at[i2].set(
                self.request_lora_mapping[i1])
        self.logit_bias[i1], self.logit_bias[i2] = self.logit_bias[
            i2], self.logit_bias[i1]

        if self.allowed_token_ids_mask_cpu is not None:
            tmp_mask_row = jnp.copy(self.allowed_token_ids_mask_cpu[i1])
            self.allowed_token_ids_mask_cpu = self.allowed_token_ids_mask_cpu.at[
                i1].set(self.allowed_token_ids_mask_cpu[i2])
            self.allowed_token_ids_mask_cpu = self.allowed_token_ids_mask_cpu.at[
                i2].set(tmp_mask_row)
        self.block_table.swap_row(i1, i2)

    def condense(self, empty_req_indices: list[int]) -> None:
        num_reqs = self.num_reqs
        if num_reqs == 0:
            self._req_ids.clear()
            self.req_output_token_ids.clear()
            return

        last_req_index = num_reqs + len(empty_req_indices) - 1

        # Convert lists to JAX arrays for batch updates if possible, or iterate carefully
        # For now, direct porting of logic with JAX array updates

        current_token_ids_cpu = self.token_ids_cpu
        current_num_tokens = self.num_tokens
        current_num_tokens_no_spec = self.num_tokens_no_spec
        current_num_prompt_tokens = self.num_prompt_tokens
        current_num_computed_tokens_cpu = self.num_computed_tokens_cpu
        current_temperature_cpu = self.temperature_cpu
        current_top_p_cpu = self.top_p_cpu
        current_top_k_cpu = self.top_k_cpu
        current_frequency_penalties_cpu = self.frequency_penalties_cpu
        current_presence_penalties_cpu = self.presence_penalties_cpu
        current_repetition_penalties_cpu = self.repetition_penalties_cpu
        current_min_p_cpu = self.min_p_cpu
        current_request_lora_mapping = self.request_lora_mapping
        current_allowed_token_ids_mask_cpu = self.allowed_token_ids_mask_cpu

        while empty_req_indices:
            while last_req_index in empty_req_indices:
                last_req_index -= 1

            empty_index = empty_req_indices.pop()
            if empty_index >= last_req_index:
                break

            req_id = self._req_ids[last_req_index]
            output_token_ids = self.req_output_token_ids[last_req_index]
            assert req_id is not None
            self._req_ids[empty_index] = req_id
            self._req_ids[last_req_index] = None
            self.req_output_token_ids[empty_index] = output_token_ids
            self.req_output_token_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index

            num_tokens_val = current_num_tokens[last_req_index].item()
            current_token_ids_cpu = current_token_ids_cpu.at[
                empty_index, :num_tokens_val].set(
                    current_token_ids_cpu[last_req_index, :num_tokens_val])
            current_num_tokens = current_num_tokens.at[empty_index].set(
                num_tokens_val)
            current_num_tokens_no_spec = current_num_tokens_no_spec.at[
                empty_index].set(current_num_tokens_no_spec[last_req_index])
            current_num_prompt_tokens = current_num_prompt_tokens.at[
                empty_index].set(current_num_prompt_tokens[last_req_index])
            current_num_computed_tokens_cpu = current_num_computed_tokens_cpu.at[
                empty_index].set(
                    current_num_computed_tokens_cpu[last_req_index])

            self.block_table.move_row(last_req_index, empty_index)

            current_temperature_cpu = current_temperature_cpu.at[
                empty_index].set(current_temperature_cpu[last_req_index])
            current_top_p_cpu = current_top_p_cpu.at[empty_index].set(
                current_top_p_cpu[last_req_index])
            current_top_k_cpu = current_top_k_cpu.at[empty_index].set(
                current_top_k_cpu[last_req_index])
            current_frequency_penalties_cpu = current_frequency_penalties_cpu.at[
                empty_index].set(
                    current_frequency_penalties_cpu[last_req_index])
            current_presence_penalties_cpu = current_presence_penalties_cpu.at[
                empty_index].set(
                    current_presence_penalties_cpu[last_req_index])
            current_repetition_penalties_cpu = current_repetition_penalties_cpu.at[
                empty_index].set(
                    current_repetition_penalties_cpu[last_req_index])
            current_min_p_cpu = current_min_p_cpu.at[empty_index].set(
                current_min_p_cpu[last_req_index])

            generator = self.generators.pop(last_req_index, None)
            if generator is not None:
                self.generators[empty_index] = generator

            min_token = self.min_tokens.pop(last_req_index, None)
            if min_token is not None:
                self.min_tokens[empty_index] = min_token

            current_request_lora_mapping = current_request_lora_mapping.at[
                empty_index].set(current_request_lora_mapping[last_req_index])
            self.logit_bias[empty_index] = self.logit_bias[last_req_index]

            if current_allowed_token_ids_mask_cpu is not None:
                current_allowed_token_ids_mask_cpu = current_allowed_token_ids_mask_cpu.at[
                    empty_index].set(
                        current_allowed_token_ids_mask_cpu[last_req_index])

            bad_words_token_ids = self.bad_words_token_ids.pop(
                last_req_index, None)
            if bad_words_token_ids is not None:
                self.bad_words_token_ids[empty_index] = bad_words_token_ids
            last_req_index -= 1

        self.token_ids_cpu = current_token_ids_cpu
        self.num_tokens = current_num_tokens
        self.num_tokens_no_spec = current_num_tokens_no_spec
        self.num_prompt_tokens = current_num_prompt_tokens
        self.num_computed_tokens_cpu = current_num_computed_tokens_cpu
        self.temperature_cpu = current_temperature_cpu
        self.top_p_cpu = current_top_p_cpu
        self.top_k_cpu = current_top_k_cpu
        self.frequency_penalties_cpu = current_frequency_penalties_cpu
        self.presence_penalties_cpu = current_presence_penalties_cpu
        self.repetition_penalties_cpu = current_repetition_penalties_cpu
        self.min_p_cpu = current_min_p_cpu
        self.request_lora_mapping = current_request_lora_mapping
        self.allowed_token_ids_mask_cpu = current_allowed_token_ids_mask_cpu

        del self._req_ids[self.num_reqs:]
        del self.req_output_token_ids[self.num_reqs:]

    def refresh_sampling_metadata(self):
        self.sampling_metadata = self._make_sampling_metadata()

    def _make_sampling_metadata(self) -> SamplingMetadata:
        num_reqs = self.num_reqs
        temperature_for_metadata: Optional[jax.Array] = None
        top_p_for_metadata: Optional[jax.Array] = None
        top_k_for_metadata: Optional[jax.Array] = None
        min_p_for_metadata: Optional[jax.Array] = None
        prompt_token_ids: Optional[jax.Array] = None
        allowed_token_ids_mask_for_metadata: Optional[jax.Array] = None

        if not self.all_greedy:
            self.temperature = self.temperature.at[:num_reqs].set(
                self.temperature_cpu[:num_reqs])
            temperature_for_metadata = self.temperature[:num_reqs]

        if not self.no_top_p:
            self.top_p = self.top_p.at[:num_reqs].set(
                self.top_p_cpu[:num_reqs])
            top_p_for_metadata = self.top_p[:num_reqs]
        if not self.no_top_k:
            self.top_k = self.top_k.at[:num_reqs].set(
                self.top_k_cpu[:num_reqs])
            top_k_for_metadata = self.top_k[:num_reqs]
        if not self.no_min_p:
            self.min_p = self.min_p.at[:num_reqs].set(
                self.min_p_cpu[:num_reqs])
            min_p_for_metadata = self.min_p[:num_reqs]

        if not self.no_penalties:
            self.frequency_penalties = self.frequency_penalties.at[:num_reqs].set(
                self.frequency_penalties_cpu[:num_reqs])
            self.presence_penalties = self.presence_penalties.at[:num_reqs].set(
                self.presence_penalties_cpu[:num_reqs])
            self.repetition_penalties = self.repetition_penalties.at[:num_reqs].set(
                self.repetition_penalties_cpu[:num_reqs])
            prompt_token_ids = self._make_prompt_token_ids_tensor()

        if not self.no_allowed_token_ids:
            assert self.allowed_token_ids_mask is not None and self.allowed_token_ids_mask_cpu is not None
            self.allowed_token_ids_mask = self.allowed_token_ids_mask.at[:num_reqs].set(
                self.allowed_token_ids_mask_cpu[:num_reqs])
            allowed_token_ids_mask_for_metadata = self.allowed_token_ids_mask[:
                                                                              num_reqs]

        return SamplingMetadata(
            temperature=temperature_for_metadata,
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=top_p_for_metadata,
            top_k=top_k_for_metadata,
            min_p=min_p_for_metadata,
            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=self.frequency_penalties[:num_reqs]
            if not self.no_penalties else None,
            presence_penalties=self.presence_penalties[:num_reqs]
            if not self.no_penalties else None,
            repetition_penalties=self.repetition_penalties[:num_reqs]
            if not self.no_penalties else None,
            output_token_ids=cast(list[list[int]], self.req_output_token_ids),
            min_tokens=self.min_tokens,
            no_penalties=self.no_penalties,
            logit_bias=self.logit_bias[:num_reqs],
            allowed_token_ids_mask=allowed_token_ids_mask_for_metadata,
            bad_words_token_ids=self.bad_words_token_ids,
        )

    def _make_prompt_token_ids_tensor(self) -> jax.Array:
        if self.num_reqs == 0:
            return jax.device_put(jnp.empty((0, 0), dtype=jnp.int64),
                                  self.device)

        max_prompt_len = self.num_prompt_tokens[:self.num_reqs].max().item()
        if max_prompt_len == 0:  # Handle case where all prompts are empty
            return jax.device_put(
                jnp.empty((self.num_reqs, 0), dtype=jnp.int64), self.device)

        prompt_token_ids_host = jnp.empty(
            (self.num_reqs, max_prompt_len),
            dtype=jnp.int64,
        )

        prompt_token_ids_host = prompt_token_ids_host.at[:, :].set(
            self.token_ids_cpu[:self.num_reqs, :max_prompt_len].astype(
                jnp.int64))

        for i in range(self.num_reqs):
            num_prompt_tokens_i = self.num_prompt_tokens[i].item()
            if num_prompt_tokens_i < max_prompt_len:
                prompt_token_ids_host = prompt_token_ids_host.at[
                    i, num_prompt_tokens_i:].set(self.vocab_size)

        return jax.device_put(prompt_token_ids_host, self.device)

    def make_lora_inputs(
        self,
        num_scheduled_tokens: jax.Array  # Changed from np.ndarray
    ) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]:
        req_lora_mapping_slice = self.request_lora_mapping[:self.num_reqs]
        prompt_lora_mapping = tuple(
            map(int, req_lora_mapping_slice
                ))  # Convert JAX array elements to Python int for tuple

        # jnp.repeat requires repeats to be an int or sequence of ints.
        # num_scheduled_tokens is a JAX array of token counts per request.
        # Ensure num_scheduled_tokens are Python integers for repeat if it's small or convert to list
        if num_scheduled_tokens.size > 0:  # Check if there are any tokens scheduled
            # Convert JAX array elements to Python int for jnp.repeat if needed, or ensure dtype is int.
            repeats_for_tokens = num_scheduled_tokens.astype(
                jnp.int32)  # Ensure integer type
            token_lora_mapping_array = jnp.repeat(
                req_lora_mapping_slice,
                repeats_for_tokens,
                total_repeat_length=jnp.sum(repeats_for_tokens).item())
            token_lora_mapping = tuple(map(int, token_lora_mapping_array))
        else:
            token_lora_mapping = tuple()

        active_lora_requests: set[LoRARequest] = set(
            self.lora_id_to_lora_request.values())

        return prompt_lora_mapping, token_lora_mapping, active_lora_requests

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    @property
    def all_greedy(self) -> bool:
        return len(self.random_reqs) == 0

    @property
    def all_random(self) -> bool:
        return len(self.greedy_reqs) == 0

    @property
    def no_top_p(self) -> bool:
        return len(self.top_p_reqs) == 0

    @property
    def no_top_k(self) -> bool:
        return len(self.top_k_reqs) == 0

    @property
    def no_min_p(self) -> bool:
        return len(self.min_p_reqs) == 0

    @property
    def no_penalties(self) -> bool:
        return (len(self.presence_penalties_reqs) == 0
                and len(self.frequency_penalties_reqs) == 0
                and len(self.repetition_penalties_reqs) == 0)

    @property
    def max_num_logprobs(self) -> Optional[int]:
        return max(self.num_logprobs.values()) if self.num_logprobs else None

    @property
    def no_prompt_logprob(self) -> bool:
        return not self.num_prompt_logprobs

    @property
    def no_allowed_token_ids(self) -> bool:
        return len(self.has_allowed_token_ids) == 0
