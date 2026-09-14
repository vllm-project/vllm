# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor

if TYPE_CHECKING:
    from vllm.config import VllmConfig

WHISPER_TIMESTAMP_RULES_KEY = "_whisper_timestamp_rules"


@dataclass(frozen=True)
class _WhisperTimestampRules:
    eos_token_id: int
    no_timestamps_token_id: int
    max_initial_timestamp_index: int | None
    begin_index: int

    @property
    def timestamp_begin(self) -> int:
        return self.no_timestamps_token_id + 1

    @classmethod
    def from_sampling_params(
        cls, sampling_params: SamplingParams
    ) -> "_WhisperTimestampRules | None":
        extra_args = sampling_params.extra_args
        values: Any = extra_args and extra_args.get(WHISPER_TIMESTAMP_RULES_KEY)
        if values is None:
            return None
        if not isinstance(values, dict):
            raise ValueError(f"{WHISPER_TIMESTAMP_RULES_KEY} must be a dictionary")

        return cls(
            eos_token_id=values["eos_token_id"],
            no_timestamps_token_id=values["no_timestamps_token_id"],
            max_initial_timestamp_index=values["max_initial_timestamp_index"],
            begin_index=values["begin_index"],
        )


@dataclass(frozen=True)
class _WhisperTimestampRequest:
    rules: _WhisperTimestampRules
    prompt_token_ids: Sequence[int]
    output_token_ids: Sequence[int]

    def sampled_tokens(self) -> Sequence[int]:
        prompt_tokens = self.prompt_token_ids[self.rules.begin_index :]
        if not prompt_tokens:
            return self.output_token_ids
        if not self.output_token_ids:
            return prompt_tokens
        return (*prompt_tokens, *self.output_token_ids)


class WhisperTimestampLogitsProcessor(LogitsProcessor):
    """Enforce Whisper's segment timestamp decoding rules per request."""

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ) -> None:
        self.requests: dict[int, _WhisperTimestampRequest] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    @staticmethod
    def _new_request(
        sampling_params: SamplingParams,
        prompt_token_ids: list[int] | None,
        output_token_ids: list[int],
    ) -> _WhisperTimestampRequest | None:
        if (
            rules := _WhisperTimestampRules.from_sampling_params(sampling_params)
        ) is None:
            return None
        return _WhisperTimestampRequest(
            rules=rules,
            prompt_token_ids=prompt_token_ids or (),
            output_token_ids=output_token_ids,
        )

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        process_dict_updates(self.requests, batch_update, self._new_request)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for request_index, request in self.requests.items():
            self._apply_rules(logits[request_index], request)
        return logits

    @staticmethod
    def _apply_rules(scores: torch.Tensor, request: _WhisperTimestampRequest) -> None:
        rules = request.rules
        timestamp_begin = rules.timestamp_begin
        if timestamp_begin >= scores.shape[-1]:
            raise ValueError(
                f"Whisper timestamp token {timestamp_begin} is outside vocabulary "
                f"of size {scores.shape[-1]}"
            )

        scores[rules.no_timestamps_token_id] = -float("inf")
        sampled_tokens = request.sampled_tokens()
        last_was_timestamp = bool(
            sampled_tokens and sampled_tokens[-1] >= timestamp_begin
        )
        penultimate_was_timestamp = (
            len(sampled_tokens) < 2 or sampled_tokens[-2] >= timestamp_begin
        )

        if last_was_timestamp:
            if penultimate_was_timestamp:
                scores[timestamp_begin:] = -float("inf")
            else:
                scores[: rules.eos_token_id] = -float("inf")

        timestamps = [
            token_id for token_id in sampled_tokens if token_id >= timestamp_begin
        ]
        if timestamps:
            timestamp_last = timestamps[-1]
            if not (last_was_timestamp and not penultimate_was_timestamp):
                timestamp_last += 1
            scores[timestamp_begin:timestamp_last] = -float("inf")

        if not sampled_tokens:
            scores[:timestamp_begin] = -float("inf")
            if rules.max_initial_timestamp_index is not None:
                last_allowed = timestamp_begin + rules.max_initial_timestamp_index
                scores[last_allowed + 1 :] = -float("inf")

        logprobs = torch.nn.functional.log_softmax(scores.float(), dim=-1)
        timestamp_logprob = logprobs[timestamp_begin:].logsumexp(dim=-1)
        max_text_token_logprob = logprobs[:timestamp_begin].max()
        force_timestamp = timestamp_logprob > max_text_token_logprob
        scores[:timestamp_begin].masked_fill_(force_timestamp, -float("inf"))
